//! The evaluation order the expander derives for an expanded model.
//!
//! The explicit sweep evaluates each step once per evaluation point, so a
//! step that reads a value its own sweep recomputes later reads the
//! previous point's number. A type author cannot control that across
//! instances, which is why the expander orders the sweep itself (plan KTD14):
//!
//! - a step comes after every step whose result it reads;
//! - steps that do not depend on each other keep their positional order;
//! - the steps of a cycle (a strongly connected part of the read relation)
//!   stay contiguous for the engine to condense (a linear block is solved
//!   simultaneously, a flow network through its distribution operators),
//!   and inside it each step still follows what it reads, but for the reads
//!   that close the cycle: a depth-first walk from the positionally first
//!   member decides which ones. Positional order inside a cycle would
//!   sweep a component's equations before its distribution operators and
//!   hand every reader of an allocation the previous point's value.

use raichu_expr::AttrRef;
use raichu_model::{strongly_connected_components, Model, ModelError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// The derived evaluation order of `model`, or `None` when the model has no
/// explicit step (a purely discrete model needs no order).
///
/// # Errors
/// The structural errors [`Model::evaluation_step_reads`] raises.
pub fn derive_evaluation_order(model: &Model) -> Result<Option<Vec<AttrRef>>, ModelError> {
    let steps = model.evaluation_step_reads()?;
    if steps.is_empty() {
        return Ok(None);
    }
    let mut writer: HashMap<&AttrRef, usize> = HashMap::new();
    for (index, step) in steps.iter().enumerate() {
        for written in &step.writes {
            writer.insert(written, index);
        }
    }
    let depends_on: Vec<Vec<usize>> = steps
        .iter()
        .map(|step| {
            let mut deps: Vec<usize> = step
                .reads
                .iter()
                .filter_map(|read| writer.get(read).copied())
                .collect();
            deps.sort_unstable();
            deps.dedup();
            deps
        })
        .collect();

    let parts: Vec<Vec<usize>> = strongly_connected_components(&depends_on)
        .into_iter()
        .map(|mut members| {
            members.sort_unstable();
            within_cycle(&members, &depends_on)
        })
        .collect();
    let mut part_of = vec![0; steps.len()];
    for (part, members) in parts.iter().enumerate() {
        for &member in members {
            part_of[member] = part;
        }
    }
    let mut waiting = vec![0usize; parts.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); parts.len()];
    for (step, deps) in depends_on.iter().enumerate() {
        let mut upstream: Vec<usize> = deps
            .iter()
            .map(|&d| part_of[d])
            .filter(|&p| p != part_of[step])
            .collect();
        upstream.sort_unstable();
        upstream.dedup();
        for p in upstream {
            if !readers[p].contains(&part_of[step]) {
                readers[p].push(part_of[step]);
                waiting[part_of[step]] += 1;
            }
        }
    }

    // Kahn over the condensation, releasing the ready part whose first
    // member comes first positionally: declaration order breaks every tie.
    let first: Vec<usize> = parts
        .iter()
        .map(|members| members.iter().copied().min().unwrap_or(usize::MAX))
        .collect();
    let mut ready: BinaryHeap<Reverse<(usize, usize)>> = (0..parts.len())
        .filter(|&part| waiting[part] == 0)
        .map(|part| Reverse((first[part], part)))
        .collect();
    let mut order = Vec::with_capacity(steps.len());
    while let Some(Reverse((_, part))) = ready.pop() {
        for &member in &parts[part] {
            order.push(AttrRef {
                component: steps[member].component.clone(),
                attribute: steps[member].step.clone(),
            });
        }
        for &reader in &readers[part] {
            waiting[reader] -= 1;
            if waiting[reader] == 0 {
                ready.push(Reverse((first[reader], reader)));
            }
        }
    }
    Ok(Some(order))
}

/// The members of one strongly connected part, in sweep order: a
/// depth-first walk over what each member reads, from the positionally
/// first member, emitting a member after everything it reads. Every read
/// inside the part is then fresh except the reads that close a cycle,
/// which a cycle cannot avoid.
fn within_cycle(members: &[usize], depends_on: &[Vec<usize>]) -> Vec<usize> {
    if members.len() < 2 {
        return members.to_vec();
    }
    let inside: HashSet<usize> = members.iter().copied().collect();
    let mut visited = HashSet::new();
    let mut out = Vec::with_capacity(members.len());
    for &root in members {
        if !visited.insert(root) {
            continue;
        }
        let mut frames = vec![(root, 0usize)];
        while let Some(&mut (node, ref mut next)) = frames.last_mut() {
            if let Some(&dep) = depends_on[node].get(*next) {
                *next += 1;
                if inside.contains(&dep) && visited.insert(dep) {
                    frames.push((dep, 0));
                }
                continue;
            }
            frames.pop();
            out.push(node);
        }
    }
    out
}
