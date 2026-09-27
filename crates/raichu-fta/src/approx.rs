//! Minimal cut sets under cutoffs, for a module too large for its exact
//! diagram (the approximate engine).
//!
//! The extraction is top-down, in the manner of MOCUS as Rauzy (2003)
//! reworked it: a *partial* cut set is a set of variables already chosen
//! plus a list of gates still to expand; expanding a conjunction adds its
//! arguments, expanding a disjunction branches on them, expanding a vote
//! branches on its first argument being in or out. Partials are explored
//! depth first and pruned by three cutoffs (Rauzy 2020, section 11.3):
//!
//! - an **order** cutoff, on the number of variables already chosen;
//! - a **probability** cutoff, on an upper bound of the partial's
//!   probability;
//! - a **count** cutoff, keeping the most probable cut sets: once more are
//!   found, the least probable is dropped and the probability cutoff
//!   raised to the next one (the threshold "adjusted dynamically").
//!
//! **What RAICHU adds: the neglected mass.** Every cut set the cutoffs lose
//! contains a pruned partial, and the event "that partial occurred" has at
//! most the probability bound that decided the pruning. The sum of those
//! bounds is therefore an upper bound on the probability of everything
//! neglected, and the retained family's upper bound plus that sum is a
//! guaranteed upper bound on the top event, whatever the cutoffs. The
//! bound is valid for monotone (coherent) formulas, which is the only kind
//! this engine accepts.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// An argument of a local gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arg {
    /// A variable (a basic event or a submodule), by level.
    Level(u32),
    /// Another gate of the module, by index.
    Gate(u32),
    /// A constant.
    Const(bool),
}

/// The connective of a local gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    And,
    Or,
    AtLeast(usize),
}

/// A module's formula over its own variables, gates listed children
/// first.
#[derive(Debug, Clone)]
pub(crate) struct LocalFormula {
    pub(crate) gates: Vec<(Kind, Vec<Arg>)>,
    pub(crate) root: u32,
}

/// The cutoffs of an extraction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cutoffs {
    pub(crate) max_order: Option<usize>,
    pub(crate) min_probability: f64,
    pub(crate) max_cut_sets: usize,
    pub(crate) max_expansions: u64,
}

/// What an extraction found.
#[derive(Debug, Clone)]
pub(crate) struct Extracted {
    /// The retained cut sets, each a strictly increasing list of levels;
    /// not necessarily minimal.
    pub(crate) sets: Vec<Vec<u32>>,
    /// An upper bound on the probability of every cut set lost to a
    /// cutoff.
    pub(crate) neglected: f64,
    /// Partials expanded.
    pub(crate) expansions: u64,
    /// Whether any cutoff removed anything.
    pub(crate) truncated: bool,
    /// Whether the count cutoff dropped a found cut set.
    pub(crate) count_reached: bool,
    /// Whether the expansion budget ran out.
    pub(crate) expansions_exhausted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Item {
    Level(u32),
    Gate(u32),
    Vote { gate: u32, start: usize, k: usize },
}

#[derive(Debug, Clone)]
struct Partial {
    levels: Vec<u32>,
    items: Vec<Item>,
    /// The product of the chosen variables' probabilities.
    p: f64,
}

/// Extract the cut sets of `formula` under `cutoffs`, `p[level]` being an
/// upper bound on each variable's probability (its probability for a
/// basic event).
pub(crate) fn extract(formula: &LocalFormula, p: &[f64], cutoffs: &Cutoffs) -> Extracted {
    let supports = Supports::new(formula, p.len());
    let bound = gate_bounds(formula, p, supports.as_ref());
    let item_bound = |item: &Item| -> f64 {
        match *item {
            Item::Level(l) => p[l as usize],
            Item::Gate(g) => bound[g as usize],
            Item::Vote { gate, start, k } => vote_bound(formula, &bound, p, gate, start, k),
        }
    };
    // The probability of a partial is at most that of its chosen variables
    // (independent, so their product) and at most that of any gate still
    // to hold. Gates may share variables, so in general they bound it by
    // their minimum; but items whose supports are disjoint from each other
    // and from the chosen variables are independent of them, and multiply.
    // Dropping an item that overlaps bounds it by 1, which stays valid.
    let partial_bound = |partial: &Partial| -> f64 {
        let mut items: Vec<(f64, &Item)> =
            partial.items.iter().map(|i| (item_bound(i), i)).collect();
        let by_min = items.iter().map(|(b, _)| *b).fold(partial.p, f64::min);
        let Some(supports) = supports.as_ref() else {
            return by_min;
        };
        let mut union = supports.empty();
        for &l in &partial.levels {
            supports.set(&mut union, l);
        }
        items.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut product = partial.p;
        for (b, item) in items {
            if supports.disjoint_item(&union, item, formula) {
                supports.merge_item(&mut union, item, formula);
                product *= b;
            }
        }
        by_min.min(product)
    };

    let mut threshold = cutoffs.min_probability;
    let mut neglected = 0.0;
    let mut truncated = false;
    let mut count_reached = false;
    let mut expansions_exhausted = false;
    let mut expansions = 0u64;
    // Retained sets, with a min-heap on their probability for the count
    // cutoff; a dropped set is marked dead rather than removed.
    let mut found: Vec<(Vec<u32>, f64, bool)> = Vec::new();
    // Every set found, dropped ones included: a vote reaches one set by
    // several branches, and copies must not fill the count cutoff. Sets
    // that turn out non-minimal are removed at the end; they are less
    // probable than the sets they contain, so the count cutoff drops them
    // first.
    let mut seen: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    // Partial states already expanded. A formula that nests redundant
    // votes over the same variables reaches one state by exponentially
    // many paths; its completions were explored (or pruned and counted)
    // the first time, so a repeat is skipped at no cost to the bound. The
    // memo stops growing past a fixed size and the search stays correct,
    // only slower.
    const MAX_VISITED: usize = 5_000_000;
    let mut visited: std::collections::HashSet<(Vec<u32>, Vec<Item>)> =
        std::collections::HashSet::new();
    let mut alive = 0usize;
    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();

    let mut stack = vec![Partial {
        levels: Vec::new(),
        items: vec![Item::Gate(formula.root)],
        p: 1.0,
    }];
    while let Some(mut partial) = stack.pop() {
        let q = partial_bound(&partial);
        let too_long = cutoffs.max_order.is_some_and(|k| partial.levels.len() > k);
        if q < threshold || too_long {
            neglected += q;
            truncated = true;
            continue;
        }
        let mut key_items = partial.items.clone();
        key_items.sort_unstable();
        let key = (partial.levels.clone(), key_items);
        if visited.contains(&key) {
            continue;
        }
        if visited.len() < MAX_VISITED {
            visited.insert(key);
        }
        expansions += 1;
        if expansions > cutoffs.max_expansions {
            // Out of budget: everything still open is neglected, each
            // partial by its own bound, which keeps the bound valid.
            neglected += q;
            neglected += stack.iter().map(&partial_bound).sum::<f64>();
            truncated = true;
            expansions_exhausted = true;
            break;
        }
        let Some(item) = take_item(&mut partial, formula) else {
            // Complete: a cut set, unless already found.
            if !seen.insert(partial.levels.clone()) {
                continue;
            }
            let index = found.len();
            heap.push(Reverse((OrdF64(partial.p), index)));
            found.push((partial.levels, partial.p, true));
            alive += 1;
            if alive > cutoffs.max_cut_sets {
                if let Some(Reverse((OrdF64(pmin), dropped))) = heap.pop() {
                    found[dropped].2 = false;
                    alive -= 1;
                    neglected += pmin;
                    truncated = true;
                    count_reached = true;
                    if let Some(Reverse((OrdF64(next), _))) = heap.peek() {
                        threshold = threshold.max(*next);
                    }
                }
            }
            continue;
        };
        match item {
            Item::Level(l) => {
                if let Err(at) = partial.levels.binary_search(&l) {
                    partial.levels.insert(at, l);
                    partial.p *= p[l as usize];
                }
                stack.push(partial);
            }
            Item::Gate(g) => {
                let (kind, args) = &formula.gates[g as usize];
                match kind {
                    Kind::And => {
                        let mut feasible = true;
                        for arg in args {
                            match *arg {
                                Arg::Level(l) => push_item(&mut partial.items, Item::Level(l)),
                                Arg::Gate(c) => push_item(&mut partial.items, Item::Gate(c)),
                                Arg::Const(true) => {}
                                Arg::Const(false) => feasible = false,
                            }
                        }
                        if feasible {
                            stack.push(partial);
                        }
                    }
                    Kind::Or => {
                        // Branches pushed least probable first, so the
                        // most probable is explored first and fills the
                        // count cutoff with the sets that matter.
                        let mut branches: Vec<(f64, Item)> = Vec::with_capacity(args.len());
                        if args.contains(&Arg::Const(true)) {
                            // The disjunction holds: no branch needed.
                            stack.push(partial);
                            continue;
                        }
                        for arg in args {
                            match *arg {
                                Arg::Level(l) => branches.push((p[l as usize], Item::Level(l))),
                                Arg::Gate(c) => branches.push((bound[c as usize], Item::Gate(c))),
                                Arg::Const(_) => {}
                            }
                        }
                        branches.sort_by(|a, b| a.0.total_cmp(&b.0));
                        for (_, branch) in branches {
                            let mut next = partial.clone();
                            push_item(&mut next.items, branch);
                            stack.push(next);
                        }
                    }
                    Kind::AtLeast(k) => {
                        push_item(
                            &mut partial.items,
                            Item::Vote {
                                gate: g,
                                start: 0,
                                k: *k,
                            },
                        );
                        stack.push(partial);
                    }
                }
            }
            Item::Vote { gate, start, k } => {
                let args = &formula.gates[gate as usize].1;
                if k == 0 {
                    stack.push(partial);
                    continue;
                }
                if args.len() - start < k {
                    continue;
                }
                // Out: the vote over the remaining arguments.
                let mut out = partial.clone();
                push_item(
                    &mut out.items,
                    Item::Vote {
                        gate,
                        start: start + 1,
                        k,
                    },
                );
                // In: the argument, and one fewer to find.
                let mut take = partial;
                push_item(
                    &mut take.items,
                    Item::Vote {
                        gate,
                        start: start + 1,
                        k: k - 1,
                    },
                );
                let mut feasible = true;
                match args[start] {
                    Arg::Level(l) => push_item(&mut take.items, Item::Level(l)),
                    Arg::Gate(c) => push_item(&mut take.items, Item::Gate(c)),
                    Arg::Const(true) => {}
                    Arg::Const(false) => feasible = false,
                }
                stack.push(out);
                if feasible {
                    stack.push(take);
                }
            }
        }
    }
    let sets = found
        .into_iter()
        .filter(|(_, _, alive)| *alive)
        .map(|(levels, _, _)| levels)
        .collect();
    Extracted {
        sets,
        neglected,
        expansions,
        truncated,
        count_reached,
        expansions_exhausted,
    }
}

/// Add `item` to a partial's pending items unless it is already there: a
/// gate reached by two paths is expanded once.
fn push_item(items: &mut Vec<Item>, item: Item) {
    if !items.contains(&item) {
        items.push(item);
    }
}

/// The next item to expand, after dropping the pending items the chosen
/// variables already satisfy (a variable already chosen, a disjunction
/// holding one of them, a vote with enough of them): on a coherent
/// formula they add nothing, and branching on them is what makes a naive
/// expansion explode (Rauzy 2003). Variables go first, conjunctions next,
/// then the disjunction or vote with the fewest branches.
fn take_item(partial: &mut Partial, formula: &LocalFormula) -> Option<Item> {
    let chosen = |l: u32| partial.levels.binary_search(&l).is_ok();
    let holds = |arg: &Arg| match *arg {
        Arg::Level(l) => chosen(l),
        Arg::Const(v) => v,
        Arg::Gate(_) => false,
    };
    let satisfied = |item: &Item| match *item {
        Item::Level(l) => chosen(l),
        Item::Gate(g) => {
            let (kind, args) = &formula.gates[g as usize];
            match kind {
                Kind::Or => args.iter().any(holds),
                Kind::And => args.iter().all(holds),
                Kind::AtLeast(k) => args.iter().filter(|a| holds(a)).count() >= *k,
            }
        }
        Item::Vote { gate, start, k } => {
            formula.gates[gate as usize].1[start..]
                .iter()
                .filter(|a| holds(a))
                .count()
                >= k
        }
    };
    let pending: Vec<Item> = partial
        .items
        .iter()
        .copied()
        .filter(|i| !satisfied(i))
        .collect();
    partial.items = pending;
    let branching = |item: &Item| -> usize {
        match *item {
            Item::Level(_) => 0,
            Item::Gate(g) => match formula.gates[g as usize].0 {
                Kind::And => 1,
                Kind::Or => 2 + formula.gates[g as usize].1.len(),
                Kind::AtLeast(_) => 3,
            },
            Item::Vote { .. } => 3,
        }
    };
    let (index, _) = partial
        .items
        .iter()
        .enumerate()
        .min_by_key(|(_, item)| branching(item))?;
    Some(partial.items.swap_remove(index))
}

/// An upper bound on each gate's probability, children first: a
/// conjunction by its least probable argument, or by the product over
/// arguments with pairwise disjoint supports when that is smaller (they
/// are independent then; shared variables forbid the product in general),
/// a disjunction by the sum, a vote of `k` by the expected number of
/// arguments holding over `k` (Markov).
fn gate_bounds(formula: &LocalFormula, p: &[f64], supports: Option<&Supports>) -> Vec<f64> {
    let mut bound = vec![1.0; formula.gates.len()];
    for (g, (kind, args)) in formula.gates.iter().enumerate() {
        let arg = |a: &Arg| match *a {
            Arg::Level(l) => p[l as usize],
            Arg::Gate(c) => bound[c as usize],
            Arg::Const(v) => f64::from(u8::from(v)),
        };
        let value = match kind {
            Kind::And => {
                let by_min = args.iter().map(arg).fold(1.0, f64::min);
                match supports {
                    Some(supports) => {
                        let mut ordered: Vec<(f64, &Arg)> =
                            args.iter().map(|a| (arg(a), a)).collect();
                        ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let mut union = supports.empty();
                        let mut product = 1.0;
                        for (b, a) in ordered {
                            if supports.disjoint_arg(&union, a) {
                                supports.merge_arg(&mut union, a);
                                product *= b;
                            }
                        }
                        by_min.min(product)
                    }
                    None => by_min,
                }
            }
            Kind::Or => args.iter().map(arg).sum::<f64>().min(1.0),
            Kind::AtLeast(k) => (args.iter().map(arg).sum::<f64>() / (*k).max(1) as f64).min(1.0),
        };
        bound[g] = value;
    }
    bound
}

fn vote_bound(
    formula: &LocalFormula,
    bound: &[f64],
    p: &[f64],
    gate: u32,
    start: usize,
    k: usize,
) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let sum: f64 = formula.gates[gate as usize].1[start..]
        .iter()
        .map(|a| match *a {
            Arg::Level(l) => p[l as usize],
            Arg::Gate(c) => bound[c as usize],
            Arg::Const(v) => f64::from(u8::from(v)),
        })
        .sum();
    (sum / k as f64).min(1.0)
}

/// The variables each gate depends on, as bitsets over the levels. Built
/// only when they fit a fixed budget of words; without them the bounds
/// fall back to minima, which stay valid, only looser.
struct Supports {
    words: usize,
    gates: Vec<Vec<u64>>,
}

impl Supports {
    const MAX_WORDS: usize = 2_000_000;

    fn new(formula: &LocalFormula, levels: usize) -> Option<Self> {
        let words = levels.div_ceil(64).max(1);
        if words.saturating_mul(formula.gates.len()) > Self::MAX_WORDS {
            return None;
        }
        let mut gates: Vec<Vec<u64>> = Vec::with_capacity(formula.gates.len());
        for (_, args) in &formula.gates {
            let mut bits = vec![0u64; words];
            for a in args {
                match *a {
                    Arg::Level(l) => bits[l as usize / 64] |= 1 << (l % 64),
                    Arg::Gate(c) => {
                        for (w, x) in bits.iter_mut().zip(&gates[c as usize]) {
                            *w |= x;
                        }
                    }
                    Arg::Const(_) => {}
                }
            }
            gates.push(bits);
        }
        Some(Self { words, gates })
    }

    fn empty(&self) -> Vec<u64> {
        vec![0; self.words]
    }

    fn set(&self, bits: &mut [u64], l: u32) {
        bits[l as usize / 64] |= 1 << (l % 64);
    }

    fn has(&self, bits: &[u64], l: u32) -> bool {
        bits[l as usize / 64] & (1 << (l % 64)) != 0
    }

    fn disjoint_gate(&self, bits: &[u64], g: u32) -> bool {
        bits.iter()
            .zip(&self.gates[g as usize])
            .all(|(a, b)| a & b == 0)
    }

    fn merge_gate(&self, bits: &mut [u64], g: u32) {
        for (a, b) in bits.iter_mut().zip(&self.gates[g as usize]) {
            *a |= b;
        }
    }

    fn disjoint_arg(&self, bits: &[u64], arg: &Arg) -> bool {
        match *arg {
            Arg::Level(l) => !self.has(bits, l),
            Arg::Gate(g) => self.disjoint_gate(bits, g),
            Arg::Const(_) => true,
        }
    }

    fn merge_arg(&self, bits: &mut [u64], arg: &Arg) {
        match *arg {
            Arg::Level(l) => self.set(bits, l),
            Arg::Gate(g) => self.merge_gate(bits, g),
            Arg::Const(_) => {}
        }
    }

    /// A vote item depends on its remaining arguments only; bounding it
    /// by the whole gate's support is coarser and stays valid.
    fn disjoint_item(&self, bits: &[u64], item: &Item, _formula: &LocalFormula) -> bool {
        match *item {
            Item::Level(l) => !self.has(bits, l),
            Item::Gate(g) | Item::Vote { gate: g, .. } => self.disjoint_gate(bits, g),
        }
    }

    fn merge_item(&self, bits: &mut [u64], item: &Item, _formula: &LocalFormula) {
        match *item {
            Item::Level(l) => self.set(bits, l),
            Item::Gate(g) | Item::Vote { gate: g, .. } => self.merge_gate(bits, g),
        }
    }
}

/// A float ordered totally, for the heap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
