//! Exact quantification: normalisation, module detection, one decision
//! diagram per module, and the measures read off the diagrams.
//!
//! 1. **Normalisation** pushes every negation down to the basic events
//!    (a negated vote `not atleast(k, n)` becomes `atleast(n - k + 1)` of
//!    the negated arguments), propagates constants, drops duplicate
//!    arguments of a conjunction or a disjunction, recognises a literal
//!    beside its own negation, turns a vote of `1` or of `n` into a
//!    disjunction or a conjunction, coalesces a gate into a parent of the
//!    same connective when nothing else references it, and shares
//!    identical gates.
//! 2. **Modules** are found in linear time (Dutuit and Rauzy 1996): a gate
//!    is a module when every basic event and gate below it is reached only
//!    through it, which the dates of a depth-first traversal decide. A
//!    module is quantified once, on its own diagram, and stands as a
//!    single variable in its parent's.
//! 3. **Each module** gets a reduced ordered BDD, its variables ordered by
//!    first encounter in a depth-first left-most traversal. The heuristic
//!    and the order it produced are recorded, because the size of the
//!    diagram depends on them.
//! 4. **Measures.** The probability is the root's, computed bottom-up over
//!    the modules. The Birnbaum importance of every event comes from one
//!    backward pass per module, chained through the modules
//!    (`∂top/∂e = ∂top/∂M · ∂M/∂e`). The other factors follow from the
//!    multilinearity of the top probability in each event:
//!    `P(top | e) = P + (1 - p) B` and `P(top | not e) = P - p B`.
//! 5. **Minimal cut sets**, only when every module is monotone: Rauzy's
//!    minimal solutions of each module's diagram, their exact number
//!    counted on the diagrams, and the sets themselves enumerated when
//!    that number fits the declared limit.

use std::collections::HashMap;

use serde::Serialize;

use crate::dd::{Bdd, ZERO};
use crate::{Formula, FtaError, Tree};

/// What a quantification may be given beside the tree.
#[derive(Debug, Clone)]
pub struct QuantifySettings {
    /// The instant the basic events' laws are read at. Required when any
    /// basic event the top reaches has a timed law.
    pub mission_time: Option<f64>,
    /// The most nodes all decision diagrams together may hold.
    pub max_bdd_nodes: usize,
    /// The most minimal cut sets enumerated. Their number is always
    /// computed when they are; past this limit the sets themselves are
    /// omitted and the result says why.
    pub cut_set_limit: usize,
    /// Whether to extract the minimal cut sets at all. Their extraction is
    /// usually the costliest step by far, and neither the probability nor
    /// the importance measures need it.
    pub cut_sets: bool,
}

impl Default for QuantifySettings {
    fn default() -> Self {
        Self {
            mission_time: None,
            max_bdd_nodes: 10_000_000,
            cut_set_limit: 100_000,
            cut_sets: true,
        }
    }
}

/// The importance measures of one basic event.
///
/// With `P` the top probability, `p` the event's, `P1` and `P0` the top
/// probability with the event certain and impossible:
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventImportance {
    /// The event, by index into the tree's events.
    pub event: usize,
    /// Its probability at the mission time; `None` for an event the top
    /// does not reach whose law could not be read there.
    pub probability: Option<f64>,
    /// Birnbaum's marginal importance, `P1 - P0`.
    pub birnbaum: f64,
    /// The critical importance factor, `p (P1 - P0) / P`.
    pub criticality: Option<f64>,
    /// Fussell-Vesely, in the PSA-code convention `(P - P0) / P`: the
    /// fraction of the risk removed by making the event impossible. Equal
    /// to the critical importance factor in exact arithmetic on a
    /// multilinear top probability; computed from its own definition so a
    /// cut-set-based approximation can later differ from it visibly.
    pub fussell_vesely: Option<f64>,
    /// The diagnostic importance factor, `p P1 / P`: the probability the
    /// event has occurred given that the top has.
    pub diagnostic: Option<f64>,
    /// Risk achievement worth, `P1 / P`.
    pub risk_achievement_worth: Option<f64>,
    /// Risk reduction worth, `P / P0`; `None` when `P0 = 0` (making the
    /// event impossible removes the whole risk, an infinite worth) or when
    /// `P = 0`.
    pub risk_reduction_worth: Option<f64>,
}

/// One module's diagram, recorded with the order it was built in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModuleRecord {
    /// `M0` is the top; a submodule appears in its parent's variables
    /// under this name.
    pub name: String,
    /// The variables, in diagram order: basic-event names and submodule
    /// names.
    pub variables: Vec<String>,
    /// Nodes of its BDD, terminals included: the size of the function's
    /// diagram.
    pub bdd_nodes: usize,
    /// Nodes its construction created, intermediate results included.
    pub nodes_created: usize,
    /// Its probability.
    pub probability: f64,
    /// Whether its function is monotone in its variables.
    pub monotone: bool,
}

/// How the numbers were produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    /// The variable-ordering heuristic.
    pub variable_order: &'static str,
    /// Every module, top first.
    pub modules: Vec<ModuleRecord>,
    /// Nodes of all module BDDs together.
    pub bdd_nodes: usize,
    /// Nodes of all minimal-cut-set diagrams together (0 when none was
    /// built).
    pub zbdd_nodes: usize,
    /// Nodes created by every construction, intermediate results
    /// included: what `max_bdd_nodes` bounds.
    pub nodes_created: usize,
}

/// A quantified tree.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quantification {
    /// The top-event probability at the mission time.
    pub probability: f64,
    /// The method that produced it.
    pub method: &'static str,
    /// Whether the probability is exact (no cutoff, no bound). Always true
    /// for the decision-diagram method.
    pub exact: bool,
    /// The instant the laws were read at.
    pub mission_time: Option<f64>,
    /// Whether the top is monotone in every basic event.
    pub coherent: bool,
    /// Every basic event's importance, in the tree's event order.
    pub importance: Vec<EventImportance>,
    /// The exact number of minimal cut sets (saturating at `u128::MAX`);
    /// `None` for a non-coherent tree or when they were not requested.
    pub cut_set_count: Option<u128>,
    /// The minimal cut sets, each a sorted list of event indices, by size
    /// then lexicographically; `None` when omitted, with the reason in
    /// [`Quantification::cut_sets_omitted`].
    pub minimal_cut_sets: Option<Vec<Vec<usize>>>,
    /// Why the minimal cut sets are not listed.
    pub cut_sets_omitted: Option<String>,
    /// How the numbers were produced.
    pub provenance: Provenance,
}

/// Quantify `tree` exactly.
///
/// Runs on a thread of its own with a large stack (1 GiB of address
/// space, committed as used): the diagram operations
/// recurse once per variable level and the normalisation once per gate
/// level, so a flat gate of a hundred thousand events or a chain of as
/// many gates would overflow a default stack.
pub fn quantify(tree: &Tree, settings: &QuantifySettings) -> Result<Quantification, FtaError> {
    crate::with_large_stack(|| quantify_here(tree, settings))
}

fn quantify_here(tree: &Tree, settings: &QuantifySettings) -> Result<Quantification, FtaError> {
    tree.validate()?;
    if let Some(t) = settings.mission_time {
        if !t.is_finite() || t < 0.0 {
            return Err(FtaError::BadMissionTime(t));
        }
    }
    let mut graph = Graph::default();
    let mut memo = HashMap::new();
    let root = graph.build(tree, &tree.top, true, &mut memo);
    let (graph, root) = graph.coalesce(root);

    let n_events = tree.events.len();
    // Only the events the top reaches must have a readable law.
    let mut reached = vec![false; n_events];
    if let Root::Node(n) = root {
        graph.mark_events(Lit::Node(n), &mut reached);
    }
    let mut probability = Vec::with_capacity(n_events);
    for (index, event) in tree.events.iter().enumerate() {
        match event.law.probability(&event.name, settings.mission_time) {
            Ok(p) => probability.push(Some(p)),
            Err(e) if reached[index] => return Err(e),
            Err(_) => probability.push(None),
        }
    }

    match root {
        Root::Node(n) => Quantifier::new(&graph, tree, settings, probability).run(n),
        Root::Constant(value) => Ok(constant_result(settings, value, probability)),
    }
}

fn constant_result(
    settings: &QuantifySettings,
    value: bool,
    probability: Vec<Option<f64>>,
) -> Quantification {
    let p = if value { 1.0 } else { 0.0 };
    let importance = probability
        .iter()
        .enumerate()
        .map(|(event, q)| measures(event, *q, p, 0.0))
        .collect();
    Quantification {
        probability: p,
        method: "bdd",
        exact: true,
        mission_time: settings.mission_time,
        coherent: true,
        importance,
        cut_set_count: settings.cut_sets.then_some(u128::from(value)),
        minimal_cut_sets: settings
            .cut_sets
            .then(|| if value { vec![Vec::new()] } else { Vec::new() }),
        cut_sets_omitted: (!settings.cut_sets).then(|| "not requested".to_owned()),
        provenance: Provenance {
            variable_order: VARIABLE_ORDER,
            modules: Vec::new(),
            bdd_nodes: 0,
            zbdd_nodes: 0,
            nodes_created: 0,
        },
    }
}

const VARIABLE_ORDER: &str = "depth_first_left_most";

fn measures(event: usize, q: Option<f64>, top: f64, birnbaum: f64) -> EventImportance {
    let p = q.unwrap_or(0.0);
    let p1 = top + (1.0 - p) * birnbaum;
    // `P - p B` cancels when the event carries the whole risk: a residue
    // of a few ulps of `P`, of either sign, is a zero, not a finite worth.
    let p0 = top - p * birnbaum;
    let p0 = if p0 <= 16.0 * f64::EPSILON * top {
        0.0
    } else {
        p0
    };
    let ratio = |num: f64, den: f64| if den > 0.0 { Some(num / den) } else { None };
    EventImportance {
        event,
        probability: q,
        birnbaum,
        criticality: ratio(p * birnbaum, top),
        fussell_vesely: ratio(top - p0, top),
        diagnostic: ratio(p * p1, top),
        risk_achievement_worth: ratio(p1, top),
        risk_reduction_worth: if top > 0.0 { ratio(top, p0) } else { None },
    }
}

// ---------------------------------------------------------------------
// The normalised graph.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Lit {
    Const(bool),
    Var(u32, bool),
    Node(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GOp {
    And,
    Or,
    AtLeast(usize),
}

/// The normalised top: a gate, or a constant once simplified.
enum Root {
    Node(u32),
    Constant(bool),
}

#[derive(Default)]
struct Graph {
    nodes: Vec<(GOp, Vec<Lit>)>,
    shared: HashMap<(GOp, Vec<Lit>), u32>,
}

impl Graph {
    fn build(
        &mut self,
        tree: &Tree,
        formula: &Formula,
        positive: bool,
        memo: &mut HashMap<(usize, bool), Lit>,
    ) -> Lit {
        match formula {
            Formula::Constant { value } => Lit::Const(*value == positive),
            Formula::Event { event } => Lit::Var(*event as u32, positive),
            Formula::Gate { gate } => {
                if let Some(&lit) = memo.get(&(*gate, positive)) {
                    return lit;
                }
                let lit = self.build(tree, &tree.gates[*gate].formula, positive, memo);
                memo.insert((*gate, positive), lit);
                lit
            }
            Formula::Not { arg } => self.build(tree, arg, !positive, memo),
            Formula::And { args } | Formula::Or { args } => {
                let conjunction = matches!(formula, Formula::And { .. }) == positive;
                let args = args
                    .iter()
                    .map(|a| self.build(tree, a, positive, memo))
                    .collect();
                self.make(if conjunction { GOp::And } else { GOp::Or }, args)
            }
            Formula::AtLeast { k, args } => {
                let args: Vec<Lit> = args
                    .iter()
                    .map(|a| self.build(tree, a, positive, memo))
                    .collect();
                let k = if positive {
                    *k as isize
                } else {
                    // not atleast(k, n) = atleast(n - k + 1) of the negations.
                    args.len() as isize - *k as isize + 1
                };
                self.make_vote(k, args)
            }
        }
    }

    fn make_vote(&mut self, mut k: isize, args: Vec<Lit>) -> Lit {
        let mut kept = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Lit::Const(true) => k -= 1,
                Lit::Const(false) => {}
                other => kept.push(other),
            }
        }
        if k <= 0 {
            return Lit::Const(true);
        }
        let k = k as usize;
        if k > kept.len() {
            return Lit::Const(false);
        }
        if k == 1 {
            return self.make(GOp::Or, kept);
        }
        if k == kept.len() {
            return self.make(GOp::And, kept);
        }
        kept.sort();
        self.intern(GOp::AtLeast(k), kept)
    }

    fn make(&mut self, op: GOp, args: Vec<Lit>) -> Lit {
        let (absorbing, neutral) = match op {
            GOp::And => (false, true),
            GOp::Or => (true, false),
            GOp::AtLeast(k) => return self.make_vote(k as isize, args),
        };
        let mut kept = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Lit::Const(v) if v == neutral => {}
                Lit::Const(_) => return Lit::Const(absorbing),
                other => kept.push(other),
            }
        }
        kept.sort();
        kept.dedup();
        for pair in kept.windows(2) {
            if let (Lit::Var(a, pa), Lit::Var(b, pb)) = (pair[0], pair[1]) {
                if a == b && pa != pb {
                    return Lit::Const(absorbing);
                }
            }
        }
        match kept.len() {
            0 => Lit::Const(neutral),
            1 => kept[0],
            _ => self.intern(op, kept),
        }
    }

    fn intern(&mut self, op: GOp, args: Vec<Lit>) -> Lit {
        if let Some(&id) = self.shared.get(&(op, args.clone())) {
            return Lit::Node(id);
        }
        let id = self.nodes.len() as u32;
        self.nodes.push((op, args.clone()));
        self.shared.insert((op, args), id);
        Lit::Node(id)
    }

    /// Rebuild the graph reachable from `root`, merging a gate into a
    /// parent of the same connective when that parent is its only
    /// referrer. A top that simplifies to a single literal is wrapped in
    /// a one-argument gate, so the root is a node unless constant.
    fn coalesce(self, root: Lit) -> (Graph, Root) {
        let mut refs = vec![0usize; self.nodes.len()];
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        if let Lit::Node(n) = root {
            stack.push(n);
        }
        while let Some(n) = stack.pop() {
            if seen[n as usize] {
                continue;
            }
            seen[n as usize] = true;
            for arg in &self.nodes[n as usize].1 {
                if let Lit::Node(c) = arg {
                    refs[*c as usize] += 1;
                    stack.push(*c);
                }
            }
        }
        let mut out = Graph::default();
        let mut memo: HashMap<u32, Lit> = HashMap::new();
        let copied = match root {
            Lit::Node(n) => self.copy(n, &refs, &mut out, &mut memo),
            other => other,
        };
        let root = match copied {
            Lit::Node(n) => Root::Node(n),
            Lit::Const(v) => Root::Constant(v),
            var @ Lit::Var(..) => {
                let id = out.nodes.len() as u32;
                out.nodes.push((GOp::Or, vec![var]));
                Root::Node(id)
            }
        };
        (out, root)
    }

    fn copy(&self, n: u32, refs: &[usize], out: &mut Graph, memo: &mut HashMap<u32, Lit>) -> Lit {
        if let Some(&lit) = memo.get(&n) {
            return lit;
        }
        let (op, args) = &self.nodes[n as usize];
        let mut flat = Vec::with_capacity(args.len());
        self.flatten(*op, args, refs, out, memo, &mut flat);
        let lit = out.make(*op, flat);
        memo.insert(n, lit);
        lit
    }

    fn flatten(
        &self,
        op: GOp,
        args: &[Lit],
        refs: &[usize],
        out: &mut Graph,
        memo: &mut HashMap<u32, Lit>,
        flat: &mut Vec<Lit>,
    ) {
        for arg in args {
            match arg {
                Lit::Node(c)
                    if refs[*c as usize] == 1
                        && matches!(op, GOp::And | GOp::Or)
                        && self.nodes[*c as usize].0 == op =>
                {
                    self.flatten(op, &self.nodes[*c as usize].1, refs, out, memo, flat)
                }
                Lit::Node(c) => flat.push(self.copy(*c, refs, out, memo)),
                other => flat.push(*other),
            }
        }
    }

    fn mark_events(&self, root: Lit, reached: &mut [bool]) {
        let mut stack = vec![root];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(lit) = stack.pop() {
            match lit {
                Lit::Var(e, _) => reached[e as usize] = true,
                Lit::Node(n) if !seen[n as usize] => {
                    seen[n as usize] = true;
                    stack.extend(self.nodes[n as usize].1.iter().copied());
                }
                _ => {}
            }
        }
    }

    /// The gates that are modules (Dutuit and Rauzy 1996), the root
    /// included.
    fn modules(&self, root: u32, n_events: usize) -> Vec<bool> {
        let n = self.nodes.len();
        let mut first = vec![0u64; n];
        let mut second = vec![0u64; n];
        let mut last = vec![0u64; n];
        let mut first_v = vec![0u64; n_events];
        let mut last_v = vec![0u64; n_events];
        let mut date = 0u64;
        // Iterative depth-first traversal: (node, next argument index).
        enum Step {
            Enter(Lit),
            Leave(u32),
        }
        let mut stack = vec![Step::Enter(Lit::Node(root))];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(Lit::Var(e, _)) => {
                    date += 1;
                    if first_v[e as usize] == 0 {
                        first_v[e as usize] = date;
                    }
                    last_v[e as usize] = date;
                }
                Step::Enter(Lit::Node(g)) => {
                    date += 1;
                    if first[g as usize] == 0 {
                        first[g as usize] = date;
                        stack.push(Step::Leave(g));
                        for arg in self.nodes[g as usize].1.iter().rev() {
                            stack.push(Step::Enter(*arg));
                        }
                    }
                    last[g as usize] = date;
                }
                Step::Enter(Lit::Const(_)) => {}
                Step::Leave(g) => {
                    date += 1;
                    second[g as usize] = date;
                }
            }
        }
        // Children are created before parents: ascending ids are bottom-up.
        let mut min_d = vec![u64::MAX; n];
        let mut max_d = vec![0u64; n];
        let mut module = vec![false; n];
        for g in 0..n {
            if first[g] == 0 {
                continue;
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for arg in &self.nodes[g].1 {
                match *arg {
                    Lit::Var(e, _) => {
                        lo = lo.min(first_v[e as usize]);
                        hi = hi.max(last_v[e as usize]);
                    }
                    Lit::Node(c) => {
                        let c = c as usize;
                        lo = lo.min(first[c]).min(min_d[c]);
                        hi = hi.max(last[c]).max(max_d[c]);
                    }
                    Lit::Const(_) => {}
                }
            }
            min_d[g] = lo;
            max_d[g] = hi;
            module[g] = first[g] < lo && hi < second[g];
        }
        module[root as usize] = true;
        module
    }
}

// ---------------------------------------------------------------------
// Per-module diagrams.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Event(u32),
    Module(u32),
}

struct ModuleDiagram {
    node: u32,
    bdd: Bdd,
    root: u32,
    keys: Vec<Key>,
    level_p: Vec<f64>,
    prob: Vec<f64>,
    monotone: bool,
}

struct Quantifier<'a> {
    graph: &'a Graph,
    tree: &'a Tree,
    settings: &'a QuantifySettings,
    probability: Vec<Option<f64>>,
}

impl<'a> Quantifier<'a> {
    fn new(
        graph: &'a Graph,
        tree: &'a Tree,
        settings: &'a QuantifySettings,
        probability: Vec<Option<f64>>,
    ) -> Self {
        Self {
            graph,
            tree,
            settings,
            probability,
        }
    }

    fn run(self, root: u32) -> Result<Quantification, FtaError> {
        let n_events = self.tree.events.len();
        let is_module = self.graph.modules(root, n_events);
        // Bottom-up: ascending node ids put every submodule before its
        // parent.
        // An unreachable node is never a module: its dates stay at zero.
        let module_nodes: Vec<u32> = (0..=root).filter(|&g| is_module[g as usize]).collect();
        let mut module_p: HashMap<u32, f64> = HashMap::new();
        // A module whose diagram reduced to a terminal is that constant in
        // its parent, never a variable: a variable standing for a
        // tautology would make its parent's cut sets carry the empty set
        // beside real ones.
        let mut constant: HashMap<u32, bool> = HashMap::new();
        let mut all: Vec<ModuleDiagram> = Vec::new();
        let mut used = 0usize;
        for &m in &module_nodes {
            let budget = self.settings.max_bdd_nodes.saturating_sub(used).max(2);
            let diagram = self.diagram(m, &is_module, &module_p, &constant, budget)?;
            used += diagram.bdd.len();
            module_p.insert(m, diagram.prob[diagram.root as usize]);
            if diagram.root <= crate::dd::ONE {
                constant.insert(m, diagram.root == crate::dd::ONE);
            }
            all.push(diagram);
        }
        let top = module_p.get(&root).copied().unwrap_or(0.0);

        // The live modules: the root, and every module its live parent's
        // function still depends on. A module its parent absorbed (in
        // `e + e.M`, say) takes part in nothing: not in the cut sets, not
        // in the coherence of the top.
        let mut live: std::collections::HashSet<u32> = std::collections::HashSet::from([root]);
        for diagram in all.iter().rev() {
            if !live.contains(&diagram.node) {
                continue;
            }
            let support = diagram.bdd.support(diagram.root, diagram.keys.len());
            for (level, key) in diagram.keys.iter().enumerate() {
                if let Key::Module(n) = key {
                    if support[level] {
                        live.insert(*n);
                    }
                }
            }
        }
        let diagrams: Vec<ModuleDiagram> =
            all.into_iter().filter(|d| live.contains(&d.node)).collect();

        // Top-down derivatives through the modules.
        let mut derivative: HashMap<u32, f64> = HashMap::from([(root, 1.0)]);
        let mut birnbaum = vec![0.0; n_events];
        for diagram in diagrams.iter().rev() {
            let d = derivative.get(&diagram.node).copied().unwrap_or(0.0);
            let b = diagram
                .bdd
                .birnbaum(diagram.root, &diagram.level_p, &diagram.prob);
            for (level, key) in diagram.keys.iter().enumerate() {
                match *key {
                    Key::Event(e) => birnbaum[e as usize] += d * b[level],
                    Key::Module(n) => {
                        *derivative.entry(n).or_insert(0.0) += d * b[level];
                    }
                }
            }
        }
        let importance = self
            .probability
            .iter()
            .enumerate()
            .map(|(event, q)| measures(event, *q, top, birnbaum[event]))
            .collect();

        let coherent = diagrams.iter().all(|d| d.monotone);
        let names: HashMap<u32, String> = diagrams
            .iter()
            .rev()
            .enumerate()
            .map(|(i, d)| (d.node, format!("M{i}")))
            .collect();
        let mut zbdd_nodes = 0usize;
        let mut zbdd_created = 0usize;
        let (cut_set_count, minimal_cut_sets, cut_sets_omitted) = if !self.settings.cut_sets {
            (None, None, Some("not requested".to_owned()))
        } else if coherent {
            let (count, sets, omitted, (size, created)) = self.cut_sets(&diagrams, root, used)?;
            zbdd_nodes = size;
            zbdd_created = created;
            (Some(count), sets, omitted)
        } else {
            (
                None,
                None,
                Some(
                    "the top is not monotone in its basic events (a negation \
                     survives simplification): its minimal cut sets are prime \
                     implicants, which this method does not compute"
                        .to_owned(),
                ),
            )
        };
        let modules = diagrams
            .iter()
            .rev()
            .map(|d| ModuleRecord {
                name: names.get(&d.node).cloned().unwrap_or_default(),
                variables: d
                    .keys
                    .iter()
                    .map(|k| match k {
                        Key::Event(e) => self.tree.events[*e as usize].name.clone(),
                        // A submodule the parent's function no longer
                        // depends on: listed in the order, never quantified.
                        Key::Module(n) => names
                            .get(n)
                            .cloned()
                            .unwrap_or_else(|| "absorbed".to_owned()),
                    })
                    .collect(),
                bdd_nodes: d.bdd.size(d.root),
                nodes_created: d.bdd.len(),
                probability: d.prob[d.root as usize],
                monotone: d.monotone,
            })
            .collect();
        Ok(Quantification {
            probability: top,
            method: "bdd",
            exact: true,
            mission_time: self.settings.mission_time,
            coherent,
            importance,
            cut_set_count,
            minimal_cut_sets,
            cut_sets_omitted,
            provenance: Provenance {
                variable_order: VARIABLE_ORDER,
                bdd_nodes: diagrams.iter().map(|d| d.bdd.size(d.root)).sum(),
                modules,
                zbdd_nodes,
                nodes_created: used + zbdd_created,
            },
        })
    }

    fn diagram(
        &self,
        m: u32,
        is_module: &[bool],
        module_p: &HashMap<u32, f64>,
        constant: &HashMap<u32, bool>,
        budget: usize,
    ) -> Result<ModuleDiagram, FtaError> {
        // Variable order: depth-first left-most, first encounter.
        let mut keys: Vec<Key> = Vec::new();
        let mut level_of: HashMap<Key, u32> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        self.order(
            m,
            m,
            is_module,
            constant,
            &mut keys,
            &mut level_of,
            &mut seen,
        );
        let level_p: Vec<f64> = keys
            .iter()
            .map(|k| match k {
                Key::Event(e) => self.probability[*e as usize].unwrap_or(0.0),
                Key::Module(n) => module_p.get(n).copied().unwrap_or(0.0),
            })
            .collect();
        let mut bdd = Bdd::new(budget);
        let mut memo: HashMap<u32, u32> = HashMap::new();
        let root = self.build(m, m, is_module, constant, &level_of, &mut bdd, &mut memo)?;
        let prob = bdd.probabilities(&level_p);
        let monotone = bdd.is_monotone(root);
        Ok(ModuleDiagram {
            node: m,
            bdd,
            root,
            keys,
            level_p,
            prob,
            monotone,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn order(
        &self,
        g: u32,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        keys: &mut Vec<Key>,
        level_of: &mut HashMap<Key, u32>,
        seen: &mut std::collections::HashSet<u32>,
    ) {
        if !seen.insert(g) {
            return;
        }
        for arg in &self.graph.nodes[g as usize].1 {
            let key = match *arg {
                Lit::Var(e, _) => Key::Event(e),
                Lit::Node(c) if c != m && is_module[c as usize] => {
                    if constant.contains_key(&c) {
                        continue;
                    }
                    Key::Module(c)
                }
                Lit::Node(c) => {
                    self.order(c, m, is_module, constant, keys, level_of, seen);
                    continue;
                }
                Lit::Const(_) => continue,
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = level_of.entry(key) {
                slot.insert(keys.len() as u32);
                keys.push(key);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        g: u32,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        level_of: &HashMap<Key, u32>,
        bdd: &mut Bdd,
        memo: &mut HashMap<u32, u32>,
    ) -> Result<u32, FtaError> {
        if let Some(&id) = memo.get(&g) {
            return Ok(id);
        }
        let (op, args) = &self.graph.nodes[g as usize];
        let mut ids = Vec::with_capacity(args.len());
        for arg in args {
            let id = match *arg {
                Lit::Var(e, positive) => bdd.literal(level_of[&Key::Event(e)], positive)?,
                Lit::Node(c) if c != m && is_module[c as usize] => match constant.get(&c) {
                    Some(true) => crate::dd::ONE,
                    Some(false) => ZERO,
                    None => bdd.literal(level_of[&Key::Module(c)], true)?,
                },
                Lit::Node(c) => self.build(c, m, is_module, constant, level_of, bdd, memo)?,
                Lit::Const(true) => crate::dd::ONE,
                Lit::Const(false) => ZERO,
            };
            ids.push(id);
        }
        let id = match op {
            GOp::And => crate::dd::fold_balanced(bdd, true, ids)?,
            GOp::Or => crate::dd::fold_balanced(bdd, false, ids)?,
            GOp::AtLeast(k) => bdd.at_least(*k, &ids)?,
        };
        memo.insert(g, id);
        Ok(id)
    }

    #[allow(clippy::type_complexity)]
    fn cut_sets(
        &self,
        diagrams: &[ModuleDiagram],
        root: u32,
        used: usize,
    ) -> Result<
        (
            u128,
            Option<Vec<Vec<usize>>>,
            Option<String>,
            (usize, usize),
        ),
        FtaError,
    > {
        let mut families: HashMap<u32, (Vec<Key>, crate::dd::Zbdd)> = HashMap::new();
        let mut counts: HashMap<u32, u128> = HashMap::new();
        let mut zbdd_nodes = 0usize;
        let mut zbdd_created = 0usize;
        for diagram in diagrams {
            let budget = self
                .settings
                .max_bdd_nodes
                .saturating_sub(used + zbdd_created)
                .max(2);
            let zbdd = diagram.bdd.minimal_solutions(diagram.root, budget)?;
            zbdd_nodes += zbdd.size();
            zbdd_created += zbdd.len();
            let weight: Vec<u128> = diagram
                .keys
                .iter()
                .map(|k| match k {
                    Key::Event(_) => 1,
                    Key::Module(n) => counts.get(n).copied().unwrap_or(0),
                })
                .collect();
            counts.insert(diagram.node, zbdd.weighted_count(&weight));
            families.insert(diagram.node, (diagram.keys.clone(), zbdd));
        }
        let count = counts.get(&root).copied().unwrap_or(0);
        let limit = self.settings.cut_set_limit;
        if count > limit as u128 {
            return Ok((
                count,
                None,
                Some(format!(
                    "{count} minimal cut sets, more than the limit of {limit}; \
                     raise cut_set_limit to list them"
                )),
                (zbdd_nodes, zbdd_created),
            ));
        }
        let mut expanded: HashMap<u32, Vec<Vec<usize>>> = HashMap::new();
        for diagram in diagrams {
            let Some((keys, zbdd)) = families.get(&diagram.node) else {
                continue;
            };
            let mut family: Vec<Vec<usize>> = Vec::new();
            for set in zbdd.sets() {
                let mut partial: Vec<Vec<usize>> = vec![Vec::new()];
                for level in set {
                    match keys[level as usize] {
                        Key::Event(e) => partial.iter_mut().for_each(|s| s.push(e as usize)),
                        Key::Module(n) => {
                            let Some(sub) = expanded.get(&n) else {
                                return Err(FtaError::Invalid(format!(
                                    "internal: submodule {n} expanded before its parent"
                                )));
                            };
                            let mut next = Vec::with_capacity(partial.len() * sub.len());
                            for s in &partial {
                                for t in sub {
                                    let mut u = s.clone();
                                    u.extend_from_slice(t);
                                    next.push(u);
                                }
                            }
                            partial = next;
                        }
                    }
                }
                family.extend(partial);
            }
            for set in &mut family {
                set.sort_unstable();
            }
            expanded.insert(diagram.node, family);
        }
        let mut sets = expanded.remove(&root).unwrap_or_default();
        sets.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        Ok((count, Some(sets), None, (zbdd_nodes, zbdd_created)))
    }
}
