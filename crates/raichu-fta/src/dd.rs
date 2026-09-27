//! The decision-diagram kernel: a reduced ordered BDD for one module's
//! Boolean function, and the ZBDD of its minimal solutions.
//!
//! Both diagrams live in an arena whose nodes are created children first,
//! so the arena index is a topological order: a probability is computed
//! by one forward pass, a reaching probability by one backward pass, with
//! no recursion over the diagram. Levels are positions in the module's
//! variable order, level 0 closest to the root.
//!
//! - Probability of a BDD: Shannon decomposition, linear in the diagram
//!   (Rauzy 1993).
//! - Birnbaum importance of every variable at once: the probability of
//!   reaching each node times the difference of its two cofactors, summed
//!   over the nodes labelled by the variable (Dutuit and Rauzy 2001). Exact
//!   because an ordered diagram meets a variable at most once per path.
//! - Minimal solutions of a monotone function: Rauzy's `minsol`, stored in
//!   a ZBDD, with the `without` operator removing the non-minimal sets
//!   (Rauzy 1993).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::FtaError;

/// A multiplicative hash for the diagram tables. Their keys are small
/// integers the diagram itself produces, so the default hasher's
/// resistance to chosen collisions buys nothing and costs most of the
/// construction time. Deterministic, so a run is reproducible.
#[derive(Default, Clone, Copy)]
pub(crate) struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(K);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_u8(&mut self, n: u8) {
        self.write_u64(u64::from(n));
    }
}

/// A map keyed through [`FastHasher`].
pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// The terminal `false` (the empty family, in a ZBDD).
pub(crate) const ZERO: u32 = 0;
/// The terminal `true` (the family holding only the empty set, in a ZBDD).
pub(crate) const ONE: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    level: u32,
    low: u32,
    high: u32,
}

const TERMINAL: Node = Node {
    level: u32::MAX,
    low: 0,
    high: 0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    And,
    Or,
}

/// A reduced ordered binary decision diagram.
pub(crate) struct Bdd {
    nodes: Vec<Node>,
    unique: FastMap<Node, u32>,
    computed: FastMap<(Op, u32, u32), u32>,
    budget: usize,
}

impl Bdd {
    /// An empty diagram that refuses to grow past `budget` nodes.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            nodes: vec![TERMINAL, TERMINAL],
            unique: FastMap::default(),
            computed: FastMap::default(),
            budget,
        }
    }

    /// How many nodes the diagram holds, terminals included.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    fn make(&mut self, level: u32, low: u32, high: u32) -> Result<u32, FtaError> {
        if low == high {
            return Ok(low);
        }
        let node = Node { level, low, high };
        if let Some(&id) = self.unique.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= self.budget {
            return Err(FtaError::TooLarge(format!(
                "the decision diagram outgrew its budget of {} nodes; raise \
                 max_bdd_nodes",
                self.budget
            )));
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.unique.insert(node, id);
        Ok(id)
    }

    /// The nodes reachable from `root`, terminals included: the size of
    /// the function's diagram, as opposed to [`Bdd::len`], every node the
    /// construction created.
    pub(crate) fn size(&self, root: u32) -> usize {
        reachable(&self.nodes, root)
    }

    /// Which levels the function `root` depends on: those labelling a
    /// node reachable from it. A reduced diagram keeps a variable exactly
    /// when the function depends on it.
    pub(crate) fn support(&self, root: u32, levels: usize) -> Vec<bool> {
        let mut out = vec![false; levels];
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if id <= ONE || seen[id as usize] {
                continue;
            }
            seen[id as usize] = true;
            let node = self.nodes[id as usize];
            out[node.level as usize] = true;
            stack.push(node.low);
            stack.push(node.high);
        }
        out
    }

    /// The variable at `level`, positive or negated.
    pub(crate) fn literal(&mut self, level: u32, positive: bool) -> Result<u32, FtaError> {
        if positive {
            self.make(level, ZERO, ONE)
        } else {
            self.make(level, ONE, ZERO)
        }
    }

    fn level(&self, id: u32) -> u32 {
        self.nodes[id as usize].level
    }

    fn cofactors(&self, id: u32, level: u32) -> (u32, u32) {
        let node = self.nodes[id as usize];
        if node.level == level {
            (node.low, node.high)
        } else {
            (id, id)
        }
    }

    fn apply(&mut self, op: Op, a: u32, b: u32) -> Result<u32, FtaError> {
        match (op, a, b) {
            (Op::And, ZERO, _) | (Op::And, _, ZERO) => return Ok(ZERO),
            (Op::And, ONE, x) | (Op::And, x, ONE) => return Ok(x),
            (Op::Or, ONE, _) | (Op::Or, _, ONE) => return Ok(ONE),
            (Op::Or, ZERO, x) | (Op::Or, x, ZERO) => return Ok(x),
            _ if a == b => return Ok(a),
            _ => {}
        }
        let key = (op, a.min(b), a.max(b));
        if let Some(&id) = self.computed.get(&key) {
            return Ok(id);
        }
        let level = self.level(a).min(self.level(b));
        let (a0, a1) = self.cofactors(a, level);
        let (b0, b1) = self.cofactors(b, level);
        let low = self.apply(op, a0, b0)?;
        let high = self.apply(op, a1, b1)?;
        let id = self.make(level, low, high)?;
        self.computed.insert(key, id);
        Ok(id)
    }

    /// The conjunction of two functions.
    pub(crate) fn and(&mut self, a: u32, b: u32) -> Result<u32, FtaError> {
        self.apply(Op::And, a, b)
    }

    /// The disjunction of two functions.
    pub(crate) fn or(&mut self, a: u32, b: u32) -> Result<u32, FtaError> {
        self.apply(Op::Or, a, b)
    }

    /// At least `k` of `args` hold, by the recurrence
    /// `atleast(k, [x, rest]) = x.atleast(k-1, rest) + atleast(k, rest)`,
    /// tabulated over `(k, position)` so a vote costs `O(k n)` operations.
    pub(crate) fn at_least(&mut self, k: usize, args: &[u32]) -> Result<u32, FtaError> {
        let n = args.len();
        // row[j] = atleast(k', args[i..]) for the current k', i = j.
        let mut previous: Vec<u32> = vec![ONE; n + 1]; // k' = 0
        for kk in 1..=k {
            let mut current = vec![ZERO; n + 1];
            for i in (0..n).rev() {
                if n - i < kk {
                    current[i] = ZERO;
                    continue;
                }
                let with = self.and(args[i], previous[i + 1])?;
                current[i] = self.or(with, current[i + 1])?;
            }
            previous = current;
        }
        Ok(previous[0])
    }

    /// The probability of every node, `p[level]` being the probability of
    /// the variable at that level.
    pub(crate) fn probabilities(&self, p: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nodes.len()];
        out[ONE as usize] = 1.0;
        for (id, node) in self.nodes.iter().enumerate().skip(2) {
            let q = p[node.level as usize];
            out[id] = q * out[node.high as usize] + (1.0 - q) * out[node.low as usize];
        }
        out
    }

    /// The partial derivative of the root's probability with respect to
    /// each level's probability.
    pub(crate) fn birnbaum(&self, root: u32, p: &[f64], prob: &[f64]) -> Vec<f64> {
        let mut reach = vec![0.0; self.nodes.len()];
        reach[root as usize] = 1.0;
        let mut out = vec![0.0; p.len()];
        for id in (2..=root as usize).rev() {
            let r = reach[id];
            if r == 0.0 {
                continue;
            }
            let node = self.nodes[id];
            let q = p[node.level as usize];
            reach[node.high as usize] += r * q;
            reach[node.low as usize] += r * (1.0 - q);
            out[node.level as usize] += r * (prob[node.high as usize] - prob[node.low as usize]);
        }
        out
    }

    /// Whether the function is monotone (coherent): no node where going
    /// from the low to the high branch loses a solution. Checked on the
    /// diagram rather than on the formula, so a negation that simplifies
    /// away does not count.
    pub(crate) fn is_monotone(&self, root: u32) -> bool {
        // implies[(a, b)]: every solution of a is a solution of b.
        let mut memo: FastMap<(u32, u32), bool> = FastMap::default();
        let mut stack = vec![root];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(id) = stack.pop() {
            if id <= ONE || seen[id as usize] {
                continue;
            }
            seen[id as usize] = true;
            let node = self.nodes[id as usize];
            if !self.implies(node.low, node.high, &mut memo) {
                return false;
            }
            stack.push(node.low);
            stack.push(node.high);
        }
        true
    }

    fn implies(&self, a: u32, b: u32, memo: &mut FastMap<(u32, u32), bool>) -> bool {
        if a == ZERO || b == ONE || a == b {
            return true;
        }
        if a == ONE || b == ZERO {
            return false;
        }
        if let Some(&v) = memo.get(&(a, b)) {
            return v;
        }
        let level = self.level(a).min(self.level(b));
        let (a0, a1) = self.cofactors(a, level);
        let (b0, b1) = self.cofactors(b, level);
        let v = self.implies(a0, b0, memo) && self.implies(a1, b1, memo);
        memo.insert((a, b), v);
        v
    }

    /// The minimal solutions of the monotone function `root`, as a ZBDD.
    pub(crate) fn minimal_solutions(&self, root: u32, budget: usize) -> Result<Zbdd, FtaError> {
        let mut zbdd = Zbdd::new(budget);
        let mut memo = FastMap::default();
        zbdd.root = self.minsol(root, &mut zbdd, &mut memo)?;
        Ok(zbdd)
    }

    fn minsol(
        &self,
        id: u32,
        zbdd: &mut Zbdd,
        memo: &mut FastMap<u32, u32>,
    ) -> Result<u32, FtaError> {
        if id <= ONE {
            return Ok(id);
        }
        if let Some(&z) = memo.get(&id) {
            return Ok(z);
        }
        let node = self.nodes[id as usize];
        let low = self.minsol(node.low, zbdd, memo)?;
        let high = self.minsol(node.high, zbdd, memo)?;
        let high = zbdd.without(high, low)?;
        let z = zbdd.make(node.level, low, high)?;
        memo.insert(id, z);
        Ok(z)
    }
}

fn reachable(nodes: &[Node], root: u32) -> usize {
    let mut seen = vec![false; nodes.len()];
    let mut stack = vec![root];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        if seen[id as usize] {
            continue;
        }
        seen[id as usize] = true;
        count += 1;
        if id > ONE {
            let node = nodes[id as usize];
            stack.push(node.low);
            stack.push(node.high);
        }
    }
    count
}

/// Fold `ids` with `op` as a balanced tree rather than left to right: the
/// intermediate diagrams stay the size of the halves they combine, instead
/// of the whole accumulated function meeting every argument in turn.
pub(crate) fn fold_balanced(
    bdd: &mut Bdd,
    conjunction: bool,
    mut ids: Vec<u32>,
) -> Result<u32, FtaError> {
    if ids.is_empty() {
        return Ok(if conjunction { ONE } else { ZERO });
    }
    while ids.len() > 1 {
        let mut next = Vec::with_capacity(ids.len().div_ceil(2));
        for pair in ids.chunks(2) {
            next.push(match pair {
                [a, b] if conjunction => bdd.and(*a, *b)?,
                [a, b] => bdd.or(*a, *b)?,
                [a] => *a,
                _ => ZERO,
            });
        }
        ids = next;
    }
    Ok(ids[0])
}

/// A zero-suppressed decision diagram: a family of sets of levels.
#[derive(Clone)]
pub(crate) struct Zbdd {
    nodes: Vec<Node>,
    unique: FastMap<Node, u32>,
    without_memo: FastMap<(u32, u32), u32>,
    budget: usize,
    /// The family built by [`Bdd::minimal_solutions`].
    pub(crate) root: u32,
}

impl Zbdd {
    /// An empty family store that refuses to grow past `budget` nodes.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            nodes: vec![TERMINAL, TERMINAL],
            unique: FastMap::default(),
            without_memo: FastMap::default(),
            budget,
            root: ZERO,
        }
    }

    /// How many nodes the diagram holds, terminals included.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The nodes of the family's diagram, terminals included.
    pub(crate) fn size(&self) -> usize {
        reachable(&self.nodes, self.root)
    }

    fn make(&mut self, level: u32, low: u32, high: u32) -> Result<u32, FtaError> {
        if high == ZERO {
            return Ok(low);
        }
        let node = Node { level, low, high };
        if let Some(&id) = self.unique.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= self.budget {
            return Err(FtaError::TooLarge(format!(
                "the minimal-cut-set diagram outgrew its budget of {} nodes; \
                 raise max_bdd_nodes",
                self.budget
            )));
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.unique.insert(node, id);
        Ok(id)
    }

    fn holds_empty_set(&self, mut id: u32) -> bool {
        while id > ONE {
            id = self.nodes[id as usize].low;
        }
        id == ONE
    }

    /// The sets of `p` that contain no set of `q`.
    fn without(&mut self, p: u32, q: u32) -> Result<u32, FtaError> {
        if p == ZERO || q == ZERO {
            return Ok(p);
        }
        if p == q || self.holds_empty_set(q) {
            return Ok(ZERO);
        }
        if p == ONE {
            // {∅} contains a set of q only when q holds ∅, excluded above.
            return Ok(ONE);
        }
        if let Some(&id) = self.without_memo.get(&(p, q)) {
            return Ok(id);
        }
        let np = self.nodes[p as usize];
        let nq = self.nodes[q as usize];
        let id = if q == ONE || np.level < nq.level {
            let high = self.without(np.high, q)?;
            let low = self.without(np.low, q)?;
            self.make(np.level, low, high)?
        } else if np.level > nq.level {
            // No set of p holds q's top variable: only q's sets without it
            // can be contained.
            self.without(p, nq.low)?
        } else {
            let high = self.without(np.high, nq.high)?;
            let high = self.without(high, nq.low)?;
            let low = self.without(np.low, nq.low)?;
            self.make(np.level, low, high)?
        };
        self.without_memo.insert((p, q), id);
        Ok(id)
    }

    /// The family of `sets`, each a strictly increasing list of levels.
    /// Sorted lexicographically, the sets starting with the smallest level
    /// form one contiguous block: that block goes high (without the level),
    /// the rest low, so each set is visited once per level it holds.
    pub(crate) fn family_of(&mut self, sets: &[Vec<u32>]) -> Result<u32, FtaError> {
        let mut slices: Vec<&[u32]> = sets.iter().map(Vec::as_slice).collect();
        slices.sort_unstable();
        slices.dedup();
        let has_empty = slices.first().is_some_and(|s| s.is_empty());
        let start = usize::from(has_empty);
        self.build_family(&slices[start..], has_empty)
    }

    /// `sets` sorted, none empty; `has_empty` adds the empty set.
    fn build_family(&mut self, sets: &[&[u32]], has_empty: bool) -> Result<u32, FtaError> {
        let Some(first) = sets.first() else {
            return Ok(if has_empty { ONE } else { ZERO });
        };
        let level = first[0];
        let end = sets.partition_point(|s| s[0] == level);
        let tails: Vec<&[u32]> = sets[..end].iter().map(|s| &s[1..]).collect();
        // Sorted tails put an empty one (the set {level} itself) first.
        let tail_empty = tails.first().is_some_and(|t| t.is_empty());
        let high = self.build_family(&tails[usize::from(tail_empty)..], tail_empty)?;
        let low = self.build_family(&sets[end..], has_empty)?;
        self.make(level, low, high)
    }

    /// The minimal sets of the family `f`: those containing no other set
    /// of it (Rauzy's minimal-solution operator, on a ZBDD).
    pub(crate) fn minimal(&mut self, f: u32) -> Result<u32, FtaError> {
        let mut memo = FastMap::default();
        self.minimal_memo(f, &mut memo)
    }

    fn minimal_memo(&mut self, f: u32, memo: &mut FastMap<u32, u32>) -> Result<u32, FtaError> {
        if f <= ONE {
            return Ok(f);
        }
        if let Some(&id) = memo.get(&f) {
            return Ok(id);
        }
        let node = self.nodes[f as usize];
        let low = self.minimal_memo(node.low, memo)?;
        let high = self.minimal_memo(node.high, memo)?;
        let high = self.without(high, low)?;
        let id = self.make(node.level, low, high)?;
        memo.insert(f, id);
        Ok(id)
    }

    /// The pivotal upper bound of the union of the family `root` (Rauzy
    /// 2020, definition 13.1.5): on the decomposition on the top variable
    /// `E`, `PUB = p(E) P1 + P0 - p(E) P1 P0`, with `P1` the bound of the
    /// sets holding `E` (without it) and `P0` of the others. Returns the
    /// bound and its partial derivative with respect to each level's
    /// probability, by one forward and one backward pass.
    pub(crate) fn pivotal(&self, root: u32, p: &[f64]) -> (f64, Vec<f64>) {
        let n = self.nodes.len();
        let mut value = vec![0.0; n];
        value[ONE as usize] = 1.0;
        for (id, node) in self.nodes.iter().enumerate().skip(2) {
            if id as u32 > root {
                break;
            }
            let q = p[node.level as usize];
            let a = value[node.high as usize];
            let b = value[node.low as usize];
            value[id] = q * a + b - q * a * b;
        }
        let mut gradient = vec![0.0; p.len()];
        let mut adjoint = vec![0.0; n];
        adjoint[root as usize] = 1.0;
        for id in (2..=root as usize).rev() {
            let w = adjoint[id];
            if w == 0.0 {
                continue;
            }
            let node = self.nodes[id];
            let q = p[node.level as usize];
            let a = value[node.high as usize];
            let b = value[node.low as usize];
            gradient[node.level as usize] += w * a * (1.0 - b);
            adjoint[node.high as usize] += w * q * (1.0 - b);
            adjoint[node.low as usize] += w * (1.0 - q * a);
        }
        (value[root as usize], gradient)
    }

    /// The rare-event approximation: the sum over the family of each set's
    /// probability.
    pub(crate) fn rare_event(&self, root: u32, p: &[f64]) -> f64 {
        let mut out = vec![0.0; self.nodes.len()];
        out[ONE as usize] = 1.0;
        for (id, node) in self.nodes.iter().enumerate().skip(2) {
            if id as u32 > root {
                break;
            }
            out[id] = out[node.low as usize] + p[node.level as usize] * out[node.high as usize];
        }
        out[root as usize]
    }

    /// Which levels appear in the family `root`.
    pub(crate) fn support(&self, root: u32, levels: usize) -> Vec<bool> {
        let mut out = vec![false; levels];
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if id <= ONE || seen[id as usize] {
                continue;
            }
            seen[id as usize] = true;
            let node = self.nodes[id as usize];
            out[node.level as usize] = true;
            stack.push(node.low);
            stack.push(node.high);
        }
        out
    }

    /// `Σ over sets of Π weight(level)`, saturating: the number of cut
    /// sets once each level stands for `weight` sets of its own.
    pub(crate) fn weighted_count(&self, weight: &[u128]) -> u128 {
        let mut out = vec![0u128; self.nodes.len()];
        out[ONE as usize] = 1;
        for (id, node) in self.nodes.iter().enumerate().skip(2) {
            let high = out[node.high as usize].saturating_mul(weight[node.level as usize]);
            out[id] = out[node.low as usize].saturating_add(high);
        }
        out[self.root as usize]
    }

    /// Every set of the family, each as the list of its levels.
    pub(crate) fn sets(&self) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.walk(self.root, &mut path, &mut out);
        out
    }

    fn walk(&self, id: u32, path: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if id == ZERO {
            return;
        }
        if id == ONE {
            out.push(path.clone());
            return;
        }
        let node = self.nodes[id as usize];
        path.push(node.level);
        self.walk(node.high, path, out);
        path.pop();
        self.walk(node.low, path, out);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// `without` in full generality, including the case the minimal-
    /// solution recursion never exercises (a set of `q` without the shared
    /// top variable contained in a set of `p` with it): with levels
    /// x = 0, y = 1, z = 2, `{{x, y}} without {{x, z}, {y}}` is empty
    /// because `{y}` is contained in `{x, y}`.
    #[test]
    fn without_removes_a_superset_across_the_shared_variable() {
        let mut z = Zbdd::new(100);
        let y = z.make(1, ZERO, ONE).unwrap(); // {{y}}
        let p = z.make(0, ZERO, y).unwrap(); // {{x, y}}
        let zz = z.make(2, ZERO, ONE).unwrap(); // {{z}}
        let q = z.make(0, y, zz).unwrap(); // {{x, z}, {y}}
        assert_eq!(z.without(p, q).unwrap(), ZERO);
        // And what is not contained survives: {{x, y}} without {{x, z}}.
        let q = z.make(0, ZERO, zz).unwrap();
        z.root = z.without(p, q).unwrap();
        assert_eq!(z.sets(), vec![vec![0, 1]]);
    }
}
