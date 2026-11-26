//! Graph algorithms the model layer runs on dependency relations.

/// Strongly connected components of `edges` (one node per index, edge
/// `i → j` when `edges[i]` contains `j`), Tarjan's algorithm, iterative
/// so a deep chain of equations cannot exhaust the stack.
///
/// Components come out in reverse topological order (a component before
/// any component that reaches it), each as its nodes in unspecified
/// order; a singleton without a self-edge is a node of its own and is
/// still emitted, letting the caller decide what a component means.
///
/// Shared with the engine, which reads the cycles of its
/// propagations with it: one implementation of the graph pass for the two
/// layers that need one.
pub fn strongly_connected_components(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    #[derive(Clone, Copy)]
    struct Frame {
        node: usize,
        child: usize,
    }
    let mut next_index = 0usize;
    let mut index_of = vec![usize::MAX; edges.len()];
    let mut lowlink = vec![0usize; edges.len()];
    let mut on_stack = vec![false; edges.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    for root in 0..edges.len() {
        if index_of[root] != usize::MAX {
            continue;
        }
        let mut frames = vec![Frame {
            node: root,
            child: 0,
        }];
        while let Some(frame) = frames.pop() {
            if frame.child == 0 {
                index_of[frame.node] = next_index;
                next_index += 1;
                lowlink[frame.node] = index_of[frame.node];
                stack.push(frame.node);
                on_stack[frame.node] = true;
            }
            let mut child = frame.child;
            let mut descended = false;
            while child < edges[frame.node].len() {
                let successor = edges[frame.node][child];
                if index_of[successor] == usize::MAX {
                    frames.push(Frame {
                        node: frame.node,
                        child: child + 1,
                    });
                    frames.push(Frame {
                        node: successor,
                        child: 0,
                    });
                    descended = true;
                    break;
                }
                if on_stack[successor] {
                    lowlink[frame.node] = lowlink[frame.node].min(index_of[successor]);
                }
                child += 1;
            }
            if descended {
                continue;
            }
            if lowlink[frame.node] == index_of[frame.node] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack[member] = false;
                    component.push(member);
                    if member == frame.node {
                        break;
                    }
                }
                components.push(component);
            }
            if let Some(parent) = frames.last_mut() {
                lowlink[parent.node] = lowlink[parent.node].min(lowlink[frame.node]);
            }
        }
    }
    components
}

/// Whether the `rows × columns` sparsity pattern admits an assignment of
/// one distinct column to every row (a perfect matching), Kuhn's
/// augmenting paths, rows in order and candidates ascending, so the
/// answer : and the refusal that a `false` produces : is a property of
/// the pattern alone.
pub(crate) fn has_perfect_matching(pattern: &[Vec<usize>], columns: usize) -> bool {
    if pattern.len() != columns {
        return false;
    }
    // Column → row currently assigned to it.
    let mut owner: Vec<Option<usize>> = vec![None; columns];
    for row in 0..pattern.len() {
        let mut seen = vec![false; columns];
        if !assign(row, pattern, &mut owner, &mut seen) {
            return false;
        }
    }
    true
}

/// Augmenting-path search for `row` (see [`has_perfect_matching`]).
fn assign(
    row: usize,
    pattern: &[Vec<usize>],
    owner: &mut [Option<usize>],
    seen: &mut [bool],
) -> bool {
    for &column in &pattern[row] {
        if seen[column] {
            continue;
        }
        seen[column] = true;
        match owner[column] {
            None => {
                owner[column] = Some(row);
                return true;
            }
            Some(previous) => {
                if assign(previous, pattern, owner, seen) {
                    owner[column] = Some(row);
                    return true;
                }
            }
        }
    }
    false
}
