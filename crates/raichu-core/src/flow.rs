//! The arithmetic of the **conservative distribution operator**: one
//! available quantity, one demand per consumer, one allocated quantity per
//! consumer, under a declared policy.
//!
//! Kept apart from the engine on purpose. It is pure: same inputs, same
//! outputs, no state, no clock, no randomness, so it can be exercised
//! directly by a property test over random demand vectors instead of only
//! through a simulated model.
//!
//! # Two properties this module owes the rest of the engine
//!
//! **Conservation.** No consumer receives more than it asked for, and the
//! quantities handed out never exceed what was available. Both hold by
//! construction here, not by a check afterwards: a share is capped at the
//! demand before it is written, and each capping pass redistributes only
//! what the previous one left.
//!
//! **Order independence** (KTD4 of the continuous-flow plan). Each
//! consumer's share is a function of its own demand and of the totals,
//! never of its position in the sweep, so two consumers with equal demands
//! receive equal quantities. The two genuine ties are broken by the
//! **compiled declaration index**, which the compiler bakes into
//! [`CPolicy::Priority`] as a serving order: the tie-break is therefore a
//! property of the model file, never of the engine's worklist, of a hash
//! order, or of the thread that happened to run the sweep.

/// A compiled allocation policy: the per-connection parameters resolved to
/// the connection declaration order of the operator's out port.
#[derive(Debug, Clone, PartialEq)]
pub enum CPolicy {
    /// Split proportionally to demand.
    Proportional,
    /// Split by fixed shares, one per connection (validated to sum to 1).
    Shares(Vec<f64>),
    /// Serve in full, in this order, until the quantity runs out. The
    /// order is the connection indices sorted by (declared rank,
    /// declaration index), computed once at compile time, which is what
    /// makes an equal-rank tie break by declaration index.
    Priority(Vec<usize>),
}

impl CPolicy {
    /// Weight of consumer `index` in a weighted split, or `None` for a
    /// policy that is not a weighted split.
    fn weight(&self, index: usize, demands: &[f64]) -> Option<f64> {
        match self {
            CPolicy::Proportional => demands.get(index).copied(),
            CPolicy::Shares(shares) => shares.get(index).copied(),
            CPolicy::Priority(_) => None,
        }
    }
}

/// Distribute `available` among `demands` under `policy`, writing one
/// quantity per consumer into `allocated` (same length and same order as
/// `demands`). Returns the number of **capping passes** performed, which
/// the engine reports as counted work.
///
/// `capped` is reusable scratch: the caller keeps one buffer across calls
/// so the hot path allocates nothing after the first pass. Its contents on
/// entry are irrelevant.
///
/// Inputs are taken as given: the caller (the engine) has already refused
/// non-finite values and clamped negative ones to zero, because a negative
/// availability or demand is not a quantity, and deciding that here would
/// hide it from the diagnostic.
///
/// # The capping loop
///
/// A weighted split can offer a consumer more than it asked for: fixed
/// shares of 50/50 offer 5 each out of 10, but a consumer wanting 2 must
/// not absorb the other 3. Each pass therefore caps every over-served
/// consumer at its demand and redistributes the remainder among the rest.
/// Every pass caps at least one consumer, so the loop is bounded by the
/// consumer count: it is a finite combinatorial search, not an iteration
/// to a tolerance.
pub fn allocate(
    policy: &CPolicy,
    available: f64,
    demands: &[f64],
    allocated: &mut [f64],
    capped: &mut Vec<bool>,
) -> u64 {
    let n = demands.len();
    if allocated.len() != n {
        // Cannot happen on the compiled path (the two vectors are built
        // from the same connection list); returning rather than panicking
        // keeps the library path panic-free.
        return 0;
    }
    for slot in allocated.iter_mut() {
        *slot = 0.0;
    }
    if let CPolicy::Priority(order) = policy {
        // Strict priority: serve in full, in the compiled order, until
        // the quantity runs out. The remainder tracking is sequential by
        // definition of the policy, and the order is a compile-time
        // property of the model, not of this sweep.
        let mut left = available;
        for &index in order {
            let (Some(&demand), Some(slot)) = (demands.get(index), allocated.get_mut(index)) else {
                continue;
            };
            let served = demand.min(left);
            *slot = served;
            left = (left - served).max(0.0);
        }
        return 1;
    }

    capped.clear();
    capped.resize(n, false);
    let mut passes = 0u64;
    // At most one pass per consumer plus the pass that finds nothing left
    // to cap.
    for _ in 0..=n {
        passes += 1;
        // What is left once every capped consumer has taken its demand.
        // Recomputed from `available` at each pass rather than decremented,
        // so the passes do not accumulate rounding.
        let mut left = available;
        let mut weight_total = 0.0;
        for index in 0..n {
            if capped[index] {
                left -= demands[index];
            } else {
                weight_total += policy.weight(index, demands).unwrap_or(0.0);
            }
        }
        let left = left.max(0.0);
        if weight_total <= 0.0 {
            // Every consumer is capped, or the uncapped ones carry no
            // weight at all: nothing more to hand out.
            break;
        }
        // Each share depends on its own weight and on the totals, never
        // on what the loop wrote before it: a permutation of the
        // consumers permutes the result and changes nothing else.
        let mut newly_capped = false;
        for index in 0..n {
            if capped[index] {
                allocated[index] = demands[index];
                continue;
            }
            let weight = policy.weight(index, demands).unwrap_or(0.0);
            let offered = left * weight / weight_total;
            if offered > demands[index] {
                capped[index] = true;
                allocated[index] = demands[index];
                newly_capped = true;
            } else {
                allocated[index] = offered;
            }
        }
        if !newly_capped {
            break;
        }
    }
    passes
}

// ---------------------------------------------------------------------
// The active set: which edges are saturated, and how far the state is
// from changing that answer.
// ---------------------------------------------------------------------

/// Per-edge tolerance the flow resolution converges to, and the **dead
/// band** of every active-set margin.
///
/// One constant serves both on purpose. The resolution promises each edge
/// only to within this tolerance, so a residual smaller than it is not a
/// crossing and must not end a segment; conversely a band any smaller
/// would let a freshly resolved network re-cross its own boundary on the
/// spot and chatter there.
///
/// It sits an order of magnitude above the event-location tolerance
/// ([`raichu_numeric::SolverParams::tol_event`], `1e-10`), which is what
/// makes the ordering above hold, and three orders below the `1e-6` the
/// validation contract tolerates on a cross-tool comparison of continuous
/// quantities, so it never eats that budget.
///
/// This is the default of `FlowConfig::tolerance`, which is where a
/// caller overrides it. The ordering above is what an override trades
/// away: loosened past the event-location tolerance it stays safe, but
/// tightened below it a freshly resolved network re-crosses its own
/// boundary on the spot.
pub const FLOW_TOLERANCE: f64 = 1e-9;

/// Dead band of an active-set margin for quantities of magnitude
/// `scale`: relative above unit scale, absolute below it, so a network
/// carrying megawatts and one carrying fractions of a unit both get a
/// band that means the same thing.
///
/// `tolerance` is the run's flow tolerance ([`FLOW_TOLERANCE`] by
/// default). It is a parameter rather than a read of the constant so a
/// run's dead band and its convergence test are the same number by
/// construction: a band computed from the constant while the resolution
/// settled to a looser figure would report crossings the resolution
/// never promised to avoid.
#[must_use]
pub fn flow_band(scale: f64, tolerance: f64) -> f64 {
    tolerance * scale.abs().max(1.0)
}

/// Saturation class of one consumer edge under a frozen active set.
///
/// This *is* the active set: the finite, combinatorial part of a
/// resolution, settled to exact equality before the flows are settled to
/// a tolerance. It is read off the operator's own capping outcome, never
/// recomputed by a parallel bookkeeping scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeClass {
    /// Weighted split: the consumer receives no more than it asked for
    /// and its weight still competes for the remainder.
    Open,
    /// Weighted split: the consumer is capped at its own demand and its
    /// weight has left the split.
    Capped,
    /// Priority: served in full.
    Full,
    /// Priority: the quantity ran out inside this edge.
    Partial,
    /// Priority: the quantity ran out before this edge.
    Unserved,
}

/// Read the active set of one operator off a completed [`allocate`]: the
/// capping flags for a weighted split, the served-in-full / part / not-at-all
/// classification for a priority order.
///
/// `capped` and `allocated` are the buffers [`allocate`] just wrote, so
/// this is the operator's own answer rather than a second opinion.
pub fn classify(
    policy: &CPolicy,
    demands: &[f64],
    allocated: &[f64],
    capped: &[bool],
    into: &mut Vec<EdgeClass>,
) {
    for (edge, &demand) in demands.iter().enumerate() {
        let class = match policy {
            CPolicy::Priority(_) => {
                let served = allocated.get(edge).copied().unwrap_or(0.0);
                if served >= demand {
                    EdgeClass::Full
                } else if served <= 0.0 {
                    EdgeClass::Unserved
                } else {
                    EdgeClass::Partial
                }
            }
            _ => {
                if capped.get(edge).copied().unwrap_or(false) {
                    EdgeClass::Capped
                } else {
                    EdgeClass::Open
                }
            }
        };
        into.push(class);
    }
}

/// Signed **active-set margin** of one edge, in the engine's event
/// convention: the value is negative while the edge keeps its frozen
/// class and reaches zero when the class is about to change, so the
/// solver locates the change as an ordinary boundary crossing.
///
/// `band` is the dead band ([`flow_band`]): the state must move that far
/// past the boundary before the crossing counts, which is what stops a
/// network resolved *on* a boundary from re-crossing it immediately.
///
/// The whole `classes` vector is needed, not just this edge's: under a
/// weighted split the quantity offered to one consumer depends on which
/// *other* consumers have left the split. A capped edge is tested by
/// leaving it out of the capped set and asking whether it would still be
/// over-served, which is the question its class answers.
#[must_use]
pub fn edge_margin(
    policy: &CPolicy,
    available: f64,
    demands: &[f64],
    classes: &[EdgeClass],
    edge: usize,
    band: f64,
) -> f64 {
    let Some(&class) = classes.get(edge) else {
        return f64::NEG_INFINITY;
    };
    let Some(&demand) = demands.get(edge) else {
        return f64::NEG_INFINITY;
    };
    let raw = match policy {
        CPolicy::Priority(order) => {
            // Running remainder ahead of this edge, *unclamped*: the
            // operator floors it at zero when serving, but the margin
            // needs the signed distance to know how far past empty the
            // supply is.
            let mut left = available;
            for &index in order {
                if index == edge {
                    break;
                }
                left -= demands.get(index).copied().unwrap_or(0.0);
            }
            match class {
                EdgeClass::Full => left - demand,
                EdgeClass::Partial => (left).min(demand - left),
                EdgeClass::Unserved => -left,
                // A weighted class on a priority operator cannot happen
                // (`classify` never produces one); reporting an
                // unreachable boundary is safer than a spurious crossing.
                EdgeClass::Open | EdgeClass::Capped => f64::INFINITY,
            }
        }
        _ => {
            // What this edge would be offered with the frozen capped set,
            // the edge itself excluded from that set so a capped edge is
            // asked whether it is still over-served.
            let mut left = available;
            let mut weight_total = 0.0;
            for (index, other) in classes.iter().enumerate() {
                if *other == EdgeClass::Capped && index != edge {
                    left -= demands.get(index).copied().unwrap_or(0.0);
                } else {
                    weight_total += policy.weight(index, demands).unwrap_or(0.0);
                }
            }
            let left = left.max(0.0);
            let offered = if weight_total > 0.0 {
                left * policy.weight(edge, demands).unwrap_or(0.0) / weight_total
            } else {
                0.0
            };
            match class {
                EdgeClass::Open => demand - offered,
                EdgeClass::Capped => offered - demand,
                EdgeClass::Full | EdgeClass::Partial | EdgeClass::Unserved => f64::INFINITY,
            }
        }
    };
    // `raw` is non-negative while the class holds. The event fires when
    // the state has moved a full band past the boundary.
    -raw - band
}
