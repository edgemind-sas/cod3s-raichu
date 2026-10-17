//! A basic event's probability under the measure a tree is quantified in.
//!
//! Under the probability without repair, an event is its law's
//! distribution at the mission time, repairs or not. Under the
//! unavailability, a repairable event is the probability of being in the
//! state it enters:
//!
//! - an exponential failure of rate λ with exponential repairs, their
//!   rates summed into µ (the first of competing exponential repairs to
//!   fire is exponential of the summed rate), is the two-state Markov
//!   chain started up: `q(t) = λ/(λ+µ) (1 - e^{-(λ+µ)t})`, exactly;
//! - any other pair is valued by its asymptotic unavailability
//!   `MTTR / (MTTF + MTTR)`, from the laws' means ([`Law::mean`]); a
//!   repair with no time of its own (on-demand, or immediate) has a mean
//!   time of zero, and competing repairs are combined as rates,
//!   `1 / MTTR = Σ 1 / MTTR_i`;
//! - an event whose repairs cannot all be read (a law no single law
//!   stands for), or whose failure has no time (an on-demand
//!   probability), keeps its distribution: the repair is ignored, and the
//!   envelope says so.
//!
//! An event without repair keeps its distribution under both measures.

use crate::{BasicEvent, FaultTreeMeasure, FtaError, Law};

/// How the unavailability values one event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Treatment {
    /// Nothing repairs it: its distribution.
    Unrepaired,
    /// Exponential failure and repairs: the closed form.
    ClosedForm {
        /// The failure rate λ.
        lambda: f64,
        /// The summed repair rate µ.
        mu: f64,
    },
    /// The asymptotic unavailability.
    Asymptotic {
        /// The mean time to failure.
        mttf: f64,
        /// The mean time to repair.
        mttr: f64,
    },
    /// Its repairs cannot be counted: its distribution.
    Ignored,
}

/// How the unavailability values `event`.
///
/// # Errors
/// [`FtaError::BadLaw`] for a failure or repair law whose parameters it
/// cannot take.
pub(crate) fn treatment(event: &BasicEvent) -> Result<Treatment, FtaError> {
    if event.repairs.is_empty() {
        return Ok(Treatment::Unrepaired);
    }
    let Some(laws) = event
        .repairs
        .iter()
        .map(|r| r.law.as_ref())
        .collect::<Option<Vec<&Law>>>()
    else {
        return Ok(Treatment::Ignored);
    };
    if let Law::Exponential { rate: lambda } = event.law {
        let rates: Option<Vec<f64>> = laws
            .iter()
            .map(|law| match law {
                Law::Exponential { rate } => Some(*rate),
                _ => None,
            })
            .collect();
        if let Some(rates) = rates {
            for law in &laws {
                // Checks the parameters the closed form reads.
                law.mean(&event.name)?;
            }
            return Ok(Treatment::ClosedForm {
                lambda,
                mu: rates.iter().sum(),
            });
        }
    }
    let Some(mttf) = event.law.mean(&event.name)? else {
        return Ok(Treatment::Ignored);
    };
    let mut repair_rate = 0.0;
    for law in laws {
        // An on-demand repair has no time: it is immediate.
        let mttr = law.mean(&event.name)?.unwrap_or(0.0);
        if mttr == 0.0 {
            return Ok(Treatment::Asymptotic { mttf, mttr: 0.0 });
        }
        repair_rate += 1.0 / mttr;
    }
    Ok(Treatment::Asymptotic {
        mttf,
        mttr: 1.0 / repair_rate,
    })
}

/// The probability of `event` at `mission_time` under `measure`.
///
/// # Errors
/// Those of [`Law::probability`] and [`treatment`].
pub(crate) fn event_probability(
    event: &BasicEvent,
    measure: FaultTreeMeasure,
    mission_time: Option<f64>,
) -> Result<f64, FtaError> {
    // Read first in every case: it checks the law and the mission time,
    // and it is the answer whenever the repairs are not counted.
    let distribution = event.law.probability(&event.name, mission_time)?;
    if measure == FaultTreeMeasure::ProbabilityWithoutRepair {
        return Ok(distribution);
    }
    Ok(match treatment(event)? {
        Treatment::Unrepaired | Treatment::Ignored => distribution,
        Treatment::ClosedForm { lambda, mu } => {
            let total = lambda + mu;
            // A timed law was read above, so the mission time is known.
            let t = mission_time.unwrap_or(0.0);
            if total == 0.0 {
                0.0
            } else {
                (lambda / total * -libm::expm1(-total * t)).clamp(0.0, 1.0)
            }
        }
        Treatment::Asymptotic { mttf, mttr } => {
            if mttr == 0.0 || mttf.is_infinite() {
                0.0
            } else {
                (mttr / (mttf + mttr)).clamp(0.0, 1.0)
            }
        }
    })
}
