//! Counted work: the machine-independent units every later performance
//! gate of the continuous-flow plan is measured against.
//!
//! Wall-clock varies with the machine, the allocator and the load; these
//! counts do not. They are captured here, at the branch point of the
//! continuous-flow work, because a later unit cannot capture a baseline
//! retroactively.
//!
//! Two counters (flow sweeps, allocation capping passes) report zero on
//! this corpus, because no fixture in it declares a conservative
//! distribution operator. The columns exist from this unit onward rather
//! than growing mid-plan, and their staying at zero is what says the
//! affordance is additive.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::Model;

/// The cross-validation corpus is absent from some checkouts of this
/// project. A test that pins a fixture's behaviour reports that it did
/// not run rather than failing, so the suite stays green either way.
fn fixture(name: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/tests/validation/fixtures")
        .join(format!("{name}.json"));
    match std::fs::read_to_string(&path) {
        Ok(json) => Some(json),
        Err(_) => {
            eprintln!("skipped: no cross-validation corpus at {}", path.display());
            None
        }
    }
}

fn run(json: &str, t_max: f64) -> raichu_core::SimulationResult {
    let model = Model::from_json(json).expect("fixture JSON parses");
    let compiled = CompiledModel::compile(&model).expect("fixture compiles");
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

/// Every counter is reported for an existing fixture, and two runs of the
/// same model report the same counts.
#[test]
fn counted_work_is_reported_and_reproducible() {
    let Some(json) = fixture("tank_01") else {
        return;
    };
    let first = run(&json, 40.0);
    let second = run(&json, 40.0);
    assert_eq!(
        first.work, second.work,
        "counted work is reproducible run to run"
    );

    let work = first.work;
    assert!(work.explicit_evaluations > 0, "{work:?}");
    assert!(work.solver_steps_accepted > 0, "{work:?}");
    assert!(work.segments > 0, "{work:?}");
    assert!(work.margin_evaluations > 0, "{work:?}");
    assert!(work.immediate_guard_scans > 0, "{work:?}");
    // No flow network exists yet: wired, reported, zero.
    assert_eq!(work.flow_sweeps, 0, "{work:?}");
    assert_eq!(work.allocation_capping_passes, 0, "{work:?}");
}

/// A discrete-only model runs no solver segment at all: the null point the
/// later continuous measurements are read against.
#[test]
fn a_discrete_model_reports_no_continuous_work() {
    let Some(json) = fixture("delay_001") else {
        return;
    };
    let work = run(&json, 18.0).work;
    assert_eq!(work.segments, 0, "{work:?}");
    assert_eq!(work.solver_steps_accepted, 0, "{work:?}");
    assert_eq!(work.solver_steps_rejected, 0, "{work:?}");
    assert_eq!(work.margin_evaluations, 0, "{work:?}");
}

/// The counters are serialized with the rest of the result, which is how
/// the benchmark script reads them across the Python binding.
#[test]
fn counted_work_is_serialized_with_the_result() {
    let Some(json) = fixture("tank_01") else {
        return;
    };
    let result = run(&json, 40.0);
    let json = serde_json::to_value(&result).expect("the result serializes");
    let work = &json["work"];
    for counter in [
        "explicit_evaluations",
        "solver_steps_accepted",
        "solver_steps_rejected",
        "segments",
        "flow_sweeps",
        "allocation_capping_passes",
        "margin_evaluations",
        "immediate_guard_scans",
    ] {
        assert!(
            work[counter].is_u64(),
            "counter `{counter}` is reported: {work}"
        );
    }
}
