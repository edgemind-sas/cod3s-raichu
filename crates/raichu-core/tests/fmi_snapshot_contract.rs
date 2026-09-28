//! Legacy checkpoints must not silently omit an imported FMU's state.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use raichu_core::{CoSimulationHost, CompiledModel, Engine, EngineConfig, EngineError};
use raichu_model::Model;
use serde_json::json;

#[test]
fn native_only_snapshot_apis_refuse_an_fmu() {
    let document = json!({
        "name": "fmu-snapshot-contract",
        "components": [{"name": "plant", "attributes": [
            {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ]}],
        "fmu_units": [{
            "name": "ode", "path": "Dahlquist.fmu", "step": 0.1,
            "outputs": [{"attribute": {"component": "plant", "attribute": "x"}, "variable": "x"}]
        }]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../raichu-fmi/tests/fixtures/reference-fmus/3.0");
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir, true).unwrap();
    let config = EngineConfig {
        t_max: 0.2,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config.clone(), &mut host).unwrap();

    assert!(matches!(
        engine.snapshot(),
        Err(EngineError::FmuSnapshotApi {
            operation: "snapshot capture",
            ..
        })
    ));
    let snapshot = engine.try_snapshot().unwrap();
    engine.step().unwrap();
    let advanced = engine.attribute("plant.x");
    assert!(matches!(
        engine.restore(&snapshot),
        Err(EngineError::FmuSnapshotApi {
            operation: "snapshot restore",
            ..
        })
    ));
    assert_eq!(engine.attribute("plant.x"), advanced);
    drop(engine);

    assert!(matches!(
        Engine::from_snapshot(&compiled, config.clone(), &snapshot),
        Err(EngineError::FmuSnapshotApi {
            operation: "engine reconstruction",
            ..
        })
    ));
    let restored =
        Engine::from_snapshot_with_host(&compiled, config, &snapshot, &mut host).unwrap();
    assert_eq!(restored.current_time(), 0.0);
}
