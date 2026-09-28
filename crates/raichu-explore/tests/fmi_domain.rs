//! Exact exploration rejects imported state before any native code loads.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_explore::{exact_domain_report, DomainViolation};
use raichu_model::Model;
use serde_json::json;

#[test]
fn an_fmu_is_outside_the_exact_domain() {
    let document = json!({
        "name": "opaque",
        "components": [{"name": "plant", "attributes": [
            {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ]}],
        "fmu_units": [{
            "name": "physics", "path": "not-opened.fmu", "step": 0.1,
            "outputs": [{"attribute": {"component": "plant", "attribute": "x"}, "variable": "x"}]
        }]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    assert_eq!(
        exact_domain_report(&compiled),
        vec![DomainViolation::Fmu {
            unit: "physics".to_owned()
        }]
    );
}
