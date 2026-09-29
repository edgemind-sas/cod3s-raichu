//! Splitting selection, result consistency, compatibility and typed failures.
#![allow(clippy::unwrap_used, clippy::panic)]
use raichu_model::Model;
use raichu_montecarlo::SplittingError;
use raichu_quantify::{
    quantify, read_quantification, Detail, Method, QuantifyError, Study, TargetProbability,
};
use serde_json::json;

fn method() -> Method {
    Method::from_parts(
        "splitting",
        &json!({
            "importance": {
                "kind": "attribute",
                "name": "sys.score"
            },
            "particles": 100,
            "batches": 4,
            "confidence": 0.99
        }),
    )
    .unwrap()
}
fn model() -> Model {
    serde_json::from_value(json!({
        "name": "split_contract",
        "components": [
            {
                "name": "sys",
                "attributes": [
                    {
                        "name": "score",
                        "kind": "float",
                        "init": {
                            "kind": "float",
                            "value": 0.0
                        }
                    }
                ],
                "automata": [
                    {
                        "name": "life",
                        "states": [
                            "safe",
                            "mid",
                            "lost"
                        ],
                        "init": "safe",
                        "transitions": [
                            {
                                "name": "first",
                                "source": "safe",
                                "targets": [
                                    "mid"
                                ],
                                "distrib": "exp",
                                "rate": 0.5,
                                "effects": [
                                    {
                                        "target": {
                                            "component": "sys",
                                            "attribute": "score"
                                        },
                                        "value": {
                                            "op": "const",
                                            "value": {
                                                "kind": "float",
                                                "value": 1.0
                                            }
                                        }
                                    }
                                ]
                            },
                            {
                                "name": "second",
                                "source": "mid",
                                "targets": [
                                    "lost"
                                ],
                                "distrib": "exp",
                                "rate": 0.5
                            }
                        ]
                    }
                ]
            }
        ],
        "targets": [
            {
                "name": "lost",
                "component": "sys",
                "automaton": "life",
                "state": "lost"
            }
        ]
    }))
    .unwrap()
}
#[test]
fn splitting_method_is_available() {
    assert_eq!(method().name(), "splitting");
}
#[test]
fn splitting_round_trip_and_older_version_refusal() {
    let q = quantify(&model(), &Study::new("lost", 1.0), &method()).unwrap();
    assert_eq!(q.version, 3);
    let TargetProbability::SplittingEstimate { batches, .. } = &q.probability else {
        panic!("expected splitting")
    };
    assert_eq!(*batches, 4);
    let Detail::Splitting(detail) = &q.detail else {
        panic!("expected splitting detail")
    };
    assert_eq!(detail.batches.len(), 4);
    let text = q.to_json().unwrap();
    let back = read_quantification(&text).unwrap();
    assert_eq!(back, q);
    assert_eq!(back.to_json().unwrap(), text);
    // Preserve the engine's float bits before changing only one member.
    let base: serde_json::Value = serde_json::to_value(&q).unwrap();
    let mut old = base.clone();
    old["version"] = json!(2);
    assert!(read_quantification(&old.to_string())
        .unwrap_err()
        .to_string()
        .contains("version 3"));
    for (field, value) in [
        ("estimate", json!(0.123456)),
        ("batches", json!(99)),
        ("extinct_batches", json!(99)),
        ("method", json!("wilson")),
    ] {
        let mut forged = base.clone();
        forged["probability"][field] = value;
        assert!(read_quantification(&forged.to_string()).is_err(), "{field}");
    }
    let mut forged = base;
    forged["provenance"]["seed"] = json!(77);
    assert!(read_quantification(&forged.to_string()).is_err());
}
#[test]
fn splitting_settings_are_named_and_caps_return_no_estimate() {
    let err = Method::from_parts(
        "splitting",
        &json!({
            "importance": {
                "kind": "attribute",
                "name": "sys.score"
            },
            "typo": 10
        }),
    )
    .unwrap_err();
    assert!(matches!(err,QuantifyError::SettingNotApplicable{setting,..} if setting=="typo"));
    let m = Method::from_parts(
        "splitting",
        &json!({
            "importance": {
                "kind": "attribute",
                "name": "sys.score"
            },
            "particles": 100,
            "batches": 2,
            "max_iterations": 0
        }),
    )
    .unwrap();
    assert!(matches!(
        quantify(&model(), &Study::new("lost", 1.0), &m),
        Err(QuantifyError::Splitting(
            SplittingError::IterationLimit { .. }
        ))
    ));
}
