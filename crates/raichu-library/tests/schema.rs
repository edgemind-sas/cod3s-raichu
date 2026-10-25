//! The published JSON Schemas of the three documents: derived from the
//! reader's types, committed under `docs/reference/schema/`, and checked
//! against what serde writes and what the reader accepts.
//!
//! `RAICHU_UPDATE_SCHEMA=1 cargo test -p raichu-library --test schema`
//! rewrites the committed files, the published copies under
//! `docs/reference/schema/` and the ones the Python package ships.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_expr::{AggOp, AttrRef, BoolOp, CmpOp, Expr, PortRef, StateRef, Value as V};
use raichu_library::schema::{document_schemas, DIALECT};
use raichu_library::{expand, parse_library, parse_study, parse_system};
use raichu_model::{
    Allocation, AllocationPolicy, ConsumerParam, Distrib, Indicator, IndicatorTarget,
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/reference/schema")
}

/// The copy the Python package ships (`pyraichu.library.schema`).
fn package_schema_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bindings/pyraichu/python/pyraichu/schema")
}

fn library_validator() -> jsonschema::Validator {
    jsonschema::validator_for(&document_schemas().library).unwrap()
}

fn system_validator() -> jsonschema::Validator {
    jsonschema::validator_for(&document_schemas().system).unwrap()
}

fn study_validator() -> jsonschema::Validator {
    jsonschema::validator_for(&document_schemas().study).unwrap()
}

fn assert_valid(validator: &jsonschema::Validator, document: &Value) {
    let errors: Vec<String> = validator
        .iter_errors(document)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{document}\n{}", errors.join("\n"));
}

fn assert_invalid(validator: &jsonschema::Validator, document: &Value) {
    assert!(
        !validator.is_valid(document),
        "the schema accepts {document}"
    );
}

fn minimal_library() -> Value {
    json!({
        "raichu_library": {"format": 1},
        "library": {
            "name": "demo",
            "version": "1.0.0",
            "types": [{
                "name": "Lamp",
                "parameters": [{"name": "power", "kind": "number", "default": 1.0}],
                "body": {"attributes": [
                    {"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}
                ]}
            }]
        }
    })
}

fn minimal_system() -> Value {
    json!({
        "raichu_system": {"format": 1},
        "system": {
            "name": "room",
            "library": {"name": "demo", "version": "1.0.0"},
            "instances": [{"name": "L1", "type": "Lamp", "parameters": {"power": 2.0}}]
        }
    })
}

fn minimal_study() -> Value {
    json!({
        "raichu_study": {"format": 1},
        "study": {
            "name": "s1",
            "system": "room",
            "modes": [{"mode": "burnt", "instances": ["L1"],
                       "occurrence": {"distrib": "exp", "rate": 0.1}}],
            "simulation": {"method": "monte_carlo", "nb_runs": 10, "horizon": 5.0}
        }
    })
}

/// A library type whose body is one automaton with `transition`.
fn library_with_transition(transition: Value) -> Value {
    json!({
        "raichu_library": {"format": 1},
        "library": {
            "name": "demo",
            "version": "1.0.0",
            "types": [{
                "name": "Pump",
                "parameters": [{"name": "rate", "kind": "number", "default": 0.1}],
                "body": {"automata": [{
                    "name": "life", "states": ["ok", "ko"], "init": "ok",
                    "transitions": [transition]
                }]}
            }]
        }
    })
}

fn every_distrib() -> Vec<Distrib> {
    let all = vec![
        Distrib::Delay { time: 2.0 },
        Distrib::Inst { probs: vec![0.25] },
        Distrib::Watched,
        Distrib::Exp {
            rate: Some(0.1),
            rate_expr: None,
        },
        Distrib::Exp {
            rate: None,
            rate_expr: Some(Expr::attr("Pump", "lambda")),
        },
        Distrib::Weibull {
            shape: 1.5,
            scale: 10.0,
        },
        Distrib::Lognormal {
            mu: 0.0,
            sigma: 1.0,
        },
        Distrib::Gamma {
            shape: 2.0,
            scale: 3.0,
        },
        Distrib::Uniform {
            low: 1.0,
            high: 2.0,
        },
        Distrib::Empirical {
            points: vec![(0.0, 0.5), (1.0, 1.0)],
        },
    ];
    // A new variant fails to compile here until it is listed above.
    for distrib in &all {
        match distrib {
            Distrib::Delay { .. }
            | Distrib::Inst { .. }
            | Distrib::Watched
            | Distrib::Exp { .. }
            | Distrib::Weibull { .. }
            | Distrib::Lognormal { .. }
            | Distrib::Gamma { .. }
            | Distrib::Uniform { .. }
            | Distrib::Empirical { .. } => {}
        }
    }
    all
}

fn every_expr() -> Vec<Expr> {
    let x = || Box::new(Expr::attr("Pump", "x"));
    let all = vec![
        Expr::Const {
            value: V::Float(1.0),
        },
        Expr::Const { value: V::Int(3) },
        Expr::bool(true),
        Expr::attr("Pump", "x"),
        Expr::PortAgg {
            port: PortRef {
                component: "Pump".into(),
                port: "in".into(),
            },
            agg: AggOp::Median,
            channel: Some("flow".into()),
        },
        Expr::Cmp {
            cmp: CmpOp::Ge,
            lhs: x(),
            rhs: x(),
        },
        Expr::Bool {
            bool_op: BoolOp::Not,
            args: vec![Expr::bool(false)],
        },
        Expr::StateActive {
            state: StateRef {
                component: "Pump".into(),
                automaton: "life".into(),
                state: "ok".into(),
            },
        },
        Expr::Add {
            args: vec![*x(), *x()],
        },
        Expr::Sub { lhs: x(), rhs: x() },
        Expr::Mul { args: vec![*x()] },
        Expr::Div { lhs: x(), rhs: x() },
        Expr::Min { args: vec![*x()] },
        Expr::Max { args: vec![*x()] },
        Expr::If {
            cond: Box::new(Expr::bool(true)),
            then: x(),
            otherwise: x(),
        },
        Expr::Sin { arg: x() },
        Expr::Exp { arg: x() },
        Expr::Time,
    ];
    for expr in &all {
        match expr {
            Expr::Const { .. }
            | Expr::Attr { .. }
            | Expr::PortAgg { .. }
            | Expr::Cmp { .. }
            | Expr::Bool { .. }
            | Expr::StateActive { .. }
            | Expr::Add { .. }
            | Expr::Sub { .. }
            | Expr::Mul { .. }
            | Expr::Div { .. }
            | Expr::Min { .. }
            | Expr::Max { .. }
            | Expr::If { .. }
            | Expr::Sin { .. }
            | Expr::Exp { .. }
            | Expr::Time => {}
        }
    }
    all
}

#[test]
fn the_committed_schemas_are_the_generated_ones() {
    let schemas = document_schemas();
    let update = std::env::var("RAICHU_UPDATE_SCHEMA").is_ok_and(|v| v == "1");
    let mut stale = Vec::new();
    for (file, schema) in [
        ("raichu-library.schema.json", &schemas.library),
        ("raichu-system.schema.json", &schemas.system),
        ("raichu-study.schema.json", &schemas.study),
    ] {
        let generated = format!("{}\n", serde_json::to_string_pretty(schema).unwrap());
        for dir in [schema_dir(), package_schema_dir()] {
            let path = dir.join(file);
            if update {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(&path, &generated).unwrap();
            } else if std::fs::read_to_string(&path).ok().as_deref() != Some(generated.as_str()) {
                stale.push(path.display().to_string());
            }
        }
    }
    assert!(
        stale.is_empty(),
        "{stale:?} differ from the generated schemas; \
         run `RAICHU_UPDATE_SCHEMA=1 cargo test -p raichu-library --test schema` and commit"
    );
}

#[test]
fn the_schemas_are_draft_2020_12_schemas() {
    let schemas = document_schemas();
    for schema in [&schemas.library, &schemas.system, &schemas.study] {
        assert_eq!(schema["$schema"], DIALECT);
        jsonschema::meta::validate(schema).unwrap();
    }
}

#[test]
fn the_minimal_documents_validate() {
    assert_valid(&library_validator(), &minimal_library());
    assert_valid(&system_validator(), &minimal_system());
    assert_valid(&study_validator(), &minimal_study());
}

#[test]
fn a_document_of_another_kind_or_revision_does_not_validate() {
    assert_invalid(&system_validator(), &minimal_library());
    let mut study = minimal_study();
    study["raichu_study"]["format"] = json!(2);
    assert_invalid(&study_validator(), &study);
    let mut study = minimal_study();
    study["model"] = json!({});
    assert_invalid(&study_validator(), &study);
}

#[test]
fn every_law_serde_writes_validates_and_a_misspelled_law_does_not() {
    let study = study_validator();
    let library = library_validator();
    for distrib in every_distrib() {
        let law = serde_json::to_value(&distrib).unwrap();
        let mut document = minimal_study();
        document["study"]["modes"][0]["occurrence"] = law.clone();
        document["study"]["modes"][0]["repair"] = law.clone();
        assert_valid(&study, &document);
        // The same law flattened into a transition of a type body.
        let mut transition = json!({"name": "fail", "source": "ok", "targets": ["ko"]});
        for (key, value) in law.as_object().unwrap() {
            transition[key] = value.clone();
        }
        assert_valid(&library, &library_with_transition(transition));
    }
    let mut document = minimal_study();
    document["study"]["modes"][0]["occurrence"] =
        json!({"distrib": "weibul", "shape": 1.5, "scale": 10.0});
    assert_invalid(&study, &document);
    // A field of another law is refused too.
    document["study"]["modes"][0]["occurrence"] = json!({"distrib": "exp", "shape": 1.5});
    assert_invalid(&study, &document);
    let transition = json!({"name": "fail", "source": "ok", "targets": ["ko"],
                            "distrib": "weibul", "shape": 1.5, "scale": 10.0});
    assert_invalid(&library, &library_with_transition(transition));
}

#[test]
fn every_policy_and_indicator_serde_writes_validates_and_a_wrong_tag_does_not() {
    let library = library_validator();
    let to = PortRef {
        component: "Load".into(),
        port: "in".into(),
    };
    let param = ConsumerParam { to, value: 1.0 };
    for policy in [
        AllocationPolicy::Proportional,
        AllocationPolicy::Shares {
            shares: vec![param.clone()],
        },
        AllocationPolicy::Priority {
            priorities: vec![param.clone()],
        },
    ] {
        let allocation = Allocation {
            name: "split".into(),
            port: "out".into(),
            available: Expr::attr("$self", "stock"),
            demand: "demand".into(),
            allocated: "allocated".into(),
            policy,
        };
        let mut document = minimal_library();
        document["library"]["types"][0]["body"]["allocations"] =
            json!([serde_json::to_value(&allocation).unwrap()]);
        assert_valid(&library, &document);
        document["library"]["types"][0]["body"]["allocations"][0]["policy"] = json!("fair");
        assert_invalid(&library, &document);
    }

    let system = system_validator();
    let attr = AttrRef {
        component: "L1".into(),
        attribute: "on".into(),
    };
    for target in [
        IndicatorTarget::Attribute { attr: attr.clone() },
        IndicatorTarget::State {
            component: "L1".into(),
            automaton: "burnt".into(),
            state: "occ".into(),
        },
        IndicatorTarget::Predicate {
            attr,
            cmp: CmpOp::Eq,
            value: V::Bool(true),
        },
    ] {
        let indicator = Indicator {
            name: "lit".into(),
            target,
        };
        let mut document = minimal_system();
        document["system"]["indicators"] = json!([serde_json::to_value(&indicator).unwrap()]);
        assert_valid(&system, &document);
        document["system"]["indicators"][0]["target"] = json!("attr");
        assert_invalid(&system, &document);
    }
}

#[test]
fn every_expression_validates_as_a_condition_and_mode_active_nests() {
    let study = study_validator();
    for expr in every_expr() {
        let mut document = minimal_study();
        document["study"]["feared_events"] = json!([{
            "name": "fe", "condition": serde_json::to_value(&expr).unwrap()
        }]);
        assert_valid(&study, &document);
    }
    let mut document = minimal_study();
    let nested = json!({"op": "bool", "bool_op": "and", "args": [
        {"op": "mode_active", "mode": "L1.burnt"},
        {"op": "bool", "bool_op": "not", "args": [{"op": "mode_active", "mode": "L1.burnt"}]}
    ]});
    document["study"]["feared_events"] = json!([{"name": "fe", "condition": nested}]);
    document["study"]["modes"][0]["condition"] = json!({"op": "mode_active", "mode": "L1.burnt"});
    assert_valid(&study, &document);
    document["study"]["modes"][0]["condition"] =
        json!({"op": "mode_active", "mode": "L1.burnt", "instance": "L1"});
    assert_invalid(&study, &document);
}

#[test]
fn a_param_leaf_validates_in_a_type_body_but_not_as_an_instance_name() {
    let transition = json!({"name": "fail", "source": "ok", "targets": ["ko"],
                            "distrib": "exp", "rate": {"$param": "rate"}});
    assert_valid(&library_validator(), &library_with_transition(transition));

    let mut system = minimal_system();
    system["system"]["instances"][0]["name"] = json!({"$param": "rate"});
    assert_invalid(&system_validator(), &system);
    // A study law is not templated either.
    let mut study = minimal_study();
    study["study"]["modes"][0]["occurrence"]["rate"] = json!({"$param": "rate"});
    assert_invalid(&study_validator(), &study);
}

#[test]
fn an_unknown_key_in_a_body_object_fails_the_templated_schema() {
    let library = library_validator();
    let guard = json!({"op": "const", "value": {"kind": "bool", "value": true}});
    let transition = json!({"name": "fail", "source": "ok", "targets": ["ko"],
                            "distrib": "delay", "time": 1.0, "guard": guard});
    assert_valid(&library, &library_with_transition(transition.clone()));
    let mut misspelled = transition;
    let guard = misspelled.as_object_mut().unwrap().remove("guard").unwrap();
    misspelled["gaurd"] = guard;
    assert_invalid(&library, &library_with_transition(misspelled));

    // Inside an attribute, and inside a `$for_each`.
    let mut document = minimal_library();
    document["library"]["types"][0]["body"]["attributes"][0]["initial"] = json!(1);
    assert_invalid(&library, &document);
    let mut document = minimal_library();
    document["library"]["types"][0]["body"]["attributes"] = json!([
        {"$for_each": "flows", "as": "f", "items": [{"name": "${f}", "kind": "bool", "iinit": true}]}
    ]);
    assert_invalid(&library, &document);
    // A template form with an extra key is refused as well.
    document["library"]["types"][0]["body"]["attributes"] = json!([
        {"$for_each": "flows", "as": "f", "items": [], "where": true}
    ]);
    assert_invalid(&library, &document);
}

/// A templated library the reader expands, which the schema accepts:
/// `$param` in a law and an initial value, `$for_each` over attributes
/// and transitions, `$choose` on an expression, a mode catalogue with a
/// `mode_active` condition.
#[test]
fn a_templated_library_the_reader_expands_validates() {
    let library = json!({
        "raichu_library": {"format": 1},
        "library": {
            "name": "demo",
            "version": "1.0.0",
            "types": [{
                "name": "Pump",
                "parameters": [
                    {"name": "rate", "kind": "number", "default": 0.1},
                    {"name": "level", "kind": "number", "default": 2.0},
                    {"name": "flows", "kind": "list", "items": "string", "default": ["water", "air"]},
                    {"name": "vote", "kind": "enum", "values": ["any", "all"], "default": "any"}
                ],
                "body": {
                    "attributes": [
                        {"name": "level", "kind": "float",
                         "init": {"kind": "float", "value": {"$param": "level"}}},
                        {"name": "fed", "kind": "bool", "init": {"kind": "bool", "value": true}},
                        {"$for_each": "flows", "as": "f", "items": [
                            {"name": "${f}_out", "kind": "bool", "init": {"kind": "bool", "value": true}}
                        ]}
                    ],
                    "automata": [{
                        "name": "life", "states": ["ok", "ko"], "init": "ok",
                        "transitions": [
                            {"name": "fail", "source": "ok", "targets": ["ko"],
                             "distrib": "exp", "rate": {"$param": "rate"},
                             "guard": {"$choose": "vote", "cases": {
                                 "any": {"op": "const", "value": {"kind": "bool", "value": true}},
                                 "all": {"op": "attr", "attr": {"component": "$self", "attribute": "fed"}}
                             }}}
                        ]
                    }]
                },
                "targetable": [{"attribute": "fed", "rule": "any_cuts"}],
                "modes": [
                    {"name": "blocked", "nature": "failure",
                     "effects": [{"attribute": "fed", "value": false}]},
                    {"name": "leak", "nature": "failure",
                     "effects": [{"attribute": "fed", "value": false}],
                     "condition": {"op": "mode_active", "mode": "$self.blocked"}}
                ]
            }]
        }
    });
    let system = json!({
        "raichu_system": {"format": 1},
        "system": {
            "name": "plant",
            "library": {"name": "demo", "version": "1.0.0"},
            "instances": [{"name": "P1", "type": "Pump", "parameters": {"vote": "all"}}]
        }
    });
    let study = json!({
        "raichu_study": {"format": 1},
        "study": {
            "name": "s",
            "system": "plant",
            "modes": [
                {"mode": "blocked", "type": "Pump", "occurrence": {"distrib": "exp", "rate": 0.01}},
                {"mode": "leak", "instances": ["P1"], "occurrence": {"distrib": "delay", "time": 1.0}}
            ]
        }
    });
    assert_valid(&library_validator(), &library);
    assert_valid(&system_validator(), &system);
    assert_valid(&study_validator(), &study);

    let library = parse_library(&library.to_string()).unwrap();
    let system = parse_system(&system.to_string()).unwrap();
    let study = parse_study(&study.to_string()).unwrap();
    let expansion = expand(&[library], &system, Some(&study)).unwrap();
    assert!(expansion.model.components.iter().any(|c| c.name == "P1"));
}

/// A library whose pump keeps its common-cause transition only when the
/// `ccf` parameter is true, with `when` as the conditional item.
fn library_with_conditional_transition(when: Value) -> Value {
    json!({
        "raichu_library": {"format": 1},
        "library": {
            "name": "demo",
            "version": "1.0.0",
            "types": [{
                "name": "Pump",
                "parameters": [{"name": "ccf", "kind": "bool", "default": false}],
                "body": {"automata": [{
                    "name": "life", "states": ["ok", "ko"], "init": "ok",
                    "transitions": [
                        {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.1},
                        when
                    ]
                }]}
            }]
        }
    })
}

#[test]
fn a_conditional_item_validates_and_a_malformed_one_does_not() {
    let validator = library_validator();
    let ccf = json!({"name": "common_cause", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.01});
    assert_valid(
        &validator,
        &library_with_conditional_transition(json!({"$when": "ccf", "items": [ccf.clone()]})),
    );
    assert_valid(
        &validator,
        &library_with_conditional_transition(
            json!({"$when": "ccf", "is": [true], "items": [ccf.clone()]}),
        ),
    );
    // Without `items`, and with a misspelled field, it is not one.
    assert_invalid(
        &validator,
        &library_with_conditional_transition(json!({"$when": "ccf"})),
    );
    assert_invalid(
        &validator,
        &library_with_conditional_transition(json!({"$when": "ccf", "item": [ccf]})),
    );
}

#[test]
fn the_example_libraries_validate_and_expand() {
    // Every published example written as a library: its documents match
    // the schemas, and the reader expands them.
    let examples = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let read = |path: &PathBuf| -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };
    let mut systems = 0;
    let mut expanded = 0;
    for entry in std::fs::read_dir(&examples).unwrap() {
        let folder = entry.unwrap().path().join("library");
        let library_path = folder.join("library.json");
        if !library_path.is_file() {
            continue;
        }
        let library = read(&library_path);
        assert_valid(&library_validator(), &library);
        let parsed_library = parse_library(&library.to_string()).unwrap();
        let std = raichu_library::standard_library().unwrap();
        let studies: Vec<PathBuf> = std::fs::read_dir(&folder)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("study")
            })
            .collect();
        for system_entry in std::fs::read_dir(&folder).unwrap() {
            let path = system_entry.unwrap().path();
            if !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("system")
            {
                continue;
            }
            systems += 1;
            let system = read(&path);
            assert_valid(&system_validator(), &system);
            let parsed_system = parse_system(&system.to_string()).unwrap();
            let libraries = [parsed_library.clone(), std.clone()];
            match studies.as_slice() {
                [] => {
                    expand(&libraries, &parsed_system, None)
                        .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.message));
                    expanded += 1;
                }
                studies => {
                    for study_path in studies {
                        let study = read(study_path);
                        assert_valid(&study_validator(), &study);
                        let parsed_study = parse_study(&study.to_string()).unwrap();
                        if parsed_study.system != parsed_system.name {
                            continue;
                        }
                        expand(&libraries, &parsed_system, Some(&parsed_study))
                            .unwrap_or_else(|e| panic!("{}: {}", study_path.display(), e.message));
                        expanded += 1;
                    }
                }
            }
        }
    }
    // Every system is expanded, once per study naming it or bare.
    assert!(systems >= 12, "only {systems} example systems found");
    assert!(
        expanded >= systems,
        "{expanded} expansions for {systems} systems"
    );
}
