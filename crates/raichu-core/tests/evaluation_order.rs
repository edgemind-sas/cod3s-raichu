//! Declared evaluation order of the explicit equations.
//!
//! An explicit equation `V = expr` is swept **once** per evaluation
//! point, in table order: a sweep that reads a value the same sweep is
//! about to recompute reads the previous point's number. The order is
//! therefore part of the answer, not an implementation detail, and a
//! resolved flow network needs a specific one (capability along the
//! flow, demand against it, production along it again).
//!
//! These tests pin the two halves of that contract: without a declared
//! order the table keeps its positional (declaration) order, and with
//! one it keeps exactly the declared order.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CStep;
use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::Model;

/// Two explicit equations whose sweep order is observable: `y` reads
/// `x` and is declared **before** it, so a positional sweep computes
/// `y` from `x`'s initial value (0) and only then sets `x` to 5. A
/// sweep in the order `[x, y]` sees the fresh 5 instead.
///
/// `order` is the JSON fragment inserted at model level (empty for the
/// positional case).
fn stale_read_model(order: &str) -> String {
    format!(
        r#"
{{
  {order}
  "name": "evaluation_order",
  "components": [
    {{
      "name": "C",
      "attributes": [
        {{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "y", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "equations": [
        {{"target": "y", "kind": "explicit",
         "expr": {{"op": "add", "args": [
            {{"op": "attr", "attr": {{"component": "C", "attribute": "x"}}}},
            {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}}]}}}},
        {{"target": "x", "kind": "explicit",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 5.0}}}}}}
      ]
    }}
  ],
  "indicators": [
    {{"name": "x", "target": "attribute",
     "attr": {{"component": "C", "attribute": "x"}}}},
    {{"name": "y", "target": "attribute",
     "attr": {{"component": "C", "attribute": "y"}}}}
  ]
}}
"#
    )
}

/// Value of each indicator at t = 0, i.e. after the single
/// initialization sweep of the explicit equations.
fn initial_values(json: &str) -> Vec<(String, f64)> {
    let model = Model::from_json(json).expect("fixture JSON parses");
    let compiled = CompiledModel::compile(&model).expect("model compiles");
    let result = Engine::new(&compiled, EngineConfig::default())
        .expect("engine builds")
        .run()
        .expect("simulation runs");
    result
        .indicators
        .iter()
        .map(|series| {
            let (time, value) = series.points.first().expect("an initial point");
            assert_eq!(*time, 0.0, "first point is t = 0");
            let raichu_expr::Value::Float(value) = value else {
                panic!("float indicator");
            };
            (series.name.clone(), *value)
        })
        .collect()
}

/// Wrap a model body in the mandatory format envelope, declaring the
/// evaluation-order feature (a bare body carrying an order is refused:
/// that refusal is what makes an engine predating the order say so).
fn sealed(body: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["evaluation_order"]}},
            "model": {body}}}"#
    )
}

/// Qualified names of the compiled sweep steps, in evaluation order: an
/// equation by its target attribute, a distribution operator by its own
/// qualified name.
fn explicit_sequence(model: &Model) -> Vec<String> {
    let compiled = CompiledModel::compile(model).expect("model compiles");
    compiled
        .explicit
        .iter()
        .map(|step| match step {
            CStep::Equation { target, .. } => compiled.var_names[*target].clone(),
            CStep::Allocate(allocation) => allocation.name.clone(),
        })
        .collect()
}

/// The positional flattening the compiler used before an order could be
/// declared: components in declaration order, explicit equations of each
/// in declaration order, then its distribution operators. Recomputed here
/// from the model itself, so the comparison is against the *documented
/// rule* and not against the compiler's own output.
fn positional_sequence(model: &Model) -> Vec<String> {
    model
        .evaluation_steps()
        .into_iter()
        .map(|(component, step)| format!("{component}.{step}"))
        .collect()
}

/// Four explicit equations spread over two components and interleaved
/// with an ODE, so the positional flattening is not the trivial
/// identity.
fn spread_model(order: &str) -> String {
    format!(
        r#"
{{
  {order}
  "name": "spread",
  "components": [
    {{
      "name": "A",
      "attributes": [
        {{"name": "a1", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "a2", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "a3", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "equations": [
        {{"target": "a1", "kind": "explicit",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}}}},
        {{"target": "a2", "kind": "ode",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 0.5}}}}}},
        {{"target": "a3", "kind": "explicit",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 3.0}}}}}}
      ]
    }},
    {{
      "name": "B",
      "attributes": [
        {{"name": "b1", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "b2", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "equations": [
        {{"target": "b1", "kind": "explicit",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 4.0}}}}}},
        {{"target": "b2", "kind": "explicit",
         "expr": {{"op": "const", "value": {{"kind": "float", "value": 5.0}}}}}}
      ]
    }}
  ]
}}
"#
    )
}

/// One `evaluation_order` entry.
fn entry(component: &str, attribute: &str) -> String {
    format!(r#"{{"component": "{component}", "attribute": "{attribute}"}}"#)
}

/// Baseline, pinned before the declared order existed: with no order
/// field the sweep follows declaration order, so `y` reads the stale
/// `x` and lands on 1, not 6.
#[test]
fn positional_order_reads_the_previous_value() {
    assert_eq!(
        initial_values(&stale_read_model("")),
        vec![("x".to_owned(), 5.0), ("y".to_owned(), 1.0)]
    );
}

/// A model with no order field compiles to the **same evaluation
/// sequence** as before, not merely to the same result: the compiled
/// table is the positional flattening, entry for entry.
#[test]
fn no_order_field_keeps_the_positional_sequence() {
    let model = Model::from_json(&spread_model("")).expect("fixture JSON parses");
    assert_eq!(
        explicit_sequence(&model),
        positional_sequence(&model),
        "an order-less model must compile to the flattening it compiled to before"
    );
    assert_eq!(
        positional_sequence(&model),
        vec!["A.a1", "A.a3", "B.b1", "B.b2"]
    );
}

/// A declared order reorders the compiled table, on a permutation that
/// no positional rule could produce.
#[test]
fn declared_order_reorders_the_compiled_table() {
    let order = format!(
        r#""evaluation_order": [{}, {}, {}, {}],"#,
        entry("B", "b2"),
        entry("A", "a3"),
        entry("B", "b1"),
        entry("A", "a1")
    );
    let model = Model::from_json(&sealed(&spread_model(&order))).expect("fixture JSON parses");
    assert_eq!(
        explicit_sequence(&model),
        vec!["B.b2", "A.a3", "B.b1", "A.a1"]
    );
    assert_ne!(explicit_sequence(&model), positional_sequence(&model));
}

/// The declared order is *evaluated*, not merely stored: swapping the
/// sweep so `x` precedes `y` turns the stale read into a fresh one.
#[test]
fn declared_order_is_evaluated() {
    let order = format!(
        r#""evaluation_order": [{}, {}],"#,
        entry("C", "x"),
        entry("C", "y")
    );
    assert_eq!(
        initial_values(&sealed(&stale_read_model(&order))),
        vec![("x".to_owned(), 5.0), ("y".to_owned(), 6.0)]
    );
}
