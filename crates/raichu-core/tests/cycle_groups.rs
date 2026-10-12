//! Whether the minimal-sequence reduction may cancel each cycle group,
//! decided from the compiled model: the acceptance matrix of issue #100.
//!
//! Each model is a small topology chosen for one question: does the
//! group's effect reach the feared event through a quantity that
//! remembers it?

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, CycleGroupDecision, DecisionOrigin};
use raichu_model::Model;
use serde_json::{json, Value};

// ---- expression and model builders ------------------------------------

fn c(v: f64) -> Value {
    json!({"op": "const", "value": {"kind": "float", "value": v}})
}
fn cb(v: bool) -> Value {
    json!({"op": "const", "value": {"kind": "bool", "value": v}})
}
fn a(comp: &str, attr: &str) -> Value {
    json!({"op": "attr", "attr": {"component": comp, "attribute": attr}})
}
fn s(comp: &str, aut: &str, st: &str) -> Value {
    json!({"op": "state_active", "state": {"component": comp, "automaton": aut, "state": st}})
}
fn cmp(op: &str, lhs: Value, rhs: Value) -> Value {
    let name = match op {
        "==" => "eq",
        "<" => "lt",
        "<=" => "le",
        ">" => "gt",
        ">=" => "ge",
        other => panic!("no comparison `{other}`"),
    };
    json!({"op": "cmp", "cmp": name, "lhs": lhs, "rhs": rhs})
}
fn and(args: Vec<Value>) -> Value {
    json!({"op": "bool", "bool_op": "and", "args": args})
}
fn or(args: Vec<Value>) -> Value {
    json!({"op": "bool", "bool_op": "or", "args": args})
}
fn not(arg: Value) -> Value {
    json!({"op": "bool", "bool_op": "not", "args": [arg]})
}
fn ite(cond: Value, then: Value, otherwise: Value) -> Value {
    json!({"op": "if", "cond": cond, "then": then, "otherwise": otherwise})
}
fn fvar(name: &str, v: f64) -> Value {
    json!({"name": name, "kind": "float", "init": {"kind": "float", "value": v}})
}
fn bvar(name: &str, v: bool) -> Value {
    json!({"name": name, "kind": "bool", "init": {"kind": "bool", "value": v}})
}

/// A repairable `health` automaton (`ok`/`failed`) whose failure and
/// repair form the cycle group `group`.
fn health(group: &str) -> Value {
    json!({"name": "health", "states": ["ok", "failed"], "init": "ok", "transitions": [
        {"name": "fail", "source": "ok", "targets": ["failed"], "distrib": "exp", "rate": 0.1,
         "monitored": true, "cycle_group": group, "kind": "failure"},
        {"name": "repair", "source": "failed", "targets": ["ok"], "distrib": "exp", "rate": 1.0,
         "monitored": true, "cycle_group": group, "kind": "repair"}]})
}

fn component(
    name: &str,
    attributes: Vec<Value>,
    automata: Vec<Value>,
    equations: Vec<Value>,
) -> Value {
    json!({"name": name, "attributes": attributes, "automata": automata, "equations": equations})
}

fn unit(name: &str) -> Value {
    component(name, vec![], vec![health(name)], vec![])
}

/// A watcher `name.ev` entering `occ` at once when `guard` holds.
fn watcher(name: &str, guard: Value) -> Value {
    component(
        name,
        vec![],
        vec![
            json!({"name": "ev", "states": ["ok", "occ"], "init": "ok", "transitions": [
            {"name": "trip", "source": "ok", "targets": ["occ"], "distrib": "delay", "time": 0.0,
             "guard": guard}]}),
        ],
        vec![],
    )
}

fn target(name: &str, comp: &str, aut: &str, st: &str) -> Value {
    json!({"name": name, "component": comp, "automaton": aut, "state": st})
}

fn compile(components: Vec<Value>, targets: Vec<Value>) -> CompiledModel {
    let model: Model = serde_json::from_value(json!({
        "name": "cycle_groups", "components": components, "targets": targets}))
    .expect("model deserializes");
    CompiledModel::compile(&model).expect("model compiles")
}

fn decision<'a>(model: &'a CompiledModel, component: &str) -> &'a CycleGroupDecision {
    model
        .cycle_groups
        .iter()
        .find(|d| d.component == component)
        .unwrap_or_else(|| panic!("no decision for `{component}`"))
}

fn cancellable(model: &CompiledModel, component: &str) -> bool {
    let d = decision(model, component);
    assert_eq!(d.targets.len(), 1);
    d.targets[0].cancellable
}

// ---- the hydrogen topology --------------------------------------------

/// The shape of the hydrogen installation of the documentation: a
/// producer fills a storage (an ODE), a compressor must run, and the user
/// loses supply when the compressor is down or the storage is empty. The
/// user's own state drives the draw on the storage, which is the path
/// through the target automaton the search must not take.
fn hydrogen(user_states: &[&str]) -> CompiledModel {
    let store = component(
        "STORE",
        vec![fvar("level", 10.0), fvar("draw", 0.0)],
        vec![],
        vec![
            json!({"target": "draw", "kind": "explicit",
                   "expr": ite(s("USER", "supply", "supplied"), c(0.5), c(0.0))}),
            json!({"target": "level", "kind": "ode",
                   "expr": json!({"op": "sub",
                                  "lhs": ite(s("PROD", "health", "ok"), c(1.0), c(0.0)),
                                  "rhs": a("STORE", "draw")})}),
        ],
    );
    let lost = or(vec![
        s("COMP", "health", "failed"),
        cmp("<=", a("STORE", "level"), c(0.0)),
    ]);
    let mut transitions = vec![
        json!({"name": "trip", "source": "supplied", "targets": ["lost"],
        "distrib": "delay", "time": 0.0, "guard": lost}),
    ];
    if user_states.len() > 2 {
        // A second non-target state the user can sit in: the target
        // automaton is no longer constant before the target.
        transitions.push(
            json!({"name": "pause", "source": "supplied", "targets": ["paused"],
            "distrib": "exp", "rate": 0.01}),
        );
        transitions.push(
            json!({"name": "resume", "source": "paused", "targets": ["supplied"],
            "distrib": "exp", "rate": 1.0}),
        );
    }
    let user = component(
        "USER",
        vec![],
        vec![
            json!({"name": "supply", "states": user_states, "init": "supplied",
                    "transitions": transitions}),
        ],
        vec![],
    );
    compile(
        vec![unit("PROD"), unit("COMP"), store, user],
        vec![target("loss_of_supply", "USER", "supply", "lost")],
    )
}

#[test]
fn hydrogen_compressor_is_cancellable_and_producer_is_kept() {
    let model = hydrogen(&["supplied", "lost"]);
    assert!(
        cancellable(&model, "COMP"),
        "{:?}",
        decision(&model, "COMP")
    );
    let producer = &decision(&model, "PROD").targets[0];
    assert!(!producer.cancellable);
    assert_eq!(producer.origin, DecisionOrigin::Analysis);
    assert_eq!(producer.memory.as_deref(), Some("STORE.level"));
    assert_eq!(
        producer.witness.first().map(String::as_str),
        Some("PROD.health")
    );
    assert_eq!(
        producer.witness.last().map(String::as_str),
        Some("USER.supply")
    );
}

#[test]
fn a_target_automaton_with_two_non_target_states_is_traversed() {
    // The compressor then reaches the storage through the user's state.
    let model = hydrogen(&["supplied", "paused", "lost"]);
    assert!(!cancellable(&model, "COMP"));
    assert!(!cancellable(&model, "PROD"));
}

// ---- memory and memoryless shapes -------------------------------------

#[test]
fn a_two_out_of_three_vote_cancels_every_group() {
    let down = |n: &str| s(n, "health", "failed");
    let vote = or(vec![
        and(vec![down("A"), down("B")]),
        and(vec![down("A"), down("C")]),
        and(vec![down("B"), down("C")]),
    ]);
    let model = compile(
        vec![unit("A"), unit("B"), unit("C"), watcher("SYS", vote)],
        vec![target("down", "SYS", "ev", "occ")],
    );
    for name in ["A", "B", "C"] {
        assert!(cancellable(&model, name), "{name}");
    }
}

#[test]
fn an_explicit_chain_without_memory_leaves_the_group_cancellable() {
    let pump = component(
        "PUMP",
        vec![fvar("flow", 1.0), fvar("pressure", 1.0)],
        vec![health("PUMP")],
        vec![
            json!({"target": "flow", "kind": "explicit",
                   "expr": ite(s("PUMP", "health", "ok"), c(1.0), c(0.0))}),
            json!({"target": "pressure", "kind": "explicit",
                   "expr": json!({"op": "mul", "args": [c(2.0), a("PUMP", "flow")]})}),
        ],
    );
    let model = compile(
        vec![
            pump,
            watcher("SYS", cmp("<", a("PUMP", "pressure"), c(1.0))),
        ],
        vec![target("low_pressure", "SYS", "ev", "occ")],
    );
    assert!(cancellable(&model, "PUMP"));
}

/// A heater warms a room (an ODE); the feared event is a cold room.
#[test]
fn a_heater_feeding_a_room_temperature_is_kept() {
    let room = component(
        "ROOM",
        vec![fvar("temperature", 20.0)],
        vec![],
        vec![json!({"target": "temperature", "kind": "ode",
                    "expr": json!({"op": "sub",
                                   "lhs": ite(s("HEATER", "health", "ok"), c(2.0), c(0.0)),
                                   "rhs": c(1.0)})})],
    );
    let model = compile(
        vec![
            unit("HEATER"),
            room,
            watcher("SYS", cmp("<", a("ROOM", "temperature"), c(15.0))),
        ],
        vec![target("cold", "SYS", "ev", "occ")],
    );
    let heater = &decision(&model, "HEATER").targets[0];
    assert!(!heater.cancellable);
    assert_eq!(heater.memory.as_deref(), Some("ROOM.temperature"));
}

/// A controller between the group and the target, without any ODE.
fn through_controller(controller: Value) -> CompiledModel {
    let sensor = component(
        "SENSOR",
        vec![fvar("load", 0.0)],
        vec![],
        vec![json!({"target": "load", "kind": "explicit",
                    "expr": ite(s("A", "health", "failed"), c(10.0), c(0.0))})],
    );
    compile(
        vec![
            unit("A"),
            sensor,
            controller,
            watcher("SYS", s("CTRL", "mode", "on")),
        ],
        vec![target("tripped", "SYS", "ev", "occ")],
    )
}

fn controller(on: Value, off: Value) -> Value {
    component(
        "CTRL",
        vec![],
        vec![
            json!({"name": "mode", "states": ["off", "on"], "init": "off", "transitions": [
            {"name": "start", "source": "off", "targets": ["on"], "distrib": "delay", "time": 0.0,
             "guard": on},
            {"name": "stop", "source": "on", "targets": ["off"], "distrib": "delay", "time": 0.0,
             "guard": off}]}),
        ],
        vec![],
    )
}

#[test]
fn a_band_controller_between_the_group_and_the_target_keeps_the_group() {
    // Switches on above 5 and off below 1: between the two, the mode
    // remembers which threshold was crossed last.
    let model = through_controller(controller(
        cmp(">", a("SENSOR", "load"), c(5.0)),
        cmp("<", a("SENSOR", "load"), c(1.0)),
    ));
    let kept = &decision(&model, "A").targets[0];
    assert!(!kept.cancellable);
    assert_eq!(kept.memory.as_deref(), Some("CTRL.mode"));
}

#[test]
fn a_memoryless_observer_between_the_group_and_the_target_leaves_it_cancellable() {
    // Entered and left at once on a condition and its negation: the
    // ObjEvent shape, whose state is a function of the load.
    let high = cmp(">", a("SENSOR", "load"), c(5.0));
    let model = through_controller(controller(high.clone(), not(high)));
    assert!(cancellable(&model, "A"));
}

#[test]
fn a_standby_failing_on_demand_records_its_own_failure() {
    // B's failure is armed while A is down: B's state persists after A's
    // repair, but B's failure is an event of its own in the sequence.
    let mut b = unit("B");
    b["automata"][0]["transitions"][0]["guard"] = s("A", "health", "failed");
    let model = compile(
        vec![unit("A"), b, watcher("SYS", s("B", "health", "failed"))],
        vec![target("standby_lost", "SYS", "ev", "occ")],
    );
    assert!(cancellable(&model, "A"));
}

#[test]
fn a_timer_that_survives_the_repair_keeps_the_group() {
    // The feared event comes 2 h after A fails, even if A is repaired
    // in between.
    let alarm = component(
        "SYS",
        vec![],
        vec![
            json!({"name": "ev", "states": ["ok", "occ"], "init": "ok", "transitions": [
            {"name": "trip", "source": "ok", "targets": ["occ"], "distrib": "delay", "time": 2.0,
             "on_interruption": "continue", "guard": s("A", "health", "failed")}]}),
        ],
        vec![],
    );
    let model = compile(
        vec![unit("A"), alarm],
        vec![target("late", "SYS", "ev", "occ")],
    );
    assert!(!cancellable(&model, "A"));
}

/// `A` writes the flag `F.set` through its transition effects; the
/// feared event reads the flag.
fn latched(repair_writes: bool) -> CompiledModel {
    let mut a_unit = unit("A");
    a_unit["attributes"] = json!([bvar("set", false)]);
    a_unit["automata"][0]["transitions"][0]["effects"] =
        json!([{"target": {"component": "A", "attribute": "set"}, "value": cb(true)}]);
    a_unit["automata"][0]["transitions"][1]["effects"] =
        json!([{"target": {"component": "A", "attribute": "set"}, "value": cb(!repair_writes)}]);
    compile(
        vec![a_unit, watcher("SYS", cmp("==", a("A", "set"), cb(true)))],
        vec![target("flagged", "SYS", "ev", "occ")],
    )
}

#[test]
fn a_flag_the_repair_restores_leaves_the_group_cancellable() {
    assert!(cancellable(&latched(true), "A"));
}

#[test]
fn a_flag_the_repair_does_not_restore_keeps_the_group() {
    let model = latched(false);
    let kept = &decision(&model, "A").targets[0];
    assert!(!kept.cancellable);
    assert_eq!(kept.memory.as_deref(), Some("A.set"));
}

#[test]
fn a_rate_reading_an_integrated_quantity_keeps_the_group() {
    // The feared event's rate grows with a temperature A's failure heats.
    let tank = component(
        "TANK",
        vec![fvar("temperature", 20.0)],
        vec![],
        vec![json!({"target": "temperature", "kind": "ode",
                    "expr": ite(s("A", "health", "failed"), c(1.0), c(-1.0))})],
    );
    let rupture = component(
        "SYS",
        vec![],
        vec![
            json!({"name": "ev", "states": ["ok", "occ"], "init": "ok", "transitions": [
            {"name": "trip", "source": "ok", "targets": ["occ"], "distrib": "exp",
             "rate_expr": json!({"op": "mul", "args": [c(0.001), a("TANK", "temperature")]})}]}),
        ],
        vec![],
    );
    let model = compile(
        vec![unit("A"), tank, rupture],
        vec![target("rupture", "SYS", "ev", "occ")],
    );
    assert!(!cancellable(&model, "A"));
}

// ---- targets, declarations, determinism -------------------------------

#[test]
fn a_group_gets_one_decision_per_target() {
    let pool = component(
        "POOL",
        vec![fvar("level", 5.0)],
        vec![],
        vec![json!({"target": "level", "kind": "ode",
                    "expr": ite(s("A", "health", "ok"), c(0.0), c(-1.0))})],
    );
    let model = compile(
        vec![
            unit("A"),
            pool,
            watcher("DRY", cmp("<=", a("POOL", "level"), c(0.0))),
            watcher("ALARM", s("A", "health", "failed")),
        ],
        vec![
            target("dry", "DRY", "ev", "occ"),
            target("alarm", "ALARM", "ev", "occ"),
        ],
    );
    let d = decision(&model, "A");
    let by_target = |name: &str| {
        d.targets
            .iter()
            .find(|t| t.target == name)
            .unwrap()
            .cancellable
    };
    assert!(!by_target("dry"));
    assert!(by_target("alarm"));
    assert!(d.cancellable_for(Some("alarm")));
    assert!(!d.cancellable_for(Some("dry")));
    // A trajectory that reached neither has no cause to keep.
    assert!(d.cancellable_for(None));
}

fn declared(value: &str) -> CompiledModel {
    let mut a_unit = unit("A");
    for t in 0..2 {
        a_unit["automata"][0]["transitions"][t]["cycle_cancellation"] = json!(value);
    }
    let room = component(
        "ROOM",
        vec![fvar("temperature", 20.0)],
        vec![],
        vec![json!({"target": "temperature", "kind": "ode",
                    "expr": ite(s("A", "health", "ok"), c(1.0), c(-1.0))})],
    );
    compile(
        vec![
            a_unit,
            unit("B"),
            room,
            watcher(
                "SYS",
                or(vec![
                    cmp("<", a("ROOM", "temperature"), c(15.0)),
                    s("B", "health", "failed"),
                ]),
            ),
        ],
        vec![target("cold", "SYS", "ev", "occ")],
    )
}

#[test]
fn a_declaration_overrides_the_analysis() {
    let always = declared("always");
    let a_decision = &decision(&always, "A").targets[0];
    assert!(a_decision.cancellable);
    assert_eq!(a_decision.origin, DecisionOrigin::Declared);
    assert!(cancellable(&always, "B"));

    let mut never_b = declared("auto");
    assert!(!cancellable(&never_b, "A"));
    // Declaring `never` on B, whose failure acts at once.
    let mut model: Model = serde_json::from_value(json!({"name": "x", "components": [
        unit("B")], "targets": []}))
    .unwrap();
    for t in &mut model.components[0].automata[0].transitions {
        t.cycle_cancellation = raichu_model::CycleCancellation::Never;
    }
    let mut components: Vec<Value> = vec![serde_json::to_value(&model.components[0]).unwrap()];
    components.push(watcher("SYS", s("B", "health", "failed")));
    never_b = compile(components, vec![target("down", "SYS", "ev", "occ")]);
    let b_decision = &decision(&never_b, "B").targets[0];
    assert!(!b_decision.cancellable);
    assert_eq!(b_decision.origin, DecisionOrigin::Declared);
}

#[test]
fn two_halves_of_a_group_cannot_disagree() {
    let mut a_unit = unit("A");
    a_unit["automata"][0]["transitions"][0]["cycle_cancellation"] = json!("never");
    let model: Model =
        serde_json::from_value(json!({"name": "x", "components": [a_unit]})).unwrap();
    let error = model.validate().unwrap_err().to_string();
    assert!(error.contains("A.health.repair"), "{error}");
    assert!(error.contains("cycle group `A`"), "{error}");
}

#[test]
fn a_declaration_needs_a_cycle_group_and_its_feature() {
    let mut a_unit = unit("A");
    for t in 0..2 {
        a_unit["automata"][0]["transitions"][t]["cycle_group"] = Value::Null;
        a_unit["automata"][0]["transitions"][t]["cycle_cancellation"] = json!("never");
    }
    let model: Model =
        serde_json::from_value(json!({"name": "x", "components": [a_unit]})).unwrap();
    assert!(model
        .validate()
        .unwrap_err()
        .to_string()
        .contains("no `cycle_group`"));

    // A bare body declares no feature: `never` is refused at load.
    let body = json!({"name": "x", "components": [unit("A")]}).to_string();
    let with = body.replacen(
        r#""kind":"failure""#,
        r#""kind":"failure","cycle_cancellation":"never""#,
        1,
    );
    assert_ne!(with, body);
    let error = Model::from_json(&with).unwrap_err().to_string();
    assert!(error.contains("cycle_cancellation"), "{error}");
}

#[test]
fn the_decisions_are_deterministic() {
    let first = hydrogen(&["supplied", "lost"]).cycle_groups;
    for _ in 0..5 {
        assert_eq!(hydrogen(&["supplied", "lost"]).cycle_groups, first);
    }
    let names: Vec<_> = first.iter().map(|d| d.component.as_str()).collect();
    assert_eq!(names, vec!["COMP", "PROD"]);
}

// ---- review findings: memory the first version missed -----------------

#[test]
fn a_target_automaton_that_latches_an_intermediate_state_keeps_the_group() {
    // ok -> degraded on A's failure, degraded -> lost on B's: once
    // degraded, A's repair changes nothing.
    let system = component(
        "SYS",
        vec![],
        vec![
            json!({"name": "ev", "states": ["ok", "degraded", "lost"], "init": "ok", "transitions": [
            {"name": "degrade", "source": "ok", "targets": ["degraded"], "distrib": "delay", "time": 0.0,
             "guard": s("A", "health", "failed")},
            {"name": "lose", "source": "degraded", "targets": ["lost"], "distrib": "delay", "time": 0.0,
             "guard": s("B", "health", "failed")}]}),
        ],
        vec![],
    );
    let model = compile(
        vec![unit("A"), unit("B"), system],
        vec![target("loss", "SYS", "ev", "lost")],
    );
    let kept = &decision(&model, "A").targets[0];
    assert!(!kept.cancellable);
    assert_eq!(kept.memory.as_deref(), Some("SYS.ev"));
}

#[test]
fn a_monitored_group_with_an_unrecorded_trip_is_not_exempt() {
    // R has a recorded failure cycle, and an unrecorded trip on A's
    // failure from which it never returns.
    let mut r = unit("R");
    r["automata"][0]["states"] = json!(["ok", "failed", "tripped"]);
    r["automata"][0]["transitions"]
        .as_array_mut()
        .unwrap()
        .push(
            json!({"name": "trip", "source": "ok", "targets": ["tripped"], "distrib": "delay",
                     "time": 0.0, "guard": s("A", "health", "failed")}),
        );
    let model = compile(
        vec![
            unit("A"),
            unit("B"),
            r,
            watcher(
                "SYS",
                and(vec![
                    s("R", "health", "tripped"),
                    s("B", "health", "failed"),
                ]),
            ),
        ],
        vec![target("loss", "SYS", "ev", "occ")],
    );
    assert!(!cancellable(&model, "A"));
}

#[test]
fn a_flag_only_one_repair_of_the_group_restores_keeps_the_group() {
    // Two failure modes of A in one group both set the flag; only the
    // second one's repair clears it.
    let mode = |name: &str, clears: bool| {
        let mut automaton = health("A");
        automaton["name"] = json!(name);
        automaton["transitions"][0]["effects"] =
            json!([{"target": {"component": "A", "attribute": "set"}, "value": cb(true)}]);
        if clears {
            automaton["transitions"][1]["effects"] =
                json!([{"target": {"component": "A", "attribute": "set"}, "value": cb(false)}]);
        }
        automaton
    };
    let a_unit = component(
        "A",
        vec![bvar("set", false)],
        vec![mode("m1", false), mode("m2", true)],
        vec![],
    );
    let model = compile(
        vec![
            a_unit,
            unit("B"),
            watcher(
                "SYS",
                and(vec![
                    cmp("==", a("A", "set"), cb(true)),
                    s("B", "health", "failed"),
                ]),
            ),
        ],
        vec![target("loss", "SYS", "ev", "occ")],
    );
    assert!(!cancellable(&model, "A"));
}

#[test]
fn an_instantaneous_observer_written_with_inst_is_memoryless() {
    let high = cmp(">", a("SENSOR", "load"), c(5.0));
    let mut observer = controller(high.clone(), not(high));
    for t in 0..2 {
        let transition = &mut observer["automata"][0]["transitions"][t];
        transition["distrib"] = json!("inst");
        transition["probs"] = json!([]);
        transition.as_object_mut().unwrap().remove("time");
    }
    let model = through_controller(observer);
    assert!(cancellable(&model, "A"), "{:?}", decision(&model, "A"));
}

#[test]
fn complementary_guards_that_read_their_own_state_are_a_latch() {
    // Up on `A failed or on`, down on its negation: once on, never off.
    let up = or(vec![s("A", "health", "failed"), s("CTRL", "mode", "on")]);
    let model = through_controller(controller(up.clone(), not(up)));
    assert!(!cancellable(&model, "A"));
}
