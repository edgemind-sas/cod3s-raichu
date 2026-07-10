"""The muscadet plugin: ObjFlow, ObjFM and ObjEvent object types,
iso-functional with their cod3s definitions (v1 subset).

Specification schemas (all pure JSON):

ObjFlow::

    { "type": "ObjFlow", "name": "S",
      "flows_in":  [ { "name": "is_ok", "logic": "or" | "and" | k } ],
      "flows_out": [ { "name": "is_ok", "var_prod_default": true,
                       "var_prod_cond": ["…"],
                       "tempo":   { "enable_time": 2, "disable_time": 1,
                                    "init_enable": false },       # FlowOutTempo
                       "trigger": { "time_up": 0, "time_down": 0,
                                    "logic": "and" | "or" | k } } ],  # FlowOutOnTrigger
      "failure_modes": [ { "name": "fm", "distrib": "delay" | "exp",
                           "failure": 4, "repair": 2,
                           "failure_cond": "is_ok_fed_out" } ] }

ObjFM (cod3s `component.py:932`, behaviour `internal`, one automaton on
the declared target set)::

    { "type": "ObjFM", "name": "fm", "targets": ["T1", "T2"],
      "failure": { "distrib": "delay", "time": 5 },
      "repair":  { "distrib": "delay", "time": 3 },
      "failure_cond": [[ { "obj": "T1", "attr": "flow",
                           "ope": "==", "value": true } ]],
      "repair_cond": true,
      "failure_effects": { "flow": false },
      "repair_effects":  { "flow": true } }

ObjEvent (cod3s `component.py:825`)::

    { "type": "ObjEvent", "name": "ER",
      "cond": [[ { "obj": "box", "attr": "working",
                   "ope": "==", "value": false } ]],
      "inner_logic": "all", "outer_logic": "any",
      "cond_operator": "==", "cond_value": true,
      "tempo_occ": 2, "tempo_not_occ": 1 }

Condition-tree leaves reference either a variable (``"attr"``) or an
automaton state (``"automaton"`` + ``"state"``) of ``"obj"``.
"""

from __future__ import annotations

from typing import Any

from .. import muscadet as authoring

__all__ = ["MuscadetPlugin"]


# --- expression helpers (core schema) --------------------------------------


def _const(value: Any) -> dict:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    else:
        kind, value = "float", float(value)
    return {"op": "const", "value": {"kind": kind, "value": value}}


_OPE = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def _leaf(leaf: dict) -> dict:
    """One condition-tree leaf → core comparison expression."""
    obj = leaf["obj"]
    ope = _OPE[leaf.get("ope", "==")]
    if "attr" in leaf:
        lhs = {"op": "attr", "attr": {"component": obj, "attribute": leaf["attr"]}}
    else:
        lhs = {
            "op": "state_active",
            "state": {
                "component": obj,
                "automaton": leaf["automaton"],
                "state": leaf["state"],
            },
        }
    return {"op": "cmp", "cmp": ope, "lhs": lhs, "rhs": _const(leaf["value"])}


_LOGIC = {"all": "and", "any": "or"}


def _cond_tree(
    cond: Any,
    inner_logic: str = "all",
    outer_logic: str = "any",
    cond_operator: str = "==",
    cond_value: bool = True,
) -> dict:
    """cod3s condition specification → core boolean expression.

    Accepts the cod3s shapes: a bare bool, a single leaf dict, a list of
    leaves (one AND group) or a list of lists (OR of AND groups) —
    mirroring `sanitize_cond_format`.
    """
    if isinstance(cond, bool):
        base = _const(cond)
    else:
        if isinstance(cond, dict):
            cond = [[cond]]
        elif cond and all(isinstance(c, dict) for c in cond):
            cond = [cond]
        groups = [
            {"op": "bool", "bool_op": _LOGIC[inner_logic], "args": [_leaf(c) for c in group]}
            for group in cond
        ]
        base = {"op": "bool", "bool_op": _LOGIC[outer_logic], "args": groups}
    if cond_operator == "==" and cond_value is True:
        return base
    return {"op": "cmp", "cmp": _OPE[cond_operator], "lhs": base, "rhs": _const(cond_value)}


def _negate(expr: dict) -> dict:
    return {"op": "bool", "bool_op": "not", "args": [expr]}


def _law(spec: dict) -> dict:
    """Occurrence-law specification → core transition fields
    (cod3s `{"cls": …}` and plugin `{"law"/"distrib": …}` spellings accepted)."""
    kind = spec.get("law") or spec.get("cls") or spec.get("distrib")
    if kind == "delay":
        return {"distrib": "delay", "time": float(spec["time"])}
    if kind == "exp":
        return {"distrib": "exp", "rate": float(spec.get("rate", spec.get("lambda", 0.0)))}
    raise ValueError(f"unsupported law specification: {spec}")


# --- object expansions ------------------------------------------------------


def _expand_objflow(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Delegate to the `pyraichu.muscadet` authoring layer (same
    semantics, serialized entry point)."""
    obj = authoring.ObjFlow.__new__(authoring.ObjFlow)
    obj.name = spec["name"]
    obj.flows_in = []
    obj.flows_out = []
    obj.failure_modes = []
    for flow in spec.get("flows_in", []):
        obj.add_flow_in(name=flow["name"], logic=flow.get("logic", "or"))
    for flow in spec.get("flows_out", []):
        if "tempo" in flow:
            tempo = flow["tempo"]
            obj.add_flow_out_tempo(
                name=flow["name"],
                enable_time=tempo.get("enable_time", 0.0),
                disable_time=tempo.get("disable_time", 0.0),
                init_enable=tempo.get("init_enable", False),
                var_prod_default=flow.get("var_prod_default", False),
                var_prod_cond=flow.get("var_prod_cond"),
            )
        elif "trigger" in flow:
            trigger = flow["trigger"]
            obj.add_flow_out_on_trigger(
                name=flow["name"],
                trigger_time_up=trigger.get("time_up", 0.0),
                trigger_time_down=trigger.get("time_down", 0.0),
                trigger_logic=trigger.get("logic", "or"),
                var_prod_default=flow.get("var_prod_default", False),
                var_prod_cond=flow.get("var_prod_cond"),
            )
        else:
            obj.add_flow_out(
                name=flow["name"],
                var_prod_default=flow.get("var_prod_default", False),
                var_prod_cond=flow.get("var_prod_cond"),
            )
    for mode in spec.get("failure_modes", []):
        if mode.get("law", "delay") == "delay":
            obj.add_delay_failure_mode(
                name=mode["name"],
                failure_time=mode["failure"],
                repair_time=mode["repair"],
                targets=mode.get("targets"),
                failure_cond=mode.get("failure_cond"),
            )
        else:
            obj.add_exp_failure_mode(
                name=mode["name"],
                failure_rate=mode["failure"],
                repair_rate=mode["repair"],
                targets=mode.get("targets"),
                failure_cond=mode.get("failure_cond"),
            )
    return [obj._build()], [], []


def _expand_objfm(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """cod3s ObjFM expansion, including **common-cause orders**: with
    N targets, cod3s creates one automaton per target combination of
    every *active* order (`fm__cc_i_j`, states `occ__cc_i_j` /
    `rep__cc_i_j`, transitions named after the target states —
    underscore-separated indices since cod3s 1.9.0). Per-order laws
    come as lists in ``"failure"`` / ``"repair"`` (``null`` = inactive
    order, mirroring `drop_inactive_automata`).

    Effects follow a *reinitialization* semantics: the
    target variable holds its failure value while ANY impacting
    combination sits in its failure state, and its INITIAL value
    otherwise (the engine reinitializes such variables at every step
    before re-applying the occurrence enforcers — a design property of
    the internal mode: repair_effects are unnecessary by
    construction)."""
    import itertools

    name = spec["name"]
    targets = spec["targets"]
    order_max = len(targets)
    failure_state = spec.get("failure_state", "occ")
    repair_state = spec.get("repair_state", "rep")
    failure_effects: dict = spec.get("failure_effects", {})
    repair_effects: dict = spec.get("repair_effects", {})

    inner = spec.get("cond_inner_logic", "all")
    outer = spec.get("cond_outer_logic", "any")

    failure_laws = (
        list(spec["failure"]) if isinstance(spec["failure"], list) else [spec["failure"]]
    )
    repair_laws = (
        list(spec["repair"]) if isinstance(spec["repair"], list) else [spec["repair"]]
    )
    failure_laws += [None] * (order_max - len(failure_laws))
    repair_laws += [None] * (order_max - len(repair_laws))

    automata = []
    impacting: dict[str, list[tuple[str, str]]] = {t: [] for t in targets}
    for order in range(1, order_max + 1):
        f_law = failure_laws[order - 1]
        r_law = repair_laws[order - 1]
        if f_law is None or r_law is None:
            continue  # inactive order
        for comb in itertools.combinations(range(order_max), order):
            suffix = (
                "__cc_" + "_".join(str(i + 1) for i in comb) if order_max > 1 else ""
            )
            aut_name = f"fm{suffix}"
            occ = f"{failure_state}{suffix}"
            rep = f"{repair_state}{suffix}"
            automata.append(
                {
                    "name": aut_name,
                    "states": [rep, occ],
                    "init": rep,
                    "transitions": [
                        {
                            "name": "failure" if order_max == 1 else occ,
                            "source": rep,
                            "targets": [occ],
                            "guard": _cond_tree(
                                spec.get("failure_cond", True), inner, outer
                            ),
                            **_law(f_law),
                        },
                        {
                            "name": "repair" if order_max == 1 else rep,
                            "source": occ,
                            "targets": [rep],
                            "guard": _cond_tree(
                                spec.get("repair_cond", True), inner, outer
                            ),
                            **_law(r_law),
                        },
                    ],
                }
            )
            for idx in comb:
                impacting[targets[idx]].append((aut_name, occ))

    # Internal-mode reinitialization semantics: the target variables
    # are *reinitialized to their initial value at every step*, then the
    # occ-state enforcers of the failed combinations re-apply their
    # failure values. Net semantics:
    # `var = failure value while ANY impacting combination is failed,
    # its initial value otherwise` — an OR over the occ states, with
    # the initial value (not a repair effect) as the rest state. This
    # is why cod3s models omit repair_effects in internal mode (and
    # why adding them hangs the simulator on multi-order ObjFMs).
    def target_init(target, variable):
        for component in model.get("components", []):
            if component.get("name") != target:
                continue
            for entry in component.get("attributes", []):
                if entry.get("name") == variable:
                    return {"op": "const", "value": entry["init"]}
        return None

    effects = []
    for target in targets:
        gates = [
            {
                "op": "state_active",
                "state": {"component": name, "automaton": aut, "state": st},
            }
            for aut, st in impacting[target]
        ]
        if not gates:
            continue
        gate = gates[0] if len(gates) == 1 else {
            "op": "bool",
            "bool_op": "or",
            "args": gates,
        }
        for variable, fail_value in failure_effects.items():
            if variable in repair_effects:
                # Explicit repair value (RAICHU extension: safe here,
                # hangs cod3s multi-order internal ObjFMs).
                otherwise = _const(repair_effects[variable])
            else:
                otherwise = target_init(target, variable)
                if otherwise is None:
                    if isinstance(fail_value, bool):
                        otherwise = _const(not fail_value)
                    else:
                        raise ValueError(
                            f"ObjFM `{name}`: cannot resolve the initial "
                            f"value of `{target}.{variable}` (reinit "
                            "semantics) — declare repair_effects"
                        )
            effects.append(
                {
                    "target": {"component": target, "attribute": variable},
                    "value": {
                        "op": "if",
                        "cond": gate,
                        "then": _const(fail_value),
                        "otherwise": otherwise,
                    },
                }
            )
    functions = [{"name": "apply_effects", "effects": effects}] if effects else []

    component = {
        "name": name,
        "attributes": [],
        "ports": [],
        "interfaces": [],
        "automata": automata,
        "sensitive_functions": functions,
        "equations": [],
    }
    return [component], [], []


def _expand_objfm_inst(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """cod3s `ObjFMInst` expansion — **failure on solicitation**
    (user guide `objfm-inst.md`, ADR 2026-07-05).

    The demand is ``failure_cond``; on each demand *front* the mode fails
    with probability ``gamma`` — one Bernoulli draw, instantaneously —
    and is repaired by an exponential ``mu``. Realised as a **3-state**
    automaton (`rep` / `occ` / `not_occ`):

    - `rep --occ [inst, guard=demand]--> {occ: gamma, not_occ: 1-gamma}` —
      the draw (a branching instantaneous transition, RAICHU brique 2);
    - `not_occ --not_occ [inst p=1, guard=NOT demand]--> rep` — the
      deterministic re-arm; `not_occ` absorbs the front so no re-draw
      happens while the demand holds (anti-Zeno);
    - `occ --rep [exp(mu), guard=repair_cond]--> rep` — the repair.

    ``failure_effects`` apply on entering `occ`; the resting value is the
    target's initial value (reinitialization semantics, as `internal`).

    Single target for now; per-target common-cause draws (``failure_param``
    as a per-order list) are a follow-up.
    """
    name = spec["name"]
    targets = spec["targets"]
    if len(targets) != 1:
        raise NotImplementedError(
            f"ObjFMInst `{name}`: only a single target is supported for now "
            f"(got {targets}); declare one ObjFMInst per target"
        )
    target = targets[0]
    failure_state = spec.get("failure_state", "occ")
    repair_state = spec.get("repair_state", "rep")
    absorb_state = spec.get("absorb_state", "not_occ")
    failure_effects: dict = spec.get("failure_effects", {})
    repair_effects: dict = spec.get("repair_effects", {})
    inner = spec.get("cond_inner_logic", "all")
    outer = spec.get("cond_outer_logic", "any")

    # gamma from the `failure` inst-law spec (or `failure_param`).
    fspec = spec.get("failure", spec.get("failure_param"))
    if isinstance(fspec, list):
        raise NotImplementedError(
            f"ObjFMInst `{name}`: per-order common-cause `failure_param` "
            "lists are a follow-up; pass a scalar gamma"
        )
    gamma = float(
        fspec.get("prob", fspec.get("gamma")) if isinstance(fspec, dict) else fspec
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"ObjFMInst `{name}`: gamma must be in [0, 1] (got {gamma})")

    # mu: exponential repair (`repair` law spec, or `repair_param` scalar).
    # `mu = 0` means *no repair* — `occ` is absorbing (one failure per
    # demand, never repaired), matching cod3s `is_occ_law_repair_active`.
    rspec = spec.get("repair", spec.get("repair_param", 0.0))
    repair_law = (
        _law(rspec) if isinstance(rspec, dict) else {"distrib": "exp", "rate": float(rspec)}
    )
    mu = float(repair_law.get("rate", 0.0))

    demand = _cond_tree(spec.get("failure_cond", True), inner, outer)
    not_demand = _negate(demand)
    repair_guard = _cond_tree(spec.get("repair_cond", True), inner, outer)

    transitions = [
        {  # the draw: rep -> occ (gamma) | not_occ (1 - gamma)
            "name": failure_state,
            "source": repair_state,
            "targets": [failure_state, absorb_state],
            "guard": demand,
            "distrib": "inst",
            "probs": [gamma],
        },
        {  # deterministic re-arm when the demand falls (inst p = 1)
            "name": absorb_state,
            "source": absorb_state,
            "targets": [repair_state],
            "guard": not_demand,
            "distrib": "inst",
            "probs": [],
        },
    ]
    if mu > 0.0:
        transitions.append(
            {  # exponential repair
                "name": repair_state,
                "source": failure_state,
                "targets": [repair_state],
                "guard": repair_guard,
                **repair_law,
            }
        )

    automaton = {
        "name": "fm",
        "states": [repair_state, failure_state, absorb_state],
        "init": repair_state,
        "transitions": transitions,
    }

    def target_init(variable):
        for component in model.get("components", []):
            if component.get("name") != target:
                continue
            for entry in component.get("attributes", []):
                if entry.get("name") == variable:
                    return {"op": "const", "value": entry["init"]}
        return None

    # Reinitialization semantics: fail value while `occ`, initial value
    # (or explicit repair value) otherwise.
    gate = {
        "op": "state_active",
        "state": {"component": name, "automaton": "fm", "state": failure_state},
    }
    effects = []
    for variable, fail_value in failure_effects.items():
        if variable in repair_effects:
            otherwise = _const(repair_effects[variable])
        else:
            otherwise = target_init(variable)
            if otherwise is None:
                if isinstance(fail_value, bool):
                    otherwise = _const(not fail_value)
                else:
                    raise ValueError(
                        f"ObjFMInst `{name}`: cannot resolve the initial value "
                        f"of `{target}.{variable}` — declare repair_effects"
                    )
        effects.append(
            {
                "target": {"component": target, "attribute": variable},
                "value": {
                    "op": "if",
                    "cond": gate,
                    "then": _const(fail_value),
                    "otherwise": otherwise,
                },
            }
        )
    functions = [{"name": "apply_effects", "effects": effects}] if effects else []

    component = {
        "name": name,
        "attributes": [],
        "ports": [],
        "interfaces": [],
        "automata": [automaton],
        "sensitive_functions": functions,
        "equations": [],
    }
    return [component], [], []


def _expand_objevent(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    name = spec["name"]
    aut = spec.get("event_aut_name", "ev")
    occ = spec.get("occ_state_name", "occ")
    not_occ = spec.get("not_occ_state_name", "not_occ")
    cond = _cond_tree(
        spec["cond"],
        spec.get("inner_logic", "all"),
        spec.get("outer_logic", "any"),
        spec.get("cond_operator", "=="),
        spec.get("cond_value", True),
    )
    component = {
        "name": name,
        "attributes": [],
        "ports": [],
        "interfaces": [],
        "automata": [
            {
                "name": aut,
                "states": [not_occ, occ],
                "init": not_occ,
                "transitions": [
                    {
                        # cod3s names the transitions after the target
                        # state (`trans_name_12_fmt="{st2}"`).
                        "name": occ,
                        "source": not_occ,
                        "targets": [occ],
                        "guard": cond,
                        "distrib": "delay",
                        "time": float(spec.get("tempo_occ", 0)),
                    },
                    {
                        "name": not_occ,
                        "source": occ,
                        "targets": [not_occ],
                        "guard": _negate(cond),
                        "distrib": "delay",
                        "time": float(spec.get("tempo_not_occ", 0)),
                    },
                ],
            }
        ],
        "sensitive_functions": [],
        "equations": [],
    }
    return [component], [], []


class MuscadetPlugin:
    """muscadet object types over the RAICHU core."""

    EXPANDERS = {
        "ObjFlow": staticmethod(_expand_objflow),
        "ObjFM": staticmethod(_expand_objfm),
        "ObjFMInst": staticmethod(_expand_objfm_inst),
        "ObjEvent": staticmethod(_expand_objevent),
    }

    def expand_object(
        self, spec: dict[str, Any], model: dict[str, Any]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        kind = spec.get("type")
        expander = self.EXPANDERS.get(kind)
        if expander is None:
            raise ValueError(
                f"muscadet plugin: unknown object type `{kind}` "
                f"(supported: {sorted(self.EXPANDERS)})"
            )
        return expander(spec, model)
