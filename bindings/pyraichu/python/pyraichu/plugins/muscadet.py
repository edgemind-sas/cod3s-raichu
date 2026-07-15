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


def _cond_groups(cond: Any, inner_logic: str = "all") -> list[dict]:
    """Normalize a cond spec to the list of per-group boolean expressions
    (each an inner-logic aggregation of its leaves) — the OR-of-AND groups
    that `_cond_tree` ORs together and that the logic gate aggregates by
    `kind`. Mirrors `sanitize_cond_format`."""
    if isinstance(cond, bool):
        return [_const(cond)]
    if isinstance(cond, dict):
        cond = [[cond]]
    elif cond and all(isinstance(c, dict) for c in cond):
        cond = [cond]
    return [
        {"op": "bool", "bool_op": _LOGIC[inner_logic], "args": [_leaf(c) for c in group]}
        for group in cond
    ]


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


def _ccf_suffix(comb: tuple, order_max: int) -> str:
    """cod3s common-cause combination suffix (`__cc_i_j`, 1-based,
    underscore-separated since cod3s 1.9.0); empty for the single-target
    order-1 case. Shared by the ObjFM and ObjFMInst CCF expansions."""
    return "__cc_" + "_".join(str(i + 1) for i in comb) if order_max > 1 else ""


def _state_active(component: str, automaton: str, state: str) -> dict:
    """`state_active` core expression."""
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def _bool_expr(op: str, args: list) -> dict:
    """`and`/`or` over args, collapsing the singleton."""
    return args[0] if len(args) == 1 else {"op": "bool", "bool_op": op, "args": args}


def _attr_is(component: str, attribute: str, value: bool) -> dict:
    """`component.attribute == value` core comparison."""
    return {
        "op": "cmp",
        "cmp": "eq",
        "lhs": {"op": "attr", "attr": {"component": component, "attribute": attribute}},
        "rhs": _const(value),
    }


def _target_init(model: dict, target: str, variable: str) -> dict | None:
    """Declared initial value of a target attribute (the reinitialization rest
    state), or None when the attribute is not found on the component."""
    for component in model.get("components", []):
        if component.get("name") != target:
            continue
        for entry in component.get("attributes", []):
            if entry.get("name") == variable:
                return {"op": "const", "value": entry["init"]}
    return None


def _reinit_effect(
    name: str,
    target: str,
    gate: dict,
    failure_effects: dict,
    repair_effects: dict,
    model: dict,
) -> list[dict]:
    """Reinitialization effects for ONE target under a given `gate`: each
    attribute holds its failure value while `gate` is true, its rest-state
    value otherwise (an explicit ``repair_effects`` entry, else the declared
    initial value, else the boolean complement). `gate` is the OR of impacting
    combinations (internal / ObjFMInst) or the target's own mirror failure
    state (external)."""
    effects = []
    for variable, fail_value in failure_effects.items():
        if variable in repair_effects:
            otherwise = _const(repair_effects[variable])
        else:
            otherwise = _target_init(model, target, variable)
            if otherwise is None:
                if isinstance(fail_value, bool):
                    otherwise = _const(not fail_value)
                else:
                    raise ValueError(
                        f"ObjFM `{name}`: cannot resolve the initial value of "
                        f"`{target}.{variable}` (reinitialization semantics) — "
                        "declare repair_effects"
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
    return effects


def _reinit_effects(
    name: str,
    targets: list,
    impacting: dict,
    failure_effects: dict,
    repair_effects: dict,
    model: dict,
) -> list[dict]:
    """Internal-mode reinitialization effects (shared by the internal ObjFM
    and ObjFMInst CCF expansions): per target, the gate is the OR over the
    impacting combinations' failure (`occ`) states."""
    effects = []
    for target in targets:
        gates = [_state_active(name, aut, st) for aut, st in impacting[target]]
        if not gates:
            continue
        effects += _reinit_effect(
            name, target, _bool_expr("or", gates), failure_effects, repair_effects, model
        )
    return effects


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
        # Occurrence-law kind: accept the cod3s `cls`, the plugin `law`,
        # and the post-migration `distrib` spelling (as `_law` does).
        kind = mode.get("law") or mode.get("cls") or mode.get("distrib") or "delay"
        if kind == "delay":
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
    """cod3s ObjFM expansion — three behaviours over N targets and every
    *active* common-cause order (`fm__cc_i_j`, states `occ__cc_i_j` /
    `rep__cc_i_j`, transitions named after the target states;
    underscore-separated indices since cod3s 1.9.0; per-order laws as
    ``null``-padded lists in ``"failure"`` / ``"repair"``):

    - ``internal`` (default): the ObjFM writes each target attribute directly,
      held at its failure value while ANY impacting combination is failed, its
      initial value otherwise (reinitialization — repair_effects unnecessary by
      construction);
    - ``external`` (mutual lock): a boolean control attribute
      ``ctrl_{name}_{target}`` = OR(impacting occ) drives a mirror automaton
      grafted into each target; a combination can only (re)fail once its
      targets are repaired and (re)repair once they are failed;
    - ``external_rep_indep`` (trigger): the ObjFM resets instantly and each
      target latches the failure until it repairs on its own order-1 law."""
    import itertools

    behaviour = spec.get("behaviour", "internal")
    if behaviour not in ("internal", "external", "external_rep_indep"):
        raise ValueError(
            f"ObjFM `{spec['name']}`: unknown behaviour `{behaviour}` "
            "(expected 'internal', 'external' or 'external_rep_indep')"
        )
    rep_indep = behaviour == "external_rep_indep"
    external = behaviour == "external" or rep_indep

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
    # Sequence monitoring: internal ObjFMs carry the sequence events on
    # their own occ/rep transitions; external modes carry them on the
    # TARGET's grafted mirror instead (cod3s drops the external ObjFM's own
    # events — in rep_indep its instant occ+rep pair would always cancel).
    monitored = not external
    for order in range(1, order_max + 1):
        f_law = failure_laws[order - 1]
        r_law = repair_laws[order - 1]
        if f_law is None:
            continue  # inactive order (cod3s `drop_inactive_automata`)
        for comb in itertools.combinations(range(order_max), order):
            suffix = _ccf_suffix(comb, order_max)
            aut_name = f"fm{suffix}"
            occ = f"{failure_state}{suffix}"
            rep = f"{repair_state}{suffix}"
            fail_guard = _cond_tree(spec.get("failure_cond", True), inner, outer)
            repair_guard = _cond_tree(spec.get("repair_cond", True), inner, outer)
            if external:
                # Mutual lock: fail only once all combo targets are repaired.
                fail_guard = _bool_expr(
                    "and",
                    [fail_guard]
                    + [_state_active(targets[i], name, repair_state) for i in comb],
                )
            # Repair transition: rep_indep resets instantly (structural,
            # law-independent); otherwise only an ACTIVE repair law builds
            # one — a non-repairable mode keeps its failure with an
            # absorbing occ state (cod3s `is_occ_law_repair_active`).
            if rep_indep:
                repair_guard = _const(True)
                repair_fields: dict | None = {"distrib": "delay", "time": 0.0}
            elif r_law is None:
                repair_fields = None
            else:
                repair_fields = _law(r_law)
                if external:
                    # Repair only once all combo targets are failed.
                    repair_guard = _bool_expr(
                        "and",
                        [repair_guard]
                        + [_state_active(targets[i], name, failure_state) for i in comb],
                    )
            transitions = [
                {
                    "name": "failure" if order_max == 1 else occ,
                    "source": rep,
                    "targets": [occ],
                    "guard": fail_guard,
                    "monitored": monitored,
                    "cycle_group": aut_name,
                    **_law(f_law),
                }
            ]
            if repair_fields is not None:
                transitions.append(
                    {
                        "name": "repair" if order_max == 1 else rep,
                        "source": occ,
                        "targets": [rep],
                        "guard": repair_guard,
                        "monitored": monitored,
                        "cycle_group": aut_name,
                        **repair_fields,
                    }
                )
            automata.append(
                {
                    "name": aut_name,
                    "states": [rep, occ],
                    "init": rep,
                    "transitions": transitions,
                }
            )
            for idx in comb:
                impacting[targets[idx]].append((aut_name, occ))

    if not external:
        # INTERNAL (default): reinitialization semantics — the ObjFM writes each
        # target attribute directly, held at its failure value while ANY
        # impacting combination is failed (the OR gate), its initial value
        # otherwise. cod3s models omit repair_effects here (adding them hangs
        # the simulator on multi-order ObjFMs).
        effects = _reinit_effects(
            name, targets, impacting, failure_effects, repair_effects, model
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

    def target_component(cname):
        for component in model.get("components", []):
            if component.get("name") == cname:
                return component
        return None

    # EXTERNAL / EXTERNAL_REP_INDEP (cod3s FEAT_OBJFM_SPECS Rev 1.1): a boolean
    # control attribute `ctrl_{name}_{target}` on the ObjFM drives a mirror
    # automaton grafted into each target; the failure/repair effects apply
    # through the target's mirror state. In `external` the control follows
    # `OR(impacting ObjFM automata in occ)`; in `external_rep_indep` the
    # trigger is transient, so the control also latches on the target's own
    # occ state (held until the target repairs on its own).
    ctrl_attributes = []
    ctrl_functions = []
    for target in targets:
        ctrl = f"ctrl_{name}_{target}"
        ctrl_attributes.append(
            {"name": ctrl, "kind": "bool", "init": {"kind": "bool", "value": False}}
        )
        occ_gates = [_state_active(name, aut, st) for aut, st in impacting[target]]
        if rep_indep:
            occ_gates = occ_gates + [_state_active(target, name, failure_state)]
        ctrl_functions.append(
            {
                "name": f"update_{ctrl}",
                "effects": [
                    {
                        "target": {"component": name, "attribute": ctrl},
                        "value": _bool_expr("or", occ_gates),
                    }
                ],
            }
        )

    objfm_component = {
        "name": name,
        "attributes": ctrl_attributes,
        "ports": [],
        "interfaces": [],
        "automata": automata,
        "sensitive_functions": ctrl_functions,
        "equations": [],
    }

    for target in targets:
        comp = target_component(target)
        if comp is None:
            raise ValueError(
                f"ObjFM `{name}` ({behaviour}): target component `{target}` is "
                "not (yet) present — external modes graft a mirror automaton "
                "into the target in place, so each target must be declared "
                "before this ObjFM (unlike `internal`, which resolves targets "
                "by name at load time)"
            )
        # The mirror automaton is named after the ObjFM and its effect function
        # `apply_{name}`; guard against grafting over a member the target
        # already owns (a duplicate-name model whose validation error would not
        # point back here).
        apply_fn = f"apply_{name}"
        if any(a.get("name") == name for a in comp.get("automata", [])):
            raise ValueError(
                f"ObjFM `{name}` ({behaviour}): target `{target}` already has an "
                f"automaton named `{name}` — rename the ObjFM or the automaton"
            )
        if any(f.get("name") == apply_fn for f in comp.get("sensitive_functions", [])):
            raise ValueError(
                f"ObjFM `{name}` ({behaviour}): target `{target}` already has a "
                f"sensitive function named `{apply_fn}` — rename the ObjFM"
            )
        ctrl = f"ctrl_{name}_{target}"
        if rep_indep:
            # The target owns its repair: order-1 law of the ObjFM, gated by the
            # user repair_cond evaluated on the target.
            if repair_laws[0] is None:
                raise ValueError(
                    f"ObjFM `{name}` (external_rep_indep): the order-1 repair "
                    "law is inactive but drives each target's repair"
                )
            repair_transition = {
                "name": repair_state,
                "source": failure_state,
                "targets": [repair_state],
                "guard": _cond_tree(spec.get("repair_cond", True), inner, outer),
                "monitored": True,
                "cycle_group": name,
                **_law(repair_laws[0]),
            }
        else:
            # external: the target mirror follows the control attribute.
            repair_transition = {
                "name": repair_state,
                "source": failure_state,
                "targets": [repair_state],
                "guard": _attr_is(name, ctrl, False),
                "monitored": True,
                "cycle_group": name,
                "distrib": "delay",
                "time": 0.0,
            }
        # The target mirror carries the external mode's sequence events
        # (cod3s drops the external ObjFM's own occ/rep and pairs the
        # target's `{fm}` occ/rep instead); the filter's (component,
        # group) key makes the pair unique per (target, ObjFM).
        comp.setdefault("automata", []).append(
            {
                "name": name,
                "states": [repair_state, failure_state],
                "init": repair_state,
                "transitions": [
                    {
                        "name": failure_state,
                        "source": repair_state,
                        "targets": [failure_state],
                        "guard": _attr_is(name, ctrl, True),
                        "monitored": True,
                        "cycle_group": name,
                        "distrib": "delay",
                        "time": 0.0,
                    },
                    repair_transition,
                ],
            }
        )
        effects = _reinit_effect(
            name,
            target,
            _state_active(target, name, failure_state),
            failure_effects,
            repair_effects,
            model,
        )
        if effects:
            comp.setdefault("sensitive_functions", []).append(
                {"name": f"apply_{name}", "effects": effects}
            )

    return [objfm_component], [], []


def _expand_objfm_inst(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """cod3s `ObjFMInst` expansion — **failure on solicitation**
    (user guide `objfm-inst.md`, ADR 2026-07-05), including **common
    cause** (test_objfm_inst_002_ccf).

    The demand is ``failure_cond``; on each demand *front* the mode fails
    with probability ``gamma`` — one Bernoulli draw, instantaneously —
    and is repaired by an exponential ``mu``. Each cc-combination is a
    **3-state** automaton (`rep` / `occ` / `not_occ`):

    - `rep --[inst, guard=demand]--> {occ: gamma_k, not_occ: 1-gamma_k}` —
      the draw (a branching instantaneous transition, RAICHU brique 2);
    - `not_occ --[inst p=1, guard=NOT demand]--> rep` — the deterministic
      re-arm; `not_occ` absorbs the front so no re-draw happens while the
      demand holds (anti-Zeno);
    - `occ --[exp(mu_k), guard=repair_cond]--> rep` — the repair
      (omitted when ``mu_k = 0`` — occ absorbing).

    With N targets, `failure_param = [gamma_1, …, gamma_n]` (per order)
    generates the 2^N−1 combination automata (`__cc_` suffix), one per
    non-empty subset of an *active* order; each draws **independently** on
    a shared front. ``failure_effects`` apply while ANY impacting
    combination sits in its `occ` (reinitialization semantics, as the
    internal CCF). A single scalar `gamma` with one target is the order-1
    special case (no suffix, automaton `fm`).
    """
    import itertools

    name = spec["name"]
    targets = spec["targets"]
    order_max = len(targets)
    if order_max == 0:
        raise ValueError(f"ObjFMInst `{name}`: `targets` must be non-empty")
    failure_state = spec.get("failure_state", "occ")
    repair_state = spec.get("repair_state", "rep")
    absorb_state = spec.get("absorb_state", "not_occ")
    failure_effects: dict = spec.get("failure_effects", {})
    repair_effects: dict = spec.get("repair_effects", {})
    inner = spec.get("cond_inner_logic", "all")
    outer = spec.get("cond_outer_logic", "any")

    # Per-order gammas (`failure`/`failure_param`) and mus (`repair`/
    # `repair_param`): a scalar means one value; a list is per order,
    # padded with `None` (inactive order) to the target count. A scalar
    # repair is *broadcast* to every order (not just order 1).
    fspec = spec.get("failure", spec.get("failure_param"))
    if fspec is None:
        raise ValueError(
            f"ObjFMInst `{name}`: missing `failure` (or `failure_param`) — "
            "the per-order Bernoulli gamma(s)"
        )
    failure_specs = list(fspec) if isinstance(fspec, list) else [fspec]
    failure_specs += [None] * (order_max - len(failure_specs))
    rspec = spec.get("repair", spec.get("repair_param", 0.0))
    if isinstance(rspec, list):
        repair_specs = list(rspec) + [None] * (order_max - len(rspec))
    else:
        repair_specs = [rspec] * order_max

    def gamma_of(fs):
        if fs is None:
            return None
        raw = fs.get("prob", fs.get("gamma")) if isinstance(fs, dict) else fs
        if raw is None:
            raise ValueError(
                f"ObjFMInst `{name}`: a failure spec is missing its `prob`/`gamma`"
            )
        g = float(raw)
        if not 0.0 <= g <= 1.0:
            raise ValueError(f"ObjFMInst `{name}`: gamma must be in [0, 1] (got {g})")
        return g

    def repair_of(rs):
        """(build_repair, law): whether an occ->rep transition is
        generated and its law. `None` or an exponential rate 0 means no
        repair (occ absorbing, cod3s `is_occ_law_repair_active` false); a
        delay law or a positive exp rate builds a real repair."""
        if rs is None:
            return False, None
        law = _law(rs) if isinstance(rs, dict) else {"distrib": "exp", "rate": float(rs)}
        if law["distrib"] == "exp" and float(law.get("rate", 0.0)) == 0.0:
            return False, law
        return True, law

    demand = _cond_tree(spec.get("failure_cond", True), inner, outer)
    not_demand = _negate(demand)
    repair_guard = _cond_tree(spec.get("repair_cond", True), inner, outer)

    automata = []
    impacting: dict[str, list[tuple[str, str]]] = {t: [] for t in targets}
    for order in range(1, order_max + 1):
        gamma = gamma_of(failure_specs[order - 1])
        if gamma is None:
            continue  # inactive order (dropped, like the internal CCF)
        build_repair, repair_law = repair_of(repair_specs[order - 1])
        for comb in itertools.combinations(range(order_max), order):
            suffix = _ccf_suffix(comb, order_max)
            occ = f"{failure_state}{suffix}"
            rep = f"{repair_state}{suffix}"
            absorb = f"{absorb_state}{suffix}"
            transitions = [
                {  # the draw: rep -> occ (gamma_k) | not_occ (1 - gamma_k)
                    "name": occ,
                    "source": rep,
                    "targets": [occ, absorb],
                    "guard": demand,
                    "distrib": "inst",
                    "probs": [gamma],
                },
                {  # deterministic re-arm when the demand falls (inst p = 1)
                    "name": absorb,
                    "source": absorb,
                    "targets": [rep],
                    "guard": not_demand,
                    "distrib": "inst",
                    "probs": [],
                },
            ]
            if build_repair:
                transitions.append(
                    {
                        "name": rep,
                        "source": occ,
                        "targets": [rep],
                        "guard": repair_guard,
                        **repair_law,
                    }
                )
            automata.append(
                {
                    "name": f"fm{suffix}",
                    "states": [rep, occ, absorb],
                    "init": rep,
                    "transitions": transitions,
                }
            )
            for idx in comb:
                impacting[targets[idx]].append((f"fm{suffix}", occ))

    # Reinitialization semantics: each target's variable holds its fail
    # value while ANY impacting combination sits in its `occ` (an OR over
    # those occ states), the initial (or explicit repair) value otherwise —
    # identical to the internal CCF, hence the shared helper.
    effects = _reinit_effects(
        name, targets, impacting, failure_effects, repair_effects, model
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
                        "monitored": True,
                        "cycle_group": name,
                        "distrib": "delay",
                        "time": float(spec.get("tempo_occ", 0)),
                    },
                    {
                        "name": not_occ,
                        "source": occ,
                        "targets": [not_occ],
                        "guard": _negate(cond),
                        "monitored": True,
                        "cycle_group": name,
                        "distrib": "delay",
                        "time": float(spec.get("tempo_not_occ", 0)),
                    },
                ],
            }
        ],
        "sensitive_functions": [],
        "equations": [],
    }
    # A feared event marked `"target": true` becomes a sequence-analysis
    # target: reaching its `occ` state ends and labels a trajectory.
    if spec.get("target"):
        model.setdefault("targets", []).append(
            {"name": name, "component": name, "automaton": aut, "state": occ}
        )
    return [component], [], []


def _expand_objlogicgate(spec: dict, model: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """muscadet `ObjLogicGate` (`muscadet/obj_logic.py`): a combinational
    boolean gate, **automaton-free**. A `result` attribute is recomputed —
    edge-triggered by a sensitive function whenever a referenced input
    changes — as `kind` over the condition leaves. By convention `cond` is
    one unit clause per source (`[[s1], [s2], …]`), so `kind` alone chooses
    the aggregation:

    - ``or``  → any source true;
    - ``and`` → all sources true;
    - ``k``   → at least ``k`` sources true (count ≥ k).

    Each ``out_elements`` entry exports ``result`` through one out port; a
    gate feeding several targets (broadcast) is simply several out ports /
    connections. Mirrors muscadet's ``result`` variable + sensitive method
    (`obj_logic.py:95-138`)."""
    name = spec["name"]
    kind = spec.get("kind", "or")
    inner = spec.get("inner_logic", "all")
    groups = _cond_groups(spec.get("cond", []), inner)
    if not groups:
        # An empty condition would silently evaluate as a CONSTANT gate
        # (empty OR = false, empty AND = true) — fail at build time instead.
        raise ValueError(
            f"ObjLogicGate `{name}`: empty or missing `cond` — declare at "
            "least one source leaf"
        )

    if kind == "or":
        value = {"op": "bool", "bool_op": "or", "args": groups}
    elif kind == "and":
        value = {"op": "bool", "bool_op": "and", "args": groups}
    elif kind == "k":
        k = spec.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(
                f"ObjLogicGate `{name}`: kind 'k' needs an integer threshold "
                f"k >= 1 (got {k!r})"
            )
        # count of true source flags ≥ k
        count = {
            "op": "add",
            "args": [
                {"op": "if", "cond": g, "then": _const(1), "otherwise": _const(0)}
                for g in groups
            ],
        }
        value = {"op": "cmp", "cmp": "ge", "lhs": count, "rhs": _const(k)}
    else:
        raise ValueError(
            f"ObjLogicGate `{name}`: unknown kind `{kind}` (expected 'or', 'and' or 'k')"
        )

    ports = [
        {"name": f"{elem}_out", "dir": "out", "attr": "result"}
        for elem in spec.get("out_elements", [])
    ]
    component = {
        "name": name,
        "attributes": [
            {"name": "result", "kind": "bool", "init": {"kind": "bool", "value": False}}
        ],
        "ports": ports,
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [
            {
                "name": f"recompute_{name}",
                "effects": [
                    {"target": {"component": name, "attribute": "result"}, "value": value}
                ],
            }
        ],
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
        "ObjLogicGate": staticmethod(_expand_objlogicgate),
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
