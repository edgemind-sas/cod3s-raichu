"""muscadet-style authoring layer over RAICHU (M3 — `raichu-muscadet`).

Recovers muscadet's productivity idioms — smart flow components,
declarative failure modes, one-line connections — as a thin builder
that *generates a native RAICHU model* (ports/interfaces, sensitive
functions, expression trees).

The authoring surface mirrors `muscadet.ObjFlow`:

>>> import pyraichu.muscadet as mu
>>> class Block(mu.ObjFlow):
...     def add_flows(self):
...         self.add_flow_in(name="is_ok")
...         self.add_flow_out(name="is_ok", var_prod_cond=["is_ok"])
>>> system = mu.System("rbd")
>>> system.add_component(Block, "B1")
>>> ...
>>> system.connect("S", "is_ok", "B1", "is_ok")
>>> result = system.simulate(t_max=24.0)

Semantics generated per flow (mirroring `muscadet/flow.py`):

- flow in  ``f``: bool ``f_fed_in`` := aggregation (``or``/``and``/
  ``k >= n``) of the connected producers' ``f_fed_out``;
- flow out ``f``: bool ``f_fed_out`` := production condition (AND of
  the declared ``var_prod_cond`` in-flows, or the ``var_prod_default``
  constant) AND ``f_fed_available_out`` (driven by failure modes);
- failure modes: two-state automaton (``ok``/``nok``) with delay or
  exponential laws; ``nok`` forces ``f_fed_available_out`` to ``False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Type

from . import Model, SimulationResult, McEstimates, load_model, simulate, monte_carlo

__all__ = ["ObjFlow", "System"]


def _var(component: str, variable: str) -> dict[str, Any]:
    return {"op": "attr", "attr": {"component": component, "attribute": variable}}


def _state_active(component: str, automaton: str, state: str) -> dict[str, Any]:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


@dataclass
class _FlowIn:
    name: str
    logic: str = "or"  # "or" | "and" | int k (k-out-of-n)
    k: int | None = None


@dataclass
class _FlowOut:
    name: str
    var_prod_default: bool = False
    # Flat list = one AND group; list-of-lists = DNF (outer-OR of
    # inner-AND groups, the platform-export `prod_cond` form).
    var_prod_cond: list[str] | list[list[str]] = field(default_factory=list)
    # muscadet `FlowOutTempo`: {"enable_time", "disable_time",
    # "init_enable"} — a disabled↔enabled automaton whose delayed
    # transitions are gated on the production condition (reset on
    # interruption); the flow is fed while `enabled`.
    tempo: dict[str, Any] | None = None
    # muscadet `FlowOutOnTrigger`: {"time_up", "time_down", "logic"} —
    # a down↔up automaton on a dedicated trigger in-port with
    # *inhibition* logic (up while the trigger inputs are absent); the
    # flow is fed while `up` AND the production condition holds.
    trigger: dict[str, Any] | None = None


@dataclass
class _FailureMode:
    name: str
    law: str  # "delay" | "exp"
    failure_param: float
    repair_param: float
    targets: list[str] = field(default_factory=list)  # affected out-flows
    failure_cond: str | None = None  # local variable gating the failure


class ObjFlow:
    """A muscadet-style smart flow component. Subclass and override
    :meth:`add_flows`."""

    def __init__(self, name: str):
        self.name = name
        self.flows_in: list[_FlowIn] = []
        self.flows_out: list[_FlowOut] = []
        self.failure_modes: list[_FailureMode] = []
        self.add_flows()

    def add_flows(self) -> None:  # pragma: no cover - overridden
        """Declare flows (override in subclasses)."""

    def add_flow_in(self, name: str, logic: str | int = "or") -> None:
        """Declare an incoming flow aggregated with `logic`
        (``"or"``, ``"and"`` or an integer k for k-out-of-n)."""
        if isinstance(logic, int):
            self.flows_in.append(_FlowIn(name=name, logic="k", k=logic))
        else:
            self.flows_in.append(_FlowIn(name=name, logic=logic))

    def add_flow_out(
        self,
        name: str,
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """Declare an outgoing flow, produced unconditionally
        (``var_prod_default=True``) or when the named in-flows are fed."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
            )
        )

    def add_delay_failure_mode(
        self,
        name: str,
        failure_time: float,
        repair_time: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
    ) -> None:
        """Deterministic failure/repair mode driving the out-flows'
        availability (all out-flows unless `targets` names some).
        `failure_cond` names a local variable gating the failure
        (muscadet's `cond_occ_12`)."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="delay",
                failure_param=failure_time,
                repair_param=repair_time,
                targets=list(targets or []),
                failure_cond=failure_cond,
            )
        )

    def add_flow_out_tempo(
        self,
        name: str,
        enable_time: float = 0.0,
        disable_time: float = 0.0,
        init_enable: bool = False,
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """muscadet `FlowOutTempo`: the flow feeds while a
        disabled↔enabled automaton sits in `enabled`; the enable
        (resp. disable) transition is a delay of `enable_time`
        (`disable_time`) guarded on the production condition (resp. its
        negation), reset on interruption."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                tempo={
                    "enable_time": enable_time,
                    "disable_time": disable_time,
                    "init_enable": init_enable,
                },
            )
        )

    def add_flow_out_on_trigger(
        self,
        name: str,
        trigger_time_up: float = 0.0,
        trigger_time_down: float = 0.0,
        trigger_logic: str | int = "or",
        var_prod_default: bool = False,
        var_prod_cond: list[str] | None = None,
    ) -> None:
        """muscadet `FlowOutOnTrigger`: the flow feeds while a down↔up
        automaton sits in `up`, with *inhibition* logic — `up` is armed
        while the trigger aggregate (`"and"`, `"or"` or k-out-of-n over
        the `{name}_trigger_in` port) is false, `down` while it is
        true; both transitions are delays, reset on interruption."""
        self.flows_out.append(
            _FlowOut(
                name=name,
                var_prod_default=var_prod_default,
                var_prod_cond=list(var_prod_cond or []),
                trigger={
                    "time_up": trigger_time_up,
                    "time_down": trigger_time_down,
                    "logic": trigger_logic,
                },
            )
        )

    def add_exp_failure_mode(
        self,
        name: str,
        failure_rate: float,
        repair_rate: float,
        targets: list[str] | None = None,
        failure_cond: str | None = None,
    ) -> None:
        """Exponential failure/repair mode (statistical regime)."""
        self.failure_modes.append(
            _FailureMode(
                name=name,
                law="exp",
                failure_param=failure_rate,
                repair_param=repair_rate,
                targets=list(targets or []),
                failure_cond=failure_cond,
            )
        )

    # --- model generation --------------------------------------------

    def _build(self) -> dict[str, Any]:
        me = self.name
        variables: list[dict] = []
        ports: list[dict] = []
        functions: list[dict] = []
        automata: list[dict] = []

        for flow in self.flows_in:
            fed_in = f"{flow.name}_fed_in"
            variables.append(
                {"name": fed_in, "kind": "bool", "init": {"kind": "bool", "value": False}}
            )
            ports.append({"name": f"{flow.name}_in", "dir": "in"})
            agg: dict[str, Any]
            port_ref = {"component": me, "port": f"{flow.name}_in"}
            if flow.logic == "and":
                agg = {"op": "port_agg", "port": port_ref, "agg": "all"}
            elif flow.logic == "k":
                agg = {
                    "op": "cmp",
                    "cmp": "ge",
                    "lhs": {"op": "port_agg", "port": port_ref, "agg": "sum"},
                    "rhs": {"op": "const", "value": {"kind": "int", "value": flow.k}},
                }
            else:
                agg = {"op": "port_agg", "port": port_ref, "agg": "any"}
            functions.append(
                {
                    "name": f"update_{fed_in}",
                    "effects": [
                        {"target": {"component": me, "attribute": fed_in}, "value": agg}
                    ],
                }
            )

        for mode in self.failure_modes:
            failure_transition: dict[str, Any] = {
                "name": "failure",
                "source": "ok",
                "targets": ["nok"],
                "distrib": mode.law,
                ("time" if mode.law == "delay" else "rate"): mode.failure_param,
            }
            if mode.failure_cond is not None:
                failure_transition["guard"] = _var(me, mode.failure_cond)
            automata.append(
                {
                    "name": mode.name,
                    "states": ["ok", "nok"],
                    "init": "ok",
                    "transitions": [
                        failure_transition,
                        {
                            "name": "repair",
                            "source": "nok",
                            "targets": ["ok"],
                            "distrib": mode.law,
                            ("time" if mode.law == "delay" else "rate"): mode.repair_param,
                        },
                    ],
                }
            )

        for flow in self.flows_out:
            fed_out = f"{flow.name}_fed_out"
            # muscadet-aligned name (`FlowOut.var_fed_available_out`): real
            # studies' failure_effects target `{flow}_fed_available_out`.
            available = f"{flow.name}_fed_available_out"
            variables.append(
                {"name": fed_out, "kind": "bool", "init": {"kind": "bool", "value": False}}
            )
            variables.append(
                {"name": available, "kind": "bool", "init": {"kind": "bool", "value": True}}
            )
            ports.append({"name": f"{flow.name}_out", "dir": "out", "attr": fed_out})

            # Availability follows the failure-mode automata targeting
            # this flow (all of them must sit in `ok`).
            relevant = [
                m for m in self.failure_modes if not m.targets or flow.name in m.targets
            ]
            if relevant:
                ok_terms = [_state_active(me, m.name, "ok") for m in relevant]
                avail_expr = (
                    ok_terms[0]
                    if len(ok_terms) == 1
                    else {"op": "bool", "bool_op": "and", "args": ok_terms}
                )
                functions.append(
                    {
                        "name": f"update_{available}",
                        "effects": [
                            {
                                "target": {"component": me, "attribute": available},
                                "value": avail_expr,
                            }
                        ],
                    }
                )

            # Production condition. `var_prod_cond` is either a flat list
            # (one AND group — the historical form) or a DNF list-of-lists
            # (outer-OR of inner-AND groups — the platform-export
            # `prod_cond` form). A referenced flow resolves to this
            # component's `_fed_in` (in-flow) or `_fed_out` (out-flow —
            # the diagnostic-mirror pattern; the fixpoint handles the
            # intra-component dependency without a topological sort).
            in_names = {f.name for f in self.flows_in}
            out_names = {f.name for f in self.flows_out}

            def prod_ref(cond: str) -> dict:
                if cond in in_names:
                    return _var(me, f"{cond}_fed_in")
                if cond in out_names:
                    return _var(me, f"{cond}_fed_out")
                raise ValueError(
                    f"ObjFlow `{me}`: production condition of "
                    f"`{flow.name}` references unknown flow `{cond}` "
                    "(neither an in-flow nor an out-flow of this component)"
                )

            if flow.var_prod_cond:
                groups = (
                    flow.var_prod_cond
                    if all(isinstance(g, list) for g in flow.var_prod_cond)
                    else [flow.var_prod_cond]
                )
                or_terms = []
                for group in groups:
                    and_terms = [prod_ref(cond) for cond in group]
                    or_terms.append(
                        and_terms[0]
                        if len(and_terms) == 1
                        else {"op": "bool", "bool_op": "and", "args": and_terms}
                    )
                prod_expr: dict[str, Any] = (
                    or_terms[0]
                    if len(or_terms) == 1
                    else {"op": "bool", "bool_op": "or", "args": or_terms}
                )
            else:
                prod_expr = {
                    "op": "const",
                    "value": {"kind": "bool", "value": bool(flow.var_prod_default)},
                }

            if flow.tempo is not None:
                # FlowOutTempo: fed while `enabled`; the production
                # condition only gates the (delayed, reset) enable and
                # disable transitions — a lost condition keeps feeding
                # until the disable delay elapses.
                aut = f"{flow.name}_tempo"
                automata.append(
                    {
                        "name": aut,
                        "states": ["disabled", "enabled"],
                        "init": "enabled" if flow.tempo["init_enable"] else "disabled",
                        "transitions": [
                            {
                                "name": f"{flow.name}_enable",
                                "source": "disabled",
                                "targets": ["enabled"],
                                "guard": prod_expr,
                                "distrib": "delay",
                                "time": float(flow.tempo["enable_time"]),
                            },
                            {
                                "name": f"{flow.name}_disable",
                                "source": "enabled",
                                "targets": ["disabled"],
                                "guard": {
                                    "op": "bool",
                                    "bool_op": "not",
                                    "args": [prod_expr],
                                },
                                "distrib": "delay",
                                "time": float(flow.tempo["disable_time"]),
                            },
                        ],
                    }
                )
                gate_terms = [_state_active(me, aut, "enabled")]
            elif flow.trigger is not None:
                # FlowOutOnTrigger: inhibition logic — `up` arms while
                # the trigger aggregate is false; fed while `up` AND
                # the production condition holds.
                aut = f"{flow.name}_trigger"
                port_name = f"{flow.name}_trigger_in"
                ports.append({"name": port_name, "dir": "in"})
                port_ref = {"component": me, "port": port_name}
                logic = flow.trigger["logic"]
                if logic == "and":
                    trigger_agg: dict[str, Any] = {
                        "op": "port_agg",
                        "port": port_ref,
                        "agg": "all",
                    }
                elif logic == "or":
                    trigger_agg = {"op": "port_agg", "port": port_ref, "agg": "any"}
                elif isinstance(logic, int):
                    trigger_agg = {
                        "op": "cmp",
                        "cmp": "ge",
                        "lhs": {"op": "port_agg", "port": port_ref, "agg": "sum"},
                        "rhs": {"op": "const", "value": {"kind": "int", "value": logic}},
                    }
                else:
                    raise ValueError(
                        "trigger logic must be 'and', 'or', or a positive integer"
                    )
                automata.append(
                    {
                        "name": aut,
                        "states": ["down", "up"],
                        "init": "down",
                        "transitions": [
                            {
                                "name": f"{flow.name}_trigger_up",
                                "source": "down",
                                "targets": ["up"],
                                "guard": {
                                    "op": "bool",
                                    "bool_op": "not",
                                    "args": [trigger_agg],
                                },
                                "distrib": "delay",
                                "time": float(flow.trigger["time_up"]),
                            },
                            {
                                "name": f"{flow.name}_trigger_down",
                                "source": "up",
                                "targets": ["down"],
                                "guard": trigger_agg,
                                "distrib": "delay",
                                "time": float(flow.trigger["time_down"]),
                            },
                        ],
                    }
                )
                gate_terms = [_state_active(me, aut, "up"), prod_expr]
            else:
                gate_terms = [prod_expr]

            fed_expr = {
                "op": "bool",
                "bool_op": "and",
                "args": gate_terms + [_var(me, available)],
            }
            functions.append(
                {
                    "name": f"update_{fed_out}",
                    "effects": [
                        {
                            "target": {"component": me, "attribute": fed_out},
                            "value": fed_expr,
                        }
                    ],
                }
            )

        return {
            "name": me,
            "attributes": variables,
            "ports": ports,
            "interfaces": [],
            "automata": automata,
            "sensitive_functions": functions,
            "equations": [],
        }


class System:
    """A muscadet-style system: add components, connect flows, simulate
    through the RAICHU engine."""

    def __init__(self, name: str):
        self.name = name
        self.comp: dict[str, ObjFlow] = {}
        self._connections: list[dict] = []

    def add_component(self, cls: Type[ObjFlow], name: str) -> ObjFlow:
        """Instantiate `cls` under `name` and register it."""
        component = cls(name)
        self.comp[name] = component
        return component

    def connect(self, source: str, flow_out: str, target: str, flow_in: str) -> None:
        """Connect `source`'s out-flow to `target`'s in-flow."""
        self._connections.append(
            {
                "from": {"component": source, "port": f"{flow_out}_out"},
                "to": {"component": target, "port": f"{flow_in}_in"},
            }
        )

    def connect_trigger(self, source: str, target: str, flow: str) -> None:
        """Connect `source`'s out-flow to `target`'s trigger in-port
        (muscadet `connect_trigger`)."""
        self._connections.append(
            {
                "from": {"component": source, "port": f"{flow}_out"},
                "to": {"component": target, "port": f"{flow}_trigger_in"},
            }
        )

    def auto_connect(self, source: str, target: str) -> None:
        """Connect every same-named (out, in) flow pair — the muscadet
        convenience."""
        for flow_out in self.comp[source].flows_out:
            for flow_in in self.comp[target].flows_in:
                if flow_out.name == flow_in.name:
                    self.connect(source, flow_out.name, target, flow_in.name)

    def build_dict(self) -> dict[str, Any]:
        """Generate the native RAICHU model as a plain dict, with one
        indicator per flow variable (muscadet naming: `comp_var`) —
        also the fixture-generation entry point."""
        components = [c._build() for c in self.comp.values()]
        indicators = []
        for component in components:
            for variable in component["attributes"]:
                if variable["name"].endswith(("_fed_in", "_fed_out")):
                    indicators.append(
                        {
                            "name": f"{component['name']}_{variable['name']}",
                            "target": "attribute",
                            "attr": {
                                "component": component["name"],
                                "attribute": variable["name"],
                            },
                        }
                    )
        return {
            "name": self.name,
            "components": components,
            "connections": self._connections,
            "indicators": indicators,
        }

    def build_model(self) -> Model:
        """Generate and validate the native RAICHU model."""
        return load_model(json.dumps(self.build_dict()))

    def simulate(self, t_max: float, **kwargs: Any) -> SimulationResult:
        """One trajectory through the RAICHU engine."""
        return simulate(self.build_model(), t_max=t_max, **kwargs)

    def monte_carlo(
        self, nb_runs: int, t_max: float, samples: list[float], **kwargs: Any
    ) -> McEstimates:
        """Monte-Carlo estimation through the RAICHU driver."""
        return monte_carlo(
            self.build_model(), nb_runs=nb_runs, t_max=t_max, samples=samples, **kwargs
        )
