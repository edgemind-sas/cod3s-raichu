"""Mixture groups of a system: which components pool into one volume."""

from __future__ import annotations

from typing import Any

from ._expr import _clamped_at_zero, _float, _state_active, _sum, _var
from ._records import _content_attribute, _ContinuousEdge
from ._vocabulary import RUN_GATE_STATES


class _SystemMixturesMixin:
    """Methods of :class:`System` mixture groups of a system: which components pool into one volume."""

    def _resolve_mixture_groups(self, edges: list[_ContinuousEdge]) -> None:
        """Bind every mixture group to the one volume that composes for it
        (muscadet's `resolve_mixture_groups`, R51), and write what the group
        asks on each of its inputs: ``R . m_f / S``, ``S`` being the volume's
        weighted occupied volume, read off its integrated contents.

        Several groups may draw one volume (muscadet's R52): each asks its own
        rate at the composition, and the producer's split serves them.

        Refused by name, as muscadet refuses them: a group whose flows do not
        all arrive from ONE capacity of ONE producer, a group leaving out a
        constituent of that volume, a producer serving any held flow of it
        to a consumer that is not one of its groups, a volume drawn by
        several groups whose outputs allocate other than proportionally, a
        flow a rule of the consumer also consumes, and a FED volume drawn as a
        mixture at ``fill_rate=0`` (muscadet 5.18.0: asking only its composed
        share, it would never admit a constituent it does not hold, nor start
        when empty). Recomputed from scratch
        at every generation, so a model generated twice binds the same
        way."""
        for obj in self.comp.values():
            for capacity in obj.capacities:
                capacity.serves_mixture = False
            for flow in obj.flows_continuous_in:
                flow.mixture_demand = None
        # (producer, capacity name) -> the INPUTS drawing it as a mixture,
        # `(consumer, flow_in)` pairs, which is what the exclusivity rule
        # exempts, and the groups, which the allocation rule counts. Keyed on
        # the input and not on the consumer: a connection here joins ports,
        # so a machine may wire an UNGROUPED input to a held output too, and
        # that one is a per-flow request like any other.
        drawers: dict[tuple[str, str], set[tuple[str, str]]] = {}
        bound: dict[tuple[str, str], list[str]] = {}
        for consumer, obj in self.comp.items():
            for group in obj.mixtures:
                for edge in edges:
                    if edge.consumer == consumer and edge.flow_in in group.flows:
                        supplier = self.comp[edge.producer]
                        held = supplier._capacity_of(edge.flow_out)
                        if held is not None:
                            drawers.setdefault((edge.producer, held.name), set()).add(
                                (consumer, edge.flow_in)
                            )
        for consumer, obj in self.comp.items():
            for group in obj.mixtures:
                where = f"System `{self.name}`: mixture group `{consumer}.{group.name}`"
                feeding = {
                    flow: [e for e in edges if e.consumer == consumer and e.flow_in == flow]
                    for flow in group.flows
                }
                producers = {e.producer for found in feeding.values() for e in found}
                if any(len(found) != 1 for found in feeding.values()) or len(producers) != 1:
                    raise ValueError(
                        f"{where}: its flows must all arrive from one capacity of "
                        f"one producer, and they arrive from "
                        f"{ {f: [e.producer for e in found] for f, found in feeding.items()} }"
                    )
                producer = producers.pop()
                supplier = self.comp[producer]
                outputs = {flow: found[0].flow_out for flow, found in feeding.items()}
                held = [supplier._capacity_of(out) for out in outputs.values()]
                names = {c.name if c is not None else None for c in held}
                if len(names) != 1 or None in names:
                    raise ValueError(
                        f"{where}: its flows must all arrive from one capacity of "
                        f"one producer, and `{producer}` holds them in "
                        f"{sorted(n or 'no capacity' for n in names)}"
                    )
                capacity = held[0]
                left_out = [
                    flow
                    for flow in capacity.flow_names
                    if flow not in outputs.values()
                ]
                if left_out:
                    raise ValueError(
                        f"{where}: capacity `{capacity.name}` of `{producer}` also "
                        f"holds {left_out}, which the group does not draw. A "
                        "machine moves a volume of whatever the room holds, so a "
                        "group names every constituent of the volume it draws"
                    )
                mixtures = drawers.get((producer, capacity.name), set())
                others = [
                    e for e in edges
                    if e.producer == producer
                    and e.flow_out in capacity.flow_names
                    and (e.consumer, e.flow_in) not in mixtures
                ]
                if others:
                    raise ValueError(
                        f"{where}: `{producer}` also serves "
                        f"{sorted({e.consumer for e in others})} out of the volume "
                        "the group draws, and a volume drawn as a mixture serves "
                        "mixture groups alone: a composed share against a per-flow "
                        "request is not arbitrated"
                    )
                bound.setdefault((producer, capacity.name), []).append(
                    f"{consumer}.{group.name}"
                )
                consumed = {
                    flow for rule_set in obj.rule_sets for flow in rule_set.consumed()
                }
                ruled = sorted(set(group.flows) & consumed)
                if ruled:
                    raise ValueError(
                        f"{where}: {ruled} are consumed by a rule of `{consumer}` "
                        "too, and an input has one demand"
                    )
                capacity.serves_mixture = True
                occupied = _sum(
                    [
                        {
                            "op": "mul",
                            "args": [
                                _var(producer, _content_attribute(capacity.name, entry.name)),
                                _float(entry.weight),
                            ],
                        }
                        for entry in capacity.flows
                    ]
                )
                # The rate is READ, never written in: a failure mode may have
                # clamped it (R52), and a mode writing below zero states a
                # direction, which is the connection's and not the rate's.
                # The command BRANCHES on it, never multiplies.
                rate: dict[str, Any] = _clamped_at_zero(
                    _var(consumer, group.rate_attribute)
                )
                if group.commanded:
                    rate = {
                        "op": "if",
                        "cond": _state_active(consumer, group.gate, RUN_GATE_STATES[0]),
                        "then": rate,
                        "otherwise": _float(0.0),
                    }
                for flow, out in outputs.items():
                    share = {
                        "op": "div",
                        "lhs": _var(producer, _content_attribute(capacity.name, out)),
                        "rhs": occupied,
                    }
                    demand = {
                        "op": "if",
                        "cond": {"op": "cmp", "cmp": "gt", "lhs": occupied, "rhs": _float(0.0)},
                        "then": {"op": "mul", "args": [rate, share]},
                        "otherwise": _float(0.0),
                    }
                    target = next(f for f in obj.flows_continuous_in if f.name == flow)
                    target.mixture_demand = demand
        for (producer, capacity_name), groups in bound.items():
            supplier = self.comp[producer]
            capacity = next(c for c in supplier.capacities if c.name == capacity_name)
            fed = sorted(
                {
                    e.flow_in
                    for e in edges
                    if e.consumer == producer and e.flow_in in capacity.flow_names
                }
            )
            if fed and not capacity.fill_rate:
                raise ValueError(
                    f"System `{self.name}`: capacity `{capacity_name}` of "
                    f"`{producer}` is drawn as a mixture ({', '.join(groups)}) "
                    f"and fed on {', '.join(fed)}, with fill_rate=0. At that "
                    "rate it asks upstream only what the machines take at its "
                    "composition, so a constituent it does not hold would never "
                    "enter and an empty volume would never fill. Declare a "
                    "fill_rate: math.inf takes whatever the producers deliver"
                )
            if len(groups) < 2:
                continue
            for out in supplier.flows_continuous_out:
                if out.name not in capacity.flow_names:
                    continue
                if out.allocation != "proportional":
                    raise ValueError(
                        f"System `{self.name}`: capacity `{capacity_name}` of "
                        f"`{producer}` is drawn by several mixture groups "
                        f"({', '.join(groups)}), and its output `{out.name}` "
                        f"declares allocation={out.allocation!r}. Machines on one "
                        "volume are served pro rata of their composed shares, "
                        "which is the proportional allocation: declare none on "
                        "that output"
                    )
