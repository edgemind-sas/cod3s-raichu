"""The continuous network of a system: its edges, its emission and its evaluation order."""

from __future__ import annotations

import heapq
from typing import Any

from ._expr import _clamped_at_zero, _float, _float_attribute, _sum, _var
from ._graph import _channel_attr, _edge_name, _strongly_connected
from ._records import (
    _content_attribute,
    _ContinuousEdge,
    _MeasurementIn,
    _MeasurementOut,
    _rate_alias,
)
from ._vocabulary import RATE_CAPABILITY, RATE_DELIVERED


class _SystemNetworkMixin:
    """Methods of :class:`System` the continuous network of a system: its edges, its emission and its evaluation order."""

    # --- continuous network ------------------------------------------

    def _continuous_edges(self) -> list[_ContinuousEdge]:
        """The continuous connections, resolved against the two
        declarations they join.

        A connection joining a continuous flow to a boolean one is
        refused here, naming both ends: it would otherwise reach the
        engine as a type mismatch on an aggregation."""
        edges: list[_ContinuousEdge] = []
        for connection in self._connections:
            source, destination = connection["from"], connection["to"]
            producer = self.comp.get(source["component"])
            consumer = self.comp.get(destination["component"])
            flow_out = next(
                (
                    flow.name
                    for flow in (producer.flows_continuous_out if producer else [])
                    if f"{flow.name}_out" == source["port"]
                ),
                None,
            )
            flow_in = next(
                (
                    flow.name
                    for flow in (consumer.flows_continuous_in if consumer else [])
                    if f"{flow.name}_in" == destination["port"]
                ),
                None,
            )
            if flow_out is None and flow_in is None:
                continue
            if flow_out is None or flow_in is None:
                stranger = source if flow_out is None else destination
                if stranger["component"] not in self.comp:
                    raise ValueError(
                        f"System `{self.name}`: connection "
                        f"{source['component']}.{source['port']} -> "
                        f"{destination['component']}.{destination['port']} "
                        f"carries a continuous flow into `{stranger['component']}`, "
                        "which this system did not build. The continuous "
                        "network is resolved over the components it declares, "
                        "so a quantity crossing into another one is accounted "
                        "for nowhere"
                    )
                raise ValueError(
                    f"System `{self.name}`: connection "
                    f"{source['component']}.{source['port']} -> "
                    f"{destination['component']}.{destination['port']} joins a "
                    "continuous flow to a boolean one; a connection carries "
                    "one or the other"
                )
            edges.append(
                _ContinuousEdge(
                    name=_edge_name(connection),
                    producer=source["component"],
                    flow_out=flow_out,
                    consumer=destination["component"],
                    flow_in=flow_in,
                )
            )
        return edges

    def _allocation_params(
        self,
        edges: list[_ContinuousEdge],
        declared: dict[Any, float],
        keyword: str,
    ) -> list[dict[str, Any]]:
        """A ``{consumer: value}`` declaration as the schema's
        ConsumerParam list, keyed by destination port.

        Keying by destination rather than by position is what keeps a
        share attached to its consumer when a connection is inserted or
        reordered, so the authoring key is the consumer, never an
        index."""
        producer, flow = edges[0].producer, edges[0].flow_out
        params: list[dict[str, Any]] = []
        matched: set[Any] = set()
        for edge in edges:
            key = next(
                (
                    candidate
                    for candidate in ((edge.consumer, edge.flow_in), edge.consumer)
                    if candidate in declared
                ),
                None,
            )
            if key is None:
                raise ValueError(
                    f"System `{self.name}`: continuous out-flow `{flow}` of "
                    f"`{producer}` declares `{keyword}` without an entry for "
                    f"consumer `{edge.consumer}`; a keyed policy must cover "
                    "every connection of the flow"
                )
            matched.add(key)
            params.append(
                {
                    "to": {"component": edge.consumer, "port": f"{edge.flow_in}_in"},
                    "value": float(declared[key]),
                }
            )
        unknown = [key for key in declared if key not in matched]
        if unknown:
            raise ValueError(
                f"System `{self.name}`: continuous out-flow `{flow}` of "
                f"`{producer}` declares `{keyword}` for {unknown}, which no "
                "connection of the flow ends at"
            )
        return params

    def _releasing_inputs(
        self, in_edges: dict[tuple[str, str], list[_ContinuousEdge]]
    ) -> set[tuple[str, str]]:
        """The ``(component, flow)`` inputs that return what they do not
        need to their suppliers.

        Every supplier of one input is asked the input's whole demand, so
        together they may offer more than it needs. A **rule** consumes
        its need and no more, and muscadet's `release_unused_supply`
        hands the surplus back pro rata of what each supplier offered:
        without it the input kept everything offered and the difference
        vanished, a battery beside a bus emptying at twice the rate its
        electrolyser consumed.

        **A single supplier releases too.** The rule runs at the draw its
        scarcest arrival allows (:meth:`ObjFlow._rule_draw_scale`), which
        may be less than the scale its demand was published at: a recipe
        limited by one input consumes less of the others than it asked
        for, and the rest goes back rather than vanishing. The need is
        therefore what the rules consume at their draw,
        ``{flow}_consumed_in``, never the demand they published.

        Which inputs release is :meth:`ObjFlow._releases_surplus`: one a
        volume upstream of the rules holds is filled by what arrives
        instead, as muscadet's does.

        **A pure sink fed twice or more releases too** (muscadet 6.1.0,
        :meth:`ObjFlow._is_sink`): each supplier serves it its whole
        published demand, so it kept the sum of the offers, two sources of
        9 and 3 delivering 12 to a demand of 6. Its need is that demand.
        A single supplier already serves no more than it, so it releases
        nothing and the input is left as it was."""
        releasing: set[tuple[str, str]] = set()
        for (name, flow), edges in in_edges.items():
            obj = self.comp.get(name)
            if obj is None:
                continue
            if obj._releases_surplus(flow) or (
                len(edges) >= 2
                and obj._capacity_of(flow) is None
                and obj._is_sink(flow)
            ):
                releasing.add((name, flow))
        return releasing

    def _emit_continuous_network(
        self, components: list[dict[str, Any]], edges: list[_ContinuousEdge]
    ) -> list[dict[str, str]] | None:
        """Add the connection-dependent material to `components` and
        return the evaluation order it needs, or ``None`` when the
        system declares no continuous flow.

        The equations emitted here are the ones a component cannot write
        on its own, because they name the connections: the per-edge
        demand and capability channels and the two totals on a producer,
        the two aggregated channels on a consumer.

        `edges` is the resolved connection graph, taken as a parameter
        rather than derived again: :meth:`build_dict` has already
        resolved it, and re-deriving it here would re-run every
        continuous/boolean compatibility refusal for a second answer that
        cannot differ, nothing between the two touching the connections
        or the flow declarations.
        """
        if not any(
            component.flows_continuous_in or component.flows_continuous_out
            for component in self.comp.values()
        ):
            return None

        by_name = {component["name"]: component for component in components}
        out_edges: dict[tuple[str, str], list[_ContinuousEdge]] = {}
        in_edges: dict[tuple[str, str], list[_ContinuousEdge]] = {}
        for edge in edges:
            out_edges.setdefault((edge.producer, edge.flow_out), []).append(edge)
            in_edges.setdefault((edge.consumer, edge.flow_in), []).append(edge)
        releasing = self._releasing_inputs(in_edges)

        def held_empty(edge: _ContinuousEdge) -> dict[str, Any] | None:
            """Whether `edge` leaves a volume that has run out of the flow
            and so passes on what arrives, as a location of that volume, or
            ``None`` when the supplier holds no such volume. What `transmits`
            says does not enter: an empty volume with a way in serves what
            crosses it either way (:meth:`ObjFlow._capacity_bounded_capability`)."""
            supplier = self.comp[edge.producer]
            capacity = supplier._capacity_of(edge.flow_out)
            if capacity is None or not any(
                declared.name == edge.flow_out
                for declared in supplier.flows_continuous_in
            ):
                return None
            return supplier._held_empty(capacity, edge.flow_out)

        def delivered(name: str, port: str, edge: _ContinuousEdge) -> dict[str, Any]:
            """What `edge` actually hands its consumer: its allocation,
            less its share of the surplus a releasing input returns."""
            channel = (
                "delivered" if (edge.consumer, edge.flow_in) in releasing else "alloc"
            )
            return _var(name, _channel_attr(port, channel, edge.name))

        for name, obj in self.comp.items():
            component = by_name[name]
            produced_by_rules = obj._produced_outputs()
            mixed = obj._mixed_outputs()
            for flow in obj.flows_continuous_out:
                port = f"{flow.name}_out"
                served = out_edges.get((name, flow.name), [])
                capability = _var(name, f"{flow.name}_capability_out")

                # What the output could deliver, published WHOLE to every
                # consumer, as muscadet's `incoming_capability` sums it
                # (R-20): a capability is asked before any demand exists
                # to apportion it against, so it is not apportioned. Two
                # consumers of one source each read all of it and size
                # their claim as if the other were absent; the split
                # settles what each receives, and what a rule makes is
                # bounded by that receipt (`ObjFlow._rule_draw_scale`).
                #
                # Netted per consumer (the capability less the others'
                # allocations), it made each consumer's claim depend on
                # the others' shares: an unbounded fill claim weighed at
                # what was left after a rival rather than at the supply,
                # and a rule bounded by it ratcheted down with its own
                # share.
                for edge in served:
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "capability", edge.name),
                            "kind": "explicit",
                            "expr": capability,
                        }
                    )

                # Demand back-propagation: each consumer's demand,
                # published on the edge that serves it.
                for edge in served:
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "demand", edge.name),
                            "kind": "explicit",
                            "expr": _var(edge.consumer, f"{edge.flow_in}_demand_in"),
                        }
                    )
                component["equations"].append(
                    {
                        "target": f"{flow.name}_demand_out",
                        "kind": "explicit",
                        "expr": _sum(
                            [
                                _var(name, _channel_attr(port, "demand", edge.name))
                                for edge in served
                            ]
                        ),
                    }
                )
                component["equations"].append(
                    {
                        "target": f"{flow.name}_fed_out",
                        "kind": "explicit",
                        "expr": _sum([delivered(name, port, edge) for edge in served]),
                    }
                )
                # Into a releasing input, each edge delivers its
                # allocation scaled by what the consumer accepts, on a
                # channel of its own so the consumer can sum it.
                released = [
                    edge for edge in served if (edge.consumer, edge.flow_in) in releasing
                ]
                if released:
                    declared_port = next(
                        declared for declared in component["ports"]
                        if declared["name"] == port
                    )
                    declared_port["channels"].append({"name": "delivered"})
                for edge in released:
                    share: dict[str, Any] = _var(
                        edge.consumer, f"{edge.flow_in}_accepted_share"
                    )
                    empty = held_empty(edge)
                    if empty is not None:
                        share = {
                            "op": "if",
                            "cond": empty,
                            "then": _var(
                                edge.consumer, f"{edge.flow_in}_accepted_held_share"
                            ),
                            "otherwise": share,
                        }
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "delivered", edge.name),
                            "kind": "explicit",
                            "expr": {
                                "op": "mul",
                                "args": [
                                    _var(name, _channel_attr(port, "alloc", edge.name)),
                                    share,
                                ],
                            },
                        }
                    )

                # An output nobody reads distributes nothing: no
                # operator, and the two totals above stand at zero.
                if not served:
                    continue
                # What is there to distribute: the capability of a flow
                # nothing transforms, what the rule actually made when
                # one does, and a mixed volume's pooled draw
                # (:meth:`ObjFlow._pooled_serve`).
                available = (
                    _var(name, f"{flow.name}_produced_out")
                    if flow.name in produced_by_rules
                    else _var(name, f"{flow.name}_pooled_out")
                    if flow.name in mixed
                    else capability
                )
                # What the split weighs each consumer at: its demand AS
                # DECLARED (muscadet 5.24.0), so consumers in parallel on
                # a saturating supply are all served one fraction of what
                # they need: 12 between claims of 6 and 18 is 3 and 9.
                # Truncated at the quantity available, as muscadet did
                # from 5.11.0 to 5.23.0, the same claims weighed 6 and 12
                # and the small consumer was favoured. An unbounded claim
                # is the limit of a growing one: it takes everything and
                # the finite claims beside it are served nothing.
                declared_port = next(
                    declared for declared in component["ports"]
                    if declared["name"] == port
                )
                declared_port["channels"].append({"name": "claim"})
                # A volume drawn as a MIXTURE hands each machine its composed
                # share scaled to what it delivers (R52), muscadet's
                # `allocate_output`. The split weighing claims as declared,
                # the scaling gives what the raw claims would: 100 and 1
                # behind a ceiling of 50.5 are served 50 and 0.5 either way.
                # The total demand is a demand-band quantity and the
                # delivery reads it, so no band reads a later one.
                drawn = flow.name in mixed and mixed[flow.name].serves_mixture
                total = _var(name, f"{flow.name}_demand_out")
                for edge in served:
                    asked = _var(name, _channel_attr(port, "demand", edge.name))
                    claim = (
                        {
                            "op": "if",
                            "cond": {
                                "op": "cmp",
                                "cmp": "gt",
                                "lhs": total,
                                "rhs": _float(0.0),
                            },
                            "then": {
                                "op": "div",
                                "lhs": {"op": "mul", "args": [asked, available]},
                                "rhs": total,
                            },
                            "otherwise": _float(0.0),
                        }
                        if drawn
                        else asked
                    )
                    component["equations"].append(
                        {
                            "target": _channel_attr(port, "claim", edge.name),
                            "kind": "explicit",
                            "expr": claim,
                        }
                    )
                allocation: dict[str, Any] = {
                    "name": f"{flow.name}_alloc",
                    "port": port,
                    "available": available,
                    "demand": "claim",
                    "allocated": "alloc",
                    "policy": flow.allocation,
                }
                if flow.allocation == "shares":
                    allocation["shares"] = self._allocation_params(
                        served, flow.allocation_shares or {}, "allocation_shares"
                    )
                elif flow.allocation == "priority":
                    allocation["priorities"] = self._allocation_params(
                        served,
                        flow.allocation_priorities or {},
                        "allocation_priorities",
                    )
                component.setdefault("allocations", []).append(allocation)

            for flow in obj.flows_continuous_in:
                port_ref = {"component": name, "port": f"{flow.name}_in"}
                connected = bool(in_edges.get((name, flow.name)))
                # A DECLARED supply replaces the aggregation: an input
                # nobody feeds reads `var_in_default`, and no sum over a
                # port answers that. Declaring nothing is the other case
                # and is not the same one: the input then reads zero,
                # which is exactly what the sum over its empty port
                # answers, so the two spellings carry the same number and
                # only one of them says WHERE the number comes from.
                #
                # Writing the constant there would erase the port from
                # the document, and with it the only thing that tells a
                # forgotten wire from a deliberate boundary value. That
                # is what `unfed_triggers` reads to name the wire, and it
                # is why the boolean inputs above keep their aggregation
                # under the same condition.
                declared = flow.var_in_default != 0.0
                releases = (name, flow.name) in releasing
                if releases:
                    # The share of what was offered the rule accepts: its
                    # need over the offers when they exceed it, all of it
                    # otherwise. Never a division by zero: a ratio is only
                    # taken where the offers exceed a need >= 0.
                    #
                    # A transiting volume that has RUN OUT passes on what
                    # arrives first, and the other suppliers share what is
                    # left. Scaled with them, it would hand on less than
                    # it receives, refill, leave its empty bound, offer
                    # its whole request, drain, and chatter on the bound
                    # for as long as the rule runs: the reference has the
                    # same dynamic, hidden by its fixed step. Passing on
                    # what arrives is that sliding motion's own average,
                    # the content held at zero.
                    offered = obj._offered(flow.name)
                    # A rule needs what it consumes at its draw, a sink
                    # the demand it published.
                    need = _var(
                        name,
                        f"{flow.name}_demand_in"
                        if obj._is_sink(flow.name)
                        else f"{flow.name}_consumed_in",
                    )
                    held = [
                        {
                            "op": "if",
                            "cond": empty,
                            "then": _var(
                                edge.producer,
                                _channel_attr(f"{edge.flow_out}_out", "alloc", edge.name),
                            ),
                            "otherwise": _float(0.0),
                        }
                        for edge in in_edges[(name, flow.name)]
                        for empty in [held_empty(edge)]
                        if empty is not None
                    ]

                    def ratio(wanted, available):
                        return {
                            "op": "if",
                            "cond": {"op": "cmp", "cmp": "gt", "lhs": available, "rhs": wanted},
                            "then": {"op": "div", "lhs": wanted, "rhs": available},
                            "otherwise": _float(1.0),
                        }

                    if held:
                        passed = _var(name, f"{flow.name}_accepted_held")
                        component["attributes"].append(
                            _float_attribute(f"{flow.name}_accepted_held", 0.0)
                        )
                        component["attributes"].append(
                            _float_attribute(f"{flow.name}_accepted_held_share", 1.0)
                        )
                        component["equations"] += [
                            {
                                "target": f"{flow.name}_accepted_held",
                                "kind": "explicit",
                                "expr": _sum(held),
                            },
                            {
                                "target": f"{flow.name}_accepted_held_share",
                                "kind": "explicit",
                                "expr": ratio(need, passed),
                            },
                        ]
                        # What the other suppliers are left to cover, and
                        # what they offered.
                        need = _clamped_at_zero({"op": "sub", "lhs": need, "rhs": passed})
                        offered = {"op": "sub", "lhs": offered, "rhs": passed}
                    component["attributes"].append(
                        _float_attribute(f"{flow.name}_accepted_share", 1.0)
                    )
                    component["equations"].append(
                        {
                            "target": f"{flow.name}_accepted_share",
                            "kind": "explicit",
                            "expr": ratio(need, offered),
                        }
                    )
                for channel, target in (
                    ("capability", f"{flow.name}_capability_in"),
                    ("delivered" if releases else "alloc", f"{flow.name}_fed_in"),
                ):
                    component["equations"].append(
                        {
                            "target": target,
                            "kind": "explicit",
                            "expr": (
                                _float(flow.var_in_default)
                                if declared and not connected
                                else {
                                    "op": "port_agg",
                                    "port": port_ref,
                                    "agg": "sum",
                                    "channel": channel,
                                }
                            ),
                        }
                    )

        return self._evaluation_order(components, out_edges, in_edges, releasing)

    def _flow_order(
        self, edges: dict[tuple[str, str], list[_ContinuousEdge]]
    ) -> list[str]:
        """The components in flow-graph topological order (producers
        before their consumers), ties broken by declaration order.

        A conservative flow network is cyclic by nature, so a cycle is
        expected and legal: the nodes it holds back are released in
        declaration order rather than refused.

        Kahn's algorithm over a heap of **declaration indices**: take the
        lowest index whose producers are all placed, and when a cycle
        leaves none of them, the lowest index **on a cycle** whose
        producers off that cycle are all placed. Lowest index *is*
        declared first, so the tie-break is never expressed a second
        time. The break releases a member of the ring and never a
        component merely downstream of it: released early, that one
        would be visited before one of its own suppliers, and every
        input it sums would read that supplier a sweep late."""
        names = list(self.comp)
        position = {name: index for index, name in enumerate(names)}
        successors: list[list[int]] = [[] for _ in names]
        indegree = [0] * len(names)
        for group in edges.values():
            for edge in group:
                if edge.producer == edge.consumer:
                    continue
                successors[position[edge.producer]].append(position[edge.consumer])
                indegree[position[edge.consumer]] += 1

        ring_of, _ = _strongly_connected(successors)
        # What each node still waits for from OUTSIDE its own ring: at a
        # stall, the source rings of what is left have none, and a
        # singleton at zero would have been ready.
        from_outside = [0] * len(names)
        for producer, targets in enumerate(successors):
            for consumer in targets:
                if ring_of[producer] != ring_of[consumer]:
                    from_outside[consumer] += 1

        ready = [index for index in range(len(names)) if indegree[index] == 0]
        heapq.heapify(ready)
        placed = [False] * len(names)
        order: list[str] = []
        while len(order) < len(names):
            while ready and placed[ready[0]]:
                heapq.heappop(ready)
            if ready:
                chosen = heapq.heappop(ready)
            else:
                chosen = next(
                    index
                    for index in range(len(names))
                    if not placed[index] and from_outside[index] == 0
                )
            placed[chosen] = True
            order.append(names[chosen])
            # A node released by a cycle break still frees its
            # successors, exactly as one taken in order does.
            for successor in successors[chosen]:
                indegree[successor] -= 1
                if ring_of[chosen] != ring_of[successor]:
                    from_outside[successor] -= 1
                if indegree[successor] == 0 and not placed[successor]:
                    heapq.heappush(ready, successor)
        return order

    def _measurement_order(self, step) -> None:
        """Sweep the measurement readers and the instruments in the order
        their readings flow: a reader after the publication it reads, an
        instrument after the channel it republishes. With no instrument in
        the model this is the readers in declaration order, as before.

        A reading that feeds back into its own source through instruments
        has no order to be swept in, and is refused naming what it joins."""
        publisher_of = {
            (c["to"]["component"], c["to"]["port"]): (
                c["from"]["component"],
                c["from"]["port"],
            )
            for c in self._connections
        }
        instruments = {
            (name, publication.name): publication
            for name, obj in self.comp.items()
            for publication in obj.measurements_out
        }
        instrument_of_port = {
            (name, f"{alias}_out"): key
            for key, publication in instruments.items()
            for name in (key[0],)
            for alias, _ in publication.published()
        }
        readers = [
            (name, link)
            for name, obj in self.comp.items()
            for link in obj.measurements_in
        ]
        done_readers: set[tuple[str, str]] = set()
        done_instruments = {
            key for key, publication in instruments.items() if publication.source is None
        }

        def reader_ready(name: str, link: _MeasurementIn) -> bool:
            for variable in link.channels():
                publisher = publisher_of.get((name, f"{variable}_in"))
                instrument = instrument_of_port.get(publisher) if publisher else None
                if instrument is not None and instrument not in done_instruments:
                    return False
            return True

        def instrument_ready(key: tuple[str, str], publication: _MeasurementOut) -> bool:
            name = key[0]
            if any(link.name == publication.source for link in self.comp[name].measurements_in):
                return (name, publication.source) in done_readers
            return True

        progress = True
        while progress:
            progress = False
            for name, link in readers:
                if (name, link.name) in done_readers or not reader_ready(name, link):
                    continue
                for variable in link.channels():
                    step(name, variable)
                done_readers.add((name, link.name))
                progress = True
            for key, publication in instruments.items():
                if key in done_instruments or not instrument_ready(key, publication):
                    continue
                for _, attribute in publication.published():
                    step(key[0], attribute)
                done_instruments.add(key)
                progress = True
        stuck = [f"{n}.{link.name}" for n, link in readers if (n, link.name) not in done_readers]
        stuck += [f"{k[0]}.{k[1]}" for k in instruments if k not in done_instruments]
        if stuck:
            raise ValueError(
                f"System `{self.name}`: the measurement readings {stuck} feed "
                "back into their own sources through republishing instruments, "
                "so no order can sweep them"
            )

    def _evaluation_order(
        self,
        components: list[dict[str, Any]],
        out_edges: dict[tuple[str, str], list[_ContinuousEdge]],
        in_edges: dict[tuple[str, str], list[_ContinuousEdge]],
        releasing: set[tuple[str, str]],
    ) -> list[dict[str, str]]:
        """The three-band sweep order: capability along the flow, demand
        back against it, production along it again.

        The engine sweeps the explicit equations once per evaluation
        point, so this order *is* part of the answer: swept the other
        way round, a consumer would size its demand on the capability
        the previous evaluation point left behind."""
        order: list[dict[str, str]] = []
        listed: set[tuple[str, str]] = set()

        def step(component: str, attribute: str) -> None:
            if (component, attribute) not in listed:
                listed.add((component, attribute))
                order.append({"component": component, "attribute": attribute})

        topological = self._flow_order(out_edges)

        def after_last_supplier(publish, skip=frozenset()):
            """An input sums what ALL its suppliers publish, so it is
            swept once the last of them is, not at the first: a later
            supplier would otherwise be read at the previous evaluation's
            value, which the discrete fixpoint hides and a varying source
            exposes. In a ring the last supplier comes after the consumer,
            and the input is still swept after it: summed at the
            consumer's own visit, the input would read the ring one
            evaluation late in the balance it integrates, where only the
            pooled draw reads it late now (`_transit`)."""
            awaiting = {
                key: len(edges) for key, edges in in_edges.items() if key not in skip
            }

            def supplied(edge: _ContinuousEdge) -> None:
                key = (edge.consumer, edge.flow_in)
                if key in awaiting:
                    awaiting[key] -= 1
                    if awaiting[key] == 0:
                        publish(*key)

            return supplied

        capability_supplied = after_last_supplier(
            lambda consumer, flow_in: step(consumer, f"{flow_in}_capability_in")
        )

        def unconnected_inputs(suffix: str) -> None:
            """An input no producer feeds appears in no producer's band,
            and what it reads still has to be swept: its declared
            constant, or the aggregation over its empty port."""
            for name, obj in self.comp.items():
                for flow in obj.flows_continuous_in:
                    if not in_edges.get((name, flow.name)):
                        step(name, f"{flow.name}_{suffix}")

        # 0. Held levels, before anything reads them. A capacity's fills
        # and its two totals read the integrated contents only, so they
        # are swept first, and an observer reads what a capacity has just
        # published.
        for name, obj in self.comp.items():
            for capacity in obj.capacities:
                for entry in capacity.flows:
                    step(name, f"{capacity.name}_fill_{entry.name}")
                step(name, f"{capacity.name}_fill")
                step(name, _content_attribute(capacity.name))
                # After the total they divide by, and before the
                # observers below read them.
                if capacity.publishes_ratios:
                    for entry in capacity.flows:
                        step(name, f"{capacity.name}_ratio_{entry.name}")
        self._measurement_order(step)
        # What multiplies a production, and what a gradient asks for:
        # neither reads the flow network, both are read by it. The
        # transfers come after the measurements, whose readings are the
        # potentials a conductive law is written over.
        for name, obj in self.comp.items():
            for flow in obj.flows_continuous_in:
                if flow.profile is not None:
                    step(name, f"{flow.name}_in_profile")
            for flow in obj.flows_continuous_out:
                if obj._derated(flow.name):
                    step(name, f"{flow.name}_effective_rate")
                if obj._taps_on(flow.name):
                    step(name, f"{flow.name}_tap_rate")
                if flow.profile is not None:
                    step(name, f"{flow.name}_out_profile")
            for pair in obj.transfers:
                step(name, f"{pair.name}_requested")

        # 1. Capability, along the flow. An input no producer feeds
        # supplies a fixed quantity, and a rule set may read it: swept
        # first, so a scale never sizes itself on a value the same pass
        # has not reached.
        unconnected_inputs("capability_in")
        for name, obj in self.comp.items():
            for flow in obj.flows_continuous_in:
                if flow.publish_rate == RATE_CAPABILITY:
                    step(name, _rate_alias(flow.name))
        for producer in topological:
            # A rule set turns the capability of what it consumes into
            # the capability of what it produces, so its scale is swept
            # between the two.
            for rule_set in self.comp[producer].rule_sets:
                step(producer, f"{rule_set.name}_capability_scale")
            # A two-stream pair caps what it moves by what its origin
            # carries, so both bases are swept before the pair, and the
            # pair before the capabilities it adjusts.
            for flow in self.comp[producer].flows_continuous_out:
                if self.comp[producer]._has_transfer_delta(flow.name):
                    step(producer, f"{flow.name}_transfer_base")
            for pair in self.comp[producer].transfers:
                if not pair.is_conduit:
                    step(producer, f"{pair.name}_moved")
            # What each tapped stream carried before its taps drew, for
            # every stream of the component at once and ahead of the
            # capabilities: an output that RECEIVES a tap reads the base
            # of the output it is fed from, and the two are ordered by
            # the routing, not by which was declared first.
            for flow_name in self.comp[producer]._tap_ordered_outputs():
                if self.comp[producer]._taps_on(flow_name):
                    step(producer, f"{flow_name}_pre_tap_capability_out")
            for flow in self.comp[producer].flows_continuous_out:
                port = f"{flow.name}_out"
                step(producer, f"{flow.name}_capability_out")
                # A channel carrying what COULD cross belongs to this
                # band and not to the delivery band: read after the
                # delivery, it would close a loop through the observer's
                # own decision, which is the reason the two quantities
                # are distinguishable at all.
                if flow.publish_rate == RATE_CAPABILITY:
                    step(producer, _rate_alias(flow.name))
                served = out_edges.get((producer, flow.name), [])
                for edge in served:
                    step(producer, _channel_attr(port, "capability", edge.name))
                for edge in served:
                    capability_supplied(edge)

        # 2. Demand, back against the flow.
        for producer in reversed(topological):
            for flow in self.comp[producer].flows_continuous_out:
                port = f"{flow.name}_out"
                served = out_edges.get((producer, flow.name), [])
                for edge in served:
                    step(edge.consumer, f"{edge.flow_in}_demand_in")
                for edge in served:
                    step(producer, _channel_attr(port, "demand", edge.name))
                step(producer, f"{flow.name}_demand_out")
            # A volume downstream of the rules claims on what its
            # consumers asked, and the scale reads the claim.
            for claimed in self.comp[producer]._claimed_outputs().values():
                step(producer, claimed)
            # The rule's own scale: after its outputs have been asked,
            # and before its inputs ask, which a producer's own visit
            # (later in this reversed walk) is what steps.
            for rule_set in self.comp[producer].rule_sets:
                step(producer, f"{rule_set.name}_scale")
        unconnected_inputs("demand_in")

        # 3. Production, along the flow again. What a releasing input
        # accepts reads the offers of ALL its suppliers, and a later
        # supplier may be reached after an earlier one's own delivery
        # (a battery fed by the bus beside it), so that share, the
        # deliveries it scales and the totals summing them are swept at
        # the end of the band. Nothing else in the band reads them.
        deferred: list[tuple[str, dict[str, Any], list[_ContinuousEdge]]] = []

        def publish_fed(consumer: str, flow_in: str) -> None:
            """What one consumer was fed, with its published rate when
            declared."""
            step(consumer, f"{flow_in}_fed_in")
            if any(
                declared.name == flow_in and declared.publish_rate == RATE_DELIVERED
                for declared in self.comp[consumer].flows_continuous_in
            ):
                step(consumer, _rate_alias(flow_in))

        # A releasing input is swept at the end of the band, below.
        fed = after_last_supplier(publish_fed, skip=releasing)

        def delivery(producer: str, flow, served: list[_ContinuousEdge]) -> None:
            """What an output delivered, then what each consumer it
            serves was fed."""
            step(producer, f"{flow.name}_fed_out")
            if flow.publish_rate == RATE_DELIVERED:
                step(producer, _rate_alias(flow.name))
            for edge in served:
                fed(edge)

        for producer in topological:
            # What the rules draw reads what arrived, settled by the
            # producers visited before this one, in declaration order
            # since each set is offered what the earlier ones left;
            # everything this component makes reads the draw.
            for rule_set in self.comp[producer].rule_sets:
                if self.comp[producer]._runs_at_a_draw(rule_set):
                    step(producer, f"{rule_set.name}_draw_scale")
            # What the rules consume of an input that releases its
            # surplus, at the draw just swept: the release at the end of
            # the band measures the offers against it.
            for flow in self.comp[producer].flows_continuous_in:
                if self.comp[producer]._releases_surplus(flow.name):
                    step(producer, f"{flow.name}_consumed_in")
            for flow_name in self.comp[producer]._claimed_outputs():
                step(producer, f"{flow_name}_made_out")
            # A mixed volume's pooled excess reads what arrived, settled
            # by the suppliers visited before this component, and the
            # demands of the band before; what each output hands out
            # reads the excess, and the allocations below read that.
            mixed = self.comp[producer]._mixed_outputs()
            # An input nobody feeds transits its declared constant, and
            # is otherwise swept only at the end of the band: taken now,
            # the excess never reads it a sweep late.
            for flow in self.comp[producer].flows_continuous_in:
                if flow.name in mixed and not in_edges.get((producer, flow.name)):
                    step(producer, f"{flow.name}_fed_in")
            for capacity in self.comp[producer].capacities:
                if any(held is capacity for held in mixed.values()):
                    step(producer, self.comp[producer]._beyond_attribute(capacity))
            for flow_name in mixed:
                step(producer, f"{flow_name}_pooled_out")
            produced_by_rules = self.comp[producer]._produced_outputs()
            for flow_name in self.comp[producer]._tap_ordered_outputs():
                if (
                    self.comp[producer]._taps_on(flow_name)
                    and flow_name in produced_by_rules
                ):
                    step(producer, f"{flow_name}_pre_tap_produced_out")
            for flow in self.comp[producer].flows_continuous_out:
                served = out_edges.get((producer, flow.name), [])
                if flow.name in produced_by_rules:
                    step(producer, f"{flow.name}_produced_out")
                if served:
                    # The claims read what is available, settled just
                    # above, and the split reads the claims.
                    for edge in served:
                        step(producer, _channel_attr(f"{flow.name}_out", "claim", edge.name))
                    step(producer, f"{flow.name}_alloc")
                if any((edge.consumer, edge.flow_in) in releasing for edge in served):
                    # Only what the release scales waits: every OTHER
                    # consumer of this output reads its plain allocation,
                    # settled now, and is swept here, before its own
                    # visit reads what arrived.
                    for edge in served:
                        if (edge.consumer, edge.flow_in) not in releasing:
                            fed(edge)
                    deferred.append((producer, flow, served))
                    continue
                delivery(producer, flow, served)
            # What a conduit moved is what its flow delivered, so it is
            # read once that delivery is settled.
            waiting = {flow.name for owner, flow, _ in deferred if owner == producer}
            for pair in self.comp[producer].transfers:
                if pair.is_conduit and pair.source not in waiting:
                    step(producer, f"{pair.name}_moved")
        # Every offer is settled now: what each releasing input accepts,
        # then what each of its suppliers delivered, and the totals.
        position = {name: index for index, name in enumerate(self.comp)}
        for name, flow_name in sorted(
            releasing, key=lambda key: (position[key[0]], key[1])
        ):
            for attribute in (
                f"{flow_name}_accepted_held",
                f"{flow_name}_accepted_held_share",
            ):
                if any(
                    equation["target"] == attribute
                    for component in components
                    if component["name"] == name
                    for equation in component["equations"]
                ):
                    step(name, attribute)
            step(name, f"{flow_name}_accepted_share")
        # Every delivery and every supplier's total first, and only then
        # what each releasing input was fed: an input sums the deliveries
        # of ALL its suppliers, so reading it after the first one would
        # take the others from the previous evaluation.
        for producer, flow, served in deferred:
            port = f"{flow.name}_out"
            for edge in served:
                if (edge.consumer, edge.flow_in) in releasing:
                    step(producer, _channel_attr(port, "delivered", edge.name))
            step(producer, f"{flow.name}_fed_out")
            if flow.publish_rate == RATE_DELIVERED:
                step(producer, _rate_alias(flow.name))
            # What a conduit moved is what its flow delivered.
            for pair in self.comp[producer].transfers:
                if pair.is_conduit and pair.source == flow.name:
                    step(producer, f"{pair.name}_moved")
        for producer, flow, served in deferred:
            for edge in served:
                if (edge.consumer, edge.flow_in) in releasing:
                    publish_fed(edge.consumer, edge.flow_in)
        unconnected_inputs("fed_in")

        # What the atmosphere adds to an OPEN volume reads the deliveries,
        # every one of them now swept (R53), and is read by nothing but the
        # contents' derivatives.
        for name, obj in self.comp.items():
            for capacity in obj.capacities:
                if not capacity.is_open:
                    continue
                for entry in capacity.flows:
                    step(name, f"{capacity.name}_vent_{entry.name}")
                step(name, f"{capacity.name}_makeup")

        # The order must cover the declared steps exactly, and the
        # engine refuses an omission rather than completing it: close it
        # over what the components actually declare, so a step this
        # layer gains later cannot fall out of the sweep silently.
        #
        # What the closure adds is swept in DEPENDENCY order, not in
        # declaration order: a controller declared before the sensor it
        # reads would otherwise compare the reading the previous sweep
        # left behind, and a crossing it watches would be located at the
        # first scan point past the true date (4/3 h fired at 1.30625 h
        # on the H2 plant) or loop on the boundary.
        for name, target in self._dependency_ordered(
            [
                (component["name"], equation)
                for component in components
                for equation in component["equations"]
                if equation["kind"] == "explicit"
                and (component["name"], equation["target"]) not in listed
            ],
            components,
        ):
            step(name, target)
        for component in components:
            for allocation in component.get("allocations", []):
                step(component["name"], allocation["name"])
        return order

    def _dependency_ordered(
        self,
        pending: list[tuple[str, dict[str, Any]]],
        components: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """`pending` explicit equations as ``(component, target)``, each
        after every other pending one it reads, ties and cycles kept in
        the order given.

        A read is an attribute the expression names, or the attribute an
        out-port publishes into an in-port the expression aggregates
        (``port_agg`` with no channel), resolved through the connections.
        A per-connection channel is a quantity of the flow network, swept
        by its own band, and is not followed here."""
        published = {
            (component["name"], port["name"]): port.get("attr")
            for component in components
            for port in component.get("ports", [])
            if port.get("dir") == "out"
        }
        feeding: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for connection in self._connections:
            source, destination = connection["from"], connection["to"]
            attribute = published.get((source["component"], source["port"]))
            if attribute is not None:
                feeding.setdefault(
                    (destination["component"], destination["port"]), []
                ).append((source["component"], attribute))

        def reads(expr: Any, found: set[tuple[str, str]]) -> None:
            if isinstance(expr, dict):
                if expr.get("op") == "attr":
                    found.add((expr["attr"]["component"], expr["attr"]["attribute"]))
                elif expr.get("op") == "port_agg" and not expr.get("channel"):
                    port = expr["port"]
                    found.update(feeding.get((port["component"], port["port"]), []))
                for value in expr.values():
                    reads(value, found)
            elif isinstance(expr, list):
                for value in expr:
                    reads(value, found)

        keys = [(name, equation["target"]) for name, equation in pending]
        position = {key: index for index, key in enumerate(keys)}
        readers: list[list[int]] = [[] for _ in keys]
        missing = [0] * len(keys)
        for index, (_, equation) in enumerate(pending):
            found: set[tuple[str, str]] = set()
            reads(equation["expr"], found)
            for read in found:
                source = position.get(read)
                if source is not None and source != index:
                    readers[source].append(index)
                    missing[index] += 1
        # A cycle among these equations is released as a block, in
        # declaration order, and only once everything it reads from
        # outside the block is placed: its strongly connected components
        # (Tarjan) are ordered by Kahn's algorithm, lowest declaration
        # first. A real cycle is refused by the engine anyway; what this
        # keeps is an order that stays complete and never places an
        # equation before one it reads that is not on its cycle.
        # `readers` points from a read equation to its readers; Tarjan
        # walks the reversed edges (reader -> what it reads), which keeps
        # the numbering the order has always had.
        reads_of: list[list[int]] = [[] for _ in keys]
        for source, targets in enumerate(readers):
            for reader in targets:
                reads_of[reader].append(source)
        component_of, members = _strongly_connected(reads_of)
        needs = [0] * len(members)
        followers: list[set[int]] = [set() for _ in members]
        for source, targets in enumerate(readers):
            for reader in targets:
                a, b = component_of[source], component_of[reader]
                if a != b and b not in followers[a]:
                    followers[a].add(b)
                    needs[b] += 1
        ready = [(block[0], number) for number, block in enumerate(members) if needs[number] == 0]
        heapq.heapify(ready)
        ordered: list[tuple[str, str]] = []
        while ready:
            _, number = heapq.heappop(ready)
            ordered.extend(keys[member] for member in members[number])
            for follower in followers[number]:
                needs[follower] -= 1
                if needs[follower] == 0:
                    heapq.heappush(ready, (members[follower][0], follower))
        return ordered
