"""Rule diagnostics of a system: refused loops and the rings torn to break them."""

from __future__ import annotations

from typing import Any

from ._graph import _CycleSearchExhausted, _find_cycle
from ._records import _ContinuousEdge


class _SystemRulesMixin:
    """Methods of :class:`System` rule diagnostics of a system: refused loops and the rings torn to break them."""

    # --- rule diagnostics ---------------------------------------------

    def _validate_rules(self, edges: list[_ContinuousEdge]) -> None:
        """The three refusals the rule vocabulary makes sayable, in
        declaration order: a contested output nobody apportions, a loop
        of rate comparisons, and a self-feeding cycle that creates
        matter.

        A **commanded capacity** enters here for the second alone: a
        discharge command compares rates in the same vocabulary a rule
        guard does, and it drives the flow it releases, so it closes the
        same loop with no fixpoint. The third stays the rule
        vocabulary's, and not only because it would answer nothing
        without one: its search walks the whole flow graph under a budget
        it REFUSES the model for spending, and a model with no rule has
        no cycle it could report anyway, every connection carrying its
        quantity unchanged."""
        commanded = any(
            capacity.commanded
            for obj in self.comp.values()
            for capacity in obj.capacities
        ) or any(
            group.commanded for obj in self.comp.values() for group in obj.mixtures
        )
        rules = any(obj.rule_sets for obj in self.comp.values())
        if not rules and not commanded:
            return
        for obj in self.comp.values():
            obj._apportionment_shares()
        self._refuse_rate_comparison_loops(edges)
        if rules:
            self._refuse_unbounded_rule_cycles(edges)

    def _rule_graph(
        self, edges: list[_ContinuousEdge]
    ) -> dict[tuple[str, str, str], list[tuple[tuple[str, str, str], float, Any]]]:
        """The graph matter travels on: one node per flow endpoint
        ``(component, flow, side)``, one edge per connection (carrying
        the quantity unchanged) and one per ``(consumed, produced)`` pair
        of a rule (carrying the ratio of their coefficients).

        Every rule of every set contributes, not only the active one: a
        guard is a run-time fact, and a diagnostic that only held for the
        rule that happens to be selected would be no diagnostic."""
        successors: dict[
            tuple[str, str, str], list[tuple[tuple[str, str, str], float, Any]]
        ] = {}
        for edge in edges:
            successors.setdefault((edge.producer, edge.flow_out, "out"), []).append(
                ((edge.consumer, edge.flow_in, "in"), 1.0, None)
            )
        for name, obj in self.comp.items():
            for rule_set in obj.rule_sets:
                for index, rule in enumerate(rule_set.rules):
                    for consumed, taken in rule.cons.items():
                        if taken <= 0:
                            continue
                        for produced, made in rule.prod.items():
                            if made <= 0:
                                continue
                            successors.setdefault(
                                (name, consumed, "in"), []
                            ).append(
                                (
                                    (name, produced, "out"),
                                    made / taken,
                                    (name, rule_set.name, index),
                                )
                            )
        return successors

    def _refuse_rate_comparison_loops(self, edges: list[_ContinuousEdge]) -> None:
        """Refuse a loop of rate comparisons (R15).

        A guard comparing a **rate** is instantaneous: it reads a
        quantity the very sweep it takes part in produces. When two rule
        sets each guard on a rate the other drives, the selection has no
        fixpoint: each mode is the answer to the other, and the pair
        chatters or settles on whichever was evaluated first. A guard on
        an integrated **level** closes no such loop, because the
        integration carries the value across the sweep, which is why the
        two kinds of comparison are told apart rather than counted
        together.

        **A commanded capacity is a node of the same graph**, and the
        same reasoning word for word: its discharge command compares
        rates in the rule guard's vocabulary and it drives what it
        releases, so a command reading a rate its own discharge moves
        has no more of a fixpoint than a rule selected on one. muscadet
        documents that angle as a blind spot of its own loop detectors
        (#172, "a comparison on a rate is not seen"); here it is the one
        place a capacity had to be written into.

        The two kinds of node close loops with each other as readily as
        with their own kind, which is why they share one graph rather
        than being looked for twice."""
        carried: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
        for edge in edges:
            carried.setdefault((edge.producer, edge.flow_out, "out"), []).append(
                (edge.consumer, edge.flow_in, "in")
            )

        # One node per switching decision, keyed by what it is so the
        # diagnostic can name it: `(component, what, name)`.
        influence: dict[tuple[str, str, str], set] = {}
        reads: dict[tuple[str, str, str], list] = {}

        def declare(key, driven_flows, operands, obj, name):
            # What this decision drives *before another one does*: its
            # own outputs and the inputs they feed. A rate that travels
            # further does so through another decision, which is then
            # the next node of the loop rather than a hop inside this one.
            driven = {(name, flow, "out") for flow in driven_flows}
            for seed in list(driven):
                driven.update(carried.get(seed, ()))
            influence[key] = driven
            compared = []
            for operand in operands:
                if operand.op is None:
                    continue
                _, rate = obj._operand_read(f"ObjFlow `{name}`", operand)
                if rate is not None:
                    compared.append((name, rate[0], rate[1]))
            reads[key] = compared

        for name, obj in self.comp.items():
            for rule_set in obj.rule_sets:
                declare(
                    (name, "rule set", rule_set.name),
                    rule_set.produced(),
                    [operand for rule in rule_set.rules for operand in rule.cond],
                    obj,
                    name,
                )
            delivered = {flow.name for flow in obj.flows_continuous_out}
            for capacity in obj.capacities:
                if not capacity.commanded:
                    continue
                declare(
                    (name, "capacity", capacity.name),
                    [flow for flow in capacity.flow_names if flow in delivered],
                    [
                        operand
                        for group in capacity.serve_cond
                        for operand in group
                    ],
                    obj,
                    name,
                )
            # A commanded MACHINE drives what the volume it draws delivers
            # (R52): the producer's outputs feeding the group, and the
            # group's own inputs they arrive on. Written out rather than
            # through `declare`, whose driven rates are the component's own
            # outputs; a machine has none.
            for group in obj.mixtures:
                if not group.commanded:
                    continue
                key = (name, "mixture group", group.name)
                feeding = [
                    edge
                    for edge in edges
                    if edge.consumer == name and edge.flow_in in group.flows
                ]
                influence[key] = {
                    (edge.producer, edge.flow_out, "out") for edge in feeding
                } | {(name, edge.flow_in, "in") for edge in feeding}
                compared = []
                for operand in (op for row in group.run_cond for op in row):
                    if operand.op is None:
                        continue
                    _, rate = obj._operand_read(f"ObjFlow `{name}`", operand)
                    if rate is not None:
                        compared.append((name, rate[0], rate[1]))
                reads[key] = compared

        arcs: dict[tuple[str, str, str], list[tuple[tuple[str, str, str], tuple]]] = {}
        for driver, driven_nodes in influence.items():
            for reader, endpoints in reads.items():
                shared = [node for node in endpoints if node in driven_nodes]
                if shared:
                    arcs.setdefault(driver, []).append((reader, shared[0]))

        cycle = _find_cycle(list(influence), arcs)
        if cycle is None:
            return
        rendered = " -> ".join(
            f"{node[1]} `{node[2]}` of `{node[0]}` (driving the rate "
            f"`{endpoint[1]}`)"
            for node, endpoint in cycle
        )
        raise ValueError(
            f"System `{self.name}`: the guards of {rendered} form a loop "
            "of rate comparisons: each drives a rate the next one switches "
            "on, so the selection has no fixpoint. Compare an integrated "
            "level instead, which the integration carries across the sweep."
        )

    def _refuse_unbounded_rule_cycles(self, edges: list[_ContinuousEdge]) -> None:
        """Refuse a rule cycle that creates matter from nothing (R25).

        A cycle whose coefficient product exceeds 1 makes more of a thing
        than it consumed, and if all of its inputs originate inside it
        nothing bounds what it makes. **The same cycle fed by a finite
        external input is bounded by that input and is a model**, not an
        error: it is refused only when it is closed on itself. A capacity
        anywhere on the cycle breaks it too, the integration carrying the
        quantity across instead of resolving it instantaneously.

        **What this verdict does not settle.** A rule cycle is also a
        cycle of *capability* equations, and the engine refuses those on
        its own account: a rule's output capability is derived from its
        inputs', so a cycle of rules closes a chain of explicit
        equations whatever its coefficients. A cycle this diagnostic
        accepts therefore still meets that refusal, and today a capacity
        on the cycle does not lift it either, since a buffered input is
        read through the upstream capability rather than through what
        the volume can serve. Making the volume replace the flow it
        buffers is what would close that gap, and it is a decision about
        what "available" means, not a detail of this diagnostic."""
        feeding: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        for edge in edges:
            feeding.setdefault((edge.consumer, edge.flow_in, "in"), set()).add(
                (edge.producer, edge.flow_out, "out")
            )
        try:
            for cycle in self._elementary_cycles(self._rule_graph(edges)):
                applied = [step[3] for step in cycle if step[3] is not None]
                if not applied:
                    continue
                gain = 1.0
                for step in cycle:
                    gain *= step[2]
                if gain <= 1.0 + 1e-12:
                    continue
                on_cycle = {step[0] for step in cycle}
                if any(
                    self.comp[component]._capacity_of(flow)
                    for component, flow, _ in on_cycle
                ):
                    continue
                if self._cycle_is_fed_externally(applied, on_cycle, feeding):
                    continue
                path = " -> ".join(
                    f"`{component}`.`{flow}`" for component, flow, _ in
                    [step[0] for step in cycle] + [cycle[0][0]]
                )
                owners: list[tuple[str, str]] = []
                for component, rule_set, _ in applied:
                    if (component, rule_set) not in owners:
                        owners.append((component, rule_set))
                named = ", ".join(
                    f"rule set `{rule_set}` of `{component}`"
                    for component, rule_set in owners
                )
                raise ValueError(
                    f"System `{self.name}`: {named} form the cycle {path}, whose "
                    f"coefficient product is {gain:g} and whose inputs all "
                    "originate inside it: it would create matter from nothing. "
                    "Feed the cycle from outside it, or break it with a capacity."
                )
        except _CycleSearchExhausted as exhausted:
            raise ValueError(
                f"System `{self.name}`: the rule-cycle search stopped after "
                f"expanding its budget of {exhausted.budget} arcs over "
                f"{exhausted.nodes} flow endpoints, before it had enumerated "
                "every cycle. The refusal of a cycle that creates matter from "
                "nothing therefore cannot be given, and the model is refused "
                "rather than built on a guard that stopped guarding. Break the "
                "rule graph into fewer interconnected flows, or interpose a "
                "capacity, which breaks any cycle it sits on."
            ) from None

    def _ring_torn_outputs(self, edges: list[_ContinuousEdge]) -> dict[str, set[str]]:
        """Per component, the rule outputs whose connection closes a
        recirculation ring on a volume: the edge the ring is torn at.

        A fan drawing air out of a room and blowing it back in is a ring
        of instantaneous values: the fan's rule is bounded by the demand
        on its return output, which is the room's inbound demand, which
        carries the fan's own demand for room air through the transit.
        The engine refuses such a chain as having no solution, and
        rightly so while nothing on it integrates.

        A volume on the ring changes that, and muscadet accepts the ring
        for it (a ring with no volume is its R30 refusal). It tears the
        edge **entering** the volume and reads the torn quantities one
        evaluation late. The same edge is torn here at its **other end**:
        the rule producing into it is no longer bounded by the volume's
        demand, and the volume absorbs whatever the two ends momentarily
        disagree on, which is what a volume is. No late read is needed,
        so the sweep keeps its consistent solution where the reference
        shifts the volume's level once by one integration stage.

        An edge is torn when it feeds a flow its consumer holds in a
        volume, its producer makes that flow by a rule, and the flow
        graph leads from the volume back to that output: through the
        connections, the rules (consumed to produced), the pass-throughs
        and the transiting volumes. A ring closing through a producer
        that is not a rule (a plain pipe) is left alone, so the engine's
        refusal still names it: tearing it would need the pipe's demand
        cut, which is another statement.

        **The tear is taken only where the volume provably absorbs what
        the two ends disagree on** (:meth:`_tear_conserves`), since a
        connection never delivers more than its consumer asks: a torn
        rule making more than the volume accepts would lose the surplus,
        and one left with no other bound would run unbounded. Every other
        ring keeps the engine's refusal."""
        successors: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
        for node, steps in self._rule_graph(edges).items():
            successors.setdefault(node, set()).update(step[0] for step in steps)
        for name, obj in self.comp.items():
            declared_in = {flow.name for flow in obj.flows_continuous_in}
            declared_out = {flow.name for flow in obj.flows_continuous_out}
            transiting = {
                flow
                for capacity in obj.capacities
                if capacity.transmits
                for flow in capacity.flow_names
            }
            for flow in declared_in & declared_out:
                if flow in transiting or obj._passes_through(flow):
                    successors.setdefault((name, flow, "in"), set()).add(
                        (name, flow, "out")
                    )

        def reaches(start, goal) -> bool:
            seen, pending = {start}, [start]
            while pending:
                node = pending.pop()
                if node == goal:
                    return True
                for following in successors.get(node, ()):
                    if following not in seen:
                        seen.add(following)
                        pending.append(following)
            return False

        torn: dict[str, set[str]] = {}
        for edge in edges:
            producer = self.comp.get(edge.producer)
            consumer = self.comp.get(edge.consumer)
            if producer is None or consumer is None:
                continue
            held = {
                flow for capacity in consumer.capacities for flow in capacity.flow_names
            }
            made = {
                flow for rule_set in producer.rule_sets for flow in rule_set.produced()
            }
            if edge.flow_in not in held or edge.flow_out not in made:
                continue
            if reaches(
                (edge.consumer, edge.flow_in, "in"),
                (edge.producer, edge.flow_out, "out"),
            ) and self._tear_conserves(edge, edges):
                torn.setdefault(edge.producer, set()).add(edge.flow_out)
        return torn

    def _tear_conserves(
        self, edge: _ContinuousEdge, edges: list[_ContinuousEdge]
    ) -> bool:
        """Whether tearing the ring at `edge` keeps every quantity
        bounded and conserved, which holds for a direct recirculation:

        - the torn output feeds the volume and nothing else, so no other
          consumer loses the bound its demand put on the rule;
        - the volume hands the same flow straight back to the producer,
          and every rule making the torn output draws at least as much of
          that returned flow as it makes, so it never blows back more
          than the volume let out, which the volume always accepts
          (full, it accepts exactly what leaves it);
        - every such rule makes something else a consumer reads, which
          still bounds it once the torn output no longer does.

        A ventilated room is exactly this: the fan returns the air it
        drew and exhausts to the atmosphere. A fan that only recirculates,
        one that makes more air than it draws, or a return read off the
        ring fails a condition and keeps the engine's refusal."""
        producer = self.comp[edge.producer]
        if [
            other
            for other in edges
            if other.producer == edge.producer and other.flow_out == edge.flow_out
        ] != [edge]:
            return False
        returned = {
            back.flow_in
            for back in edges
            if back.producer == edge.consumer
            and back.consumer == edge.producer
            and back.flow_out == edge.flow_in
        }
        if not returned:
            return False
        bounding = {
            other.flow_out
            for other in edges
            if other.producer == edge.producer and other.flow_out != edge.flow_out
        }
        for rule_set in producer.rule_sets:
            for rule in rule_set.rules:
                made = rule.prod.get(edge.flow_out, 0.0)
                if made <= 0:
                    continue
                drawn = max((rule.cons.get(flow, 0.0) for flow in returned), default=0.0)
                if drawn < made:
                    return False
                if not any(
                    rule.prod.get(flow, 0.0) > 0 for flow in bounding
                ):
                    return False
        return True

    def _cycle_is_fed_externally(self, applied, on_cycle, feeding) -> bool:
        """Whether anything outside the cycle feeds it, which is what
        bounds what it can make.

        Two ways in: a rule on the cycle consumes a flow the cycle does
        not carry (connected or standing at its declared default, either
        way a finite quantity), or a flow the cycle does carry is also
        fed by a producer that is not on it."""
        for component, set_name, index in applied:
            obj = self.comp[component]
            rule_set = next(s for s in obj.rule_sets if s.name == set_name)
            for flow, taken in rule_set.rules[index].cons.items():
                if taken > 0 and (component, flow, "in") not in on_cycle:
                    return True
        for node in on_cycle:
            if node[2] != "in":
                continue
            if any(producer not in on_cycle for producer in feeding.get(node, ())):
                return True
        return False
