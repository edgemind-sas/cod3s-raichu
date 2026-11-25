"""Graph helpers over the component connections."""

from __future__ import annotations

from typing import Any


class _CycleSearchExhausted(Exception):
    """The rule-cycle search spent its budget before it enumerated every
    cycle, so it saw only part of the graph.

    Raised out of the walk rather than returned, because a partial
    enumeration is not a shorter answer to the same question: a cycle
    that creates matter may sit in the part never reached, and a guard
    that quietly stops guarding is worse than no guard. The caller turns
    it into a refusal naming what the search covered."""

    def __init__(self, nodes: int, budget: int):
        super().__init__(f"{budget} arcs expanded over {nodes} nodes")
        self.nodes = nodes
        self.budget = budget


def _strongly_connected(graph: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """The strongly connected components of a graph given as adjacency
    lists (Tarjan, iterative): the component number of each node, and
    each component's members in increasing index. A node on no cycle is
    a component of its own."""
    component_of = [-1] * len(graph)
    members: list[list[int]] = []
    index_of = [-1] * len(graph)
    low = [0] * len(graph)
    on_stack = [False] * len(graph)
    stack: list[int] = []
    counter = 0
    for root in range(len(graph)):
        if index_of[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            node, edge = work.pop()
            if edge == 0:
                index_of[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            if edge < len(graph[node]):
                work.append((node, edge + 1))
                child = graph[node][edge]
                if index_of[child] == -1:
                    work.append((child, 0))
                elif on_stack[child]:
                    low[node] = min(low[node], index_of[child])
                continue
            for child in graph[node]:
                if on_stack[child] and component_of[child] == -1:
                    low[node] = min(low[node], low[child])
            if low[node] == index_of[node]:
                block = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component_of[member] = len(members)
                    block.append(member)
                    if member == node:
                        break
                members.append(sorted(block))
    return component_of, members


def _channel_attr(port: str, channel: str, edge: str) -> str:
    """The attribute the compiler materialises for one (connection,
    channel) pair, per the schema's naming rule."""
    return f"{port}__{channel}__{edge}"


def _edge_name(connection: dict[str, Any]) -> str:
    """The edge a connection is named after: its own ``name`` when it
    has one, its destination otherwise (the schema's default)."""
    if connection.get("name"):
        return str(connection["name"])
    destination = connection["to"]
    return f"{destination['component']}__{destination['port']}"


def connected_in_flows(
    connections: list[dict[str, Any]], component: str
) -> set[str]:
    """The in-ports of `component` at least one connection feeds, each
    minus its `_in` suffix.

    The one piece of system knowledge :meth:`ObjFlow._build` needs at the
    input end, and it is needed once: a trigger port, read as
    `{flow}_trigger`, since a `FlowOutOnTrigger` whose trigger nothing
    feeds is armed from the initial instant and the automaton is declared
    in `up` rather than reaching it through a zero-delay transition.

    The boolean in-flows read it no more, and that is the point of their
    cure: what an in-flow answers out of connection is written into its
    own aggregating expression (`count(p) >= 1`), so the rule holds
    whatever the connection list says and a component generates the same
    document wherever it is instantiated.

    Ports are matched by the schema's `{port}_in` naming, the same rule
    :func:`_channel_attr` and `_continuous_edges` read them by."""
    return {
        connection["to"]["port"][: -len("_in")]
        for connection in connections
        if connection["to"]["component"] == component
        and connection["to"]["port"].endswith("_in")
    }


def _find_cycle(nodes, arcs):
    """The first cycle a depth-first walk closes over `arcs`, as a list
    of ``(node, label)`` steps, or ``None``.

    Declaration order in, declaration order out: the reported cycle is
    the first one the walk closes, so one model always yields one
    diagnostic."""
    unseen, active, done = 0, 1, 2
    mark = {node: unseen for node in nodes}
    path: list = []

    def walk(node):
        mark[node] = active
        for target, label in arcs.get(node, ()):
            path.append((node, label))
            state = mark.get(target, unseen)
            if state == active:
                start = next(
                    index for index, (seen, _) in enumerate(path) if seen == target
                )
                return path[start:]
            if state == unseen:
                found = walk(target)
                if found is not None:
                    return found
            path.pop()
        mark[node] = done
        return None

    for node in nodes:
        if mark[node] == unseen:
            found = walk(node)
            if found is not None:
                return found
    return None
