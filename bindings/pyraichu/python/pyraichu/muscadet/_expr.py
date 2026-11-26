"""Expression builders: the JSON expression nodes the emitted model is written with."""

from __future__ import annotations

import re
from typing import Any

from ._vocabulary import _CMP_PREDICATES


def _var(component: str, variable: str) -> dict[str, Any]:
    return {"op": "attr", "attr": {"component": component, "attribute": variable}}


def _state_active(component: str, automaton: str, state: str) -> dict[str, Any]:
    return {
        "op": "state_active",
        "state": {"component": component, "automaton": automaton, "state": state},
    }


def _float(value: float) -> dict[str, Any]:
    return {"op": "const", "value": {"kind": "float", "value": float(value)}}


def _float_attribute(name: str, init: float) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "float",
        "init": {"kind": "float", "value": float(init)},
    }


def _sum(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """The sum of `terms`, degrading to the term itself and to zero
    rather than emitting a one-argument or empty ``add``."""
    if not terms:
        return _float(0.0)
    if len(terms) == 1:
        return terms[0]
    return {"op": "add", "args": terms}


def _min(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """The minimum of `terms`, degrading to the term itself rather than
    emitting a one-argument ``min``.

    This is the operator a rule's scale is written with (KTD14): it is
    continuous and piecewise linear, so the trajectory keeps a **kink**
    where the limiting term changes identity, and needs no crossing
    transition to be located. An empty list has no minimum and is a
    caller error rather than a silent zero."""
    if not terms:
        raise ValueError("a minimum needs at least one term")
    if len(terms) == 1:
        return terms[0]
    return {"op": "min", "args": terms}


def _max(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """The maximum of `terms`, degrading to the term itself rather than
    emitting a one-argument ``max``. An empty list is a caller error,
    as for :func:`_min`."""
    if not terms:
        raise ValueError("a maximum needs at least one term")
    if len(terms) == 1:
        return terms[0]
    return {"op": "max", "args": terms}


def _const_bool(value: bool) -> dict[str, Any]:
    return {"op": "const", "value": {"kind": "bool", "value": bool(value)}}


def _boolean_comparison(read: dict[str, Any], op: str, value: float) -> dict[str, Any]:
    """A comparison whose left side reads a **two-valued** quantity, settled
    here instead of at every evaluation.

    muscadet thresholds such an operand on ``float(var_fed.value())``
    (``muscadet.flow._prod_cond_compare_reader``), so a state that is False or
    True is compared as 0 or 1. There are only two left-hand sides, so the
    comparison is one of four expressions -- the read, its negation, or a
    constant -- and writing it as one is not an optimisation: this engine's
    ordering comparison is typed and refuses a boolean left side outright
    (``ordering comparison on booleans``), so the comparison HAS to be resolved
    on this side to mean what it means over there.

    The constants are the honest answer to a threshold no state can cross:
    ``ctrl > 3`` never holds and ``ctrl >= 0`` always does. Refusing them
    instead would refuse a condition muscadet runs."""
    predicate = _CMP_PREDICATES[op]
    threshold = float(value)
    at_false = bool(predicate(0.0, threshold))
    at_true = bool(predicate(1.0, threshold))
    if at_false == at_true:
        return _const_bool(at_true)
    if at_true:
        return read
    return {"op": "bool", "bool_op": "not", "args": [read]}


def _bool(bool_op: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    """`terms` joined by `bool_op`, degrading to the term itself rather
    than emitting a one-argument ``bool``.

    The sibling of :func:`_sum` and :func:`_min`, and there for the same
    reason: a conjunction or a disjunction is written wherever one is
    needed, over a list whose length the caller rarely knows, and the
    one-term case is the common one."""
    if len(terms) == 1:
        return terms[0]
    return {"op": "bool", "bool_op": bool_op, "args": terms}


def _matches_flow(pattern: str, name: str) -> bool:
    """Whether an effect pattern names `name`, **matched whole**.

    A full match rather than a bare search, which is the whole of the
    rule: unanchored, ``"H2"`` would name ``H2O`` as well, and a
    declaration meant for one output would silently derate its
    neighbour. The pattern is **grouped** before it is matched, because
    alternation binds looser than anchoring: written ``^H2|O2$``, the
    two-flow pattern ``"H2|O2"`` reads as "starts with H2, or ends with
    O2" and names ``H2O``, ``XO2`` and ``H2X`` after all. Grouped, it
    names the two flows it was written for and nothing else."""
    return re.fullmatch(f"(?:{pattern})", name) is not None


def _negated(expr: dict[str, Any]) -> dict[str, Any]:
    """``-expr``, written as a subtraction from zero: the schema has no
    unary minus, and a `mul` by -1 reads worse at the same cost."""
    return {"op": "sub", "lhs": _float(0.0), "rhs": expr}


def _clamped_at_zero(expr: dict[str, Any]) -> dict[str, Any]:
    """``max(0, expr)``: a magnitude, where a negative value means the
    other direction and is handled by the sign routing rather than by
    letting it through as a quantity."""
    return {"op": "max", "args": [_float(0.0), expr]}


def _renamed_reads(
    expr: Any, component: str, renames: dict[str, str]
) -> Any:
    """`expr` with every read of `component`'s attribute ``{flow}_demand_out``
    redirected to ``renames[flow]``.

    How a volume downstream of the rules puts its claim where the rule
    scale reads the consumers' demand, without the scale having to know
    about volumes: the scale is written once, over the demand, and the
    one component whose output is claimed reads the claim instead."""
    if not renames:
        return expr
    targets = {f"{flow}_demand_out": name for flow, name in renames.items()}
    if isinstance(expr, list):
        return [_renamed_reads(item, component, renames) for item in expr]
    if not isinstance(expr, dict):
        return expr
    if expr.get("op") == "attr":
        attr = expr["attr"]
        if attr.get("component") == component and attr.get("attribute") in targets:
            return _var(component, targets[attr["attribute"]])
        return expr
    return {key: _renamed_reads(value, component, renames) for key, value in expr.items()}
