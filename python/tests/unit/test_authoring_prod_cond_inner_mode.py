"""A production condition means what muscadet says it means, on the
authoring route too.

muscadet reads a production condition as two levels combined by
``var_prod_cond_inner_mode`` (``muscadet/flow.py``, ``prod_cond_holds``):
``"or"``, its default, is ``all(any(group))``, a conjunction of
disjunctions; ``"and"`` is ``any(all(group))``. A flat list is one clause
per element. ``pyraichu.muscadet``'s authoring classes read the same list
as the OR of conjunctions and took no mode, so a two-group condition that
states none meant ``a and b`` there and ``a or b`` here.

The authoring classes now carry the mode with muscadet's default, the way
a capacity already carries ``serve_cond_inner_mode``. The two other routes
keep what they compute: the muscadet-facing declaration passes the mode it
read, and the plugin's ``flows_out`` section keeps its own documented
reading when it states none (see ``test_plugin_prod_cond_parity.py``).
"""

from __future__ import annotations

import itertools

import pyraichu.muscadet as mu
import pytest
from conftest import sampled


def muscadet_holds(condition, inner_mode: str, fed: dict[str, bool]) -> bool:
    """muscadet's definition, written out: a flat list is one clause per
    element, and the mode decides which level is the disjunction."""
    clauses = [
        list(element) if isinstance(element, list) else [element]
        for element in condition
    ]
    if inner_mode == "or":
        return all(any(fed[name] for name in clause) for clause in clauses)
    return any(all(fed[name] for name in clause) for clause in clauses)


def a_board(condition, inner_mode: str | None, fed_a: bool, fed_b: bool):
    def source(fed: bool, flow: str):
        class Source(mu.ObjFlow):
            def add_flows(self):
                self.add_flow_out(name=flow, var_prod_default=fed)

        return Source

    keywords = {} if inner_mode is None else {"var_prod_cond_inner_mode": inner_mode}

    class Board(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_in(name="a")
            self.add_flow_in(name="b")
            self.add_flow_out(name="out", var_prod_cond=condition, **keywords)

    system = mu.System("board")
    system.add_component(source(fed_a, "a"), "A")
    system.add_component(source(fed_b, "b"), "B")
    system.add_component(Board, "BRD")
    system.connect("A", "a", "BRD", "a")
    system.connect("B", "b", "BRD", "b")
    return system


CONDITIONS = [[["a"], ["b"]], [["a", "b"]], ["a", "b"], [["a"]], ["a"]]
FEEDS = list(itertools.product([False, True], repeat=2))


@pytest.mark.parametrize("condition", CONDITIONS, ids=str)
@pytest.mark.parametrize("inner_mode", ["or", "and"])
def test_every_shape_and_mode_reads_as_muscadet_defines_it(condition, inner_mode):
    for fed_a, fed_b in FEEDS:
        result = a_board(condition, inner_mode, fed_a, fed_b).simulate(
            t_max=1.0, samples=[1.0]
        )
        expected = muscadet_holds(condition, inner_mode, {"a": fed_a, "b": fed_b})
        assert sampled(result, "BRD_out_fed_out", 1.0) is expected, (fed_a, fed_b)


@pytest.mark.parametrize("condition", CONDITIONS, ids=str)
def test_a_condition_stating_no_mode_takes_muscadet_s_default(condition):
    for fed_a, fed_b in FEEDS:
        result = a_board(condition, None, fed_a, fed_b).simulate(
            t_max=1.0, samples=[1.0]
        )
        expected = muscadet_holds(condition, "or", {"a": fed_a, "b": fed_b})
        assert sampled(result, "BRD_out_fed_out", 1.0) is expected, (fed_a, fed_b)


def test_a_third_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="var_prod_cond_inner_mode"):
        a_board([["a"], ["b"]], "xor", True, True).build_dict()
