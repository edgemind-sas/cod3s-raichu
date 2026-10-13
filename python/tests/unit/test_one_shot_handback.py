"""One-shot effects beside held ones: the hand-back of a clamped value, on
several targets and on on-demand (inst) directions.

cod3s 1.20.0 (edgemind-sas/cod3s#8) accepts a variable held by a mode on one
face and written ONCE entering the other: typically a clamp held while the
mode is failed (`occ_effects`) and the nominal value handed back at the
repair (`not_occ_effects_trans`), on a variable that does not reinitialize.
In the external behaviours the one-shot writes ride on each target's mirror
automaton, so they fire when the target itself changes state. cod3s 1.21.0
(edgemind-sas/cod3s#9) accepts them on several targets and on inst
directions, and refuses only what overlapping combinations make ill-posed.

Every expected trajectory below is the reference engine's own output for the
same mode declarations, measured 2026-10-12 on PyCATSHOO 1.3.8.0 with cod3s
1.21.0 and muscadet 5.14.1 (target variables `ceiling` and `gate` declared
`setReinitialized(False)`, `working` reinitialized). Deterministic laws, one
trajectory, so the contract is equality.
"""

from __future__ import annotations

import pyraichu
import pytest
from pyraichu.mode_objects import failure_mode_object
from pyraichu.muscadet import declare
from pyraichu.plugins import expand_model

N, C, REST = 40.0, 0.0, 25.0
BEHAVIOURS = ("internal", "external", "external_rep_indep")

S_SHORT = [0.0, 0.5, 1.5, 2.5, 3.1, 3.5]
S_CCF = [0.0, 2.5, 3.5, 4.5, 12.5, 13.5, 14.5]
S_INST = [0.0, 0.5, 1.2, 2.2, 2.7, 3.7, 4.2]

HANDBACK = {"occ_effects": {"ceiling": C}, "not_occ_effects_trans": {"ceiling": N}}

#: The demand an on-demand mode draws on: on at 1, off at 1.5, on at 2.5, ...
DEMAND = [[{"obj": "D", "attr": "demand", "ope": "==", "value": True}]]


def store(name: str) -> dict:
    return {
        "name": name,
        "attributes": [
            {"name": "ceiling", "kind": "float", "init": {"kind": "float", "value": N}},
            {"name": "gate", "kind": "bool", "init": {"kind": "bool", "value": False}},
            {
                "name": "working",
                "kind": "bool",
                "init": {"kind": "bool", "value": True},
            },
        ],
        "ports": [],
        "interfaces": [],
        "automata": [],
        "sensitive_functions": [],
        "equations": [],
    }


def demander() -> dict:
    on = {
        "op": "state_active",
        "state": {"component": "D", "automaton": "cycle", "state": "on"},
    }
    return {
        "name": "D",
        "attributes": [
            {"name": "demand", "kind": "bool", "init": {"kind": "bool", "value": False}}
        ],
        "ports": [],
        "interfaces": [],
        "equations": [],
        "automata": [
            {
                "name": "cycle",
                "states": ["idle", "on"],
                "init": "idle",
                "transitions": [
                    {
                        "name": "on",
                        "source": "idle",
                        "targets": ["on"],
                        "distrib": "delay",
                        "time": 1.0,
                    },
                    {
                        "name": "idle",
                        "source": "on",
                        "targets": ["idle"],
                        "distrib": "delay",
                        "time": 0.5,
                    },
                ],
            }
        ],
        "sensitive_functions": [
            {
                "name": "dem",
                "effects": [
                    {"target": {"component": "D", "attribute": "demand"}, "value": on}
                ],
            }
        ],
    }


def mode(name, targets, behaviour="internal", **fields):
    spec = {
        "cls": "ObjMode2S",
        "fm_name": name,
        "targets": list(targets),
        "behaviour": behaviour,
        "occ_state": "occ",
        "not_occ_state": "rep",
    }
    spec.update(fields)
    return failure_mode_object(spec)


def delay(ttf, ttr):
    return {
        "occ_law": {"cls": "delay", "time": ttf},
        "not_occ_law": {"cls": "delay", "time": ttr},
    }


def model(targets, modes, variables, demand=False):
    components = [store(t) for t in targets] + ([demander()] if demand else [])
    body = expand_model(
        {
            "name": "one_shot",
            "components": components,
            "connections": [],
            "indicators": [],
            "plugins": {"muscadet": {"objects": list(modes)}},
        }
    )
    body["indicators"] = [
        {
            "name": f"{t}_{v}",
            "target": "attribute",
            "attr": {"component": t, "attribute": v},
        }
        for t in targets
        for v in variables
    ]
    return pyraichu.load_model(body)


def timeline(targets, modes, schedule, variable="ceiling", demand=False, seed=1):
    result = pyraichu.simulate(
        model(targets, modes, [variable], demand),
        t_max=schedule[-1],
        samples=schedule,
        seed=seed,
    )
    return {
        t: [float(v) for _, v in result.samples[f"{t}_{variable}"]] for t in targets
    }


# --- the hand-back of a clamped value ---------------------------------------


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_one_mode_hands_the_value_back_at_its_repair(behaviour):
    """occ at 1, repaired at 3 (and failing again at 4, after the samples)."""
    got = timeline(
        ["S"], [mode("cell", ["S"], behaviour, **delay(1.0, 2.0), **HANDBACK)], S_SHORT
    )
    assert got["S"] == [N, N, C, C, N, N]


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_two_modes_hold_the_clamp_until_the_last_repair(behaviour):
    """cell: occ 1, rep 3. bms: occ 2, rep 3.25. cell's hand-back at 3 is
    overwritten by bms's clamp, bms's at 3.25 hands the value back."""
    modes = [
        mode("cell", ["S"], behaviour, **delay(1.0, 2.0), **HANDBACK),
        mode("bms", ["S"], behaviour, **delay(2.0, 1.25), **HANDBACK),
    ]
    assert timeline(["S"], modes, S_SHORT)["S"] == [N, N, C, C, C, N]


#: Delay laws over two targets, order 1 failing at 2 for 2, order 2 failing at
#: 3 for 10. Measured per behaviour: in `internal` the order-2 combination
#: holds both targets until 13 whatever the order-1 repairs; the external
#: behaviours follow each target's mirror under the mutual lock.
CCF_EXPECTED = {
    "internal": [N, C, C, C, C, N, C],
    "external": [N, C, C, N, N, N, C],
    "external_rep_indep": [N, C, C, N, N, N, C],
}


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_a_common_cause_hand_back_follows_every_combination(behaviour):
    modes = [
        mode(
            "m",
            ["A", "B"],
            behaviour,
            occ_law={"cls": "delay", "time": [2.0, 3.0]},
            not_occ_law={"cls": "delay", "time": [2.0, 10.0]},
            **HANDBACK,
        )
    ]
    got = timeline(["A", "B"], modes, S_CCF)
    assert got == {"A": CCF_EXPECTED[behaviour], "B": CCF_EXPECTED[behaviour]}


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_a_demand_failure_hands_the_value_back_at_its_timed_repair(behaviour):
    """A certain draw on each demand front, repaired after 1: occ 1, rep 2,
    occ 2.5, rep 3.5, occ 4."""
    modes = [
        mode(
            "m",
            ["S"],
            behaviour,
            occ_law={"cls": "inst", "prob": 1.0},
            occ_cond=DEMAND,
            not_occ_law={"cls": "delay", "time": 1.0},
            **HANDBACK,
        )
    ]
    assert timeline(["S"], modes, S_INST, demand=True)["S"] == [N, N, C, N, C, N, C]


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_held_at_rest_and_written_once_entering_the_failure(behaviour):
    """The symmetric opposite-face pair: the rest value from t = 0, the
    written value while failed."""
    modes = [
        mode(
            "m",
            ["S"],
            behaviour,
            **delay(1.0, 2.0),
            not_occ_effects={"ceiling": REST},
            occ_effects_trans={"ceiling": C},
        )
    ]
    assert timeline(["S"], modes, S_SHORT)["S"] == [REST, REST, C, C, REST, REST]


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_order_one_alone_over_three_targets_pulses_each_target(behaviour):
    modes = [
        mode(
            "m",
            ["A", "B", "C"],
            behaviour,
            occ_law={"cls": "delay", "time": [1.0, None, None]},
            not_occ_law={"cls": "delay", "time": [2.0, None, None]},
            occ_effects_trans={"gate": True},
            not_occ_effects_trans={"gate": False},
        )
    ]
    got = timeline(["A", "B", "C"], modes, S_SHORT, variable="gate")
    assert got == {t: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0] for t in "ABC"}


def test_the_hand_back_is_absorbed_into_the_held_effect():
    """Written by the held conditional on both of its branches, so not
    written on the repair edge a second time (an edge write beside a level
    writer of the same attribute is refused by the core)."""
    body = expand_model(
        {
            "name": "one_shot",
            "components": [store("S")],
            "connections": [],
            "indicators": [],
            "plugins": {
                "muscadet": {
                    "objects": [mode("m", ["S"], **delay(1.0, 2.0), **HANDBACK)]
                }
            },
        }
    )
    component = next(c for c in body["components"] if c["name"] == "m")
    edges = [t for a in component["automata"] for t in a["transitions"]]
    assert not any(t.get("effects") for t in edges)
    (effect,) = component["sensitive_functions"][0]["effects"]
    assert effect["value"]["then"]["value"]["value"] == C
    assert effect["value"]["otherwise"]["value"]["value"] == N


# --- draws: the write is guarded on the landed branch -----------------------


def _agreement(modes, variable, nb_seeds=40):
    """On every trajectory, `variable` written by the one-shot effects
    agrees with the clamp the mode holds on `working`: equal means at every
    instant are only possible if they agree trajectory by trajectory, which
    is how cod3s 1.21.0 checks the same shapes."""
    schedule = [0.0, 0.7, 1.3, 2.2, 3.1, 4.6, 5.9, 7.4, 9.0, 11.3, 14.0]
    seen_failed = False
    for seed in range(nb_seeds):
        result = pyraichu.simulate(
            model(["S"], modes, [variable, "working"], demand=True),
            t_max=schedule[-1],
            samples=schedule,
            seed=seed,
        )
        working = [bool(v) for _, v in result.samples["S_working"]]
        pulsed = [float(v) for _, v in result.samples[f"S_{variable}"]]
        if variable == "gate":
            assert [bool(v) for v in pulsed] == [not w for w in working], seed
        else:
            assert pulsed == [N if w else C for w in working], seed
        seen_failed = seen_failed or not all(working)
    assert seen_failed


@pytest.mark.parametrize("behaviour", ["internal", "external"])
def test_a_lost_occurrence_draw_writes_nothing(behaviour):
    modes = [
        mode(
            "m",
            ["S"],
            behaviour,
            occ_law={"cls": "inst", "prob": 0.5},
            occ_cond=DEMAND,
            not_occ_law={"cls": "exp", "rate": 1.0},
            occ_effects={"working": False},
            occ_effects_trans={"gate": True},
            not_occ_effects_trans={"gate": False},
        )
    ]
    _agreement(modes, "gate")


@pytest.mark.parametrize("variable", ["gate", "ceiling"])
def test_a_lost_return_draw_keeps_the_mode_failed_and_writes_nothing(variable):
    modes = [
        mode(
            "m",
            ["S"],
            occ_law={"cls": "exp", "rate": 0.4},
            not_occ_law={"cls": "inst", "prob": 0.5},
            not_occ_cond=DEMAND,
            occ_effects={"working": False, "ceiling": C},
            occ_effects_trans={"gate": True},
            not_occ_effects_trans={"gate": False, "ceiling": N},
        )
    ]
    _agreement(modes, variable)


def test_the_legacy_on_demand_dialect_carries_one_shot_effects():
    """`ObjFMInst` no longer refuses them: its draw is the engine's."""
    obj = failure_mode_object(
        {
            "cls": "ObjFMInst",
            "fm_name": "m",
            "targets": ["S"],
            "failure_cond": DEMAND,
            "failure_param": [0.5],
            "repair_param": [1.0],
            "failure_effects": {"working": False},
            "failure_effects_trans": {"gate": True},
            "repair_effects_trans": {"gate": False},
        }
    )
    assert obj["failure_effects_trans"] == {"gate": True}
    _agreement([obj], "gate")


# --- what stays refused ------------------------------------------------------

E3 = {
    "occ_law": {"cls": "exp", "rate": [0.3, 0.2, 0.1]},
    "not_occ_law": {"cls": "exp", "rate": [1.0, 0.5, 0.5]},
}


def refused(targets, modes):
    with pytest.raises(ValueError) as error:
        model(targets, modes, ["ceiling"])
    return str(error.value)


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
@pytest.mark.parametrize(
    "held, pulsed",
    [
        ("occ_effects", "occ_effects_trans"),
        ("not_occ_effects", "not_occ_effects_trans"),
    ],
)
def test_held_and_written_once_on_the_same_face_is_refused(behaviour, held, pulsed):
    fields = {held: {"ceiling": C}, pulsed: {"ceiling": N}}
    message = refused(["S"], [mode("m", ["S"], behaviour, **delay(1.0, 2.0), **fields)])
    assert "SAME face" in message


@pytest.mark.parametrize(
    "fields, words",
    [
        (
            {"occ_effects_trans": {"gate": True}, "not_occ_effects": {"gate": False}},
            "held at rest",
        ),
        (
            {
                "occ_effects_trans": {"gate": True},
                "not_occ_effects_trans": {"gate": False},
            },
            "on both faces",
        ),
    ],
    ids=["write-occ-held-rest", "both-faces-written"],
)
def test_ill_posed_shapes_over_overlapping_combinations_are_refused(fields, words):
    message = refused(["A", "B", "C"], [mode("m", ["A", "B", "C"], **E3, **fields)])
    assert words in message
    assert (
        "orders [1, 2, 3] built): 4 combination automata write each target" in message
    )


def test_a_value_held_on_both_faces_over_overlapping_combinations_follows_every_one():
    """Held while failed (`occ_effects`) and held at rest with another value
    (`not_occ_effects`) over two targets and two orders. Order 1 fails at 2
    for 2 (again at 6, 10, 14), order 2 at 3 for 10. A target holds the
    failure value while ANY combination covering it is failed, the rest
    value only while ALL of them are at rest: the order-1 repairs at 4, 8
    and 12 leave it failed under the order-2 combination, which repairs at
    13. cod3s refused this shape in 1.21.0 (cod3s#4) and applies this rule
    since 1.22.0 (cod3s#10); the cross-validation replay of
    `repair_effects_ccf` agrees with it event for event."""
    modes = [
        mode(
            "m",
            ["A", "B"],
            occ_law={"cls": "delay", "time": [2.0, 3.0]},
            not_occ_law={"cls": "delay", "time": [2.0, 10.0]},
            occ_effects={"ceiling": C},
            not_occ_effects={"ceiling": REST},
        )
    ]
    expected = [REST, C, C, C, C, REST, C]
    assert timeline(["A", "B"], modes, S_CCF) == {"A": expected, "B": expected}


def test_overlapping_combinations_accept_the_hand_back_and_the_external_behaviours():
    model(["A", "B", "C"], [mode("m", ["A", "B", "C"], **E3, **HANDBACK)], ["ceiling"])
    model(
        ["A", "B", "C"],
        [
            mode(
                "m",
                ["A", "B", "C"],
                "external",
                **E3,
                occ_effects_trans={"gate": True},
                not_occ_effects_trans={"gate": False},
            )
        ],
        ["gate"],
    )
    model(
        ["A", "B", "C"],
        [
            mode(
                "m",
                ["A", "B", "C"],
                **E3,
                occ_effects={"gate": True},
                not_occ_effects={"gate": True},
            )
        ],
        ["gate"],
    )


def test_two_modes_clamping_different_values_are_refused():
    message = refused(
        ["S"],
        [
            mode("a", ["S"], **delay(1.0, 2.0), **HANDBACK),
            mode(
                "b",
                ["S"],
                **delay(2.0, 1.25),
                occ_effects={"ceiling": 20.0},
                not_occ_effects_trans={"ceiling": N},
            ),
        ],
    )
    assert "DIFFERENT clamp values" in message


def test_a_hand_back_beside_a_clamp_never_handed_back_is_refused():
    message = refused(
        ["S"],
        [
            mode("a", ["S"], **delay(1.0, 2.0), **HANDBACK),
            mode("b", ["S"], **delay(2.0, 1.25), occ_effects={"ceiling": C}),
        ],
    )
    assert "declare the hand-back on every mode" in message


def test_a_hand_back_value_other_than_the_initial_one_is_refused():
    message = refused(
        ["S"],
        [
            mode(
                "m",
                ["S"],
                **delay(1.0, 2.0),
                occ_effects={"ceiling": C},
                not_occ_effects_trans={"ceiling": 30.0},
            )
        ],
    )
    assert "not its initial value 40.0" in message


# --- the declaration route: a persistent gate, a controller endpoint --------

FLOW = "is_ok"
GATE = f"{FLOW}_fed_available_out"
S_GATE = [0.0, 0.5, 1.5, 2.5, 3.1, 3.5, 4.5]


def _flow_out(produced: bool, reset: bool) -> dict:
    return {
        "cls": "FlowOut",
        "name": FLOW,
        "var_type": "bool",
        "var_fed_default": False,
        "component_authorized": [{"class_name_bkd": ".*"}],
        "var_is_active_default": True,
        "var_fed_available_out_init": True,
        "var_fed_available_out_reset": reset,
        "var_prod_cond_inner_mode": "or",
        "var_prod_default": produced,
        "negate": False,
    }


def _declared_mode(name, ttf, ttr, target, variable, clamp, nominal, behaviour=None):
    entry = {
        "name": f"{target}__{name}",
        "kind": "two_state_mode",
        "cls": "ObjMode2S",
        "mode_name": name,
        "targets": [target],
        "target_name": target,
        "occ_law": {"cls": "delay", "time": [ttf]},
        "occ_param_name": ["ttf"],
        "occ_param": [ttf],
        "not_occ_state": "rep",
        "not_occ_law": {"cls": "delay", "time": [ttr]},
        "not_occ_param_name": ["ttr"],
        "not_occ_param": [ttr],
        "occ_effects": {variable: clamp},
        "not_occ_effects_trans": {variable: nominal},
    }
    if behaviour is not None:
        entry["behaviour"] = behaviour
    return entry


def _declared_timeline(components, component, variable, connections=()):
    document = {
        "version": "1.0.0",
        "name": "one_shot",
        "components": {entry["name"]: entry for entry in components},
        "connections": list(connections),
    }
    body = declare.build_document(document)
    body = body.get("model", body)
    body["indicators"] = [
        {
            "name": "v",
            "target": "attribute",
            "attr": {"component": component, "attribute": variable},
        }
    ]
    result = pyraichu.simulate(
        pyraichu.load_model(body), t_max=S_GATE[-1], samples=S_GATE
    )
    return [float(v) for _, v in result.samples["v"]]


@pytest.mark.parametrize(
    "reset, two, expected",
    [
        (False, False, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        (False, True, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        (True, True, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
    ids=["persistent-one", "persistent-two", "reinitialized-two"],
)
def test_an_availability_gate_is_handed_back(reset, two, expected):
    """Up to 0.91.0 a persistent gate refused a held write beside a one-shot
    one, and two modes holding it. Measured on muscadet 5.14.1."""
    source = {
        "name": "S",
        "cls": "ObjFlow",
        "flows": [_flow_out(True, True)],
        "capacities": [],
        "failure_modes": [],
    }
    block = {
        "name": "B",
        "cls": "ObjFlow",
        "flows": [
            {
                "cls": "FlowIn",
                "name": FLOW,
                "var_type": "bool",
                "var_fed_default": False,
                "component_authorized": [{"class_name_bkd": ".*"}],
                "var_in_default": False,
                "var_available_in_default": True,
                "logic": "or",
            },
            dict(
                _flow_out(False, reset), var_prod_cond=[[{"name": FLOW, "port": "in"}]]
            ),
        ],
        "capacities": [],
        "failure_modes": [],
    }
    modes = [_declared_mode("cell", 1.0, 2.0, "B", GATE, False, True)]
    if two:
        modes.append(_declared_mode("bms", 2.0, 1.25, "B", GATE, False, True))
    connection = {
        "source": "S",
        "source_box": f"{FLOW}_out",
        "target": "B",
        "target_box": f"{FLOW}_in",
    }
    assert (
        _declared_timeline([source, block, *modes], "B", GATE, [connection]) == expected
    )


@pytest.mark.parametrize("behaviour", BEHAVIOURS)
def test_a_blinded_controller_output_is_handed_back(behaviour):
    """The platform's controller endpoint, which muscadet never reinitializes."""
    controller = {
        "name": "CTRL",
        "kind": "controller",
        "cls": "ObjCtrl",
        "source_cls": "ObjCtrl",
        "controls_in": [],
        "controls_out": [{"name": "fill", "kind": "bool", "default": False}],
    }
    modes = [
        _declared_mode(
            "cell", 1.0, 2.0, "CTRL", "fill_signal_available", False, True, behaviour
        ),
        _declared_mode(
            "bms", 2.0, 1.25, "CTRL", "fill_signal_available", False, True, behaviour
        ),
    ]
    got = _declared_timeline([controller, *modes], "CTRL", "fill_signal_available")
    assert got == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
