"""Building a component from a declaration, and the order that makes it work.

The companion of `test_declare_vocabulary`: that one pins what a declaration
may say, this one pins what building it produces.

What it pins, beyond the flows that come out:

- **the section order is imposed, not read off the mapping.** A capacity, a
  failure mode and a rule all name flows; a rule guard may gate on the
  automaton a failure mode declares; a transfer pair refuses a flow a rule
  already carries. Declared in any other order each of those refusals is
  unreachable, so the entry point orders the sections itself and a declaration
  written back to front builds the same component as one written in order;
- **what only the build can answer is answered at the build**, with the
  section and the entry named on top of the authoring layer's own diagnostic:
  a rule naming an undeclared flow, a capacity holding one, a conduit metering
  what a rule already carries;
- **a declaration and the class it was read back from build the same model.**
  That is the claim the whole declaration path rests on, and it is asserted
  here on the generated model rather than argued: the electrolysis plant is
  written twice, once as four subclasses and once as four declarations, and
  the two generated models are compared key for key.
"""

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu

from test_declare_vocabulary import HEAT_PUMP

# --- the reference declaration builds ---------------------------------


def test_the_heat_pump_declaration_builds_its_flows_and_its_rule_set():
    """Two continuous flows, two discrete flows and one rule set, from a
    mapping and no subclass."""
    system = mu.System(name="HEATING")
    pump = declare.build_component(system, HEAT_PUMP)

    assert [flow.name for flow in pump.flows_continuous_in] == ["elec"]
    assert [flow.name for flow in pump.flows_continuous_out] == ["heat"]
    assert [flow.name for flow in pump.flows_in] == ["call"]
    assert [flow.name for flow in pump.flows_out] == ["healthy"]
    assert [rule_set.name for rule_set in pump.rule_sets] == ["heat_pump"]

    assert system.comp["PUMP"] is pump


def test_the_declared_component_reaches_the_generated_model():
    """A declaration is not a document: what it builds is generated and
    simulated like anything else this layer authors."""
    system = mu.System(name="HEATING")
    declare.build_component(system, HEAT_PUMP)
    declare.build_component(
        system,
        {
            "name": "GRID",
            "flows": [
                {
                    "cls": "FlowContinuousOut",
                    "name": "elec",
                    "var_fed_default": 2.0,
                },
                {"cls": "FlowOut", "name": "call", "var_prod_default": True},
            ],
        },
    )
    declare.build_component(
        system,
        {
            "name": "ROOM",
            "flows": [
                {
                    "cls": "FlowContinuousIn",
                    "name": "heat",
                    "var_demand_default": 100.0,
                }
            ],
        },
    )
    system.connect("GRID", "elec", "PUMP", "elec")
    system.connect("GRID", "call", "PUMP", "call")
    system.connect("PUMP", "heat", "ROOM", "heat")

    result = system.simulate(t_max=5.0)

    # The grid can deliver two units of electricity and the room asks for
    # more heat than the pump can make, so the rule runs at scale one: two
    # of electricity in, seven of heat out.
    assert result.indicators["PUMP_heat_fed_out"][-1][1] == pytest.approx(7.0)
    assert result.indicators["PUMP_elec_fed_in"][-1][1] == pytest.approx(2.0)


def test_the_declaration_entry_point_sits_beside_the_class_based_one():
    system = mu.System(name="HEATING")
    pump = system.add_declared_component(HEAT_PUMP)

    assert system.comp["PUMP"] is pump
    assert [flow.name for flow in pump.flows_continuous_out] == ["heat"]


# --- the order is imposed ---------------------------------------------

#: The heat pump again, with every section written in the order that would
#: break it: the rule set before the flows it names, the transfer pair
#: before the rule set that refuses it.
HEAT_PUMP_BACK_TO_FRONT = {
    "transfers": [],
    "rules": HEAT_PUMP["rules"],
    "capacities": [],
    "flows": HEAT_PUMP["flows"],
    "name": "PUMP",
}


def test_sections_declared_out_of_order_still_build():
    """The entry point imposes the order, so a mapping is a set of
    sections and not a sequence of calls."""
    ordered = mu.System(name="ORDERED")
    declare.build_component(ordered, HEAT_PUMP)

    shuffled = mu.System(name="ORDERED")
    declare.build_component(shuffled, HEAT_PUMP_BACK_TO_FRONT)

    assert shuffled.build_dict() == ordered.build_dict()


def test_a_failure_mode_is_declared_before_the_rule_guard_that_gates_on_it():
    """The order this layer departs from muscadet's on: a rule guard
    resolves the automaton it names at declaration time, and a failure
    mode is what declares that automaton."""
    system = mu.System(name="PLANT")
    component = declare.build_component(
        system,
        {
            "name": "STACK",
            # Deliberately written before the failure mode it gates on.
            "rules": [
                {
                    "name": "reaction",
                    "rules": [
                        {
                            "cond": [{"automaton": "df", "state": "ok"}],
                            "cons": {"water": 1.0},
                            "prod": {"gas": 1.0},
                        }
                    ],
                }
            ],
            "failure_modes": [
                {
                    "cls": "delay",
                    "name": "df",
                    "failure_time": 2.0,
                    "repair_time": 2.0,
                    "failure_effects": [[".*", 0.0]],
                }
            ],
            "flows": [
                {"cls": "FlowContinuousIn", "name": "water"},
                {"cls": "FlowContinuousOut", "name": "gas"},
            ],
        },
    )

    assert [mode.name for mode in component.failure_modes] == ["df"]
    assert [rule_set.name for rule_set in component.rule_sets] == ["reaction"]


# --- what only the build can answer -----------------------------------


def test_a_rule_naming_an_undeclared_flow_is_refused_naming_the_rule_and_the_flow():
    system = mu.System(name="HEATING")
    spec = dict(
        HEAT_PUMP,
        rules=[
            {
                "name": "heat_pump",
                "rules": [
                    {"cond": ["call"], "cons": {"steam": 2.0}, "prod": {"heat": 7.0}}
                ],
            }
        ],
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(system, spec)

    message = str(raised.value)
    assert "heat_pump" in message, message
    assert "steam" in message, message
    assert "rules" in message, message


def test_a_capacity_holding_an_undeclared_flow_is_refused_naming_the_section():
    system = mu.System(name="HEATING")
    spec = dict(
        HEAT_PUMP,
        capacities=[{"name": "buffer", "flow": "steam", "capacity": 10.0}],
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(system, spec)

    message = str(raised.value)
    assert "capacities" in message, message
    assert "buffer" in message, message
    assert "steam" in message, message


def test_a_name_the_system_already_holds_is_refused():
    """A duplicate instance name is the most likely defect in a generated
    study, so it is the one shape that has to name itself rather than
    silently replace what was there."""
    system = mu.System(name="HEATING")
    declare.build_component(system, HEAT_PUMP)

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(system, HEAT_PUMP)

    assert "PUMP" in str(raised.value)


# --- a class stays usable as a template -------------------------------


class SourceContinuous(mu.ObjFlow):
    """A shipped-style component: one continuous output at a declared rate."""

    def add_flows(self, flow="out", rate=0.0, **kwargs):
        self.add_flow_continuous_out(name=flow, var_fed_default=rate)


def test_the_class_based_entry_point_still_works():
    system = mu.System(name="PLANT")
    source = system.add_component(SourceContinuous, "S")

    assert [flow.name for flow in source.flows_continuous_out] == ["out"]
    assert source.flows_continuous_out[0].var_fed_default == 0.0


def test_a_declaration_carrying_initialisation_parameters_builds_with_them():
    system = mu.System(name="PLANT")
    source = declare.build_component(
        system,
        {
            "name": "S_H2O",
            "cls": SourceContinuous,
            "params": {"flow": "H2O", "rate": 2.0},
        },
    )

    assert [flow.name for flow in source.flows_continuous_out] == ["H2O"]
    assert source.flows_continuous_out[0].var_fed_default == pytest.approx(2.0)


def test_a_registered_class_is_reachable_by_name():
    """The seam a knowledge base of shipped classes plugs into: a
    declaration held in data names its class as a string."""
    declare.register_component_class(SourceContinuous)
    try:
        system = mu.System(name="PLANT")
        source = declare.build_component(
            system,
            {
                "name": "S_H2O",
                "cls": "SourceContinuous",
                "params": {"flow": "H2O", "rate": 2.0},
            },
        )
        assert [flow.name for flow in source.flows_continuous_out] == ["H2O"]
    finally:
        declare.COMPONENT_CLASSES.pop("SourceContinuous", None)


def test_the_sections_are_added_on_top_of_what_the_class_declared():
    """A class declares its ports first, so a shipped source plus one
    discrete output is a declaration and not a subclass."""
    system = mu.System(name="PLANT")
    source = declare.build_component(
        system,
        {
            "name": "S_H2O",
            "cls": SourceContinuous,
            "params": {"flow": "H2O", "rate": 2.0},
            "flows": [{"cls": "FlowOut", "name": "healthy", "var_prod_default": True}],
        },
    )

    assert [flow.name for flow in source.flows_continuous_out] == ["H2O"]
    assert [flow.name for flow in source.flows_out] == ["healthy"]


def test_a_shipped_component_read_back_as_primitives_builds_the_same_model():
    """The claim the migration path rests on: a class expanded into the
    primitives it declares generates the model the class itself does."""
    authored = mu.System(name="PLANT")
    authored.add_component(SourceContinuous, "S")

    declared = mu.System(name="PLANT")
    declare.build_component(
        declared,
        {
            "name": "S",
            "source_cls": "SourceContinuous",
            "flows": [
                {"cls": "FlowContinuousOut", "name": "out", "var_fed_default": 0.0}
            ],
        },
    )

    assert declared.build_dict() == authored.build_dict()


# --- the electrolysis plant, both ways --------------------------------
#
# muscadet's four-component reference model: a water source, a battery, an
# electrolyser stack turning four water and one electricity into one hydrogen
# and one oxygen, and a local hydrogen store. The stack carries a delay failure
# mode derating every continuous output to zero while it stands.

H2_SOURCE_RATE = 2.0
H2_BATTERY_CAPACITY = 100.0
H2_BATTERY_CONTENT = 100.0
#: What the battery can deliver. The reference publishes an unbounded
#: capability; this layer writes a model document in strict JSON, which
#: carries no infinity, so the rate is named and chosen well above the 0.5
#: the reaction draws: the water stays the limiting reagent either way.
H2_BATTERY_RATE = 10.0
H2_TANK_CAPACITY = 6.0
H2_TANK_CONTENT = 3.0
H2_TANK_FILL_RATE = 1.0
H2_CONS = {"H2O": 4.0, "Elec": 1.0}
H2_PROD = {"H2": 1.0, "O2": 1.0}
H2_FAILURE_TIME = 2.0
H2_REPAIR_TIME = 2.0


class WaterSource(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="H2O", var_fed_default=H2_SOURCE_RATE)


class Battery(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_out(name="Elec", var_fed_default=H2_BATTERY_RATE)
        self.add_capacity(
            name="battery",
            flow="Elec",
            capacity=H2_BATTERY_CAPACITY,
            content_init={"Elec": H2_BATTERY_CONTENT},
        )


class Electrolyser(mu.ObjFlow):
    def add_flows(self):
        for flow in H2_CONS:
            self.add_flow_continuous_in(name=flow)
        for flow in H2_PROD:
            self.add_flow_continuous_out(name=flow)
        self.add_delay_failure_mode(
            name="df_H2",
            failure_time=H2_FAILURE_TIME,
            repair_time=H2_REPAIR_TIME,
            failure_effects=[(".*", 0.0)],
        )
        self.add_rule_set(
            name="electrolysis",
            rules=[{"name": "electrolysis", "cons": H2_CONS, "prod": H2_PROD}],
        )


class LocalStore(mu.ObjFlow):
    def add_flows(self):
        self.add_flow_continuous_in(name="H2")
        self.add_flow_continuous_out(name="H2")
        self.add_capacity(
            name="tank",
            flow="H2",
            side="in",
            capacity=H2_TANK_CAPACITY,
            content_init={"H2": H2_TANK_CONTENT},
            fill_rate=H2_TANK_FILL_RATE,
        )


#: The same four components, as declarations. Every number is read off the
#: constants above, so the two authorings cannot drift apart on a value: what
#: is being compared is the declaration path, not the arithmetic.
H2_DECLARATIONS = [
    {
        "name": "S_H2O",
        "source_cls": "SourceContinuous",
        "flows": [
            {
                "cls": "FlowContinuousOut",
                "name": "H2O",
                "var_fed_default": H2_SOURCE_RATE,
            }
        ],
    },
    {
        "name": "B1",
        "source_cls": "CapacityContinuous",
        "flows": [
            {
                "cls": "FlowContinuousOut",
                "name": "Elec",
                "var_fed_default": H2_BATTERY_RATE,
            }
        ],
        "capacities": [
            {
                "name": "battery",
                "flow": "Elec",
                "capacity": H2_BATTERY_CAPACITY,
                "content_init": {"Elec": H2_BATTERY_CONTENT},
            }
        ],
    },
    {
        "name": "Electro",
        "source_cls": "TransformerContinuous",
        # Written after the sections that name them, to exercise the
        # imposed order on the reference model itself.
        "rules": [
            {
                "name": "electrolysis",
                "rules": [{"name": "electrolysis", "cons": H2_CONS, "prod": H2_PROD}],
            }
        ],
        "failure_modes": [
            {
                "cls": "delay",
                "name": "df_H2",
                "failure_time": H2_FAILURE_TIME,
                "repair_time": H2_REPAIR_TIME,
                "failure_effects": [[".*", 0.0]],
            }
        ],
        "flows": [{"cls": "FlowContinuousIn", "name": flow} for flow in H2_CONS]
        + [{"cls": "FlowContinuousOut", "name": flow} for flow in H2_PROD],
    },
    {
        "name": "Local",
        "source_cls": "CapacityContinuous",
        "flows": [
            {"cls": "FlowContinuousIn", "name": "H2"},
            {"cls": "FlowContinuousOut", "name": "H2"},
        ],
        "capacities": [
            {
                "name": "tank",
                "flow": "H2",
                "side": "in",
                "capacity": H2_TANK_CAPACITY,
                "content_init": {"H2": H2_TANK_CONTENT},
                "fill_rate": H2_TANK_FILL_RATE,
            }
        ],
    },
]


def wire_the_plant(system: mu.System) -> mu.System:
    """The three connections the reference declares. O2 is produced and
    deliberately wired to nothing, as in the reference."""
    system.connect("S_H2O", "H2O", "Electro", "H2O")
    system.connect("B1", "Elec", "Electro", "Elec")
    system.connect("Electro", "H2", "Local", "H2")
    return system


def the_plant_by_class() -> mu.System:
    system = mu.System(name="H2StackSys")
    system.add_component(WaterSource, "S_H2O")
    system.add_component(Battery, "B1")
    system.add_component(Electrolyser, "Electro")
    system.add_component(LocalStore, "Local")
    return wire_the_plant(system)


def the_plant_by_declaration() -> mu.System:
    system = mu.System(name="H2StackSys")
    for spec in H2_DECLARATIONS:
        system.add_declared_component(spec)
    return wire_the_plant(system)


def test_the_electrolysis_plant_declarations_validate_without_building():
    assert [declare.check_spec(spec) for spec in H2_DECLARATIONS] == [
        "S_H2O",
        "B1",
        "Electro",
        "Local",
    ]


def test_the_electrolysis_plant_generates_the_same_model_either_way():
    """Four components, no subclass: the strongest claim the declaration
    path can make, asserted on the generated model."""
    assert the_plant_by_declaration().build_dict() == the_plant_by_class().build_dict()


def test_the_electrolysis_plant_simulates_the_same_either_way():
    """Equality of the generated model already implies it; simulating both
    is what makes a divergence read as a number rather than as a diff."""
    observed = (
        "Electro_H2_fed_out",
        "Electro_O2_fed_out",
        "Electro_H2O_fed_in",
        "Electro_Elec_fed_in",
        "Local_tank_content_H2",
        "B1_battery_content_Elec",
    )

    by_class = the_plant_by_class().simulate(t_max=5.0)
    by_declaration = the_plant_by_declaration().simulate(t_max=5.0)

    for indicator in observed:
        assert by_declaration.indicators[indicator] == by_class.indicators[indicator]


def test_the_water_holds_the_reaction_to_half_a_unit():
    """The reference model's own figure, reproduced through the
    declaration path: `min(H2O/4, Elec/1)` with water arriving at 2, so
    the stack runs at 0.5 however full the battery."""
    result = the_plant_by_declaration().simulate(t_max=1.0)
    scale = H2_SOURCE_RATE / H2_CONS["H2O"]

    assert scale == pytest.approx(0.5)
    assert result.indicators["Electro_H2_fed_out"][-1][1] == pytest.approx(
        scale * H2_PROD["H2"]
    )
    # The oxygen is wired to nothing, so nothing is delivered on it: what is
    # asserted here is the reaction's scale, and the two authorings agreeing
    # on the oxygen is the equivalence test's business, not this one's.


# --- the production-condition convention ------------------------------
#
# The one conversion this path cannot get away with reading loosely.
# muscadet normalises a production condition into CONJUNCTIVE normal form: a
# flat list is one clause whose operands are OR-ed, and a list of lists is the
# AND of those clauses. The authoring layer underneath reads a list of groups
# the other way round, as the OR of conjunctions. Passed through untouched,
# `["a", "b"]` would mean `a or b` on one side and `a and b` on the other, and
# nothing would say so.


def a_gated_output(condition) -> dict:
    """A component whose one discrete output is gated on `condition`,
    over three inputs."""
    return {
        "name": "GATE",
        "flows": [{"cls": "FlowIn", "name": name} for name in ("a", "b", "c")]
        + [{"cls": "FlowOut", "name": "out", "var_prod_cond": condition}],
    }


def gate_output(condition, fed: tuple[str, ...]) -> bool:
    """Whether the gated output feeds when exactly `fed` are supplied."""
    system = mu.System(name="PROD_COND")
    system.add_declared_component(a_gated_output(condition))
    for name in fed:
        system.add_declared_component(
            {
                "name": f"SRC_{name}",
                "flows": [{"cls": "FlowOut", "name": name, "var_prod_default": True}],
            }
        )
        system.connect(f"SRC_{name}", name, "GATE", name)

    result = system.simulate(t_max=1.0)
    return result.indicators["GATE_out_fed_out"][-1][1]


def test_a_flat_production_condition_is_the_conjunction_muscadet_reads():
    """A flat list is N clauses of one operand each, AND-ed.

    muscadet's own docstring says otherwise, calling a list of strings a
    disjunction, but its normalisation makes every top-level element a
    clause of its own and the clauses are conjoined. The code is the
    authority here, and this is the case a reader is most likely to get
    backwards."""
    assert gate_output(["a", "b"], fed=("a",)) is False
    assert gate_output(["a", "b"], fed=("a", "b")) is True


def test_one_clause_of_two_operands_is_the_disjunction():
    """The nesting is what makes a disjunction, and it is exactly the
    nesting the layer underneath reads the other way round."""
    assert gate_output([["a", "b"]], fed=("a",)) is True
    assert gate_output([["a", "b"]], fed=("b",)) is True
    assert gate_output([["a", "b"]], fed=()) is False


def test_a_mixed_condition_is_expanded_clause_by_clause():
    """`(a or b) and c`, which the authoring layer underneath can only
    hold as `(a and c) or (b and c)`."""
    condition = [["a", "b"], ["c"]]

    assert gate_output(condition, fed=("a",)) is False
    assert gate_output(condition, fed=("c",)) is False
    assert gate_output(condition, fed=("a", "c")) is True
    assert gate_output(condition, fed=("b", "c")) is True


def test_an_operand_mapping_is_the_same_condition_as_its_short_form():
    """muscadet's read-back always writes the mapping form, so the two
    have to mean one thing."""
    assert gate_output([[{"name": "a"}], [{"name": "b"}]], fed=("a",)) is False
    assert gate_output([[{"name": "a"}], [{"name": "b"}]], fed=("a", "b")) is True
