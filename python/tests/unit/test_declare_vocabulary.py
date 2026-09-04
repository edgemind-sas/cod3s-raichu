"""The muscadet declaration vocabulary, validated as DATA.

A muscadet component is normally a subclass of `ObjFlow` overriding
`add_flows`. That subclass is almost never behaviour: it declares flows,
capacities, measurement channels, rule sets, transfer pairs and failure
modes, and every one of those is a declaration a mapping can carry. This
suite pins the reading side of that mapping, before anything is built.

What it pins, beyond the arithmetic of the keys:

- the vocabulary is **muscadet's own**, section for section and key for
  key, rather than a parallel one invented here. A declaration written
  for muscadet is the declaration this layer reads;
- everything checkable from the mapping alone is checked **without an
  engine being raised**, so a caller validating a batch of declarations
  learns that one of them is misspelled without building the other
  forty;
- anything the vocabulary cannot carry is refused **by name**. A key
  swallowed in silence is a component quietly missing its rule set, and
  that is indistinguishable from one that never had any;
- a Python callable is refused wherever it sits. muscadet's own
  read-back refuses the same four shapes, and this layer refuses them
  one step earlier, at validation rather than at construction.
"""

import pytest

import pyraichu.declare as declare
import pyraichu.muscadet as mu

# --- the reference declaration ----------------------------------------

#: The heat pump of muscadet's own `declare` documentation: two
#: continuous flows, two discrete flows and one rule set turning two
#: units of electricity into seven of heat while the call for heat
#: holds.
HEAT_PUMP = {
    "name": "PUMP",
    "flows": [
        {"cls": "FlowContinuousIn", "name": "elec"},
        {"cls": "FlowContinuousOut", "name": "heat"},
        {"cls": "FlowIn", "name": "call", "logic": "or"},
        {"cls": "FlowOut", "name": "healthy", "var_prod_default": True},
    ],
    "rules": [
        {
            "name": "heat_pump",
            "rules": [
                {"cond": ["call"], "cons": {"elec": 2.0}, "prod": {"heat": 7.0}},
            ],
        }
    ],
}


def a_heat_pump(**overrides) -> dict:
    """The reference declaration, with the named sections replaced."""
    spec = {key: value for key, value in HEAT_PUMP.items()}
    spec.update(overrides)
    return spec


# --- no engine is raised ----------------------------------------------


def test_a_well_formed_declaration_validates_without_raising_an_engine(monkeypatch):
    """The point of validating as data: a batch reports its misspelling
    without forty components being built to find it."""
    built = []
    original = mu.ObjFlow._init_declarations

    def spy(self):
        built.append(self)
        return original(self)

    monkeypatch.setattr(mu.ObjFlow, "_init_declarations", spy)

    assert declare.check_spec(HEAT_PUMP) == "PUMP"
    assert built == [], "check_spec built a component"


def test_validation_does_not_write_through_the_declaration():
    """A caller keeps its declaration and may build it twice."""
    spec = a_heat_pump()
    before = repr(spec)
    declare.check_spec(spec)
    assert repr(spec) == before


# --- the shape of the mapping itself ----------------------------------


def test_a_declaration_that_is_not_a_mapping_is_refused():
    with pytest.raises(declare.ComponentSpecError, match="mapping"):
        declare.check_spec(["PUMP"])


def test_a_declaration_without_a_name_is_refused():
    spec = {key: value for key, value in HEAT_PUMP.items() if key != "name"}
    with pytest.raises(declare.ComponentSpecError, match="'name'"):
        declare.check_spec(spec)


def test_a_declaration_whose_name_is_empty_is_refused():
    with pytest.raises(declare.ComponentSpecError, match="'name'"):
        declare.check_spec(a_heat_pump(name=""))


# --- unknown and uncarried sections -----------------------------------


def test_an_unknown_section_is_refused_naming_it():
    """A misspelled section is otherwise swallowed whole, and a component
    silently missing its rule set reads exactly like one that never had
    any."""
    spec = a_heat_pump()
    spec["rulez"] = spec.pop("rules")

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "rulez" in str(raised.value)
    # And it says what it does accept, so the misspelling is fixable.
    assert "rules" in str(raised.value)


def test_a_published_measurement_is_refused_naming_the_section():
    """`measurements_out` has no counterpart here: a capacity publishes
    its own level and the system wires it."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(measurements_out=[{"name": "probe"}]))

    assert "measurements_out" in str(raised.value)


def test_a_declared_automaton_is_refused_naming_the_section():
    """`automata` has no counterpart either: this layer derives every
    automaton from the declaration that needs one."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(automata=[{"name": "aut"}]))

    assert "automata" in str(raised.value)


def test_an_empty_uncarried_section_says_nothing_and_is_accepted():
    """muscadet's read-back writes every section, empty ones included. A
    section carrying nothing declares nothing, so refusing it would
    refuse every read-back of a component that has none."""
    assert declare.check_spec(a_heat_pump(measurements_out=[], automata=[])) == "PUMP"


def test_the_derived_output_automata_flag_is_refused_when_it_asks_for_them():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(create_default_out_automata=True))

    assert "create_default_out_automata" in str(raised.value)


def test_the_derived_output_automata_flag_says_nothing_when_false():
    assert declare.check_spec(a_heat_pump(create_default_out_automata=False)) == "PUMP"


def test_the_class_a_declaration_was_read_back_from_is_informational():
    """`source_cls` is what muscadet writes so a template picker can show
    the class a spec came from. It is carried and ignored."""
    assert declare.check_spec(a_heat_pump(source_cls="HeatPump")) == "PUMP"


# --- section and entry shapes -----------------------------------------


def test_a_section_that_is_not_a_list_is_refused_naming_the_section():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(capacities="battery"))

    assert "capacities" in str(raised.value)


def test_a_section_entry_that_is_not_a_mapping_is_refused_naming_the_section():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(capacities=["battery"]))

    assert "capacities" in str(raised.value)


def test_a_single_mapping_is_the_short_form_of_a_one_entry_section():
    """muscadet accepts either; so does this."""
    spec = a_heat_pump()
    spec["rules"] = spec["rules"][0]
    assert declare.check_spec(spec) == "PUMP"


# --- the first fault, in declaration order ----------------------------


def test_validation_reports_the_first_fault_in_declaration_order():
    """The sections are checked in the order the build imposes, not in
    the order the mapping happens to list them: a declaration faulty in
    two places reports the fault the build would reach first."""
    spec = {
        "name": "PUMP",
        # Written first in the mapping, checked last of the two.
        "rules": ["not a mapping"],
        "flows": ["not a mapping either"],
    }

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "flows" in str(raised.value)
    assert "rules" not in str(raised.value)


def test_the_imposed_order_puts_the_flows_before_what_names_them():
    """The order is the whole difficulty and is written down once: a
    capacity, a failure mode and a rule all name flows, so the flows come
    first; a rule guard may gate on the automaton a failure mode
    declares, so the modes come before the rules."""
    order = list(declare.DECLARATION_SECTIONS)

    assert order.index("flows") < order.index("capacities")
    assert order.index("flows") < order.index("failure_modes")
    assert order.index("capacities") < order.index("rules")
    assert order.index("failure_modes") < order.index("rules")
    assert order.index("rules") < order.index("transfers")
    assert order.index("measurements_in") < order.index("rules")


# --- a Python callable, wherever it sits ------------------------------


def a_function(*args, **kwargs):  # pragma: no cover - never called
    return 0.0


def test_a_declaration_holding_a_callable_is_refused_naming_the_field():
    """The one shape a mapping cannot carry, and the one this vocabulary
    exists to catch before it reaches the engine."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {"cls": "FlowContinuousOut", "name": "heat", "profile": a_function},
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "profile" in str(raised.value)


def test_a_callable_buried_inside_a_rule_is_refused_naming_its_path():
    spec = a_heat_pump(
        rules=[
            {
                "name": "heat_pump",
                "rules": [{"cond": a_function, "cons": {"elec": 2.0}}],
            }
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "cond" in str(raised.value)


def test_an_allocation_split_written_as_a_python_function_is_refused_by_name():
    """muscadet's own Python extension point. Declaring it here would
    reopen the reproducibility this layer's expression trees protect, so
    it is refused rather than approximated by the nearest policy."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "allocation_fun": a_function,
            },
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "allocation_fun" in str(raised.value)


def test_a_reading_combination_written_as_a_python_function_is_refused_by_name():
    spec = a_heat_pump(
        measurements_in=[{"name": "store", "combine_fun": a_function}],
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "combine_fun" in str(raised.value)


def test_a_named_reading_combination_is_refused_by_name():
    """`combine` carries data rather than a function, and is still
    refused: a channel reading a median of three instruments and one
    reading a single publisher answer differently, so accepting the key
    and summing anyway would be the silent loss the refusal exists to
    close."""
    spec = a_heat_pump(measurements_in=[{"name": "store", "combine": "median"}])

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "combine" in str(raised.value)


def test_a_bare_transfer_equation_is_refused_naming_the_shapes_that_are_declarable():
    """`Transfer` is the base carrying a Python function; only the
    conductive law has a mapping form."""
    spec = a_heat_pump(
        transfers=[
            {
                "name": "wall",
                "flows": ["heat", "heat"],
                "equation": {"cls": "Transfer"},
            }
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "ConductiveTransfer" in str(raised.value)


def test_a_bare_profile_is_refused_naming_the_shape_that_is_declarable():
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "profile": {"cls": "Profile", "continuous": True},
            },
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "SinusoidalProfile" in str(raised.value)


# --- flow entries -----------------------------------------------------


def test_a_flow_entry_without_a_class_is_refused():
    spec = a_heat_pump(flows=[{"name": "elec"}])

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "cls" in str(raised.value)


def test_an_unknown_flow_class_is_refused_naming_the_classes_that_exist():
    spec = a_heat_pump(flows=[{"cls": "FlowContinuous", "name": "elec"}])

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "FlowContinuous" in str(raised.value)
    assert "FlowContinuousIn" in str(raised.value)


def test_the_canonical_and_legacy_flow_class_names_both_read():
    """muscadet accepts `FlowDiscreteIn` and its 1.x spelling `FlowIn`,
    and a declaration may carry either."""
    canonical = a_heat_pump(
        flows=[
            {"cls": "FlowDiscreteIn", "name": "call"},
            {"cls": "FlowDiscreteOut", "name": "healthy"},
        ],
        rules=[],
    )
    assert declare.check_spec(canonical) == "PUMP"


def test_a_flow_key_this_layer_cannot_carry_is_refused_by_name():
    """`var_demand_in_default` on an OUTPUT is muscadet's aggregated
    demand read when no consumer is connected. This layer derives that
    read from the connections, so the key would be dropped."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "var_demand_in_default": 3.0,
            },
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "var_demand_in_default" in str(raised.value)


def test_a_flow_key_that_says_nothing_is_accepted():
    """The same key at its muscadet default declares nothing, and a
    read-back writes every field: refusing it would refuse every spec
    muscadet produces."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec", "var_type": "float"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "var_demand_in_default": 0.0,
            },
        ]
    )

    assert declare.check_spec(spec) == "PUMP"


def test_a_negated_production_condition_operand_is_refused_by_name():
    """A discrete production condition here reads flow names and nothing
    else; muscadet's negation and threshold operands have no counterpart
    and are refused rather than dropped."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowIn", "name": "call"},
            {
                "cls": "FlowOut",
                "name": "healthy",
                "var_prod_cond": [[{"name": "call", "negate": True}]],
            },
        ],
        rules=[],
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "negate" in str(raised.value)


# --- initialisation parameters ----------------------------------------


def test_initialisation_parameters_on_the_bare_class_are_refused():
    """`ObjFlow.add_flows` reads none, so the keys would be dropped in
    silence."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(params={"rate": 2.0}))

    assert "rate" in str(raised.value)


def test_an_unknown_component_class_is_refused_naming_it():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(cls="SourceContinuous"))

    assert "SourceContinuous" in str(raised.value)


# --- failure modes ----------------------------------------------------


def test_a_failure_mode_of_an_unknown_kind_is_refused_naming_the_kinds():
    spec = a_heat_pump(
        failure_modes=[{"cls": "weibull", "name": "fm", "failure_rate": 1.0}]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "weibull" in str(raised.value)
    assert "delay" in str(raised.value)
    assert "exp" in str(raised.value)


def test_a_failure_mode_condition_tree_is_refused_by_name():
    """muscadet's `failure_cond` accepts a condition expression; here it
    names one local variable gating the failure, and anything richer is
    refused rather than reduced."""
    spec = a_heat_pump(
        failure_modes=[
            {
                "cls": "delay",
                "name": "fm",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_cond": [[{"obj": "PUMP", "attr": "call_fed_in"}]],
            }
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "failure_cond" in str(raised.value)


def test_a_failure_mode_condition_at_its_default_says_nothing():
    spec = a_heat_pump(
        failure_modes=[
            {
                "cls": "delay",
                "name": "fm",
                "failure_time": 1.0,
                "repair_time": 1.0,
                "failure_cond": True,
                "repair_cond": True,
                "failure_state": "occ",
                "repair_state": "rep",
            }
        ]
    )

    assert declare.check_spec(spec) == "PUMP"
