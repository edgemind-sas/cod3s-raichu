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

import json

import pytest

import pyraichu
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


def test_the_derived_output_automata_flag_lets_a_declaration_through():
    """muscadet writes `create_default_out_automata` ONLY when it is `True`
    (its read-back writes "what says something"), so a key refused at that
    value is a key refused on every export that carries it. The refusal moved
    onto what its absence takes away -- an indicator naming one of the derived
    states -- and the declaration itself passes."""
    assert declare.check_spec(a_heat_pump(create_default_out_automata=True)) == "PUMP"


def test_the_derived_output_automata_flag_says_nothing_when_false():
    assert declare.check_spec(a_heat_pump(create_default_out_automata=False)) == "PUMP"


def test_the_derived_pair_is_named_after_each_discrete_output():
    """What the key asks muscadet for, spelled as muscadet spells it: one
    two-state automaton per DISCRETE output, holding `{flow}_ok` and
    `{flow}_nok`. The continuous outputs carry none, and neither do the
    inputs."""
    assert declare.derived_out_states(
        a_heat_pump(create_default_out_automata=True)
    ) == {"healthy_ok", "healthy_nok"}


def test_a_declaration_that_does_not_ask_for_the_pair_names_no_state():
    for spec in (a_heat_pump(), a_heat_pump(create_default_out_automata=False)):
        assert declare.derived_out_states(spec) == set()


def test_the_class_a_declaration_was_read_back_from_is_informational():
    """`source_cls` is what muscadet writes so a template picker can show
    the class a spec came from. It is carried and ignored."""
    assert declare.check_spec(a_heat_pump(source_cls="HeatPump")) == "PUMP"


# --- a mixture group: one volumetric rate for several constituents ----

#: A group as muscadet's `add_mixture_in` declares one (R51): the flows
#: are drawn TOGETHER, and `flow_rate` is one VOLUME per unit of time for
#: the whole group, not a rate per flow. What leaves per constituent is
#: not declarable at all -- it is fixed by the composition of the volume
#: drawn from.
MIXTURE_GROUP = {"name": "extract", "flows": ["elec"], "flow_rate": 50.0}


def test_the_mixture_section_muscadet_writes_on_every_component_is_accepted_empty():
    """muscadet 5.4.0 writes `mixtures` on EVERY flow component, `[]`
    included and without pruning. So a reader refusing the key by name
    refuses every document that muscadet exports, at the first component
    it meets, whether or not anything in the model ventilates."""
    assert declare.check_spec(a_heat_pump(mixtures=[])) == "PUMP"


def test_a_declared_mixture_group_is_refused_naming_the_section():
    """The other half, and the reason the acceptance above is not a
    silence: a group is one rate for several constituents, and every rate
    this layer carries is a rate per flow. Read as two demands it would
    give the model two degrees of freedom where the physics has one, and
    return a trajectory that is wrong rather than absent."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(a_heat_pump(mixtures=[MIXTURE_GROUP]))

    message = str(raised.value)
    assert "mixtures" in message
    assert "PUMP" in message
    # The mechanism it names in muscadet, so the refusal is traceable to
    # the declaration that provoked it...
    assert "add_mixture_in" in message
    # ...and what the seam does not carry, rather than a bare "unknown".
    assert "unknown declaration key" not in message


def test_a_mixture_group_is_refused_on_the_build_too_and_not_only_on_the_check():
    """`check_spec` and `build_component` share one expansion, so a
    refusal reachable from the mapping cannot be validated away by
    building instead of checking."""
    system = mu.System(name="ventilated")

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(system, a_heat_pump(mixtures=[MIXTURE_GROUP]))

    assert "mixtures" in str(raised.value)


def test_the_empty_mixture_section_builds_the_model_it_built_before_the_key():
    """What "accepted" has to mean: not that the document loads, but that
    it loads as the SAME model. The two documents differ by the section
    muscadet added and by nothing else, so the bodies they build are
    identical byte for byte -- an acceptance that quietly changed a
    default would pass a load and fail here."""
    before = {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "ventilated",
        "components": {"PUMP": a_heat_pump()},
        "connections": [],
    }
    after = {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "ventilated",
        "components": {"PUMP": a_heat_pump(mixtures=[])},
        "connections": [],
    }

    assert json.dumps(declare.build_document(after), sort_keys=True) == json.dumps(
        declare.build_document(before), sort_keys=True
    )


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


# --- the label a profile carries --------------------------------------
#
# muscadet serializes a declared object by its constructor's own parameter
# names, read off the signature, and `Profile.__init__` takes a `name`
# defaulted to the class name. EVERY exported profile therefore carries the
# key, whether or not a modeller wrote it, and a reader with no name for it
# refused every document declaring a time profile -- the same shape of gate
# `mixtures` was on a component, one level down.


def a_profiled_pump(**profile) -> dict:
    """The reference declaration whose output declares a time profile."""
    return a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "profile": {"cls": "SinusoidalProfile", **profile},
            },
        ],
        rules=[],
    )


def test_the_label_muscadet_writes_on_every_profile_is_read():
    """`{"cls": "SinusoidalProfile", "name": "SinusoidalProfile"}` is what a
    read-back writes for a profile nobody labelled, and it is the shape a
    platform export carries."""
    built = declare.build_component(
        mu.System("read"), a_profiled_pump(name="SinusoidalProfile", period=24.0)
    )

    assert built.flows_continuous_out[0].profile.name == "SinusoidalProfile"


def test_a_modeller_s_own_label_reaches_the_reader():
    built = declare.build_component(
        mu.System("read"), a_profiled_pump(name="solar curve", period=24.0)
    )

    assert built.flows_continuous_out[0].profile.name == "solar curve"


def test_a_profile_with_no_label_is_named_after_its_shape():
    """muscadet's own default, `name or type(self).__name__`, mirrored: a
    read-back of what this layer read says what muscadet's read-back said."""
    built = declare.build_component(mu.System("read"), a_profiled_pump(period=24.0))

    assert built.flows_continuous_out[0].profile.name == "SinusoidalProfile"


def test_a_labelled_profile_names_itself_in_a_refusal():
    """What the label BUYS here: a component declaring a curve per flow says
    which of them it wrote wrong, where naming the flow alone leaves a
    modeller counting mappings."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(
            mu.System("read"), a_profiled_pump(name="solar curve", period=0.0)
        )

    assert "`solar curve`" in str(raised.value)


def test_an_unlabelled_profile_is_refused_without_repeating_its_shape():
    """The default label is the shape's own name, and a message reading
    "time profile `SinusoidalProfile` of period 0" would say it twice."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(mu.System("read"), a_profiled_pump(period=0.0))

    message = str(raised.value)
    assert "of period 0" in message and "`SinusoidalProfile`" not in message


def test_a_label_that_is_not_a_string_is_refused_naming_the_key():
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(mu.System("read"), a_profiled_pump(name=7))

    assert "`name`" in str(raised.value)


def test_the_vocabulary_opened_on_ONE_key_and_not_on_whatever_a_mapping_holds():
    """A closed vocabulary opens by name. The refusal lists what it accepts,
    the label among them, so the next key muscadet grows is found here rather
    than by a corpus that stops reading."""
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.build_component(mu.System("read"), a_profiled_pump(colour="amber"))

    message = str(raised.value)
    assert "['colour']" in message and "name" in message


def test_the_label_is_the_last_key_of_the_declared_vocabulary():
    """Read as DATA so the bench can confront it with what muscadet writes
    (`test_muscadet_declaration_vocabulary`), rather than restating a tuple
    buried in the parser."""
    assert mu.PROFILE_KEYS == (
        "amplitude",
        "period",
        "phase_shift",
        "offset",
        "value_min",
        "value_max",
        "name",
    )


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


def test_a_runtime_state_handle_is_dropped_rather_than_read():
    """`trigger_up` is not a declaration: muscadet binds the automaton's `up`
    STATE onto the flow when it builds it (`FlowDiscreteOutOnTrigger`), and
    its read-back writes the object out because the name carries neither the
    `var_` nor the `sm_` prefix its runtime-handle rule looks for.

    This layer derives the automaton and names the state itself, so there is
    nothing in the handle to honour and nothing to lose. Read as a
    declaration, it took every model carrying a triggered output down -- which
    is `examples/isimu/power_plant`, whose backup pump is one.
    """
    spec = a_heat_pump(
        flows=[
            {
                "cls": "FlowOutOnTrigger",
                "name": "healthy",
                "trigger_time_up": 0.0,
                "trigger_time_down": 0.0,
                "trigger_logic": "and",
                "trigger_up": {"name": "up", "cls": "StateModel"},
            }
        ],
        rules=[],
    )
    assert declare.check_spec(spec) == "PUMP"


def test_a_boundary_input_declared_on_a_discrete_flow_is_carried():
    """`var_in_default` on a BOOLEAN input is what an unconnected input
    reads, and `True` is how a model grounds a chain at its physical edge:
    an external supply, a utility nobody modelled.

    It used to be inert here, accepted at muscadet's `False` and refused
    at `True`, which refused the whole component rather than the one
    field -- and refused exactly the models whose own consistency check
    tells the modeller to mark those inputs. Carried now, on the four
    logics."""
    method, keywords = declare.entry_call(
        "FlowIn",
        {"cls": "FlowIn", "name": "call", "logic": "and", "var_in_default": True},
        where="Component PUMP: flow 'call'",
    )

    assert method == "add_flow_in"
    assert keywords["var_in_default"] is True


def test_the_availability_counterpart_stays_inert_at_its_neutral_value():
    """`var_available_in_default` is the same idea on the OTHER channel,
    and muscadet's default there, `True`, is its neutral element: out of
    connection the feed channel decides alone, and wired, a producer's
    `fed` already entails its `available`. So `True` says nothing and is
    accepted; `False` would say something this layer cannot honour and is
    refused by name."""
    accepted = a_heat_pump(
        flows=[
            {"cls": "FlowIn", "name": "call", "var_available_in_default": True},
            {"cls": "FlowOut", "name": "healthy"},
        ],
        rules=[],
    )
    assert declare.check_spec(accepted) == "PUMP"

    refused = a_heat_pump(
        flows=[
            {"cls": "FlowIn", "name": "call", "var_available_in_default": False},
            {"cls": "FlowOut", "name": "healthy"},
        ],
        rules=[],
    )
    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(refused)

    assert "var_available_in_default" in str(raised.value)


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


def test_a_production_condition_operand_key_outside_the_vocabulary_is_refused():
    """The three keys muscadet reads on such an operand -- `negate`, `op` and
    `value` -- are carried since 0.32.0 and have their own suite
    (`test_prod_cond_operands.py`). What is refused is what muscadet's own
    operand does NOT carry: `release` widens a comparison into a band, and a
    band has to be held between its two edges where the production variable is
    rewritten at every evaluation."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowIn", "name": "call"},
            {
                "cls": "FlowOut",
                "name": "healthy",
                "var_prod_cond": [
                    [{"name": "call", "op": ">", "value": 0.5, "release": 0.2}]
                ],
            },
        ],
        rules=[],
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "release" in str(raised.value)


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


# --- the four keys a real muscadet read-back writes and this layer read
#     past nothing ------------------------------------------------------
#
# Four keys a muscadet read-back writes on a system where a CONTROLLER
# commands a source were in no vocabulary at all, so an export carrying a
# capacity, or a controller that commands, was refused before it reached the
# engine. Two sit on the continuous flow vocabularies, and two are the class
# stamp muscadet writes INSIDE a rule set. The confrontation that keeps the
# four from being followed by a fifth is the cross-validation suite's
# `test_muscadet_declaration_vocabulary.py`, which needs muscadet; what is
# pinned here is the reading itself.


def test_the_inner_mode_of_a_production_condition_says_nothing_on_a_continuous_output():
    """A continuous output's production is derived from its rule sets, and
    this vocabulary carries no `var_prod_cond` at all: the mode that would
    combine that condition's operands says nothing at muscadet's own
    default."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "var_prod_cond_inner_mode": "or",
            },
        ]
    )

    assert declare.check_spec(spec) == "PUMP"


def test_another_inner_mode_on_a_continuous_output_is_refused_naming_the_default():
    """`and` is a different reading of a condition, and this layer reads
    none: it is refused, naming the key and the default it is accepted
    at."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec"},
            {
                "cls": "FlowContinuousOut",
                "name": "heat",
                "var_prod_cond_inner_mode": "and",
            },
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    message = str(raised.value)
    assert "var_prod_cond_inner_mode" in message
    assert "'and'" in message
    assert "'or'" in message


def test_what_a_continuous_input_was_fed_says_nothing_at_zero():
    """On an INPUT, `var_fed_default` seeds the `{name}_fed_in` MIRROR, which
    the sensitive method overwrites with what the connections delivered and
    which nothing on this side reads: a continuous input is read through its
    connections, precisely because the mirror lags a step."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec", "var_fed_default": 0.0},
            {"cls": "FlowContinuousOut", "name": "heat"},
        ]
    )

    assert declare.check_spec(spec) == "PUMP"


def test_a_continuous_input_fed_by_default_is_refused_rather_than_read_as_its_unconnected_value():
    """The seed is NOT another spelling of `var_in_default`, which muscadet
    documents as what an input reads when nothing is connected and which this
    vocabulary carries. Carrying one onto the other would start an
    unconnected input at a value the declaration never put there, so a
    non-zero seed is refused by name."""
    spec = a_heat_pump(
        flows=[
            {"cls": "FlowContinuousIn", "name": "elec", "var_fed_default": 4.0},
            {"cls": "FlowContinuousOut", "name": "heat"},
        ]
    )

    with pytest.raises(declare.ComponentSpecError) as raised:
        declare.check_spec(spec)

    assert "var_fed_default" in str(raised.value)


def test_what_a_continuous_output_was_fed_by_default_stays_carried():
    """The same key on an OUTPUT carries a decision and keeps carrying it:
    it is what a source with no rule set produces. The change is the input
    side and nothing else."""
    method, keywords = declare.entry_call(
        "FlowContinuousOut",
        {"cls": "FlowContinuousOut", "name": "heat", "var_fed_default": 7.0},
        where="Component PUMP: flow 'heat'",
    )

    assert method == "add_flow_continuous_out"
    assert keywords["var_fed_default"] == 7.0


def test_a_source_with_no_rule_set_produces_what_it_was_fed_by_default():
    """The decision the key carries on an OUTPUT, built and run rather than
    read off a keyword: a continuous output declared at 5.0 and reached by no
    rule delivers 5.0, and the input's own seed decides nothing."""
    document = {
        "version": declare.SYSTEM_SPEC_VERSION,
        "name": "seeded_source",
        "components": {
            "SRC": {
                "name": "SRC",
                "flows": [
                    {
                        "cls": "FlowContinuousOut",
                        "name": "q",
                        "var_fed_default": 5.0,
                    }
                ],
            },
            "SINK": {
                "name": "SINK",
                "flows": [
                    {
                        "cls": "FlowContinuousIn",
                        "name": "q",
                        "var_demand_default": 9.0,
                        # The input's seed, at the value that says nothing.
                        "var_fed_default": 0.0,
                    }
                ],
            },
        },
        "connections": [
            {
                "source": "SRC",
                "source_box": "q_out",
                "target": "SINK",
                "target_box": "q_in",
                "flow": "q",
            }
        ],
    }
    session = pyraichu.interactive(
        pyraichu.load_model(json.dumps(declare.build_document(document))), t_max=1.0
    )

    assert abs(session.attribute("SINK.q_fed_in") - 5.0) < 1e-9


# --- the class stamp muscadet writes inside a rule set ----------------


#: A rule set as muscadet's own read-back writes it: every object it
#: serializes carries a `cls` key naming its class, and a rule set is dumped
#: whole -- so the stamp lands on each RULE and on each OPERAND of a rule's
#: guard, two levels below the section entry where `_keywords` already drops
#: it.
STAMPED_RULE_SET = {
    "name": "heat_pump",
    "rules": [
        {
            "name": "on_call",
            "cond": [
                {
                    "name": "call",
                    "negate": False,
                    "port": "in",
                    "op": None,
                    "value": None,
                    "cls": "RuleOperand",
                }
            ],
            "cons": {"elec": 2.0},
            "prod": {"heat": 7.0},
            "cls": "Rule",
        }
    ],
}


def test_the_class_stamp_on_a_rule_and_on_its_guard_operand_is_ignored():
    """muscadet stamps a class name on everything it serializes. Knowing
    that is this reader's business, so the stamp is dropped here rather than
    accepted one layer down."""
    assert declare.check_spec(a_heat_pump(rules=[STAMPED_RULE_SET])) == "PUMP"


def test_the_stamped_rule_set_reaches_the_authoring_layer_without_its_stamp():
    """What `entry_call` hands on carries no `cls`, at either level: the
    reader strips it, and the writing surface never sees one."""
    method, keywords = declare.entry_call(
        "rules", dict(STAMPED_RULE_SET), where="Component PUMP: rule set 'heat_pump'"
    )

    assert method == "add_rule_set"
    rule = keywords["rules"][0]
    assert "cls" not in rule
    assert "cls" not in rule["cond"][0]
    # Everything else survives untouched, guard operand included.
    assert rule["name"] == "on_call"
    assert rule["cons"] == {"elec": 2.0}
    assert rule["cond"][0] == {
        "name": "call",
        "negate": False,
        "port": "in",
        "op": None,
        "value": None,
    }


def test_the_declaration_read_is_not_written_through():
    """The stamp is dropped from a COPY: a caller validating a batch of
    declarations gets its own mappings back as it wrote them."""
    declared = {
        "name": "heat_pump",
        "rules": [dict(STAMPED_RULE_SET["rules"][0])],
    }
    declared["rules"][0]["cond"] = [dict(STAMPED_RULE_SET["rules"][0]["cond"][0])]

    declare.entry_call("rules", declared, where="Component PUMP: rule set")

    assert declared["rules"][0]["cls"] == "Rule"
    assert declared["rules"][0]["cond"][0]["cls"] == "RuleOperand"


def test_the_writing_surface_of_pyraichu_did_not_learn_the_stamp():
    """The correction went into the READER. `add_rule_set` describes the
    rules pyraichu itself accepts, and a `cls` written there still names a
    key it does not know -- teaching it one it never means would widen that
    surface for a muscadet detail."""

    class Pump(mu.ObjFlow):
        def add_flows(self):
            self.add_flow_continuous_in(name="elec")
            self.add_flow_continuous_out(name="heat")
            self.add_flow_in(name="call")

    component = Pump(name="PUMP")

    with pytest.raises(ValueError) as raised:
        component.add_rule_set(**STAMPED_RULE_SET)

    assert "cls" in str(raised.value)
