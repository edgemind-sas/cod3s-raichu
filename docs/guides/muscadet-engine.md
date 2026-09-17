# Running a muscadet model on RAICHU

[muscadet](https://github.com/edgemind-sas/muscadet) is a modelling façade for
reliability and flow networks. Since it grew an **engine extension point**, it
knows no engine but its own reference one: an engine registers itself, and a
run says where it happens.

`pyraichu` registers there. A modeller writes muscadet and picks RAICHU:

<!-- skip -->
```python
import muscadet
import muscadet.kb.rbd            # Source / Block / Target

system = muscadet.System(name="rbd")
system.add_component(name="S", cls="Source")
system.add_component(name="B", cls="Block")
system.add_component(name="T", cls="Target")
system.connect_flow(source="S", target="B", flow_name="is_ok")
system.connect_flow(source="B", target="T", flow_name="is_ok")
system.comp["B"].add_exp_failure_mode(
    name="failure", failure_rate=0.1, repair_rate=0.5,
    failure_effects=[("is_ok_fed_available_out", False)])
system.add_indicator_var(component="^T$", var="^is_ok_fed_in$", stats=["mean"])

params = {"nb_runs": 10_000, "schedule": [0, 5, 10, 20], "seed": 42}

system.simulate(params)                    # PyCATSHOO, the reference path
estimates = system.simulate(params, engine="raichu")   # RAICHU
```

**Nothing imports `pyraichu`.** The choice of engine is a string. Installing
the package is the registration: the wheel advertises itself under the
`muscadet.engines` entry point group, muscadet discovers it on first need, and
a third engine would be added the same way without a line of muscadet
changing.

Registering explicitly stays available for a caller who wants it, and is what
an application does when it pins its engines rather than discovering them:

<!-- skip -->
```python
from pyraichu.muscadet_engine import register
register()
```

## What crosses, and what comes back

What the engine receives is the **system declaration**
(`muscadet.declare.system_spec`): a document, checked by muscadet on the way
out, never the live system. The run parameters travel beside it, in cod3s's
own vocabulary, because they configure a run rather than describe a system:
the schedule's instants become the sample instants and its last one the
horizon, which is what makes the two engines answer at the same dates.

The answer is the engine's own, handed back untouched. The reference path
writes its indicator values onto the live system and returns nothing; RAICHU
returns `McEstimates`, so a study reads the result rather than the system:

<!-- skip -->
```python
target = estimates.indicators["T_is_ok_fed_in"]
list(zip(target.instants, target.mean))
```

An interactive session takes the same declaration and the same build
(`system.isimu_start(engine="raichu")`), which is deliberate: a model that
behaved differently one step at a time than in bulk would be a divergence
nothing reports. It returns a `pyraichu.Interactive` session, stepped
RAICHU's way and not cod3s's.

## The feared events a run stops at

A safety study does not only ask how often an undesired event occurs, it asks
which chains of failures lead there. Both questions are asked of **one**
system: the availability figures come off a free-cycling campaign, the
sequences off a campaign where every trajectory stops at the event's first
occurrence. So the feared events are a parameter of the **run** and not a
section of the document -- a `targets` section in the declaration would make
those two campaigns two different systems.

muscadet spells that one run keyword itself, `muscadet.engine.RUN_TARGETS`,
and it travels beside the document:

<!-- skip -->
```python
system.add_component(
    cls="ObjEvent", name="EVT_LOSS",
    cond=[[{"attr": "is_ok_fed_in", "obj": "T", "value": False}]])

estimates = system.simulate(params, engine="raichu", targets=["EVT_LOSS"])
```

A target names the **event**, and nothing but the event. muscadet checks the
name against the declaration before the engine is reached, so a typo is
refused rather than run; what reaches RAICHU is a name to translate into what
it actually stops at, an automaton and a state, which the two engines spell
differently. The translation is read off the event's own declaration
(`pyraichu.declare.event_automaton` and `event_occurrence_state`), so an event
that renamed its automaton or its occurrence state is honoured under the new
names.

**A target stops the trajectory, and that is derived rather than asked for.**
`pyraichu.monte_carlo` carries `stop_at_targets` as a knob of its own, and the
seam sets it from the presence of targets: a target that does not stop the
trajectory is not a target, and it is what the reference engine does without
being told (PyCATSHOO stops unconditionally on an `addTarget`). An explicit
`stop_at_targets=` still wins, so a study that really wants a free-cycling
campaign over a model carrying targets says so and gets it.

What the keyword is worth is the distance between two runs of one model. A
block failing at 0.1 and repaired at 0.5 plateaus at its stationary
unavailability, 0.1 / 0.6 = 0.167, so a free-cycling campaign cannot approach
1 at any instant, while a first-occurrence campaign latches and climbs
(2000 replicas, seed 4242, both engines agreeing within Monte-Carlo noise):

| instant | 0 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| free-cycling | 0.000 | 0.154 | 0.162 | 0.174 | 0.169 |
| with a target | 0.000 | 0.386 | 0.624 | 0.910 | 0.991 |

`pyraichu.muscadet_engine.build_model` takes the same keyword, and not only
the run does. A study reading sequences needs the **model** twice: once for
the campaign and once for `pyraichu.analyse_sequences`, which the seam has no
kind of run for. Building it once is what lets both calls see the same targets:

<!-- skip -->
```python
from pyraichu import analyse_sequences
from pyraichu.muscadet_engine import build_model

model = build_model(muscadet.system_spec(system), targets=["EVT_LOSS"])
cuts = analyse_sequences(model, nb_runs=2000, t_max=50.0, seed=4242)
```

An interactive session accepts the keyword too, muscadet passing it to both
kinds of run: one entry point taking a keyword the other dies on would make a
demonstration and a campaign disagree about the study they are two views of.
A session has no `stop_at_targets` to set, so reaching the event stops nothing
and whoever drives the session decides what it means.

## Where the declaration lands

`pyraichu.muscadet` used to be a second authoring interface, mirroring
muscadet's idioms over RAICHU. It is now the **internal adapter**: the
declaration is read by `pyraichu.declare.build_document`, built through that
layer, and a modeller never names it. Nothing new enters it; an alignment
between the two vocabularies lands in the declaration reader or in
`pyraichu.muscadet_engine`, never in the mirror.

"Nothing new" is about the **authoring surface**, not about what the adapter
emits. Changing the model a declaration compiles to is what the adapter is
*for*: it is where muscadet's own flow sheet is reproduced, and a shape
muscadet holds in a variable is not reproducible by an expression, whatever
the two evaluate to. So the `{flow}_prod_available` variable below lands here,
while its arbitration and its refusals land in the reader. What would breach
the rule is a new `add_flow_out` keyword, a new `ObjFlow` idiom, a second way
to say something the declaration already says.

Two places in the vocabulary needed reconciling, and both are worth knowing
because they are what a modeller meets:

- **a failure mode's grip on a discrete output.** muscadet says "this mode
  kills this output" with an effect on the output's availability, spelled
  either as the flow's name or as its `{flow}_fed_available_out` variable.
  This layer says it with the mode's `targets` and keeps its effects for what
  a mode does to a *continuous* output, which is derate it. Both spellings are
  read, and a repair effect restoring availability is absorbed: the gate
  returns on its own when the mode leaves its failing state;
- **a mode declared ON a component that gates nothing.** There an empty target
  list means *every* discrete output of the component, so a muscadet mode
  naming none of them -- one that only derates a continuous output, or only
  flips a state an indicator watches -- is **refused by name** rather than
  built into a mode that silently kills outputs the declaration leaves alone.
  The rule stops at that shape: see below.

## What a study writes on every component, set or not

A cod3s parameter object and a muscadet read-back both write **every** field
they hold, whether or not anybody set one. So a refusal keyed on the
*presence* of a key is a refusal of every study written that way, and five of
them stood between the reader and the platform's own reference corpus. What
each became:

| What arrives | How it is read |
|---|---|
| `filter_objevent_in_sequences`, `filter_objfm_in_sequences`, `monitor_patterns`, `strict_failure_modes` | accepted, not read. They configure the reference runner's tracing and its post-processing of sequences, not the system |
| `pdmp_dt` | accepted, not read, and **stated as a divergence** rather than as silence: it is the reference solver's integration step, and RAICHU locates a crossing instead of sampling a grid. On a continuous model it would change a result, so the gap belongs in `muscadet.conformance`, where a reader consults it before choosing an engine. On a discrete model it governs nothing on either side |
| `create_default_out_automata` | accepted. muscadet writes it **only when it is `True`**, so a refusal at that value refused every export. What it asks for is one ok/nok automaton per discrete output, at a rate of `1e-100`; this layer derives no such pair, so what the absence takes away is not the trajectory but an **observation**, and that is where the refusal moved: an indicator naming `{flow}_ok` or `{flow}_nok` is refused by name, and a document naming none loses nothing |
| `var_fed_available_out_init`, `var_fed_available_out_reset` | carried. The platform writes `False` on both over a standby channel, which is the dormant service function of the section below, read through the gate |
| a state indicator spelled `PycAttrIndicator` + `attr_type: "ST"` | read as the state indicator it is (see *An event, which affects nothing*) |

**`var_prod_cond_inner_mode` is the one that changed a number rather than
stopping a run**, and it is worth its own paragraph. muscadet writes a
production condition as a list of groups, and this key says how the two levels
combine. It swaps *both* at once (`muscadet/flow.py`, `prod_cond_holds`):

| Declared | muscadet evaluates | This layer reads it as |
|---|---|---|
| `"or"` (muscadet's default) | `all(any(...))`, a conjunction of disjunctions | conjunctive normal form, expanded into the disjunction below |
| `"and"` (what a platform export writes) | `any(all(...))`, a disjunction of conjunctions | already the form underneath: carried through untouched |

The layer underneath reads a list of groups as the **OR of ANDs**, so
`[["a"], ["b"], ["c"]]` is `a or b or c` under `"and"` and `a and b and c`
under `"or"`. Read for the other, an alarm voting over three detections stops
alerting as soon as **one** of them falls, while the other two hold: a study
then reads an unavailability its model never stated, and the run is green.

A condition of one group of one operand means the same thing either way, so a
test written on one proves nothing about the reading it got. That is why
`python/tests/unit/test_prod_cond_inner_mode.py` opposes the two on a
multi-clause condition, and runs it.

## A mode declared beside its target

A component declaration takes **three shapes**, and says which under its `kind`
key. Absent, or `"flow"`, it is a component holding flows. `"two_state_mode"`
is a component of its own carrying no flow, holding a two-state automaton.
The kind names the state count rather than the nature of its most common
member, because it covers **three families**: the two mode vocabularies
described here, which apply their effects to components they **name** rather
than own, and `ObjEvent`, which affects nothing and is described under
[an event, which affects nothing](#an-event-which-affects-nothing). The third
shape is `"controller"`, described under
[a controller beside the flow graph](#a-controller-beside-the-flow-graph).

A standalone mode is what `system.add_component(cls="ObjFMDelay", ...)`
builds, and what a compromise cascade is made of:

<!-- skip -->
```python
system.add_component(cls="ObjFMDelay", fm_name="phishing", targets=["HMI"],
                     failure_param=20, repair_cond=False, repair_param=1e9)
system.add_component(
    cls="ObjFMDelay", fm_name="lateral_movement", targets=["HMI"],
    failure_param=5,
    failure_cond=[[{"attr": "occ", "obj": "HMI__phishing", "value": True}]],
    repair_cond=False, repair_param=1e9)
```

Nothing on `HMI` records either mode: drop the two entries and the system has
no attack in it at all. So they are read, and read in **both spellings a mode
constructor takes** — a declaration is written the way its own class reads it:

| Class | Spelling | Where its law is |
|---|---|---|
| `ObjFMExp`, `ObjFMDelay`, `ObjFMInst`, `muscadet.ObjFailureMode*` | `failure_*` / `repair_*` | carried by the class |
| `ObjMode2S` | `occ_*` / `not_occ_*` | declared, in `occ_law` / `not_occ_law` |
| `ObjEvent` | `cond` and its comparison | its two tempos, `tempo_occ` / `tempo_not_occ` |

What is worth knowing about the shape:

- **a standalone mode may gate nothing, and that is not the case refused
  above.** On a component, `targets` names *flows* and an empty list means all
  of them, which is why the absence has to be refused there. A standalone
  mode's `targets` names *components*, and its grip is what its effects name:
  it writes what it names and nothing else, so a mode with empty
  `failure_effects` means the same thing on both engines. That is the
  state-only mode a cascade's next stage watches;
- **its condition may watch another mode's state**, across components, which
  is the reference that makes a cascade a cascade;
- **the components are built first.** A mode resolves its effects against
  every component it names, so `build_document` builds the flow graph and then
  expands the modes onto it. `build_system` answers the flow graph alone and
  refuses a document carrying modes rather than dropping them.

Both of muscadet's compromise examples run and answer what PyCATSHOO answers:
`examples/isimu/power_plant`, whose cascade holds availability gates down, and
`examples/isimu/cyber_3comp`, whose middle stage **starts a dormant output**,
which is the next section.

## A mode that starts a dormant output, and the three booleans it could use

A **service function** (a maintenance channel, a remote-diagnosis link, a
remote access) is *exploitable but unused* in nominal operation. It is in the
model because an attacker can turn it on, not because anything uses it. That
is slide 51 of the IMdR P23-4 workshop, and `examples/isimu/cyber_3comp` is
it: `Srv` carries `f_service`, the mode `mdc_b` turns it on, and the attack
travels to the process on the ordinary flow wiring.

muscadet's delivery formula offers **three** booleans a mode could write, and
they are not interchangeable:

| Variable | What it says | How far it reaches |
|---|---|---|
| `{flow}_prod_available` | this component **produces** this flow | local, but copied into `{flow}_prod`, which an indicator or a condition can read |
| `{flow}_is_active` | a local switch on the output | **purely local**: it appears in the delivery formula and nowhere else |
| `{flow}_fed_available_out` | the output **channel** is available | **exported** downstream (`{name}_available_out`), read by `FlowIn.var_fed_available`, by `check_fed=False` gates and by the visualisation |

The COD3S Platform settled its own choice on 2026-06-15
(`ADR-2026-05-22-attribute-roles-vocabulary`, addendum *Fonctions de service*):
it carries dormancy on **the availability gate** `var_fed_available_out`, with
the assumed semantics "dormant ⇒ unavailable downstream", and **not** on
`var_is_active`. Its `activate` effect forces `True` on
`{flow}_fed_available_out`, which *replaced* an earlier `prod_available`
spelling; `is_active` and `active_init` stay wired as latent roles, exposed
nowhere, their consolidation deliberately deferred.

What RAICHU carries of each, and why:

- **`{flow}_fed_available_out`**: carried, and it is the platform's route.
  Seed it with `var_fed_available_out_init` and hold it down or open with a
  mode's effects. Nothing here had to change for it;
- **`{flow}_is_active`**: carried, and emitted **only** when a caller seeds
  `var_is_active_default`, so an ordinary flow carries nothing new. It does
  not replace `prod_available`: the two overlap on this one use and nowhere
  else, and nothing but the delivery ever reads `is_active`;
- **`{flow}_prod_available`**: carried, as a **variable** and not as an
  expression. Not because the platform needs it (it does not, it goes through
  the gate) but because muscadet's declaration surface lets a mode write it,
  and an engine that claims to run a muscadet model cannot refuse a legitimate
  one. Rewriting `cyber_3comp` to go through `fed_available_init` would have
  hidden the gap rather than closed it.

The mechanism is the variable's **writer**, not the variable:

- `update_{flow}_prod_available` is emitted **only where a production
  condition is declared**. A flow that declares none has no writer at all, so
  the variable keeps its `var_prod_default` until a mode writes it, and the
  latch holds. That gap is muscadet's own: its production method is subscribed
  to the operands of `var_prod_cond`, and an empty condition leaves it
  subscribed to nothing;
- the delivery reads the **variable**, never the condition again. Re-derived
  where it is consumed, the production would step over the latch at the next
  evaluation;
- nothing has to be exempted from a per-step reset, because there is none.
  muscadet declines to reinitialise `var_prod_available` (`flow.py`: *driven
  by tempo mecanisms*); here an effect is **held** and re-evaluated for as
  long as the mode's state lasts, which holds the latch for the same reason.

**Declaring both is refused.** A flow that declares a production condition
*and* is written by a mode has two writers on one variable. muscadet resolves
them by the order events happen in (the production method fires when an
operand changes, the mode's effect when the mode does, last one standing),
and this engine has no such ordering: the held effect wins at every evaluation
and the condition never shows. Measured against PyCATSHOO on a source-fed
flow, the reference produces from t = 0 and this engine would produce only
from the mode's date. It is refused by name, for the reason a persistent
availability gate written by a mode is: a divergence a study cannot see is
worse than a model it cannot run.

## An event, which affects nothing

The third family under the same `kind`, and the one the two paragraphs above
do not describe. A `cod3s.ObjEvent` is **self-hosted**: it names no target,
writes no attribute and has no common-cause order. What it declares is a
condition over arbitrary components of the system, the comparison it is tested
by, and the two tempos it waits before occurring and before returning:

<!-- skip -->
```python
system.add_component(
    cls="ObjEvent", name="DEGRADED",
    cond=[[{"obj": "B1__hw", "attr": "occ", "value": True}],
          [{"obj": "B3__wear", "attr": "occ", "value": True}]],
    inner_logic="all", outer_logic="any",
    tempo_occ=0.25, tempo_not_occ=0.75)
```

The COD3S Platform synthesises one per study event **and one per indicator
whose formula has more than one clause**, which is an ordinary thing to write,
so a document holding one is an ordinary document. Three things are worth
knowing:

- **a condition leaf must name what it watches.** On a mode, a leaf with no
  `obj` reads the mode's own target; an event has none, and cod3s compiles its
  tree with a system-wide resolution, so the leaf says which component it
  reads;
- **a leaf may watch the state of a mode or of another event**, and each is
  read with its own vocabulary: a mode's automaton is `fm` and its states are
  `occ` / `rep`, an event's are the three names it declares
  (`event_aut_name`, `occ_state_name`, `not_occ_state_name`, defaulting to
  `ev`, `occ`, `not_occ`);
- **an event is observed by a state indicator**, because it holds no variable
  for a variable indicator to name:

<!-- skip -->
```python
system.add_indicator_state(component="^DEGRADED$", state="^occ$", stats=["mean"])
```

This is the one place a state indicator is carried, and the reason is what
the point above says: an event declares its automaton and its two states, and
both layers build them under exactly those names, so the reference means the
same thing on each. Everywhere else a state indicator is refused, naming the
event as the family that carries one — muscadet's failure mode `m` sits in
`m_occ` of automaton `{comp}_m` and this engine's in `nok` of automaton `m`,
and no mapping between the two is declared.

**cod3s has two spellings of that one observation, and writes the second.**
A `PycSTIndicator` carries the state under `state`; a `PycAttrIndicator`
carries `attr_type: "ST"` with the state in `attr_name`. Both compile to the
same `{component}.{state}` on the reference engine, and the platform's
synthesised indicators are all of the second kind:

```json
{"name": "Rail_and_Detecteur_fed", "component": "_ind_eb45a35f",
 "attr_name": "occ", "attr_type": "ST", "kind": "PycAttrIndicator"}
```

Both are read here, and to the same entry.

**The estimate comes back under the name the document DECLARED**, which is a
rule the ordinary case hides. A `VAR` indicator's declared name is already
`{component}_{attribute}`, cod3s's own default pattern, so it is found
whichever of the two a consumer looks for. A state indicator stands for no
variable and agrees with nothing: `synthetic_event` declares
`Rail_and_Detecteur_fed` on the event `_ind_eb45a35f`, and a consumer asking
for `_ind_eb45a35f_occ` gets nothing.

Measured on that package through the platform's production runner: 30 of its
40 rows, the ten of the state indicator skipped on a warning. The values are
right (looked up by declared name the forty rows equal the golden), so the
gap is the **lookup**, which belongs to whoever reads the estimates. Nothing
in this repository fixes it, and nothing here should: registering the same
observation a second time under a name the document never wrote is what the
"two indicators of one name" refusal exists to prevent.

**One name, one indicator, whichever layer wrote it.** A declared indicator
usually names an observation the layer was going to emit anyway, since the
`VAR` naming agrees on both sides, so the assemblies **reconcile** rather
than concatenate: a name nobody holds is added, a name held by the same
observation is one indicator, and a name held by a *different* observation
is refused there, where both are still in hand and the refusal can say which
two writers disagree. It is one rule
(`pyraichu.indicators.merge_indicators`), used by the declaration route and
by the plugin expansion alike. Until 2026-09-15 only the first had it: a
document declaring an indicator **and** a continuous construct came out of
`expand_model` carrying that name twice, and the engine refused the whole
model on `duplicate indicator name`, naming the modeller's indicator rather
than the rebuild that recopied it. Removing the continuous part made the
same document load, which is what made the cause hard to see.

**Two observations can meet on one name without either being declared
twice**, because the name is flattened: `{component}_{variable}` puts a
component's name and a variable's name side by side with nothing between
them. A tank `TANK` whose capacity `level` publishes `level_content` is
observed as `TANK_level_content`, and so is the signal `content` of a
controller named `TANK_level`, and naming a controller after the level it
watches is the ordinary way to name one. That is refused, and refused
with both observations printed side by side, rather than reaching the engine
as a name carried twice.

**The model says whether it wants the generated set, and both routes read
it.** Beside the indicators it declares, a model may ask for the one this
layer generates -- one per observable variable, named `{component}_{variable}`
-- through a key of its own at model level:

```json
{"name": "plant", "generated_indicators": true,
 "indicators": [{"name": "availability", "target": "attribute",
                 "attr": {"component": "SNK", "attribute": "power_fed_in"}}]}
```

The key is spelled the same and at the same level in a system declaration
(`muscadet.declare.system_spec`) and in a RAICHU document carrying a
`plugins.muscadet` section, and `System.build_dict` writes it into every
document it produces rather than leaving it to a default, so the two
authoring surfaces write one document for one model.

**Absent, it is false**, and false is the declared indicators and nothing
else. That is what costs the existing corpus nothing: every COD3S Platform
study is boolean and reaches the engine through the plugin expansion, which
emitted nothing of its own for them. Measured on two reference studies, the
production runner's `indicators.csv` is byte for byte what it was.

Until 2026-09-17 there was no key, and the answer was read off the model's
own shape: a model carrying no continuous construct never reached the plugin
expansion's rebuild, so through the plugins it observed what the document
declared and nothing more, while the declaration route emitted the generated
set in every case. Two routes then disagreed on one model, and worse, the
presence of a tank **somewhere** in a model decided what every component of
it was observed by -- taking a buffer out to compare two variants silently
took ten observations away from components that had nothing to do with it.
The rebuild still runs where there is a network to resolve; it no longer
decides what is observed.

A value that is not a boolean is refused on either route, `"false"` being a
true string. A model-level key nothing reads is **ignored**, silently: the
component level is a closed vocabulary and refuses an unknown key by name,
the model level is open, so `generated_indicator` one letter short is read by
nobody and the model observes what it declared with no message.

## A controller beside the flow graph

The third shape, `"controller"`, is a `muscadet.ObjCtrl`: a **peer** of a flow
component and not a subclass of it. A flow transports a conserved quantity, a
controller transports a reading or a signal, and nothing is allocated. It is
what the platform instantiates for any template carrying
`metadata.controller`, so every client model that commands anything by
threshold holds one, and it declares **two sections and nothing else**:

- `controls_in`, the quantities it observes: a capacity level, the rate a
  continuous output delivers, or the share one constituent is of what a volume
  holds. An input declaring an `aggregate` reduces several publishers to one
  value;
- `controls_out`, the signals it publishes: a boolean on `{name}_out`, which is
  the very box a discrete in-flow imports, or a number on `{name}_level_out`,
  which a second controller reads like any other publication. Each carries
  under `emit` the value that computes it, as a composition of exactly four
  operators — `compare`, `band`, `combine` and `republish`.

What is worth knowing about the shape:

- **the thresholds are attributes of the model**, named `{output}{path}_{edge}`
  from the node's position in the output's tree: `run_threshold`,
  `alarm_operand_1_activate`. That is what an indicator names and what a
  failure mode moves, so two instances of one class run at their own tuning
  rather than at the class's;
- **the order runs in three steps.** `build_document` builds the flow graph,
  expands the controllers onto it, then the modes — a mode reaches *into* a
  controller, so a mode expanded first would name an attribute the document
  does not yet hold. `build_system` answers the flow graph alone and refuses a
  document carrying controllers rather than dropping them;
- **a controller reading another controller's output is declared after it.**
  The sweep follows declaration order, and a reading swept before the
  publication it mirrors would lag it by one evaluation point;
- three things are refused rather than approximated: a **Python callable**
  under `emit`, whatever continuity it attests, because nothing can read a
  threshold out of arbitrary Python and so nothing could locate the crossing;
  an **equality** in a comparison, which brackets no crossing on a
  continuously-evolving quantity; and a **`min` or `max` aggregation**, which
  this engine has no variable-arity operator for and which a sum or a mean
  would answer as a different question.

A **measurement link** — what wires all of this — follows the `{x}_out` /
`{x}_in` convention of a flow without naming one, so muscadet writes it with no
`flow` key at all and it travels by the raw connection route. This reader tells
the families apart from the box names, so the key changes nothing whether a
stored document carries it or not.

One thing the link has to reconcile, because the two engines address it
differently: muscadet's message box carries **several variables** — a level, a
weighted fill, one level per constituent, the ratios — while RAICHU connects
attribute to attribute. A single anchor pair in the document therefore resolves
to the one attribute each end actually holds. Two consequences worth knowing:

- an observation input reading a constituent or a share resolves to the alias
  the *volume* publishes it under (`{capacity}_ratio_{flow}_out`), not to the
  total the anchor names;
- a controller's value output read by a flow component's measurement channel
  reaches **both** of that channel's references, level and fill. muscadet's
  `MeasurementOut` writes the fill equal to the level when no fill is given, so
  an instrument publishing 7.5 leaves the observer's level *and* its fill at 7.5;
  wired to the level alone, the fill would read 0 where muscadet reads the level.

What has no counterpart is refused, never approximated: an indicator on the
state of anything but an event, a PyCATSHOO trace, a run parameter this engine
does not read. A knob of the engine itself travels as a keyword of the run,
beside the parameters:

<!-- skip -->
```python
system.simulate(params, engine="raichu", threads=8, quantiles=[0.05, 0.95])
```

Where an engine does what muscadet defines but does it *otherwise*, the
statement belongs to muscadet's own conformance registry rather than here:
`python -m muscadet.conformance raichu`.

## A machine that moves a mixture, and the section it is declared in

muscadet 5.4.0 added `add_mixture_in` (R51), and its read-back writes a
`mixtures` section on **every** flow component, `[]` included. The key is
therefore read the way `measurements_out` and `automata` are: **accepted while
it declares nothing**, refused as soon as it declares something. A reader
refusing it by name would refuse the whole corpus at the first component it
meets, ventilated or not, which is what a refusal keyed on the *presence* of a
key always costs.

A document that declares no group builds exactly the model it built before the
section existed, and that is pinned rather than intended: the two documents
build the same body, byte for byte.

A **non-empty** group is refused, by the name of its section, and the refusal
is not a spelling gap waiting to be closed. A group is one volumetric rate `R`
for several constituents, the split being fixed by the composition of the
volume drawn from and by nothing the group declares:

```text
out_f  =  R . m_f / sum_g ( m_g . w_g )
```

Every rate this layer carries is a rate **per flow**: a rule's `cons`, a
source's `rate`. Reading a group of two constituents as two independent
demands would give the model two degrees of freedom where the physics has one,
which is the very model R51 exists to refuse, so accepting it in silence would
return a trajectory and a wrong one. The refusal names the component and the
section, says what the seam does not carry, and says what stands in its place:
today, nothing but the reference engine.

## What replaces `cod3s.ComponentInstance.to_bkd_raichu`

`cod3s` carried a second backend on its component specification: a
`class_name_bkd["raichu"]` naming `pyraichu.muscadet.System` and the component
classes behind it, so that `spec.to_bkd("raichu")` built a RAICHU system
directly.

**It disappears rather than re-points.** Its whole purpose was to give cod3s a
second target, and it did so by making cod3s name a RAICHU class path: every
specification carried two class names per component, and adding a third engine
would have meant a third. That is exactly the coupling the extension point
removes. The route now is one target and one choice:

<!-- skip -->
```python
system = spec.to_bkd("pycatshoo")        # ONE build, muscadet's
system.simulate(params, engine="raichu") # the engine is a run parameter
```

The consequences, stated so nobody looks for the old path:

- a specification declares its components once, with no engine in it. The
  `raichu` entries of `class_name_bkd` become dead weight and their removal is
  a cod3s change, not a RAICHU one;
- a model written for the second backend keeps working while `to_bkd_raichu`
  exists, and buys nothing: the same model reaches the same engine through the
  declaration, and reaches it *checked*, muscadet validating the document on
  the way out;
- the cross-validation that covered the second backend is superseded by the
  one covering the route that survives: one muscadet model, both engines, a
  live PyCATSHOO oracle rather than a recorded trajectory. See
  [Cross-validation](../benchmarks/cross-validation.md).
