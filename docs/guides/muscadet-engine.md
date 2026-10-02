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
from pyraichu.muscadet.engine import register
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
(`pyraichu.muscadet.declare.event_automaton` and `event_occurrence_state`), so an event
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

`pyraichu.muscadet.engine.build_model` takes the same keyword, and not only
the run does. A study reading sequences needs the **model** twice: once for
the campaign and once for `pyraichu.analyse_sequences`, which the seam has no
kind of run for. Building it once is what lets both calls see the same targets:

<!-- skip -->
```python
from pyraichu import analyse_sequences
from pyraichu.muscadet.engine import build_model

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
declaration is read by `pyraichu.muscadet.declare.build_document`, built through that
layer, and a modeller never names it. Nothing new enters it; an alignment
between the two vocabularies lands in the declaration reader or in
`pyraichu.muscadet.engine`, never in the mirror.

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
| `pdmp_dt` | read, as the **event resolution**: the reference solver's base step, which decides how short an episode it can catch, is honoured here as the widest spacing accepted between two points of the crossing scan (`event_resolution`, see the numerical-tuning guide). A finer request than the engine's own adds scan points, a coarser one changes nothing. A value that is not a finite positive time is refused by name. On a discrete model it governs nothing on either side |
| `create_default_out_automata` | accepted. muscadet writes it **only when it is `True`**, so a refusal at that value refused every export. What it asks for is one ok/nok automaton per discrete output, at a rate of `1e-100`; this layer derives no such pair, so what the absence takes away is not the trajectory but an **observation**, and that is where the refusal moved: an indicator naming `{flow}_ok` or `{flow}_nok` is refused by name, and a document naming none loses nothing |
| `var_fed_available_out_init`, `var_fed_available_out_reset` | carried. The platform writes `False` on both over a standby channel, which is the dormant service function of the section below, read through the gate |
| a state indicator spelled `PycAttrIndicator` + `attr_type: "ST"` | read as the state indicator it is (see *An event, which affects nothing*) |

**`var_prod_cond_inner_mode` is the one that changed a number rather than
stopping a run**, and it is worth its own paragraph. muscadet writes a
production condition as a list of groups, and this key says how the two levels
combine. It swaps *both* at once (`muscadet/flow.py`, `prod_cond_holds`, in the muscadet repository):

| Declared | muscadet evaluates | This layer reads it as |
|---|---|---|
| `"or"` (muscadet's default) | `all(any(...))`, a conjunction of disjunctions | conjunctive normal form, expanded into the disjunction below |
| `"and"` (what a platform export writes) | `any(all(...))`, a disjunction of conjunctions | already the form underneath: carried through untouched |

The layer underneath reads a list of groups as the **OR of ANDs**, so
`[["a"], ["b"], ["c"]]` is `a or b or c` under `"and"` and `a and b and c`
under `"or"`. Read for the other, an alarm voting over three detections stops
alerting as soon as **one** of them falls, while the other two hold: a study
then reads an unavailability its model never stated, and the run is green.

The authoring classes (`pyraichu.muscadet`, `add_flow_out` and its tempo and
trigger siblings) take the same key with the same default, so a model written
directly in Python reads a condition as muscadet does; this route hands them
its converted form stated as `"and"`. The plugin's `flows_out` section keeps
its own reading, the groups OR-ed and a flat list one conjunction, where no
mode is declared, and honours a declared one with muscadet's meaning.

A condition of one group of one operand means the same thing either way, so a
test written on one proves nothing about the reading it got. That is why
`python/tests/unit/test_prod_cond_inner_mode.py` opposes the two on a
multi-clause condition, and runs it.

**`negate` on an output** is muscadet's `FlowOut.negate`: the output
publishes `not (production and active and available)`, the inverter of a logic
chain. It is carried on a plain output and refused by name on a temporised or
triggered one, whose delivery is an automaton's state. One consequence is
worth knowing: a negated output can publish `fed` while its own availability
is false, which no other output does. An input it feeds still reads the feed
alone. muscadet reads `fed AND available` on an input, but the availability
term only aggregates what is wired to the separate `{flow}_available_in` box,
and `auto_connect` wires the feed or that channel, never both; this layer
carries no availability connection at all. So a downstream input of a negated
output whose gate a mode took down reads `True`, as on the reference engine
(measured 2026-10-02: an inverter down at t = 1, its consumer reads `True` at
t = 2). Up to 0.74.0 such an input also read its producers' gates and read
`False`, which stopped a detection chain from ever starting.

## An operand that says more than a flow name

A production condition is groups of operands, and an operand is either a bare
flow name or a mapping. muscadet writes the mapping form on **every** flow
operand it hands back (`muscadet/declare.py`, `_prod_cond_spec`, in the muscadet repository), because the
condition it stores is resolved and has to be walked back into a declaration;
this layer reads the five keys that walk-back writes, and no others.

| Key | What it says | How it is read |
|---|---|---|
| `name` | the flow, capacity level or measurement channel the operand reads | required |
| `port` | `"in"` or `"out"`, selecting a side | the default resolves the **input side first**, exactly as muscadet's does. `port` matters where the component carries one name on both sides |
| `negate` | deny the operand | the guard it reads, inverted |
| `op`, `value` | compare what `name` carries against a threshold | the rule-guard comparison vocabulary, `<` `<=` `>` `>=` `==` `!=` |

Three of them are read where they were refused before, and the third is not a
key at all:

- **`negate`** is the `¬` an editor of libraries puts in front of an operand,
  and muscadet's `var_prod_cond_negate` is where it ends up over there;
- **`op` and `value`** are muscadet's `var_prod_cond_compare`, and the
  vocabulary is deliberately the one a controller's rule guard already carries
  here: one comparison syntax for both directions of the discrete/continuous
  interoperation;
- **`port: "out"` on a name the component declares as an input too** was
  refused by a *resolution rule* rather than by a missing key. The reader
  resolved the input side first and refused rather than silently read the
  other. Honouring the selection is what a **transit** component needs, and
  the shape is the most ordinary there is: a board that feeds on and publishes
  that it does.

An operand that states the default reduces to the bare flow name, so a
document built before these keys were read is unchanged, byte for byte:
`{"name": "elec", "port": "in"}` and `"elec"` build the same model, and so
does `port: "out"` on a flow the component carries only as an output.

### What is refused, and why it stays refused

- **`negate` beside `op`.** A comparison already yields a truth value, so it
  is denied by the opposite operator. muscadet refuses the pair at the same
  place (`muscadet.rules.check_operand_negation`).
- **`release`**, the band that widens a comparison. A rule guard carries one
  because its mode is an automaton that holds a location between the two
  edges; a production condition writes a **variable** rewritten at every
  evaluation, and there is nothing to hold. muscadet's own operand does not
  read the key either.
- **`automaton` / `state`.** A production condition reads what crosses the
  component, not where its automata sit, and muscadet's operand carries
  neither.
- **a boolean operand naming a continuously-evolving quantity.** muscadet
  reads that as `!= 0` (its R46). Here the verdict is read at discrete epochs,
  so the crossing has to be **located** to be read at its own date, and `!= 0`
  has no side to be entered from. Locating it would mean turning it into `> 0`
  or into `< 0`, that is **supposing the sign of the quantity** in the
  modeller's place, on a rate that may well take both; reading it as it stands
  would mean settling the condition at t = 0 and never again. Neither is a
  reading somebody can be given without being told, so the shape is refused
  and the refusal names what to write: `{"op": ">", "value": 0}` says the same
  of a quantity that only rises, and says which side it is entered from. An
  equality on such a quantity is refused for the same reason, in the same
  words a capacity's discharge command uses.

### A threshold on a quantity that moves

A comparison against a continuous quantity gets a **two-state threshold
automaton** whose two transitions are `watched`, and the production condition
reads that automaton's location rather than the quantity itself. The
indirection is the mechanism, not a detour:

the condition writes a variable through a sensitive function, and a sensitive
function is re-run by a *discrete* change. An attribute an ODE moves announces
nothing of its own, so a condition reading the quantity directly is evaluated
at t = 0 and never again -- measured, on a volume filling past its threshold
with the output left unfed for the whole mission. Reading the location
subscribes the function to the automaton, whose watched transitions fire **at**
the crossing. muscadet wires the same thing the other way round, hanging a
sensitive method on the same automaton
(`muscadet.flow.add_prod_cond_threshold_automata`).

The two readings agree everywhere but at t = 0, where the automaton is still
in the location it was declared in: the initial fixpoint settles it, as it
settles a capacity's discharge gate and a rule set's mode.

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

Both of muscadet's compromise examples (in the muscadet repository) run and
answer what PyCATSHOO answers: `examples/isimu/power_plant`, whose cascade holds availability gates down, and
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
from the mode's date. It is refused by name, for the reason two modes
holding one persistent availability gate are: a divergence a study cannot
see is worse than a model it cannot run.

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
  does not yet hold. A mode may therefore name a controller among its
  `targets`, which is the cyber scenario the shape exists for: an instrument
  that is not destroyed but blinded, its `{output}_signal_available` clamped
  while the reading goes on being right. `build_system` answers the flow graph
  alone and refuses a document carrying controllers rather than dropping them;
- **one attribute is spelled differently on the two sides, and only one.**
  muscadet holds a boolean output's signal in `{output}_signal_out`, so that a
  mode's unanchored regular expression has a name of its own to anchor on;
  this layer holds it in `{output}` and exports it on `{output}_out`. An
  indicator naming the muscadet spelling is **translated** — it keeps the name
  the document declared it under, and only what it points at is read in this
  layer's spelling. Everything else a controller exposes is shared, the R44
  endpoints and a value output's `{output}_level` included;
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

## A volume, and the name its level is observed under

A capacity is the one other component family whose variables the two layers do
not spell alike, and the disagreement falls on the very reading a continuous
study is written around: what the volume **contains**. muscadet holds it in
`{capacity}_qty` and `{capacity}_qty_{flow}`, this layer in
`{capacity}_content` and `{capacity}_content_{flow}`.

An indicator naming the muscadet spelling is **translated**, exactly as a
controller's signal is: it keeps the name the document declared it under, and
only what it points at is read in this layer's spelling. So
`add_indicator_var(component="^BAT$", var="^reserve_qty$")` observes
`BAT.reserve_content` and comes back under `BAT_reserve_qty`. The weighted fill
is shared outright — `{capacity}_fill` and `{capacity}_fill_{flow}` — and so is
`{capacity}_ratio_{flow}` on a volume holding more than one constituent.

**A declared CONDITION goes through the same translation**, and it is the
reading that needs it most: a feared event armed when a vessel passes a
threshold, or a failure mode armed on a low tank level, is the most ordinary
use a safety study makes of a volume. The `cond` of an `ObjEvent` and the
`occ_cond` / `not_occ_cond` of an `ObjMode2S` may each name
`{capacity}_qty_{flow}`, and each is resolved to the attribute this layer
carries before anything is built. The two consumers share one reading, so the
day the attribute is named otherwise both follow at once; and the three
variables below are refused in a condition by the same names they are refused
in an indicator.

Three of muscadet's capacity variables have **no attribute here**, and none of
them is a spelling disagreement. Each is refused by its own name, saying what
to observe instead, rather than reaching the loader and failing there on a name
nobody can trace back to a declaration:

| muscadet variable | what this layer offers |
|---|---|
| `{c}_inflow_{f}`, `{c}_outflow_{f}` | muscadet writes them from its allocation sweeps; here the content is integrated straight from `{f}_fed_in` minus `{f}_fed_out`, which both layers name alike |
| `{c}_ratio_{f}` *on a single-constituent volume* | no ratio is published, that share being identically one wherever the volume holds anything: observe `{c}_content`, or `{c}_fill` for how full it is |

`{c}_serve_rate_{f}` is **not** one of them, and was until the ceiling stopped
being a constant folded into the service expression. It is a variable now, one
per held flow and under muscadet's own spelling, because that is what a failure
mode clamps to throttle a discharge, so an observation on it is carried through
untouched. A volume declaring no ceiling publishes the unbounded sentinel
there, so the observation answers whether the declaration named a number or
not.

The translation is keyed on the capacities the component **declares**, never on
a suffix: a variable ending in `_qty` on a component holding no volume of that
name, or holding one another component declares, is left exactly as the
document wrote it.

## A continuous flow's demand, swapped between the two layers

The demand channel is carried under **swapped** names. On muscadet an input
publishes what it asks for as `{flow}_demand_out`, and an output reads its
consumers back through the reference `{flow}_demand_in`. This layer publishes
what an input asks for as `{flow}_demand_in`, and the total asked of an output
as `{flow}_demand_out`. The quantities are the same; the names point the other
way.

An indicator naming muscadet's spelling is **translated**, like a capacity's
level: `add_indicator_var(component="^PIPE$", var="^feed_demand_out$")`
observes `PIPE.feed_demand_in`, what the pipe asks upstream, and comes back
under `PIPE_feed_demand_out`. The correspondence is published as
`pyraichu.muscadet.declare.flow_demand_variables`, so a caller reading this
layer's attributes (an interactive session, for one) translates without
restating the rule.

The generated indicator set keeps this layer's own spelling. On a component
holding a flow on **both** sides (a tank, a pass-through) the generated
`{c}_{flow}_demand_out` observes the output's demand, so a declaration of the
same name, which means the input's, would collide with it. The declaration
wins: the generated homonym steps aside and the model runs, as it does on the
reference engine. Every other collision between a declaration and the
generated set is still refused.

## An instrument that republishes what it reads

`measurements_out` (muscadet's `add_measurement_out`, R37) is carried. An
instrument is a component that publishes a reading under the very aliases a
capacity uses, so an observer cannot tell it from the volume behind it: its
`source` is a capacity or a measurement channel of the same component, and it
republishes the level, the fill and, for each constituent in `flows`, the
constituent's level, fill and share. Everything it publishes is multiplied by
`{name}_level_gain`, the endpoint a failure mode clamps: 0 is a dead
instrument, 5 a wild one. A declaration with no `source` publishes its
`level_default` and nothing refreshes it.

The readers and the instruments are swept in the order the reading flows,
volume, instrument's channel, its publication, observer, so an observer
reads this evaluation's publication rather than the previous one's; a
reading fed back into its own source through instruments has no such order
and is refused naming the channels. A continuous output's rate is not a
source: a flow publishes its own rate channel (`publish_rate`), which an
observer reads directly.

muscadet 5.6.0 writes `kind`, `rate_default` and `ratio_default` on every
measurement channel, a level one included. They are accepted at the values a
level channel carries (`"level"`, 0) and refused above them.

## A machine that moves a mixture, and the section it is declared in

muscadet 5.4.0 added `add_mixture_in` (R51), and its read-back writes a
`mixtures` section on **every** flow component, `[]` included. A document
that declares no group builds exactly the model it built before the section
existed, and that is pinned: the two documents build the same body, byte for
byte.

A **non-empty** group is carried. A group is one volumetric rate `R` for
several constituents, the split being fixed by the composition of the volume
drawn from and by nothing the group declares:

```text
out_f  =  R . m_f / sum_g ( m_g . w_g )
```

so the volume extracted, `sum_f out_f . w_f`, is exactly `R`. Two things make
it hold, and the second is the one a shortcut misses.

- **What the group asks.** When the model is generated, each group is bound to
  the one capacity of the one producer its flows arrive from, and each of its
  inputs asks `R . m_f / S`, read off that volume's integrated contents and
  declared weights (`S` is the weighted occupied volume). A group whose flows
  arrive from two producers, or two capacities, or from a producer serving the
  same outputs to somebody else, is refused by the group's name, as muscadet
  refuses it; so is a flow a rule of the consumer also consumes.
- **What the volume serves.** A volume drawn by a group serves each constituent
  exactly the composed request, what merely transits entering the composition
  first (muscadet's `draw_from_capacity`, the branch serving a mixture). The
  pooled rule a mixed volume otherwise follows, transit passed straight on and
  only the excess drawn at the composition, gives the right answer while the
  inflows cover the requests and the wrong one as soon as the stock has to
  supply: measured, it holds the supply-open trajectory and breaks the
  supply-cut one and the weighted invariant. The group is therefore a demand
  in its own right, not per-flow demands that happen to be in proportion.

Measured against muscadet's own closed forms for a room of `V` receiving air
at `Q` and hydrogen at `q` under one extractor at `R = Q`: the share follows
`q/(Q+q) . [1 - (V/(V+qt))^((Q+q)/q)]` with the supply open and
`x0 . exp(-R t / M0)` with it cut, and the extracted volume is `R` at a
hydrogen weight of 2.

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

## Replaying muscadet's own examples on both engines

Everything above is measured model by model, on questions chosen because
each one was worth asking. That leaves a gap it cannot close by itself: the
models it asks them of are the ones someone thought to write down. muscadet
ships a corpus of its own under `examples/`, and those are the models that
define what the framework means, so a bench enumerates them rather than
naming them one by one.

That bench lives in this project's cross-validation harness, which needs a
PyCATSHOO installation to run and is not part of the distribution. What it
found is below; how it is wired is an internal matter, and the numbers are
the part that is useful here.

Seventeen models at muscadet 5.6.0. Each leaves a run in exactly one of four
states, and a model in none of them fails the bench, which is the rule that
stops one from quietly ceasing to be measured.

| State | How many | What it means |
|---|---|---|
| green | 10 | both engines answered, and agree |
| refused | 4 | an engine, or the example itself, declined to go on |
| drawn | 3 | the numbers come from a draw, so no byte comparison can hold |
| disagreeing | 0 | both ran, and answered differently |

The version floor is refused rather than degraded. Below muscadet 5.6.0 the
corpus is a different corpus, and a bench that quietly measures less than it
claims is worse than one that does not run.

### What the four states are declared in

A manifest holds what each model is: how it hands over its system, which
comparison regime its laws put it in, whether a closed form can witness it.
That is declared rather than derived at run time, because a derived
classification reclassifies a model the day muscadet edits it and then
reports the reclassification as a disagreement between engines.

Every declared entry is **launched**, never skipped. A refusal that has
silently changed its message, or stopped happening, is a fact about the
engine's reach that only a launch can notice; a table read as a skip-list
would hide exactly the day a gap closed.

### Reaching a model that was never written to hand one over

Nine of the seventeen construct a system and run it in the same gesture at
module scope. The driver takes the system mid-sentence, through three
interceptions rather than one, and the second is the one that decides
whether the import survives at all: eight of those nine build an indicator
figure on the line after their run, and that call concatenates indicator
values that stay empty until the run's post-processing fills them. Patching
the figure's write and display calls does not reach it, because the call
that raises is the one that builds the figure.

The third interception widens the indicator declaration to ask the reference
engine for a per-instant dispersion. The examples declare `stats=["mean"]`
and the restitution mask is derived from that list when the simulation is
prepared, so it cannot be asked for afterwards.

The other eight expose a build entry point and declare no indicator at all,
so the bench supplies what to watch: every `*_fed_in` and `*_fed_out`
variable, which is what every model that does declare indicators chose to
watch.

### Which instants are compared, and which are not

A factory model is sampled at half-integer offsets over a horizon the
manifest names. Every transition date in this corpus is a whole number, so a
half-integer grid cannot land on one.

That matters because the reference engine observes the state **before** the
transitions due at an instant are resolved, and RAICHU observes it after;
muscadet's conformance registry declares that convention against the
reference engine. A sample posted on a transition date measures that
registered divergence and not parity. Measured on `rbd_04`: of its thousand
sampled instants exactly one diverged, the horizon itself, where two
components swap states. Dropping it takes the model to zero.

### Two engines agreeing is not two engines being right

A mistake both make together passes every comparison the bench can run. So
where the answer can be worked out, it is written down and each engine is
checked against it separately. Two such forms exist today, deliberately
fewer than the models that could carry one, and a test pins that gap so it
cannot drift shut unnoticed.

One of them is bounded to the window its model's docstring actually states.
The first version of it ran past that last date and failed against **both**
engines at once: they agreed with each other, and the witness was the thing
that was wrong. That is what a closed form is for.

## Portable combinational gates

The portable declaration seam supports `ObjLogicGate` as `kind: logic_gate`.
The separate `logic_kind` is `or`, `and`, or `k`; `k` requires a positive integer
threshold. `cond` retains equality conditions over named component variables,
and `out_elements` names the discrete exports. Live Muscadet read-back carries
the same declaration; native document building reuses the Muscadet plugin's
combinational expansion. Gate exports connect to declared discrete flow inputs.

An empty OR gate and an empty k/n gate are false; an empty AND gate is true.
Conditions and thresholds are validated before model construction. No Python
callback participates in native simulation.
