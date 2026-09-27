# Heated room

A room loses heat to a cold outside whose temperature follows the day. A
main heater holds it between 19 and 21 °C; a backup heater, set lower,
starts only when the room falls under 17 °C. Either can fail and is
repaired. The feared event is the room falling under 10 °C, which only
happens when both heaters are down long enough.

The example extends the model of the [hybrid tutorial](../tutorial/04-going-hybrid.md)
with what a real study adds: a time-varying boundary condition, a
redundant and staggered control, repairs, and a feared event followed
over a whole winter. The parameters are illustrative.

## The installation

![Heated room: a main and a backup heater with their thermostats, heat lost to a daily varying outside temperature](../assets/schematics/heated-room-light.svg#only-light){ .figure }
![Heated room: a main and a backup heater with their thermostats, heat lost to a daily varying outside temperature](../assets/schematics/heated-room-dark.svg#only-dark){ .figure }

## Hypotheses

| Aspect | Hypothesis |
|---|---|
| Room | $\dfrac{dT}{dt} = P - 0.1\,\big(T - T_{out}(t)\big)$: heating minus losses, which relax the room to the outside in about 10 h |
| Outside | $T_{out}(t) = 2 + 6 \sin\big(2\pi (t - 9)/24\big)$ °C, peaking at 15:00 at 8 °C, lowest at 03:00 at −4 °C |
| Heaters | each running and healthy heater gives $P = 3$ °C/h, enough on its own to hold the room on the coldest night |
| Thermostats | main: on below 19 °C, off above 21 °C. Backup: on below 17 °C, off above 20 °C. Both are watched crossings |
| Failures | each heater fails at rate 1/500 h⁻¹, whether it runs or not, and is repaired at rate 1/24 h⁻¹ |
| Feared event | the room under 10 °C (*frozen*), cleared above 12 °C |
| Initial state | 20 °C at midnight, main heater on, backup off, both healthy |
| Not modelled | common-cause failures, a failure to start on demand, the thermal mass of the walls |

## The model

```python
import pyraichu

# Units: hours and °C. Power is expressed as the heating rate it gives the room.
LOSS = 0.1                    # 1/h: the room relaxes to the outside in about 10 h
OUTSIDE = (2.0, 6.0, 15.0)    # mean, amplitude, hour of the daily peak
POWER = 3.0                   # °C/h delivered by one running heater
FAIL, REPAIR = 1 / 500, 1 / 24   # 1/h: 500 h between failures, 24 h to repair
FROZEN, THAWED = 10.0, 12.0   # the feared event, and the level that clears it
HEATERS = {  # name: (thermostat on below, off above, initially on)
    "main": (19.0, 21.0, True),
    "backup": (17.0, 20.0, False),
}

def num(x):
    return {"op": "const", "value": {"kind": "float", "value": x}}

def attr(component, name):
    return {"op": "attr", "attr": {"component": component, "attribute": name}}

def in_state(component, automaton, state):
    return {"op": "state_active",
            "state": {"component": component, "automaton": automaton, "state": state}}

def cmp(op, lhs, rhs):
    return {"op": "cmp", "cmp": op, "lhs": lhs, "rhs": rhs}

temperature = attr("room", "temperature")
```

A heater has two automata: its thermostat, whose two transitions are
**watched** crossings of the room temperature, and its health, whose
failure and repair are exponential. The pair is `monitored` and grouped
in one `cycle_group`, so a sequence campaign records them and cancels a
failure that was repaired before the feared event:

```python
def heater(name, on_below, off_above, running):
    delivering = {"op": "bool", "bool_op": "and", "args": [
        in_state(name, "thermostat", "on"), in_state(name, "health", "ok")]}
    return {
        "name": name,
        "attributes": [{"name": "power", "kind": "float",
                        "init": {"kind": "float", "value": POWER if running else 0.0}}],
        "ports": [{"name": "power_out", "dir": "out", "attr": "power"}],
        "automata": [
            {"name": "thermostat", "states": ["on", "off"],
             "init": "on" if running else "off", "transitions": [
                {"name": "switch_off", "source": "on", "targets": ["off"],
                 "guard": cmp("gt", temperature, num(off_above)), "distrib": "watched"},
                {"name": "switch_on", "source": "off", "targets": ["on"],
                 "guard": cmp("lt", temperature, num(on_below)), "distrib": "watched"},
            ]},
            {"name": "health", "states": ["ok", "failed"], "init": "ok", "transitions": [
                {"name": "fail", "source": "ok", "targets": ["failed"],
                 "distrib": "exp", "rate": FAIL, "monitored": True, "cycle_group": name},
                {"name": "repair", "source": "failed", "targets": ["ok"],
                 "distrib": "exp", "rate": REPAIR, "monitored": True, "cycle_group": name},
            ]},
        ],
        "sensitive_functions": [{"name": "update_power", "effects": [
            {"target": {"component": name, "attribute": "power"},
             "value": {"op": "if", "cond": delivering, "then": num(POWER), "otherwise": num(0.0)}}]}],
    }
```

The outside temperature is an expression of the simulation `time`. The
room carries the differential equation and the *comfort* automaton whose
*frozen* state is the feared event. Besides it, the model observes the
temperature, the time spent under 17 °C (a **predicate** indicator), and
each heater's failed state:

```python
mean, amplitude, peak = OUTSIDE
outside = {"op": "add", "args": [num(mean), {"op": "mul", "args": [
    num(amplitude),
    {"op": "sin", "arg": {"op": "mul", "args": [
        num(2 * 3.141592653589793 / 24),
        {"op": "sub", "lhs": {"op": "time"}, "rhs": num(peak - 6)}]}}]}]}

room = {
    "name": "room",
    "attributes": [{"name": "temperature", "kind": "float", "init": {"kind": "float", "value": 20.0}}],
    "ports": [{"name": "power_in", "dir": "in"}],
    "automata": [{"name": "comfort", "states": ["ok", "frozen"], "init": "ok", "transitions": [
        {"name": "freeze", "source": "ok", "targets": ["frozen"],
         "guard": cmp("lt", temperature, num(FROZEN)), "distrib": "watched"},
        {"name": "thaw", "source": "frozen", "targets": ["ok"],
         "guard": cmp("gt", temperature, num(THAWED)), "distrib": "watched"},
    ]}],
    # dT/dt = heating − LOSS · (T − T_outside(t))
    "equations": [{"target": "temperature", "kind": "ode", "expr": {"op": "sub",
        "lhs": {"op": "port_agg", "port": {"component": "room", "port": "power_in"}, "agg": "sum"},
        "rhs": {"op": "mul", "args": [num(LOSS), {"op": "sub", "lhs": temperature, "rhs": outside}]}}}],
}

model = pyraichu.load_model({
    "name": "heated_room",
    "components": [room] + [heater(n, *spec) for n, spec in HEATERS.items()],
    "connections": [{"from": {"component": n, "port": "power_out"},
                     "to": {"component": "room", "port": "power_in"}} for n in HEATERS],
    "targets": [{"name": "frozen", "component": "room", "automaton": "comfort", "state": "frozen"}],
    "indicators": [
        {"name": "temperature", "target": "attribute", "attr": {"component": "room", "attribute": "temperature"}},
        {"name": "cold", "target": "predicate", "attr": {"component": "room", "attribute": "temperature"},
         "cmp": "lt", "value": {"kind": "float", "value": 17.0}},
        {"name": "frozen", "target": "state", "component": "room", "automaton": "comfort", "state": "frozen"},
    ] + [
        {"name": f"{n}_failed", "target": "state", "component": n, "automaton": "health", "state": "failed"}
        for n in HEATERS
    ],
})
```

## One trajectory

```python
run = pyraichu.simulate(model, t_max=24.0 * 30, seed=12)
for event in run.events:
    if "health" in event.transition or "comfort" in event.transition:
        print(f"{event.time:7.1f} h  {event.transition}: {event.from_state} -> {event.to_state}")
```

With seed 12, the main heater fails in its first hour and is repaired 32 h
later; the backup covers the gap. At 192 h the main heater fails again,
and 10 h later so does the backup: the room cools toward the outside,
crosses 10 °C at 213 h, and recovers once the main heater is back at 225 h.

![Room and outside temperatures around the double failure, with the periods each heater was down](../assets/figures/example-heated-room-trajectory-light.svg#only-light){ .figure }
![Room and outside temperatures around the double failure, with the periods each heater was down](../assets/figures/example-heated-room-trajectory-dark.svg#only-dark){ .figure }

## Over a winter

A campaign over 90 days estimates, day by day, the probability that the
room has frozen at least once (the `reached` measure of the *frozen*
indicator) and the mean number of hours spent under 17 °C (the sojourn of
the *cold* predicate):

<!-- skip -->
```python
days = [24.0 * d for d in range(91)]
winter = pyraichu.monte_carlo(model, nb_runs=5000, t_max=24.0 * 90, samples=days, seed=1)
frozen, cold = winter.indicators["frozen"], winter.indicators["cold"]
frozen.reached_mean, frozen.reached_ci, cold.sojourn_mean, cold.sojourn_ci
```

![Probability of at least one freeze, and mean hours under 17 °C, over a 90-day winter with 95 % bands](../assets/figures/example-heated-room-winter-light.svg#only-light){ .figure }
![Probability of at least one freeze, and mean hours under 17 °C, over a 90-day winter with 95 % bands](../assets/figures/example-heated-room-winter-dark.svg#only-dark){ .figure }

## Reading the results

Over a 90-day winter the room freezes at least once with probability
0.167 (95 % interval [0.157, 0.177]), and a winter counts 0.187 freezes on
average: once in a while it freezes twice. The room spends 5.1 hours under
17 °C per winter on average ([4.8, 5.5]), of which 2.7 under 10 °C.

Both curves grow almost linearly. After the first days the two heaters are
in their steady regime, and a freeze is a rare event whose rate hardly
changes along the winter.

The campaign also carries its own check. A heater's failures and repairs
do not depend on the temperature, so its unavailability has the closed
form $\dfrac{\lambda}{\lambda+\mu}\big(1 - e^{-(\lambda+\mu)t}\big)$, which is 0.0458 at day 90. The
campaign estimates 0.0510 for the main heater, with a 95 % interval
[0.0452, 0.0575] that holds the exact value.

*Figures produced by `docs/figures/example_heated_room.py`, schematic by
`docs/figures/schematic_heated_room.py`.*
