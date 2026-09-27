# Heated tank

A tank of water is heated from below and kept between two levels by two
inlet pumps and one outlet valve. The units fail at random, stuck open or
stuck closed, more often as the water gets hotter, and a stuck unit ignores
the control. Three things can go wrong: the tank runs dry, overflows, or
boils.

This is the benchmark of **dynamic reliability**. Aldemir (1987) introduced
it with the level alone; Marseguerra and Zio (1996) added the temperature
and made it a Monte-Carlo reference. It needs both halves of a hybrid
engine at once: the level and the temperature follow differential
equations whose right-hand side changes with every failure, the control
fires when the level crosses a set-point, and the failure rates move with
the temperature. The constants below are those published by Zhang et al.
(2013), and this exact model is part of RAICHU's cross-validation against
PyCATSHOO.

## The installation

![Heated tank: two inlet pumps P1 and P2, an outlet valve V, a heat source under the tank, the control set-points at 6 and 8 m and the three top events](../assets/schematics/heated-tank-light.svg#only-light){ .figure }
![Heated tank: two inlet pumps P1 and P2, an outlet valve V, a heat source under the tank, the control set-points at 6 and 8 m and the three top events](../assets/schematics/heated-tank-dark.svg#only-dark){ .figure }

## Hypotheses

| Aspect | Hypothesis |
|---|---|
| Level | $\dfrac{dh}{dt} = G\,(n_p - n_v)$, with $n_p$ the flowing pumps, $n_v$ the flowing valve and $G = 1.5$ m/h per flowing unit |
| Temperature | $\dfrac{d\theta}{dt} = \dfrac{G\,n_p\,(\theta_{in} - \theta) + 23.88915}{h}$, inlet water at $\theta_{in} = 15$ °C |
| Initial state | $h = 7$ m, $\theta = 30.9261$ °C, P1 on, P2 off, V on: an equilibrium, nothing moves until a unit fails |
| Control | below 6 m: P1 and P2 on, V off; above 8 m: P1 and P2 off, V on. A stuck unit is not commanded |
| Unit states | on, off, stuck on, stuck off. A failure takes an on or off unit to either stuck state with equal chance |
| Failure rate | $\lambda(\theta) = \hat\lambda\,a(\theta)$ with $a(\theta) = \dfrac{b_1 e^{b_c(\theta-20)} + b_2 e^{-b_d(\theta-20)}}{b_1 + b_2}$, $b_1 = 3.0295$, $b_2 = 0.7578$, $b_c = 0.05756$, $b_d = 0.2301$ |
| Base rates $\hat\lambda$ | P1: 1/438 h⁻¹, P2: 1/350 h⁻¹, V: 1/640 h⁻¹ |
| Top events | dry-out $h \le 4$ m, overflow $h \ge 10$ m, overheat $\theta \ge 100$ °C. The first one reached ends the story: the tank is absorbed and its physics stops |
| Not modelled | repairs, common-cause failures, the heat source failing |

$a(\theta)$ is smallest at 20 °C, where it equals 1. It is already 1.5 at the
initial 30.9 °C, 8 at 60 °C and 80 at 100 °C: a hot tank wears its units
much faster, which is what couples the two halves of the model.

## The model

The model is plain data. A few helpers keep the expressions short:

```python
import pyraichu

# Constants of the benchmark (units: m, °C, h).
G = 1.5                    # level rate moved by one active unit, m/h
THETA_IN = 15.0            # inlet water temperature, °C
HEAT = 23.88915            # heating term, °C·m/h
H0, THETA0 = 7.0, 30.9261  # initial level and temperature (an equilibrium)
LOW, HIGH = 6.0, 8.0       # control set-points
DRY, OVER, HOT = 4.0, 10.0, 100.0   # the three top events
B1, B2, BC, BD = 3.0295, 0.7578, 0.05756, 0.2301
UNITS = {  # name: (base failure rate λ̂ in 1/h, initially on, is the outlet valve)
    "pump1": (1 / 438, True, False),
    "pump2": (1 / 350, False, False),
    "valve": (1 / 640, True, True),
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

level, theta = attr("tank", "level"), attr("tank", "temperature")
inflow = {"op": "port_agg", "port": {"component": "tank", "port": "pumps_in"}, "agg": "sum"}
outflow = {"op": "port_agg", "port": {"component": "tank", "port": "valve_in"}, "agg": "sum"}
```

A unit is a four-state automaton. Its start and stop transitions are
**watched**: they fire exactly when the level crosses a set-point, located
by root-finding on the solver's dense output. Its four failures draw an
exponential law whose rate is an **expression of the temperature**, so the
engine integrates the cumulative hazard along the trajectory rather than
freezing the rate between events. They are `monitored`, which records them
in the trajectory's sequence:

```python
def failure_rate(base):
    """λ(θ) = λ̂ · a(θ), a(θ) = (b1 e^{bc(θ-20)} + b2 e^{-bd(θ-20)}) / (b1 + b2),
    halved: a failure is stuck-on or stuck-off with equal chance."""
    shifted = {"op": "sub", "lhs": theta, "rhs": num(20.0)}
    a = {"op": "add", "args": [
        {"op": "mul", "args": [num(B1), {"op": "exp", "arg": {"op": "mul", "args": [num(BC), shifted]}}]},
        {"op": "mul", "args": [num(B2), {"op": "exp", "arg": {"op": "mul", "args": [num(-BD), shifted]}}]},
    ]}
    return {"op": "mul", "args": [num(base / 2 / (B1 + B2)), a]}

def unit(name, base, on, is_valve):
    # The control law: below 6 m both pumps run and the valve closes,
    # above 8 m the reverse. A stuck unit ignores it.
    start = cmp("ge", level, num(HIGH)) if is_valve else cmp("le", level, num(LOW))
    stop = cmp("le", level, num(LOW)) if is_valve else cmp("ge", level, num(HIGH))
    transitions = [
        {"name": "start", "source": "off", "targets": ["on"], "guard": start, "distrib": "watched"},
        {"name": "stop", "source": "on", "targets": ["off"], "guard": stop, "distrib": "watched"},
    ] + [
        {"name": f"{stuck}_from_{src}", "source": src, "targets": [stuck],
         "distrib": "exp", "rate_expr": failure_rate(base), "monitored": True}
        for src in ("on", "off") for stuck in ("stuck_on", "stuck_off")
    ]
    flowing = {"op": "bool", "bool_op": "or",
               "args": [in_state(name, "mode", "on"), in_state(name, "mode", "stuck_on")]}
    return {
        "name": name,
        "attributes": [{"name": "flowing", "kind": "bool", "init": {"kind": "bool", "value": on}}],
        "ports": [{"name": "flow_out", "dir": "out", "attr": "flowing"}],
        "automata": [{"name": "mode", "states": ["on", "off", "stuck_on", "stuck_off"],
                      "init": "on" if on else "off", "transitions": transitions}],
        "sensitive_functions": [{"name": "update_flowing", "effects": [
            {"target": {"component": name, "attribute": "flowing"}, "value": flowing}]}],
    }
```

The tank carries the two differential equations and the three top events,
watched as well. The top events are also the model's `targets`, which ends
a sequence campaign's trajectory at the first of them:

```python
ok = in_state("tank", "status", "ok")

def frozen_after_top_event(rate):
    # The benchmark measures the first top event: once one occurs, the
    # trajectory is absorbed and the physics stops.
    return {"op": "if", "cond": ok, "then": rate, "otherwise": num(0.0)}

tank = {
    "name": "tank",
    "attributes": [
        {"name": "level", "kind": "float", "init": {"kind": "float", "value": H0}},
        {"name": "temperature", "kind": "float", "init": {"kind": "float", "value": THETA0}},
    ],
    "ports": [{"name": "pumps_in", "dir": "in"}, {"name": "valve_in", "dir": "in"}],
    "automata": [{"name": "status", "states": ["ok", "dryout", "overflow", "overheat"],
                  "init": "ok", "transitions": [
        {"name": "dryout", "source": "ok", "targets": ["dryout"],
         "guard": cmp("le", level, num(DRY)), "distrib": "watched"},
        {"name": "overflow", "source": "ok", "targets": ["overflow"],
         "guard": cmp("ge", level, num(OVER)), "distrib": "watched"},
        {"name": "overheat", "source": "ok", "targets": ["overheat"],
         "guard": cmp("ge", theta, num(HOT)), "distrib": "watched"},
    ]}],
    "equations": [
        # dh/dt = G (pumps − valve)
        {"target": "level", "kind": "ode", "expr": frozen_after_top_event(
            {"op": "mul", "args": [num(G), {"op": "sub", "lhs": inflow, "rhs": outflow}]})},
        # dθ/dt = (G · pumps · (θ_in − θ) + heat) / h
        {"target": "temperature", "kind": "ode", "expr": frozen_after_top_event(
            {"op": "div", "lhs": {"op": "add", "args": [
                {"op": "mul", "args": [inflow, num(G), {"op": "sub", "lhs": num(THETA_IN), "rhs": theta}]},
                num(HEAT)]}, "rhs": level})},
    ],
}

model = pyraichu.load_model({
    "name": "heated_tank",
    "components": [tank] + [unit(n, *spec) for n, spec in UNITS.items()],
    "connections": [
        {"from": {"component": n, "port": "flow_out"},
         "to": {"component": "tank", "port": "valve_in" if spec[2] else "pumps_in"}}
        for n, spec in UNITS.items()
    ],
    # The three top events end a trajectory, and the failures that led
    # there are recorded as its sequence.
    "targets": [{"name": e, "component": "tank", "automaton": "status", "state": e}
                for e in ("dryout", "overflow", "overheat")],
    "indicators": [
        {"name": e, "target": "state", "component": "tank", "automaton": "status", "state": e}
        for e in ("dryout", "overflow", "overheat")
    ] + [
        {"name": a, "target": "attribute", "attr": {"component": "tank", "attribute": a}}
        for a in ("level", "temperature")
    ],
})
```

## One trajectory

```python
run = pyraichu.simulate(model, t_max=100.0, seed=3)
for event in run.events:
    print(f"{event.time:8.2f} h  {event.transition}: {event.from_state} -> {event.to_state}")
```

With seed 3, P2 sticks off while idle at 15 h. At 63 h P1 sticks off while
running: nothing fills the tank any more, the level falls, and at 6 m the
control closes the valve. Below 6 m the control would start both pumps,
but both are stuck: the level holds, and with no cold water coming in the
heat source raises the temperature by about 4 °C an hour. The valve
sticks off at 77 h, which changes nothing any more, and the water reaches
100 °C at 80 h.

![Level and temperature of the heated tank along one trajectory, with its events](../assets/figures/example-heated-tank-trajectory-light.svg#only-light){ .figure }
![Level and temperature of the heated tank along one trajectory, with its events](../assets/figures/example-heated-tank-trajectory-dark.svg#only-dark){ .figure }

## Probability of the three top events

A Monte-Carlo campaign estimates the probability that each top event has
occurred by time `t`. Because a top event absorbs the tank, the mean of its
state indicator at `t` is exactly that cumulative probability:

<!-- skip -->
```python
instants = [10.0 * k for k in range(101)]
estimates = pyraichu.monte_carlo(model, nb_runs=20_000, t_max=1000.0,
                                 samples=instants, seed=1)
overflow = estimates.indicators["overflow"]
overflow.mean, overflow.ci.low, overflow.ci.high
```

![Cumulative probability of dry-out, overflow and overheat over 1000 h, with 95 % confidence bands](../assets/figures/example-heated-tank-top-events-light.svg#only-light){ .figure }
![Cumulative probability of dry-out, overflow and overheat over 1000 h, with 95 % confidence bands](../assets/figures/example-heated-tank-top-events-dark.svg#only-dark){ .figure }

## Sequences leading to each top event

A sequence campaign stops every trajectory at its first top event and
keeps the ordered failures that led there, reduced to the minimal ones:

<!-- skip -->
```python
sequences = pyraichu.analyse_sequences(model, nb_runs=20_000, t_max=1000.0, seed=1)
```

![The most frequent minimal sequences leading to a top event, as a fraction of the trajectories](../assets/figures/example-heated-tank-sequences-light.svg#only-light){ .figure }
![The most frequent minimal sequences leading to a top event, as a fraction of the trajectories](../assets/figures/example-heated-tank-sequences-dark.svg#only-dark){ .figure }

## Reading the results

Over 1 000 h, overflow is the most likely top event (0.435, 95 % interval
[0.428, 0.442]), then overheat (0.202) and dry-out (0.082). The other 28 %
of the trajectories reach none of them.

Overheat stops growing after about 500 h because most of it takes a
single failure: the valve sticking off, 12 % of all trajectories on its
own. With the outlet closed the level climbs to 8 m, the control stops the
pumps, and no cold water reaches the tank any more.

Overflow takes two failures, a pump stuck on followed by the other pump
stuck on or the valve stuck off. The same pair appears in both orders with
different weights: *pump 2 stuck on, then valve stuck off* explains 9 % of
the trajectories, *valve stuck off, then pump 1 stuck on* 4.5 %, because a
closed valve first tends to end in overheat before a pump sticks. The
order of the failures is part of the scenario, which is what a static
fault tree cannot say. Dry-out takes three failures: both pumps stuck off
and the valve stuck on.

*Figures produced by `docs/figures/example_heated_tank.py`, schematic by
`docs/figures/schematic_heated_tank.py`.*

## References

- Aldemir, T. (1987). Computer-assisted Markov failure modeling of process
  control systems. *IEEE Transactions on Reliability*, R-36(1), 133-144.
  [doi:10.1109/TR.1987.5222318](https://doi.org/10.1109/TR.1987.5222318)
- Marseguerra, M., Zio, E. (1996). Monte Carlo approach to PSA for dynamic
  process systems. *Reliability Engineering & System Safety*, 52(3),
  227-241.
  [doi:10.1016/0951-8320(95)00131-X](https://doi.org/10.1016/0951-8320(95)00131-X)
- Zhang, H., de Saporta, B., Dufour, F., Deleuze, G. (2013). Dynamic
  reliability by using Simulink and Stateflow. *Chemical Engineering
  Transactions*, 33, 529-534. Open access, the source of the constants
  above. [doi:10.3303/CET1333089](https://doi.org/10.3303/CET1333089)
