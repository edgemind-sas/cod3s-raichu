# Two-out-of-three pumps

Three identical pumps, each repaired when it fails; the service needs at
least two of them. It is the textbook repairable *k*-out-of-*n* system,
and its Markov model has a closed-form solution (Rausand et al., 2020,
chapter 11). That makes it the place to see RAICHU's three ways of
answering one question agree: a Monte-Carlo campaign, which gives an
estimate and a confidence interval; a sequence-tree exploration, which
gives a lower and an upper bound; and the exact value.

## Hypotheses

| Aspect | Hypothesis |
|---|---|
| Pumps | three, identical and independent: failure rate $\lambda = 2 \cdot 10^{-3}$ h⁻¹ (500 h on average), repair rate $\mu = 2 \cdot 10^{-2}$ h⁻¹ (50 h), one repair crew per pump |
| Service | up while at most one pump is failed |
| Feared event | the service down, i.e. two pumps failed at once |
| Two measures | **unreliability** $F(t)$: the probability the service has been down at least once by `t`; **unavailability** $U(t)$: the probability it is down at `t` |
| Not modelled | common-cause failures, a shared repair crew, failure on demand |

## The model

Each pump is a two-state automaton. The service is an automaton whose
**instantaneous** transitions follow the count of failed pumps, and its
*down* state is the model's `target`. The failures and repairs carry their
declared `kind` and are `monitored`, which is what a sequence campaign and
the exploration read:

```python
import pyraichu

LAMBDA = 2e-3   # failure rate of one pump, 1/h
MU = 2e-2       # repair rate of one pump, 1/h (50 h on average)
PUMPS = ("P1", "P2", "P3")

def pump(name):
    return {"name": name, "automata": [{
        "name": "health", "states": ["ok", "failed"], "init": "ok", "transitions": [
            {"name": "fail", "source": "ok", "targets": ["failed"], "distrib": "exp",
             "rate": LAMBDA, "kind": "failure", "monitored": True, "cycle_group": name},
            {"name": "repair", "source": "failed", "targets": ["ok"], "distrib": "exp",
             "rate": MU, "kind": "repair", "monitored": True, "cycle_group": name},
        ]}]}

def nb_failed():
    return {"op": "add", "args": [
        {"op": "if",
         "cond": {"op": "state_active", "state": {"component": p, "automaton": "health", "state": "failed"}},
         "then": {"op": "const", "value": {"kind": "float", "value": 1.0}},
         "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}}}
        for p in PUMPS]}

def at_least(k):
    return {"op": "cmp", "cmp": "ge", "lhs": nb_failed(),
            "rhs": {"op": "const", "value": {"kind": "float", "value": float(k)}}}

system = {"name": "system", "automata": [{
    "name": "service", "states": ["up", "down"], "init": "up", "transitions": [
        {"name": "lose", "source": "up", "targets": ["down"], "distrib": "inst", "probs": [],
         "guard": at_least(2)},
        {"name": "recover", "source": "down", "targets": ["up"], "distrib": "inst", "probs": [],
         "guard": {"op": "bool", "bool_op": "not", "args": [at_least(2)]}},
    ]}]}

model = pyraichu.load_model({
    "name": "two_out_of_three",
    "components": [pump(p) for p in PUMPS] + [system],
    "targets": [{"name": "system_down", "component": "system", "automaton": "service", "state": "down"}],
    "indicators": [{"name": "down", "target": "state", "component": "system",
                    "automaton": "service", "state": "down"}],
})
```

## The exact values

**Unreliability.** Until the service first goes down, the system is a
chain on the number of failed pumps, 0 → 1 → 2, with state 2 absorbing:
from 0 a pump fails at $3\lambda$, from 1 the failed pump is repaired at $\mu$ or
a second one fails at $2\lambda$. The probability of having reached state 2 is
a difference of two exponentials, whose rates are the roots of
$s^2 + (5\lambda + \mu)\,s + 6\lambda^2 = 0$.

**Unavailability.** The pumps are independent, so each is failed at `t`
with probability $p(t) = \dfrac{\lambda}{\lambda+\mu}\big(1 - e^{-(\lambda+\mu)t}\big)$, and the service is
down when at least two are: $U(t) = 3p^2(1 - p) + p^3$.

## Exploring the sequence tree

The exploration enumerates the paths to the feared event instead of
sampling them, and bounds what its cut-offs leave out. On this Markov
model it is exact:

```python
import math

def unreliability(t):
    """P(system down by t), repairs of single failures included: the
    chain 0 -> 1 -> 2 failed pumps, with state 2 absorbing."""
    # The two decay rates are the roots of s² + (5λ + μ) s + 6λ² = 0.
    b, c = 5 * LAMBDA + MU, 6 * LAMBDA**2
    r1 = (-b + math.sqrt(b * b - 4 * c)) / 2
    r2 = (-b - math.sqrt(b * b - 4 * c)) / 2
    return 1 - (r1 * math.exp(r2 * t) - r2 * math.exp(r1 * t)) / (r1 - r2)

result = pyraichu.explore(model, "system_down", horizon=250.0,
                          min_probability=1e-12, max_length=30)
print(f"exact    {unreliability(250.0):.8f}")
print(f"explored [{result.lower:.8f}, {result.upper:.8f}], "
      f"{len(result.sequences)} sequences")
for seq in result.sequences[:3]:
    print(f"  {seq.probability:.4e}  " + " -> ".join(f"{e['obj']} {e['attr']}" for e in seq.events))
assert result.lower <= unreliability(250.0) <= result.upper + 1e-12
```

At 250 h the two bounds and the exact value agree to the eighth digit.
The dominant sequences are the six orders of two failures with no repair;
the other 6 552 add failure and repair cycles before the fatal pair.

## Three answers to one question

The same model in a Monte-Carlo campaign gives both measures at once:
the `reached` measure of the *down* indicator is the unreliability, its
mean the unavailability.

<!-- skip -->
```python
instants = [10.0 * k for k in range(101)]
campaign = pyraichu.monte_carlo(model, nb_runs=20_000, t_max=1000.0,
                                samples=instants, seed=1)
down = campaign.indicators["down"]
down.reached_mean, down.reached_ci   # unreliability F(t)
down.mean, down.ci                   # unavailability U(t)
```

![Unreliability and unavailability of the two-out-of-three system: exact curves, Monte-Carlo estimates with 95 % bands, and exploration bounds](../assets/figures/example-two-out-of-three-measures-light.svg#only-light){ .figure }
![Unreliability and unavailability of the two-out-of-three system: exact curves, Monte-Carlo estimates with 95 % bands, and exploration bounds](../assets/figures/example-two-out-of-three-measures-dark.svg#only-dark){ .figure }

![Probability of reaching the feared event by 250 h, split by the number of events in the sequence](../assets/figures/example-two-out-of-three-lengths-light.svg#only-light){ .figure }
![Probability of reaching the feared event by 250 h, split by the number of events in the sequence](../assets/figures/example-two-out-of-three-lengths-dark.svg#only-dark){ .figure }

## Reading the results

The three answers agree. At 1 000 h the exact unreliability is 0.54795;
the exploration brackets it in [0.54794, 0.54799], and the campaign of
20 000 replicas estimates 0.5444 with a 95 % interval [0.5374, 0.5512]
that holds it. The exact unavailability at 1 000 h is 0.02329; the
campaign estimates 0.02510, interval [0.02302, 0.02736].

Monte-Carlo gives both measures from one campaign and does not care
about the model's laws. The exploration gives one measure at one
horizon, with guaranteed bounds instead of an interval, and its cost
grows with the horizon: 240 sequences at 100 h, 531 438 at 1 000 h,
because a longer horizon leaves room for more failure and repair cycles
before the fatal pair.

Those cycles are what the second chart counts. By 250 h, 72 % of the
probability of the feared event comes from two failures in a row, 23 %
from sequences with one repair before them, and each further cycle
weighs 5 to 14 times less than the one before.

*Figures produced by `docs/figures/example_two_out_of_three.py`.*

## References

- Rausand, M., Barros, A., Høyland, A. (2020). Markov analysis. In
  *System Reliability Theory: Models, Statistical Methods, and
  Applications*, chapter 11, 473-544. Wiley.
  [doi:10.1002/9781119373940.ch11](https://doi.org/10.1002/9781119373940.ch11)
