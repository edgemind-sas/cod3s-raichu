# Confidence intervals

A study reports *"probability of the feared event: 0.0483"*. Nothing in
that number says whether it came from 500 replicas or from 100 000, and
therefore nothing says at what precision it is known. Every Monte-Carlo
estimate RAICHU produces carries a **confidence interval** next to the
figure, at a level the study declares.

The dispersion statistics that were already there, the standard
deviation and the quantiles, do not answer this. They describe the
**population of trajectories**: how spread out the replicas are. An
interval describes the **estimator**: how well the campaign pinned the
number down. A campaign of 500 and one of 100 000 on the same model have
the same standard deviation and very different intervals.

## Reading one

```python
import pyraichu

model = pyraichu.load_model({
    "name": "unit",
    "components": [{"name": "C", "automata": [{
        "name": "health", "states": ["ok", "ko"], "init": "ok",
        "transitions": [
            {"name": "fail", "source": "ok", "targets": ["ko"],
             "distrib": "exp", "rate": 0.01},
        ]}]}],
    "indicators": [{"name": "C_ko", "target": "state",
                    "component": "C", "automaton": "health", "state": "ko"}],
})

estimates = pyraichu.monte_carlo(
    model, nb_runs=2000, t_max=10.0, samples=[10.0], seed=1
)
ko = estimates.indicators["C_ko"]
print(f"P = {ko.mean[0]:.4f} "
      f"[{ko.ci.low[0]:.4f}, {ko.ci.high[0]:.4f}] "
      f"at {ko.ci.level:.0%} ({ko.ci.method})")
```

Each indicator carries four intervals, one per estimator: `ci` on the
sampled value, `sojourn_ci` on the cumulated sojourn,
`nb_occurrences_ci` on the occurrence count, and `reached_ci` on the
probability of having been active at least once (a proportion, so
Wilson whatever the indicator's kind). Each is a
`ConfidenceInterval` holding `level`, `method`, `low` and `high`, the
last two being series over the schedule instants, like `mean` itself.
`half_width(k)` gives the `±` a report prints.

## The level is a study parameter

It is not a constant of the library. A campaign that answers to a
regulator at 99 % states so, and the whole result is reported at 99 %:

```python
strict = pyraichu.monte_carlo(
    model, nb_runs=2000, t_max=10.0, samples=[10.0], seed=1, confidence=0.99
)
loose = pyraichu.monte_carlo(
    model, nb_runs=2000, t_max=10.0, samples=[10.0], seed=1, confidence=0.80
)
assert strict.confidence == 0.99
assert strict.indicators["C_ko"].ci.half_width() > \
       loose.indicators["C_ko"].ci.half_width()
print("99 % is wider than 80 %, on the same replicas")
```

Stating nothing applies `pyraichu.DEFAULT_CONFIDENCE`, the conventional
0.95. The level applied always comes back, on `McEstimates.confidence`
and on every interval, so a figure never travels without the precision
it is claimed at. A level outside `(0, 1)` is refused **before** the
replicas run, not after paying for them.

## How many replicas do I need?

The half-width falls as `1/√n`: a hundredfold in replicas divides it by
ten. That is the practical use of the interval, sizing the campaign
instead of guessing at it.

```python
import math

widths = {}
for nb_runs in (200, 20_000):
    est = pyraichu.monte_carlo(
        model, nb_runs=nb_runs, t_max=10.0, samples=[10.0], seed=3
    )
    widths[nb_runs] = est.indicators["C_ko"].ci.half_width()

print(f"{widths[200]:.4f} on 200 replicas, {widths[20_000]:.4f} on 20 000")
assert math.isclose(widths[200] / widths[20_000], 10.0, rel_tol=0.15)
```

## Which construction, and why

The method is decided from the **declared type of the indicator**, never
from a scan of the observed values: an integer indicator that happened
to stay in `{0, 1}` over one campaign is not a probability, and reading
it as one would make the reported precision depend on the draw.

| Estimator | Method | Why |
|---|---|---|
| Sampled value of a **state** indicator, or of a **boolean** attribute | `wilson` | The model says the per-replica draw is in `{0, 1}`: the estimator is a binomial proportion. |
| Sampled value of an int/float attribute | `normal` | A general mean; central-limit interval. |
| Cumulated sojourn, occurrence count | `normal` | Idem. |
| Anything, on fewer than two replicas | `undefined` | No dispersion is observable; the bounds repeat the estimate and claim nothing. |

Where a campaign observed no dispersion **at all** the construction is
still the one this table names, and the bound comes from the frequency
of a departure instead of from its spread: see
[below](#when-every-replica-said-the-same-thing).

**Wilson rather than the textbook `p ± z·√(p(1−p)/n)`**, for one
decisive reason. On a feared event so rare that no replica reached it,
the textbook interval collapses to `[0, 0]`: it declares the event
impossible on the strength of a finite campaign. Wilson answers what the
campaign actually establishes, `[0, z²/(n + z²)]`, which for 500
replicas at 95 % is `[0, 0.0076]`.

```python
never = pyraichu.load_model({
    "name": "rare",
    "components": [{"name": "C", "automata": [{
        "name": "health", "states": ["ok", "ko"], "init": "ok",
        "transitions": [
            {"name": "fail", "source": "ok", "targets": ["ko"],
             "distrib": "exp", "rate": 1e-12},
        ]}]}],
    "indicators": [{"name": "C_ko", "target": "state",
                    "component": "C", "automaton": "health", "state": "ko"}],
})
est = pyraichu.monte_carlo(never, nb_runs=500, t_max=1.0, samples=[1.0], seed=1)
ko = est.indicators["C_ko"]
assert ko.mean[0] == 0.0 and ko.ci.low[0] == 0.0
print(f"nothing observed in 500 replicas: P <= {ko.ci.high[0]:.4f}")
```

Wilson bounds also stay inside `[0, 1]` at every sample size, which the
textbook interval does not.

## When every replica said the same thing

The same defect reappears on the two normal estimators as soon as the
sample is **constant**. On the real campaign that exposed it, 1000
replicas of an 8760 h study, the occurrence count of the feared event
read:

```text
nb of occurrences   1.0000 +/- 0.0000   normal   [1.000000, 1.000000]
```

An interval of zero width, claiming the mean number of occurrences is
*exactly* one on the strength of 1000 replicas. It could be 0.999. The
replica-count guard does not catch this: there are a thousand replicas,
they are simply all identical.

What a constant sample establishes is not a spread. It is that **no
replica departed from the value in `n` draws**, so the frequency of a
departure is a proportion observed at zero, bounded above by
`z**2 / (n + z**2)`: the rule of three in exact form, `3.84/n` at 95 %,
and the very expression Wilson returns at `p = 0`.

Turning a frequency into bounds on a mean needs the **size** of one
departure, which is no more observable than the spread was. It is read
off the declaration of the quantity, exactly as the construction is:

| Estimator | One departure is | So the bound is |
|---|---|---|
| Occurrence count | one occurrence, and never below zero | `value ± ε`, floored at 0 |
| Cumulated sojourn of a 0/1 indicator | the elapsed time `t`, since the sojourn lives in `[0, t]` | `value ± ε·t`, floored at 0 |
| Sampled value of a state or boolean indicator | one unit; Wilson already answers | `[0, ε]` and `[1−ε, 1]` |
| Sampled value of a numeric attribute | one unit of the attribute, no declared sign | `value ± ε` |

A sojourn is charged the elapsed time and not "one unit of time" for a
reason that is not cosmetic: a unit of time is the model's own choice,
and a bound charged in units would answer differently on a model written
in hours and on the same model written in seconds.

```python
est = pyraichu.monte_carlo(never, nb_runs=500, t_max=1.0, samples=[1.0], seed=1)
ko = est.indicators["C_ko"]
assert ko.sojourn_mean[0] == 0.0 and ko.sojourn_std[0] == 0.0
for interval in (ko.ci, ko.sojourn_ci, ko.nb_occurrences_ci):
    assert interval.constant_sample[0], "no replica departed at all"
    assert interval.high[0] > 0.0, "and yet the event is not impossible"
print(f"nothing observed in 500 replicas: bounded above by "
      f"{ko.nb_occurrences_ci.high[0]:.5f}, not by 0")
```

The **construction does not change**. A dispersion of zero is a draw,
not a declaration of type, so it may not turn a sojourn into a
proportion: `method` still names the construction the model asked for,
and `constant_sample` marks the instants where the frequency bound
answered in its place. Wherever the sample moved, the bounds are the
central-limit ones, to the last bit.

Two boundaries are worth knowing. An **occurrence count has no upper
end**, so on a constant sample no finite bound on its mean holds in the
worst case: a replica that departed could have counted a thousand.
Charging one occurrence is the *narrowest* statement the observation
admits, and it is exact when the count is 0 or 1, which a target-stopped
campaign guarantees. And at the **origin of the schedule** a cumulated
sojourn is the integral over an empty interval: zero for every
trajectory of every model, so its interval is `[0, 0]` there and that
width of zero is an identity, not a claim.

## What the interval does not say

- **It is not a tolerance on the model.** It bounds the sampling error of
  the campaign, not the error of the modelling. A model with the wrong
  failure rate produces a tight interval around the wrong number.
- **The normal interval is asymptotic.** Its coverage is the nominal one
  as `n` grows. Its bounds are deliberately **not clamped** to the
  support of the quantity: a negative lower bound on a sojourn time is
  the interval reporting that the normal approximation is out of its
  range at this sample size, and clamping it to zero would erase that
  signal and leave a bound that looks sound. The constant-sample bound
  is the opposite case and *is* floored: it is a closed form and not an
  approximation, so a bound below zero there would signal nothing and
  mislead. One-sidedness at the edge of the support is what the rule of
  three has always been.
- **No Student `t`.** Substituting `t_{n−1}` for `z` corrects the one
  term that is *not* the dominant error: `t` is exact only when the
  per-replica quantity is itself Gaussian, which a sojourn time or an
  occurrence count never is. At the sizes a campaign runs the two
  coincide (`1.9647` against `1.9600` at `n = 500`), and at the sizes
  where they differ the assumption behind `t` has already failed. The
  binary case, the one where a genuinely small-sample construction
  exists, gets Wilson instead.
- **The replicas must be independent**, which the
  [reproducibility contract](reproducibility.md) guarantees: replica *k*
  draws from its own substream of the master seed.

## Cost

The intervals are closed forms over sums the reduction already holds:
they cost `O(indicators × instants)` arithmetic and never revisit a
replica. Measured with `cargo bench -p raichu-montecarlo` on the
cheapest campaign there is, a two-state model over a 20-instant
schedule:

| Replicas | Campaign | Intervals | Share |
|---|---|---|---|
| 500 | 348 µs | 278 ns | 0.08 % |
| 2 000 | 1.09 ms | 277 ns | 0.03 % |

The interval column does not move with the replica count: it is the same
work whatever the campaign size, which is why the share only falls as
studies grow.
