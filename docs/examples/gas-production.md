# Gas production with a buffer reservoir

Two production units feed a customer whose demand neither can meet
alone; a reservoir covers the shortfall while one unit, or both, are
being repaired. Production is lost when the reservoir runs dry. The
question is a production-availability one: what share of the time, and
what share of the volume, is delivered, and how often is production lost,
in part or in full?

This benchmark of **dynamic reliability for production availability**
comes from an industrial exercise (Labeau and Dutuit, 2004). It needs
what a static or purely discrete model cannot give: a continuous variable
(the reservoir level) whose speed depends on the discrete state, events
fired when that variable reaches a bound (empty, full), non-exponential
repairs, failure rates that depend on how hard a unit is working, and a
reservoir refilled at the instant both units are back. Eymard and Mercier
(2008) quantified its four cases with a regenerative Monte-Carlo
estimator and a finite-volume scheme; their Monte-Carlo values, with
confidence intervals, are the reference here.

## The benchmark

| Aspect | Hypothesis (Eymard and Mercier 2008, section 2.1, Tables 1, 2, 7 to 12) |
|---|---|
| Units | U1, at most 3200 m³/h; U2, at most 5500 m³/h; both up shares the demand |
| Demand | 7500 m³/h, constant |
| Reservoir | 220 000 m³; with one unit up it supplies the complement (4300 m³/h with U1 alone, 2000 m³/h with U2 alone), with both down the whole demand |
| Failures | exponential: λ1 = 1/20 000 h⁻¹, λ2 = 1/4000 h⁻¹ in case 1, ten times higher in cases 2 to 4 |
| Running alone | in cases 3 and 4 a unit running alone at its maximum fails at λ′ = 10 λ |
| Repairs | lognormal, ln(duration) ~ N(µ, σ²): U1 µ = 0.23, σ = 2.25 (mean 15.8 h); U2 µ = 0.50, σ = 1.83 (mean 8.8 h); a repair goes on when the other unit fails, and both can be repaired at once |
| Refill | cases 1 to 3: the reservoir is **full again at the instant** both units are up; case 4: both units run at their maximum and refill it at 1200 m³/h, failing at λ′, until it is full |
| Measures | stationary: availability A (share of time with nominal production), production availability PA (delivered over nominal volume), annual frequencies of loss of nominal production f_LNP (the reservoir runs dry) and of total loss f_TLP (both units down and the reservoir dry), with a year of 8766 h |

## Bibliographic note

- **The case.** Labeau and Dutuit (2004) present it as an exercise from
  Air Liquide submitted to the ESRA technical committee on dependability
  modelling in 2003, and part of the IMdR test-case collection. They give
  its data and quantify it with Monte-Carlo on Petri nets (reservoir
  discretised) and with a dedicated continuous Monte-Carlo code, without
  confidence intervals.
- **The reference values.** Eymard and Mercier (2008) write the system as
  a piecewise-deterministic Markov process, give its jump rates and reset
  maps in full (Tables 7 to 12), and quantify the four cases with a
  regenerative Monte-Carlo estimator (10⁷ to 10⁸ cycles, 95 % confidence
  intervals, Tables 3 to 6) and with a finite-volume scheme. Both papers
  are deposited on HAL under CC BY 4.0; the data are restated here in our
  own tables.
- **One published cell is unusable.** Table 5, case 2 prints
  f_TLP = 0.11295 with the interval [0.11367, 0.11692], which does not
  contain it. It is compared with its interval only.

## The model

The model is written from the published description, in
[`examples/gas_production/model.py`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/gas_production),
one document per case. Each part of the process maps onto a RAICHU
construct:

| Process | RAICHU |
|---|---|
| unit up or down | an automaton per unit |
| failure at λ, or λ′ when running at its maximum | exponential transition whose rate is an expression of the states (`rate_expr`) |
| lognormal repair, independent of the other unit | `lognormal` transition, whose guard reads nothing else, so a failure of the other unit never restarts it |
| reservoir level, drained or refilled at a state-dependent speed | ODE on `reservoir.level` |
| level held at 0 while empty, at capacity while full | a `store` automaton (`full`, `partial`, `empty`) whose state clamps the speed |
| the reservoir runs dry, or fills up | **watched** transitions: the crossing is located on the solver's dense output, not stepped over |
| refill at the instant both units are up (cases 1 to 3) | an **edge effect** of the repair on the level, an ODE target: a reset map, the jump of a piecewise-deterministic Markov process (Davis 1984), since RAICHU 0.80 |
| delivered production, total loss | attributes written by a sensitive function, observed as indicators |

The repair that completes "both up" writes `level := 220 000` on its
firing edge; integration then restarts from there:

<!-- skip -->
```python
repair = {"name": "repair", "source": "down", "targets": ["up"],
          "distrib": "lognormal", "mu": 0.23, "sigma": 2.25,
          "effects": [{"target": {"component": "reservoir", "attribute": "level"},
                       "value": if_(up(other), num(220_000.0), LEVEL)}]}
```

## Estimation

All published measures are stationary. Each history starts from the
regeneration state of the published estimator, both units up and the
reservoir full, and runs to a horizon T; the measures are time averages
over [0, T], with a standard error from the spread between histories.
Starting from the regeneration state biases a time average by O(1/T).
At an equal budget (4 × 10¹⁰ simulated hours), case 2 shows it:

| Horizon T | Histories | A | f_TLP (per year) |
|---|---|---|---|
| 10⁵ h | 400 000 | 0.990028 ± 1.9e-5 | 0.11270 ± 0.00038 |
| 10⁶ h | 40 000 | 0.989942 ± 2.1e-5 | 0.11411 ± 0.00044 |
| 10⁷ h | 4 000 | 0.989948 ± 2.2e-5 | 0.11404 ± 0.00046 |

At 10⁵ h the start still weighs about three standard errors; between
10⁶ and 10⁷ h nothing is measurable. The campaigns use T = 10⁶ h and
40 000 histories per case.

The level moves linearly between two events, so the integrator's step
cap is raised from its default 0.1 h to 100 h: the trajectory is the
same, located crossings included, and a campaign takes one to four
minutes on 24 threads instead of hours.

## Results

`compare.py` reads the campaigns and the published values:

| Case | Quantity | RAICHU (95 %) | Reference (95 %) | Gap (combined SE) |
|---|---|---|---|---|
| 1 | A | 0.998977 ± 1.3e-05 | 0.9989852 [0.998979, 0.9989915] | -1.2 |
| 1 | PA | 0.999518 ± 4.8e-06 | 0.9995232 [0.9995209, 0.9995255] | -1.8 |
| 1 | f_LNP | 0.0766168 ± 0.00025 | 0.0764 [0.07627, 0.07652] | +1.5 |
| 1 | f_TLP | 0.00116478 ± 4e-05 | 0.001151 [0.001132, 0.001171] | +0.6 |
| 2 | A | 0.989942 ± 4.2e-05 | 0.9898986 [0.9898195, 0.9899777] | +1.0 |
| 2 | PA | 0.995233 ± 1.5e-05 | 0.9952215 [0.9951946, 0.9952484] | +0.7 |
| 2 | f_LNP | 0.752769 ± 0.00079 | 0.7526 [0.7514, 0.7538] | +0.2 |
| 2 | f_TLP | 0.114114 ± 0.00086 | 0.11295 [0.11367, 0.11692] | inside the published interval |
| 3 | A | 0.988487 ± 4.6e-05 | 0.9884501 [0.9883955, 0.9885048] | +1.0 |
| 3 | PA | 0.994051 ± 2.1e-05 | 0.9940342 [0.994001, 0.9940584] | +0.9 |
| 3 | f_LNP | 0.829235 ± 0.00083 | 0.8291 [0.8282, 0.83] | +0.2 |
| 3 | f_TLP | 1.20845 ± 0.0076 | 1.2143 [1.2053, 1.2234] | -1.0 |
| 4 | A | 0.972784 ± 6.7e-05 | 0.9716373 [0.9712818, 0.9719929] | +6.2 |
| 4 | PA | 0.985878 ± 3e-05 | 0.9853242 [0.9851754, 0.9854729] | +7.1 |
| 4 | f_LNP | 3.38365 ± 0.003 | 3.4985 [3.4887, 3.5082] | -22.1 |
| 4 | f_TLP | 2.7891 ± 0.011 | 2.9331 [2.871, 2.9953] | -4.5 |

![Gap between RAICHU and the published regenerative Monte-Carlo estimate, per case and quantity, in combined standard errors](../assets/figures/example-gas-production-gaps-light.svg#only-light){ .figure }
![Gap between RAICHU and the published regenerative Monte-Carlo estimate, per case and quantity, in combined standard errors](../assets/figures/example-gas-production-gaps-dark.svg#only-dark){ .figure }

**Cases 1 to 3 agree.** The eleven usable cells sit within 1.8 combined
standard errors of the reference, and the twelfth inside its published
interval.

**Case 4 does not, and neither do its references.** The references
themselves disagree there:

| Case 4 | RAICHU | Eymard-Mercier, Monte-Carlo | Eymard-Mercier, finite volumes | Labeau-Dutuit, continuous Monte-Carlo |
|---|---|---|---|---|
| A | 0.97278 | 0.97164 | 0.97290 | 0.9744 |
| PA | 0.98588 | 0.98532 | 0.98570 | 0.9867 |
| f_LNP | 3.384 | 3.4985 | 3.310 | 3.300 |
| f_TLP | 2.789 | 2.933 | 2.737 | 2.584 |

The three published estimates of case 4 are many of their own
uncertainties apart. The regenerative Monte-Carlo, which agrees with
RAICHU everywhere else, is here the outlying value; RAICHU sits within
the spread of the three, closest to the finite volumes on A and PA. Case 4 is the only one with a rate that depends on the
reservoir being full, and the only one where the published tables leave
part of the behaviour implicit (when the rates switch back, what happens
with both units up and the reservoir empty). RAICHU implements the
tables as written; which reading produced each published value cannot be
established from the sources. Case 4 is therefore reported, not used as
a validation point.

## Two engine changes this example led to

- **Reset maps.** The instant refill of cases 1 to 3 is a jump of the
  continuous state. RAICHU refused an edge effect on an ODE target until
  0.80; it now accepts it as a reset map (see the
  [model schema](../reference/model-schema.md#edge-effects)).
- **Event location far from the origin.** A history of half a million
  hours hung silently: beyond about t = 5 × 10⁵ the gap between two
  adjacent doubles exceeds the default event tolerance of 10⁻¹⁰, and the
  bisection that locates a crossing could no longer shrink its bracket.
  Since 0.79.1 it stops at the resolution of the time axis.

## Reproducing

```bash
cd examples/gas_production
python model.py                                # the four model documents
for c in 1 2 3 4; do python run_campaign.py $c; done           # 1 to 4 min each
python run_campaign.py 2 --runs 400000 --horizon 1e5 --tag T1e5
python run_campaign.py 2 --runs 4000 --horizon 1e7 --tag T1e7
python compare.py                              # the table above
```

Every RAICHU number on this page is read from the files under
[`examples/gas_production/results`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/gas_production/results);
the seed is fixed (2008), so a rerun reproduces them, wall-clock times
aside. *Chart produced by `docs/figures/example_gas_production.py`.*

## References

- Labeau, P.-E., Dutuit, Y. (2004). Fiabilité dynamique et disponibilité
  de production : un cas illustratif. *Actes du 14e congrès Lambda Mu*,
  Bourges, vol. 2, 431-436. Open access (CC BY 4.0).
  [hal-01570847](https://hal.science/hal-01570847)
- Eymard, R., Mercier, S. (2008). Comparison of numerical methods for the
  assessment of production availability of a hybrid system. *Reliability
  Engineering & System Safety*, 93(1), 168-177.
  [doi:10.1016/j.ress.2006.12.001](https://doi.org/10.1016/j.ress.2006.12.001);
  author version open access (CC BY 4.0),
  [hal-00693079](https://hal.science/hal-00693079)
- Davis, M. H. A. (1984). Piecewise-deterministic Markov processes: a
  general class of non-diffusion stochastic models. *Journal of the Royal
  Statistical Society, Series B*, 46(3), 353-376.
  [doi:10.1111/j.2517-6161.1984.tb01308.x](https://doi.org/10.1111/j.2517-6161.1984.tb01308.x)
