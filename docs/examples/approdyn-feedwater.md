# Feedwater supply of a steam generator (APPRODYN)

The feedwater supply of one steam generator of a 900 MW pressurised-water
reactor: a steam header (VVP), three extraction pumps (CEX), two turbo-
pumps (TPA) and the regulating valves (ARE) that hand the water to the
steam generator, all driven through a power cycle of a ramp up, eighteen
months at full power and a ramp down. Losing the supply trips the
reactor, an unplanned unavailability. The question is the probability of
a trip within the cycle, and its cause.

This is the test case of the **APPRODYN** project (Aubry et al., 2012),
built by EDF and three academic teams to compare dynamic-reliability
methods on an industrial-size system: stochastic hybrid automata,
piecewise-deterministic Markov processes and stochastic Petri nets. Its
data are, in the authors' words, representative but not real, and
restricted to use as a test case; that is the only use made of them
here. It is the hardest of the examples on this site: not because of its
size, but because the published description leaves choices open, and the
published results do not all follow from the published data.

## The case and the scope of this example

The report specifies two halves:

- **a discrete part**: the components, their failure modes, failures on
  demand, repairs, common-cause failures, standby redundancy, and the
  plant's reconfiguration logic along the power cycle (report chapter 4);
- **a continuous part**: the water level of the steam generator, a linear
  model taken from Kothare et al. (2000), held by a PID controller acting
  on the ARE valves, with trip thresholds on the level.

The continuous part cannot be rebuilt from the report. It gives neither
the controller's gains, nor the units of the flows and levels, nor the
steam flow as a function of power, and no published result shows a level
threshold causing a trip. One of the three teams could not make the
controller hold its setpoint at all. **This example models the discrete
part only**, and says so wherever it compares with a published value.

| Aspect | Hypothesis (Aubry et al. 2012, chapter 4) |
|---|---|
| Cycle | 0 to 2 % power in 10 h, to 10 % in 10 h, to 100 % in 4 h; 12 960 h at full power; 24 h ramp down. The first trip ends the history |
| VVP header | failure 2.17e-5 /h; any failure (leak or rupture) trips the plant |
| CEX pumps | 2 running, 1 in standby; running failure 4.35e-5 /h (repair 72 h, 168 h for a rupture); the standby pump is started and may refuse (1.95e-3); losing two pumps trips the plant; 5 % of the running failures are common-cause |
| TPA turbo-pumps | each the series of a turbine part (5.9e-4 /h, refusal 3.9e-3) and an out-of-turbine part (1.46e-4 /h, refusal 5.45e-4), repairs of 2 to 144 h by mode; one runs below 60 % power, both above; losing one above 60 % forces the power back to 60 %, which fails with probability 1e-3; a TPA lost with the other unavailable, or refusing to restart at 60 %, trips the plant; 5 % of the running failures are common-cause |
| ARE valves | small-flow valve PD in service from 2 to 15 % power, large-flow GD above; failure 3.53e-5 /h, refusal to open 0.72 x 5.85e-3 when taking over (the ramp holds during the 12 h repair) |

## Bibliographic note

- **The report.** Aubry et al. (2012) define the case (chapter 4) and
  quantify it with three methods. Stochastic hybrid automata (chapter 5):
  closed-form single-path examples, then 111 and 200 histories on a
  one-month cycle, the second with all rates multiplied by 18.
  Piecewise-deterministic Markov processes in Simulink and Stateflow
  (chapter 6): **4000 histories of the full 18-month cycle, 2190 trips**,
  of which 792 by the VVP, 1301 by the ARE, 50 by the CEX and 47 by the
  TPA (Table 6.1). Stochastic Petri nets (chapter 7): no quantitative
  result. The report counts as a trip, in the Markov-process run, every
  ARE failure that forces a drop to 2 % power (p. 9).
- **Later papers.** Babykina et al. (2013) restate the automata model;
  Babykina et al. (2016) treat it at length, but the article is closed
  access and was not read for this page.
- **The level model.** Kothare et al. (2000), the source of the linear
  steam-generator model, is cited by the report and not used here.

The reference is the 4000-history Markov-process run: the full cycle, the
largest sample. The report itself flags that its methods disagree on the
causes, and gives the closed-form automata examples as worst-case paths,
not as system probabilities.

## Data choices

The report contradicts itself in places. Each choice below is the one the
model makes; every other reading is one keyword away in
[`examples/approdyn/model.py`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/approdyn).

| Point | Report | Choice here |
|---|---|---|
| Repair times | the three implementations used twice the printed values, or swapped them | printed tables; the first trip ends a history, so most repairs never complete anyway |
| Turbine refusal on demand | 3.9e-3 (data table), 3.9e-2 and 3.9e-5 elsewhere | 3.9e-3 |
| Failure of the forcing back to 60 % | 1e-3 per demand (p. 50), 0.1 in a worked example | 1e-3 |
| Demand failures | component and instrumentation both carry one | one per start |
| CEX standby failures | a rate appears in one figure only, with no derivation | not modelled |
| Common-cause failures | 5 % of the running failures of the CEX and TPA | modelled; a variant without them |
| ARE failures that trip | "detection rates" per mode with no time basis (Tables 4.11, 4.12); the Markov-process run counts every failure forcing 2 % power as a trip | two variants: every failure in service trips (the published counting), or a trip with the probability of the mode's detection rate (8 % of the failures) |
| Ramp down, sensors, external perturbations, repair failures | demands on the way down, sensor drift, turbine and condenser trips, failed repairs | not modelled; absent from the published result table |

## The model

The model is plain data written by `model.py`, one document per variant.
Every part of the case maps onto a native construct:

| Case | RAICHU |
|---|---|
| power cycle with holds | an automaton whose ramps are fixed **delays** and whose holds are **instantaneous** transitions waiting for a condition (a TPA running, a valve open) |
| end of the plateau, by the calendar | a separate automaton with a delay of 12 984 h |
| a component's failures, repairs | exponential transitions, declared `failure` or `repair` |
| a start on demand that may fail | one instantaneous transition with several destinations and their probabilities (running, or under repair for 2, 24 or 28 h) |
| standby, start order | guards on how many units run and how many are required in the current power phase |
| forcing back to 60 %, which may fail | an instantaneous draw between `forced60` and `forcing_failed` |
| a common-cause failure, a refusal to restart at 60 % | an **edge effect** of the transition latching a flag, read by the trip logic |
| the first trip and its cause | a trip automaton with one state per cause, each a **target**: a history stops at the first |

## Results

Each variant is run over 200 000 histories (about one second on 24
threads; standard error about 0.1 point). `compare.py` sets them beside
the published run:

| | Trip in 18 months | VVP | ARE | CEX | TPA |
|---|---|---|---|---|---|
| published Monte-Carlo, 4000 histories (95 %) | 54.8 % ± 1.5 | 19.8 % ± 1.2 | 32.5 % ± 1.5 | 1.2 % ± 0.3 | 1.2 % ± 0.3 |
| ARE: every failure trips; common causes | 87.5 % | 11.8 % | 19.2 % | 2.8 % | 53.7 % |
| ARE: every failure trips; no common cause | 67.2 % | 17.0 % | 27.7 % | 0.6 % | 22.0 % |
| ARE: report tables; common causes | 81.0 % | 13.7 % | 1.8 % | 3.2 % | 62.3 % |
| diagnostic: perfect TPA | 55.5 % | 19.3 % | 31.7 % | 4.6 % | 0.0 % |

![Probability of a trip within the cycle, by cause, for the published Monte-Carlo and each RAICHU variant](../assets/figures/example-approdyn-trips-light.svg#only-light){ .figure }
![Probability of a trip within the cycle, by cause, for the published Monte-Carlo and each RAICHU variant](../assets/figures/example-approdyn-trips-dark.svg#only-dark){ .figure }

**The turbo-pumps make the difference.** With the reconfiguration logic
as the report prints it, the TPA trip the plant in 22 % of the cycles
even without common causes, and in 54 % with them. A hand count gives the
same order: about 19 TPA failures per cycle with both running, each
followed by a repair of about 20 h during which the other TPA can fail
(about 1.4 %), plus the forcing failure and the restart refusal, about
2 % per failure. The report's own closed-form reasoning on the TPA
trajectories reaches 17 % (chapter 5). The published Markov-process run
attributes 1.2 % to them.

**Without TPA trips, the published run is reproduced.** The diagnostic
variant keeps everything else and makes the TPA perfect: it gives 55.5 %,
of which 19.3 % by the VVP and 31.7 % by the ARE, against 54.8 %, 19.8 %
and 32.5 % published, each within the published sample's uncertainty. The
published causes are those of a model whose turbo-pumps hardly ever trip
the plant. Whether the published run left out the TPA common causes and
most of the forcing logic, or modelled it otherwise, cannot be told from
the report. The published CEX share, 1.2 %, lies between the values
found here without common causes (0.6 %) and with them (4.6 %).

**The ARE counting rule matters as much.** Counting only the ARE failures
that the report's detection rates would reveal (8 % of them) turns the ARE
from the first cause into a minor one (1.8 %). The published run counts
them all.

## Reading the results

- **What RAICHU shows here.** A plant-level logic with holds, standby
  starts, demand failures with several outcomes, latched conditions and
  competing causes is plain data, and an 18-month campaign of 200 000
  histories runs in a second.
- **What this example validates.** The model reproduces the published
  total and the VVP and ARE shares once the TPA are neutralised; it does
  not reproduce the published TPA share, and shows that the report's
  printed logic cannot produce it.
- **What it does not.** The steam-generator level and its control are not
  modelled; the data choices above are ours; the reference itself rests on
  undocumented modelling decisions. Treat the case as a test of modelling
  choices, not as a reference value.

## Reproducing

```bash
cd examples/approdyn
python model.py                                       # the variants
python run_campaign.py approdyn_pdmp                  # about 1 s each
python run_campaign.py approdyn_pdmp_noccf
python run_campaign.py approdyn_tables
python run_campaign.py approdyn_pdmp_perfect_tpa
python compare.py                                     # the table above
```

Every RAICHU number on this page is read from the files under
[`examples/approdyn/results`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/approdyn/results);
the seed is fixed (2012). *Chart produced by
`docs/figures/example_approdyn.py`.*

## References

- Aubry, J.-F., Babykina, G., Barros, A., Brinzei, N., Deleuze, G., de
  Saporta, B., Dufour, F., Langeron, Y., Zhang, H. (2012). *Rapport final
  du projet APPRODYN : APPROches de la fiabilité DYNamique pour modéliser
  des systèmes critiques*. [hal-00740181](https://hal.science/hal-00740181)
- Babykina, G., Brinzei, N., Aubry, J.-F., Deleuze, G. (2013). Modelling a
  feed-water control system of a steam generator in the framework of the
  dynamic reliability. *ESREL 2013*, Amsterdam.
  [hal-00872422](https://hal.science/hal-00872422)
- Babykina, G., Brînzei, N., Aubry, J.-F., Deleuze, G. (2016). Modeling and
  simulation of a controlled steam generator in the context of dynamic
  reliability using a Stochastic Hybrid Automaton. *Reliability
  Engineering & System Safety*, 152, 115-136.
  [doi:10.1016/j.ress.2016.03.009](https://doi.org/10.1016/j.ress.2016.03.009)
- Kothare, M. V., Mettler, B., Morari, M., Bendotti, P., Falinower, C.-M.
  (2000). Level control in the steam generator of a nuclear power plant.
  *IEEE Transactions on Control Systems Technology*, 8, 55-69.
  [doi:10.1109/87.817692](https://doi.org/10.1109/87.817692)
