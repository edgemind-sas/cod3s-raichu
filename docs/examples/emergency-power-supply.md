# Emergency power supply of a nuclear power plant

The electrical system that keeps the safety functions of a nuclear power
plant supplied when everything else fails: two bus bars, LHA and LHB, fed
by the grid, by the plant itself, by two transformers, two diesel
generators and a last-resort gas turbine, with circuit breakers that
reconfigure the supply after each loss. The feared event is the loss of
both bus bars.

This is EDF's benchmark of **complex discrete systems** (Bouissou, 2017).
It gathers most of what makes a real dependability study hard: cascades of
instantaneous reconfigurations, failures on demand, cold redundancies,
common-cause failures, repairs and a shared repairman, rates spread over
six orders of magnitude, and a deterministic delay (battery depletion).
Its author published the model as the input file of EDF's own solvers and
did not publish the full answer, so that other tools could try. A later
paper (Bouissou, Khan, Katoen and Krcal, 2020) then quantified it with four
methods: sequence exploration (FIGSEQ), Monte-Carlo simulation (YAMS),
probabilistic model checking (STORM) and a cut-set method for large
repairable systems (I&AB, in RiskSpectrum PSA).

That makes it a validation case with **several independent references**,
and a demonstration of the capabilities RAICHU puts behind one model: exact
sequence-tree exploration, Monte-Carlo simulation, two rare-event methods
and a static fault tree quantified by binary decision diagrams.

## The benchmark

The system description and its data are in Bouissou (2017), sections 2.1
to 2.4, with the layout of the high- and low-voltage parts (figures 2 and
3). In short:

| Aspect | Hypothesis (Bouissou 2017) |
|---|---|
| Function | supply the bus bars LHA and LHB; losing one of them is tolerated |
| Feared event | loss of power on both LHA and LHB |
| Mission time | 10 000 h, about one year |
| Normal supply | transformer TS, fed by the grid or by the unit; when the grid is lost, an attempt at **house load** operation (succeeds with probability 0.8, then fails at 0.1 per hour) |
| Backups, in order | transformer TA from the grid; diesel generators DGA (for LHA) and DGB (for LHB); the gas turbine TAC (for LHA only) |
| Failures | in function (constant rate) for every component; on demand for the diesels (2e-3), the TAC (2e-3) and the circuit breakers asked to open or close (2e-4); short circuits on bus bars, transformers and breakers |
| Common causes | both lines lost together (1e-6 per hour, 200 h repair); both diesels (on demand 2e-4, in function 5e-5 per hour, 400 h repair) |
| Repairs | constant rates, from 3 h (converters) to 400 h (common-cause loss of the diesels); one repairman for the grid and the house load, so the house load cannot be restored before the grid |
| Control | breakers need the low-voltage supply (LBA, LBB) to move; batteries hold it for one hour |

The data are deliberately fictitious but of realistic orders of magnitude
(2017, section 1).

## Bibliographic note

- **The benchmark.** Bouissou (2017) describes the system, gives the data,
  and models it as a **BDMP** (Boolean logic Driven Markov Process,
  Bouissou and Bon, 2003): a fault tree whose leaves carry small Markov
  processes and whose *triggers* switch leaves from standby to required.
  The BDMP has 81 leaves (2020, section 5.2). The model is published as the Figaro 0 text that
  was the input of FIGSEQ and YAMS, with the one-hour battery depletion
  already replaced by an Erlang(2, 2/h) law to keep the model Markovian
  (2017, section 3.3). The paper gives two sequences, not the global
  values.
- **The global values.** Bouissou, Khan, Katoen and Krcal (2020) quantify
  the same model. Table 1: the repairable model at 10 000 h with FIGSEQ,
  YAMS and I&AB. Table 2: a sensitivity case, the repair of the common-cause
  loss of the lines ten times faster. Table 3: a non-repairable variant at
  100, 1000 and 10 000 h with FIGSEQ, YAMS, STORM and a static fault tree.
  The STORM figures come from a translation of the BDMP into a dynamic
  fault tree (Khan, Katoen, Volk and Bouissou, 2019).
- **I&AB.** Bouissou, Bäckström, Gamble, Krcal and Wang (2018) present the
  I&AB method in practice, on two industrial use cases.

The methods differ, and so do the models they solve: FIGSEQ and YAMS read
the Figaro 0 file itself, STORM a dynamic fault tree translated from it,
RiskSpectrum a static tree or a cut-set reading of it. The comparisons
below say which reference solves the same model as RAICHU and which does
not.

## The model

RAICHU's model is **translated from the Figaro 0 file**, not redrawn from
the paper's figures, by
[`examples/emergency_power_supply/translate.py`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/emergency_power_supply).
The script fetches the file from arXiv, checks its SHA-256, and writes one
RAICHU model per variant. The translated models are adaptations of a file
that is not ours to redistribute: they are not in the repository, and
every reader regenerates them in about ten seconds.

Figaro 0 runs an *interaction phase* after every transition: some
attributes are reset, then the interaction rules run step by step, each
step until nothing changes, and the occurrence rules read the result. The
translation carries each part onto a RAICHU construct:

| Figaro 0 | RAICHU |
|---|---|
| attribute reset at each phase (the gates' `S`, `required`, `relevant_evt`) | attribute written by a sensitive function, its expression derived step by step |
| attribute kept between phases (priority-AND memory, demand latches, repair queue) | attribute written by the edge effects of every transition, with the value the previous phase ended with |
| failure or repair at a constant rate (`DIST EXP`) | exponential transition, guarded by the occurrence condition, declared `failure` or `repair` |
| instantaneous draw (`DIST INS`, failure on demand) | instantaneous transition into a pending state; a phase controller commits the pending draws together, so the draws valid in one state are made simultaneously, as in FIGSEQ |
| the feared event | a watcher that enters `reached` when both bus bars are lost between two phases |

The translator checks its own work: on 400 random states it evaluates
every derived attribute both from the translated expressions and from a
direct interpreter of the Figaro rules, and refuses to write a model on
which they disagree. *Relevant event filtering*, the BDMP's way of
forbidding failures that cannot bring the system closer to the feared
event, is part of the file's guards, so RAICHU's model has it too.

The repairable model has 109 exponential transitions (48 failures, 61
repairs), 13 instantaneous draws and 498 derived attributes; its product
state space is never built. The variants:

| Variant | What changes | Compared with |
|---|---|---|
| `eps_benchmark` | the published model | 2020, Table 1 |
| `eps_benchmark_nonrepairable` | the file's own switch `repairable_system = FALSE`: every repair removed | 2020, Table 3 |
| `eps_benchmark_fast_line_repair` | repair of the common-cause loss of the lines in 20 h instead of 200 h | 2020, Table 2 |
| `eps_benchmark_battery_1h` | the battery lasts exactly one hour: each Erlang stage becomes a fixed half-hour delay | the YAMS row of 2020, Table 1 |

<!-- skip -->
```python
# examples/emergency_power_supply: regenerate the models, then load one
import json
import pyraichu
import translate

translate.write("eps_benchmark")
model = pyraichu.load_model(json.loads((translate.MODEL_DIR / "eps_benchmark.json").read_text()))
study = pyraichu.Study(target="LHA_and_LHB_lost", horizon=10_000.0, seed=2017)
```

## Non-repairable variant: exact exploration

Without repairs, the exploration of the sequence tree is the natural
method: it enumerates the sequences of events from the perfect state to
the feared event, each with its exact probability by the mission time, and
cuts the branches less probable than a threshold. The retained sequences
sum to a **lower bound**; adding the probability mass cut off gives an
**upper bound**. It is the algorithm FIGSEQ calls "NS" (2020, section 2.2),
run here with FIGSEQ's own thresholds.

<!-- skip -->
```python
result = pyraichu.quantify(model, pyraichu.Study("LHA_and_LHB_lost", 100.0),
                           method="exact", min_probability=1e-12)
```

| Mission time, cut-off | RAICHU bounds | Sequences retained | FIGSEQ estimate (upper bound) | FIGSEQ sequences | STORM |
|---|---|---|---|---|---|
| 100 h, 1e-12 | [3.4479e-6, 3.4570e-6] | 3674 | 3.448e-6 (3.453e-6) | 3674 | 3.495e-6 |
| 1000 h, 1e-9 | [7.9853e-3, 7.9959e-3] | 6567 | 7.986e-3 (7.993e-3) | 8567 | 7.925e-3 |
| 10 000 h, 1e-7 | [3.5913e-1, 3.6093e-1] | 12 903 | 3.59e-1 (3.61e-1) | 12 921 | 3.604e-1 |

RAICHU's lower bound matches FIGSEQ's estimate to every digit FIGSEQ
prints, at the three mission times, and its upper bounds are within
0.12 % of FIGSEQ's. At 100 h both tools retain the same 3674 sequences. At
10 000 h RAICHU retains 18 fewer (0.14 %). At 1000 h it retains 6567
against a printed 8567: the difference is exactly 2000 while the bounds
agree to 1e-4, which reads like a misprint in the table but cannot be
confirmed from the source.

STORM solves another model, a dynamic fault tree translated from the
BDMP. Its values sit outside both brackets at 100 h (1.4 % above
RAICHU's lower bound) and at 1000 h (0.8 % below), and inside RAICHU's
bracket at 10 000 h. Each exploration takes
under a second on 24 threads (FIGSEQ: 23 to 134 s on a 2020 laptop).

## Static fault tree

A BDMP without its triggers is a fault tree. Dropping the triggers and the
order constraints gives the static tree a classical tool quantifies: each
gate keeps its inputs, a priority-AND gate becomes an AND gate, a leaf
failing in function gets the law $1 - e^{-\lambda t}$ and a leaf failing
on demand the probability $\gamma$. The basic events are then independent
and never repaired. `run_static_tree.py` writes this tree as an OpenPSA
document and RAICHU quantifies it exactly, by binary decision diagrams.

| Mission time | RAICHU, static tree (BDD, exact) | RiskSpectrum, static tree (2020) | Dynamic model (RAICHU exploration, lower bound) |
|---|---|---|---|
| 100 h | 2.8121e-5 | 2.812e-5 | 3.4479e-6 |
| 1000 h | 2.7360e-2 | 2.740e-2 | 7.9853e-3 |
| 10 000 h | 4.0711e-1 | 4.071e-1 | 3.5913e-1 |

The tree has 49 gates and 61 basic events, and **53 137 minimal cut
sets, exactly RiskSpectrum's count**. The probabilities agree with
RiskSpectrum's to every printed digit at 100 h and 10 000 h; at 1000 h
they differ by 0.15 %, which the source does not explain. One choice
matters for the count and not for the probability: an *approximation OR
gate* of the BDMP (one per low-voltage line, aggregating five to seven
components) stays one basic event failing at the sum of its components'
rates, as in the dynamic model. Expanding it into the OR of its
components gives the same probability (none of them feeds another gate,
and the laws are exponential) but 262 344 cut sets.

![Unreliability of the non-repairable variant against the mission time: the static fault tree and the dynamic model, RAICHU and the published references](../assets/figures/example-emergency-power-supply-nonrepairable-light.svg#only-light){ .figure }
![Unreliability of the non-repairable variant against the mission time: the static fault tree and the dynamic model, RAICHU and the published references](../assets/figures/example-emergency-power-supply-nonrepairable-dark.svg#only-dark){ .figure }

The static tree overestimates the unreliability eight times at 100 h and
3.4 times at 1000 h. Its cut sets let the failures happen in any order,
whereas the dominant scenarios of this system are ordered: the line loss
must come first, the house load must be tried and lost, then the diesels
and the turbine are asked to start. The 2020 paper makes the same
observation (section 5.3.2). At 10 000 h almost every component has
failed with a high probability and the order no longer matters much.

## Repairable model

The published question: the probability of losing both bus bars at least
once in 10 000 h, every component being repaired. Three RAICHU methods
answer it, with the settings given in the scripts; the wall-clock times
are measured on a 24-thread x86_64 workstation.

| Method | RAICHU | Interval | Wall clock |
|---|---|---|---|
| Monte-Carlo, 2e7 histories | 3.725e-5 (745 histories) | 95 % Wilson [3.467e-5, 4.002e-5] | 5 min 46 s |
| Monte-Carlo, battery exactly 1 h | 3.725e-5 (the same 745) | identical | 6 min 21 s |
| Cross-entropy, 1e6 histories | 3.705e-5 | 95 % [3.538e-5, 3.873e-5] | 2 min 42 s |
| Splitting, 1000 particles x 100 batches | 3.597e-5 | 95 % Student [3.142e-5, 4.053e-5] | 7 s |
| Exact exploration, cut-off 1e-8 | | bounds [1.84e-5, 1.19e-2], inconclusive | 28 s |

| Reference (2020, Table 1) | Value |
|---|---|
| YAMS, Monte-Carlo, 2e7 histories, battery exactly 1 h | 3.80e-5, 90 % half-width 3e-6 (82 min) |
| FIGSEQ ("NRI"), cut-off 1e-8 / 1e-10 | 3.48e-5 / 3.84e-5 |
| RiskSpectrum I&AB | 1.35e-4 |

The Monte-Carlo campaign uses YAMS's budget, 2e7 histories, and lands
within YAMS's interval; YAMS and FIGSEQ at 1e-10 both fall inside
RAICHU's. Cross-entropy and splitting agree with them.

**The battery.** The variant with a battery lasting exactly one hour,
YAMS's setting, reaches the feared event in the same 745 of the 2e7
histories as the Erlang approximation of the file, drawn from the same
seed. The exploration explains why: none of the 323 sequences it
retains at 1e-8 involves a battery stage. The 2020 paper reports the same
insensitivity (section 5.2).

**The most probable sequence** is the one printed in 2017 (section 4),
event for event, with probability 2.02e-6 at 10 000 h: common-cause loss
of both lines, successful switch to house load, house-load failure,
successful diesel starts and breaker reconfigurations, common-cause loss
of both diesels, successful switch to the TAC, failure of the TAC.

**Why the exploration is inconclusive here.** It explores every sequence
from the perfect state, repair loops included, and adds the probability
of each one by the mission time. With frequent failures that are quickly
repaired (the unit fails at 1e-4 per hour and is repaired in 10 h), the
loops multiply the sequences, and the mass cut off at 1e-8 (1.2e-2)
dominates the upper bound. RAICHU reports the bracket and flags it
inconclusive rather than giving a number. FIGSEQ's repairable figures come
from its other algorithm, "NRI", which folds the loops through the perfect
state and neglects the time spent in degraded states (2020, section 2.2):
an approximation that suits quickly repaired systems, and a different
quantity from RAICHU's bounds.

**I&AB** is 3.6 times above the dynamic methods: it lets the barriers fail
in any order after the initiator, whereas here the order is imposed (2020,
section 5.3.1).

## Rare events: cross-entropy and splitting

The unreliability of the published model, 3.7e-5, is within reach of a
plain campaign on a workstation. The sensitivity case is ten times rarer:
with the common-cause loss of the lines repaired in 20 h instead of 200 h,
its probability drops to about 4e-6, and the 2020 paper reports that YAMS
would have needed "at least 10 h" (Table 2). Two methods make rare events
cheaper.

- **Cross-entropy** (de Boer et al., 2005) draws the constant-rate
  exponential transitions at multiplied rates, fitted over pilot
  campaigns, and weights each history by its likelihood ratio, so that the
  weighted proportion stays an unbiased estimate. The method groups the
  transitions into families of equal rate and fits one factor per family.
- **Adaptive multilevel splitting** (Cérou and Guyader, 2007) runs a
  population of histories, discards those that progress least towards the
  feared event and clones the others from the point where they progressed.
  Progress is a score: `run_rare_event.py` adds an observer that counts
  the supply paths currently lost (grid connection, unit, the two
  transformers, the diesel and turbine supplies, each bus bar). The
  automatic score built from minimal cut sets is not available here,
  because the fault-tree generator refuses the translated model (its phase
  controller reads states under a negation).

| Line repair ten times faster | Estimate | Interval | Wall clock |
|---|---|---|---|
| RAICHU Monte-Carlo, 2e7 histories | 4.10e-6 (82 histories) | 95 % [3.30e-6, 5.09e-6] | 6 min 17 s |
| RAICHU Monte-Carlo, 5e8 histories | 3.84e-6 (1918 histories) | 95 % [3.67e-6, 4.01e-6] | 1 h 39 min |
| RAICHU cross-entropy, 1e6 histories | 4.03e-6, **inconclusive** | [2.66e-6, 5.41e-6] | 3 min 6 s |
| RAICHU splitting, seed 2017 | 3.25e-6 | 95 % [2.62e-6, 3.88e-6] | 6 s |
| RAICHU splitting, mean of 100 seeds | 3.76e-6 | standard error 0.17e-6 | 9 min 24 s |
| FIGSEQ ("NRI"), cut-off 1e-11 (2020) | 3.85e-6 | | 8 min |
| I&AB (2020) | 1.46e-5 | | 7 s |

![Unreliability at 10 000 h by method, for the published model and for the line repair ten times faster: RAICHU's Monte-Carlo, cross-entropy and splitting against FIGSEQ, YAMS and I&AB](../assets/figures/example-emergency-power-supply-repairable-light.svg#only-light){ .figure }
![Unreliability at 10 000 h by method, for the published model and for the line repair ten times faster: RAICHU's Monte-Carlo, cross-entropy and splitting against FIGSEQ, YAMS and I&AB](../assets/figures/example-emergency-power-supply-repairable-dark.svg#only-dark){ .figure }

*An empty marker is an inconclusive estimate.* On the published model,
cross-entropy is conclusive with an effective sample size of 1881 and a
relative error of 2.3 %; its fitted factors read like the dominant
scenario: the common-cause loss of the lines is drawn 47 times more
often, that of the diesels 27 times, the failure of the TAC 10 times, and
the long repairs of the common-cause failures are slowed down. On the
sensitivity case the same settings leave an effective sample size of 33,
under the threshold of 50: the estimate falls near the others, but RAICHU
declares it not established. This is the regime the documentation warns
about: quick repairs over a long horizon defeat one fixed factor per
family.

Splitting is the fastest method by far, a few seconds per estimate, and
it needs a careful reading. Its interval is a Student interval over
independent batches, and on this model it can be too narrow. A first run
with 20 batches returned, on the published model, an interval that
excluded every other value on this page while being flagged conclusive;
the scripts therefore use 100 batches. Repeating the run over 100 seeds
(`run_splitting_seeds.py`) then measures the estimator itself:

| Splitting over 100 seeds | Mean | Standard error | Single intervals containing the Monte-Carlo estimate |
|---|---|---|---|
| published model | 3.73e-5 | 0.03e-5 | 93 of 100 (2e7 histories) |
| line repair ten times faster | 3.76e-6 | 0.17e-6 | 66 of 100 (5e8 histories) |

**The estimator is right.** On both variants the mean over 100 seeds
agrees with the Monte-Carlo reference: 0.1 and 0.4 standard errors away.
A first sweep over ten seeds had put the sensitivity case 1.9 standard
errors low; the 5e8-history campaign and the hundred seeds show that this
was a fluctuation, not a bias. **A single interval is not.** On the
sensitivity case, an interval announced at 95 % contains the reference two
times out of three: the estimates are heavy-tailed (their spread between
seeds, 1.7e-6, is 2.2 times the half-width a typical run announces, and
one seed in a hundred returned 1.5e-5, four times the reference), and a
Student interval over 100 batches does not see the tail. On this kind of model, read a splitting estimate through several
seeds, as above, rather than through its own interval.

## Reading the results

- **The translation is right.** On the non-repairable variant, RAICHU's
  exact exploration reproduces FIGSEQ (same algorithm, same model) to
  every printed digit, and STORM (another model, a dynamic fault tree)
  within 1.4 %. On the repairable model, the most probable sequence is the
  published one, event for event, and three independent RAICHU methods
  agree with YAMS and FIGSEQ.
- **The static tree is right, and wrong for this system.** RAICHU's
  static tree has RiskSpectrum's 53 137 minimal cut sets and its
  probabilities, and overestimates the dynamic answer up to eight times,
  because this system's dominant scenarios are ordered.
- **Each method has a regime.** Exact exploration is the method of
  choice without repairs: exact bounds and the sequences themselves in
  under a second. With quick repairs over a long mission its bracket
  widens until it says nothing, and RAICHU says so. Plain Monte-Carlo is
  unbiased and simple, and at 3.7e-5 it costs six minutes on 24 threads.
  Cross-entropy cuts the cost when one family factor captures the
  scenario, and reports when it does not. Splitting is the cheapest and
  its mean is right, but its single interval overstates its precision on
  the rarer case: average it over several seeds.
- **What this example does not show.** The data are fictitious. The
  comparison is with values printed by others, from their own runs and
  machines, so the wall-clock times compare orders of magnitude only. The
  unavailability, which the benchmark also asks for, is not computed here.
- **A limitation found on the way, and lifted.** Up to RAICHU 0.78, a
  Monte-Carlo campaign held about 170 bytes per history until the end:
  2e7 histories took 3.4 GB and 1e8 did not fit in 10 GB. Since 0.79 the
  histories are folded chunk by chunk, with the same bytes as before
  whatever the thread count (see [Parallelism](../guides/parallelism.md)):
  the 5e8-history campaign above peaked at 69 MB. It is the campaign that
  settled the splitting question.

## Reproducing

Every RAICHU estimate, bound and count on this page is read from the
files under
[`examples/emergency_power_supply/results`](https://github.com/edgemind-sas/cod3s-raichu/tree/main/examples/emergency_power_supply/results),
written by the scripts beside them; the folder's README lists the
commands. Two side measurements are not: the 262 344 cut sets of the
expanded static tree, and the memory footprints of the Monte-Carlo
campaigns. The seeds are fixed. Each result has a `.meta.json` with its
wall-clock time and machine.

## References

- Bouissou, M. (2017). A Benchmark on Reliability of Complex Discrete
  Systems: Emergency Power Supply of a Nuclear Power Plant. *Models for
  Formal Analysis of Real Systems (MARS 2017)*, EPTCS 244, 200-216. Open
  access. [doi:10.4204/EPTCS.244.8](https://doi.org/10.4204/EPTCS.244.8),
  [arXiv:1703.06575](https://arxiv.org/abs/1703.06575)
- Bouissou, M., Khan, S., Katoen, J.-P., Krcal, P. (2020). Various Ways to
  Quantify BDMPs. *Models for Formal Analysis of Real Systems (MARS 2020)*,
  EPTCS 316, 1-14. Open access.
  [doi:10.4204/EPTCS.316.1](https://doi.org/10.4204/EPTCS.316.1)
- Bouissou, M., Bäckström, O., Gamble, R., Krcal, P., Wang, W. (2018). The
  I&AB quantification method for large dynamic systems in practice: two use
  cases. *Congrès Lambda Mu 21*, Reims.
  [hal-02075088](https://hal.science/hal-02075088)
- Bouissou, M., Bon, J.-L. (2003). A new formalism that combines advantages
  of fault-trees and Markov models: Boolean logic driven Markov processes.
  *Reliability Engineering & System Safety*, 82(2), 149-163.
  [doi:10.1016/S0951-8320(03)00143-1](https://doi.org/10.1016/S0951-8320(03)00143-1)
- Khan, S., Katoen, J.-P., Volk, M., Bouissou, M. (2019). Synergizing
  Reliability Modeling Languages: BDMPs without Repairs and DFTs. *IEEE
  24th Pacific Rim International Symposium on Dependable Computing (PRDC)*,
  266-275. [doi:10.1109/PRDC47002.2019.00057](https://doi.org/10.1109/PRDC47002.2019.00057)
- de Boer, P.-T., Kroese, D. P., Mannor, S., Rubinstein, R. Y. (2005). A
  Tutorial on the Cross-Entropy Method. *Annals of Operations Research*,
  134(1), 19-67. [doi:10.1007/s10479-005-5724-z](https://doi.org/10.1007/s10479-005-5724-z)
- Cérou, F., Guyader, A. (2007). Adaptive Multilevel Splitting for Rare
  Event Analysis. *Stochastic Analysis and Applications*, 25(2), 417-443.
  [doi:10.1080/07362990601139628](https://doi.org/10.1080/07362990601139628)
