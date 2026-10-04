# Emergency power supply benchmark

The files behind the documentation example
[Emergency power supply](../../docs/examples/emergency-power-supply.md):
EDF's benchmark of the emergency power supply of a nuclear power plant
(Bouissou 2017, [doi:10.4204/EPTCS.244.8](https://doi.org/10.4204/EPTCS.244.8)),
quantified with RAICHU and compared with the values published for it.

| File | Role |
|---|---|
| `translate.py` | fetches the benchmark's Figaro 0 file from arXiv, checks its SHA-256 and translates it into RAICHU models, one per variant, under `model/` |
| `common.py` | the feared event, the mission time, model loading and the timing written beside each result |
| `run_exploration.py` | exact sequence-tree exploration: lower and upper bounds, most probable sequences |
| `run_monte_carlo.py` | plain Monte-Carlo estimate with its Wilson interval |
| `run_rare_event.py` | cross-entropy and adaptive multilevel splitting |
| `run_static_tree.py` | the static fault tree of the BDMP, quantified by binary decision diagrams |
| `results/` | every result quoted on the documentation page, as written by the scripts |

## Licence of the model

The Figaro 0 file is the work of its author, distributed by arXiv under
its non-exclusive distribution licence (and by the MARS repository under
CC BY-NC-SA 4.0). The translated models are adaptations of it, so they are
**not** part of this repository: `model/` is ignored by git and every
reader regenerates it. The scripts and the results are ours (MIT).

## Reproducing

From the repository root, with `pyraichu` installed:

```bash
cd examples/emergency_power_supply
python translate.py                      # the four variants, about 10 s
python run_exploration.py eps_benchmark_nonrepairable 100 1e-12
python run_exploration.py eps_benchmark_nonrepairable 1000 1e-9
python run_exploration.py eps_benchmark_nonrepairable 10000 1e-7
python run_exploration.py eps_benchmark 10000 1e-8
python run_static_tree.py
python run_monte_carlo.py eps_benchmark 20000000
python run_monte_carlo.py eps_benchmark_battery_1h 20000000
python run_monte_carlo.py eps_benchmark_fast_line_repair 20000000
python run_rare_event.py eps_benchmark cross_entropy
python run_rare_event.py eps_benchmark splitting
python run_rare_event.py eps_benchmark_fast_line_repair cross_entropy
python run_rare_event.py eps_benchmark_fast_line_repair splitting
```

Each Monte-Carlo campaign draws 2 x 10^7 histories, the budget of the YAMS
run it is compared with, and holds about 170 bytes per history in memory
until the end (the replicas are reduced in replica order, so that the
result does not depend on the thread count): allow 4 GB. The seeds are
fixed (2017 by default), so a rerun reproduces the results, except for the
wall-clock times.
