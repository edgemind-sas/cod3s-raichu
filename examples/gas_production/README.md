# Gas production benchmark

The files behind the documentation example
[Gas production with a buffer reservoir](../../docs/examples/gas-production.md):
the production-availability benchmark of Labeau and Dutuit (2004,
hal-01570847), quantified by Eymard and Mercier (2008,
doi:10.1016/j.ress.2006.12.001, hal-00693079), modelled and run with RAICHU.

| File | Role |
|---|---|
| `model.py` | builds the RAICHU model of each of the four cases from the published data |
| `model/gas_production_case<N>.json` | the four model documents `model.py` writes |
| `run_campaign.py` | one Monte-Carlo campaign: availability, production availability and the two loss frequencies, with their standard errors |
| `compare.py` | the published reference values, and the table comparing them with the campaigns |
| `results/` | the campaigns the documentation page quotes |

The model is ours, written from the published description; the data come
from the two papers, both deposited on HAL under CC BY 4.0. Cases 1 to 3
need RAICHU 0.80 or later (the instant refill is a reset map on an ODE
target).

## Reproducing

```bash
cd examples/gas_production
python model.py
for c in 1 2 3 4; do python run_campaign.py $c; done           # 1 to 4 min each on 24 threads
python run_campaign.py 2 --runs 400000 --horizon 1e5 --tag T1e5
python run_campaign.py 2 --runs 4000 --horizon 1e7 --tag T1e7
python compare.py
```
