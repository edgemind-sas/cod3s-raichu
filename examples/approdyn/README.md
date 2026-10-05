# APPRODYN feedwater benchmark (discrete part)

The files behind the documentation example
[Feedwater supply of a steam generator (APPRODYN)](../../docs/examples/approdyn-feedwater.md):
the test case of the APPRODYN project (Aubry et al. 2012, hal-00740181),
its discrete part modelled and run with RAICHU, and compared with the
published Monte-Carlo.

| File | Role |
|---|---|
| `model.py` | builds the RAICHU model of each variant (ARE trip rule, common-cause failures, the perfect-TPA diagnostic) from the report's data |
| `model/<variant>.json` | the model documents `model.py` writes |
| `run_campaign.py` | one Monte-Carlo campaign: probability of a trip within the 18-month cycle and its cause, with the cumulative curve |
| `compare.py` | the published result, and the table comparing it with the campaigns |
| `results/` | the campaigns the documentation page quotes |

The model is ours, written from the report's description. The report's
data are "representative but not real" and restricted by its authors to
use as a test case; they are restated here for that use only, with
citation. The steam-generator level and its control are not modelled: the
report does not specify them completely (see the documentation page).

## Reproducing

```bash
cd examples/approdyn
python model.py
for v in approdyn_pdmp approdyn_pdmp_noccf approdyn_tables approdyn_pdmp_perfect_tpa; do
    python run_campaign.py $v                           # about 1 s each on 24 threads
done
python compare.py
```
