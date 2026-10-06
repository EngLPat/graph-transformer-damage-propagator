# Analysis scripts

Scripts behind specific numbers and figures of the paper.

| Script | Produces |
|---|---|
| `regression_from_npz.py` | raster of the laminate regression figure (`fig_regression_tikz.tex`) |
| `neff_perplexity_testset.py` | perplexities of the influence-concentration analysis |
| `dice_pooled.py`, `dice_ablation.sh` | pooled Dice and damaged-area fractions of the attention-ablation arms |
| `dt001.sh`, `timing_dt001.py`, `figs_dt001.py`, `sweep_many.py` | data and panels of the material-variability figure (`fig_material_variability.tex`) |

The scripts read `data/` and `results/` under `GTP_ROOT` (default: this repository).
`dt001.sh` and `dice_ablation.sh` write into `OUT_DIR` (default: `results/paper_analysis`),
the directory that `timing_dt001.py`, `figs_dt001.py` and `sweep_many.py` take as argument.
