# Analysing Microbiome Metabarcoding Data with Tabular Foundation Models

Code and complete numerical results for the MSc thesis *Analyzing Microbiome
Metabarcoding Data with Tabular Foundation Models (TabPFN)*, University of
Rostock, 2026.

The thesis benchmarks TabPFN v3 against Lasso, Random Forest and XGBoost for
predicting environmental variables and anthropogenic trace substances from 16S
and 18S ASV profiles collected in the Warnow estuary and the adjacent Baltic
coast, under four validation designs that separate interpolation, temporal
transfer and spatial transfer.

## Data

The data are not included in this repository. They are published separately:

- 16S/18S amplicon profiles and physicochemical measurements:
  Sperlea et al. (2025), *Scientific Data* 12:1774,
  https://doi.org/10.1038/s41597-025-06249-1
- Anthropogenic trace substances:
  Vogel et al. (2025), *Environmental Pollution* 383:126916,
  https://doi.org/10.1016/j.envpol.2025.126916
  (dataset: https://doi.org/10.1594/PANGAEA.975100)

Place the processed tables in `data/processed/` before running the notebooks.

## Which code produces which result

| Thesis | Notebook | Result file |
|---|---|---|
| Chapter 4 — data preparation | `notebooks/02_preprocessing.ipynb` | — |
| Chapter 5 — validation designs | `notebooks/04_test_cv_strategies.ipynb`, `src/cv_strategies.py` | — |
| Chapter 6 — Phase 1 benchmark (TabPFN) | `notebooks/10_tabpfn_evaluation_strategies_v3.ipynb` | `results/appendix_A_full72_all_metrics.csv` |
| Chapter 6 — classical baselines | `notebooks/05_lasso_...`, `06_xgboost_...`, `07_random_forest_...` | `results/appendix_A_full72_all_metrics.csv` |
| Table 6.2 | — | `results/table_6_2_full72_allmetrics.csv` |
| Section 6.6 — fold-level diagnostics | `notebooks/10_tabpfn_evaluation_strategies_v3.ipynb` | `results/ch6_perfold_metrics.csv`, `results/ch6_full72_long.csv` |
| Section 6.6.2 — salinity LOSO per site | — | `results/ch6_salinity_loso_sitelevel.csv` |
| Chapter 7 — Phase 2 linear probing | `notebooks/13_phase2_clean.ipynb` | `results/phase2_downstream_emb_vs_raw.csv` |
| Table 7.3(e) — Hellinger control | `notebooks/13_phase2_clean.ipynb` | `results/phase2_raw_hellinger_control.csv` |
| Section 7.5 — cross-target probing | `notebooks/13_phase2_clean.ipynb` | `results/phase2_cross_target_v2.csv` |
| Chapter 7 — extracted embeddings | — | `results/embeddings/*.npz` |
| Chapter 8 — Phase 3 forecasting | `notebooks/14_phase3_hplc.ipynb` | `results/appendix_B_forecasting_216.csv`, `results/chapter8_master_forecasting_table.csv`, `results/chapter8_forecasting_all_metrics_54cells.csv` |
| Chapter 8 — model-free baselines | `notebooks/14_phase3_hplc.ipynb` | `results/persistence_baseline_six_targets.csv`, `results/forecasting_baseline_metrics.csv` |
| Chapter 8 — split-regime diagnostic | `notebooks/14_phase3_hplc.ipynb` | `results/appendix_C_split_regime.csv`, `results/phase3_split_regime_table.csv`, `results/phase3_random_split_multiseed.csv` |
| Section 8.4 — target autocorrelation | `notebooks/15_temporal_autocorrelation_all_targets.ipynb` | `results/temporal_autocorr_per_site.csv`, `results/temporal_autocorr_summary.csv` |

## Environment

- Python 3.11 (conda environment `thesis`)
- TabPFN package version 8.0.6, v3 regressor checkpoint
- scikit-learn, pandas, numpy, xgboost, matplotlib
- Computations were run on an NVIDIA L40S GPU
- Random seed 42 throughout; all models were evaluated on identical
  pre-computed folds

## Notes

The notebooks contain absolute paths of the original compute environment
(`/mnt/data/home/tv2414/tabpfn_project`). Adjust the `ROOT` / `PROJ` variable at
the top of each notebook to your own project directory before running. The
TabPFN checkpoint path in `notebooks/10_tabpfn_evaluation_strategies_v3.ipynb`
also points to a local cache directory.

The result files in `results/` are the exact outputs used for the tables and
figures of the submitted thesis; the tag `v1.0-thesis` marks that state.

## Author

Rahil Akbari Fartkhouni, MSc Informatics, University of Rostock.
Supervisors: Prof. Dr.-Ing. Stefan Lüdtke (University of Rostock),
Dr. Theodor Sperlea (Leibniz Institute for Baltic Sea Research Warnemünde).
