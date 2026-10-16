# ADNI MCI prognosis analysis code

Code for comparing gradient-boosted and neural sequence models for two-year diagnostic outcomes in ADNI, including the effect of excluding participants whose MCI diagnosis reverts to normal cognition.

This repository is private. It contains source code and the recorded Python environment. It contains no ADNI source tables, participant records, individual predictions, model weights, executed notebooks, or unpublished manuscript files. Obtain ADNI access independently and follow the [ADNI Data Use Agreement](https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf).

## Environment

The recorded environment used Python 3.12.14 on macOS with an Apple M-series CPU. Other platforms have not been validated by this export.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

The lock file records the author's existing environment; future availability of every pinned package is not guaranteed.

## Local data inputs

Place your authorised ADNIMERGE2 data files under `data/raw/adni/ADNIMERGE2/data/`. The manuscript uses ADNIMERGE2 version 0.1.1 with a recorded download date of 12 November 2025. A different data snapshot can change the cohort and results.

The preprocessing and main analyses require these `.rda` tables: `DATA_DOWNLOADED_DATE`, `MMSE`, `MOCA`, `ADAS`, `CDR`, `DXSUM`, `PTDEMOG`, `UCSFFSX7`, `APOERES` and `UCBERKELEY_AMY_6MM`. Supplementary revision analyses also read `NEUROBAT` and `TREATDIS`. No credentials or datasets are provided here.

## Prepare the cohorts

```bash
.venv/bin/python scripts/prepare_inputs.py
```

This runs the original longitudinal-data preparation, builds the diagnosis labels and MRI features, recreates the v1 participant split (70/15/15; seed 42), and builds the v2 main and sensitivity cohorts. The v2 main cohort deliberately inherits the v1 participant split. All resulting participant-level files stay local under the ignored `data/` directory.

`scripts/build_adni_longitudinal.py` was copied from the original data-preparation checkout. `scripts/prepare_inputs.py` is a new command-line wrapper for the preparation functions and the existing notebook split steps. These steps were not rerun on ADNI records when exporting this repository.

## Train and evaluate

```bash
.venv/bin/python scripts/rerun_all_v2.py
.venv/bin/python scripts/make_report_v2.py
.venv/bin/python scripts/amyloid_subgroups.py
.venv/bin/python scripts/revision_analyses.py
.venv/bin/python scripts/make_figures_v2.py
.venv/bin/python scripts/make_fig1_v3.py
```

Run commands from the repository root. `rerun_all_v2.py` runs gradient-boosted and neural models in separate processes: loading LightGBM and PyTorch in the same process caused OpenMP failures on the original macOS environment. Its neural jobs can consume substantial CPU and memory. Existing prediction files and tuning results are reused by the analysis scripts; use a fresh ignored output directory when a complete rerun is needed.

The main version is `*_v2.py`. Unversioned scripts and the corresponding source modules are retained as earlier implementations and shared dependencies. The LongForMAD implementation is an inspired reimplementation with documented departures from the original method; it is not the original authors' implementation.

## Repository layout

| Path | Purpose |
|---|---|
| `src/data_engineering/` | Diagnosis labels and MRI features |
| `src/dataset/` | Cohort rules, histories and participant splits |
| `src/modeling/` | GBM, GRU, Transformer models, training and validation |
| `src/evaluation/` | Metrics, calibration, comparisons and report functions |
| `scripts/` | Data preparation, experiments, subgroup analyses, tables and figures |
| `requirements.lock.txt` | Recorded Python dependencies |
| `SOURCE_MANIFEST.json` | Original source file hashes and provenance |

Outputs are generated locally under `data/`, `results/`, `reports/` and `paper/figures/`; these paths are ignored by Git. Never commit ADNI raw or derived participant-level data, individual predictions or credentials. The manuscript and journal submission package are managed separately.

## Export validation

All exported Python files were parsed for syntax. Original source hashes were checked, and the Git file list was restricted to source files, documentation, provenance and dependency metadata. Full training and independent reproduction from ADNI data were not rerun as part of the export.
