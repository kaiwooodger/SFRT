# Scripts Index

This folder contains both the final manuscript workflow and the broader historical script archive used during development.

If you are reviewing the PMB submission, do **not** start by scanning every file in this directory.

## Use this short path

1. Read [`00_START_HERE.md`](./00_START_HERE.md).
2. Run [`01_REPRODUCE_PMB.sh`](./01_REPRODUCE_PMB.sh).
3. Use [`02_SCRIPT_MAP.md`](./02_SCRIPT_MAP.md) to identify which scripts are core and which are historical.

## Main manuscript entry points

- `run_phase32_site_specific_template_phantoms.py`
- `run_phase33_phase32_topas_cohort.py`
- `run_phase34_phase32_bio_cohort.py`
- `run_phase35_subset_repeat_uncertainty.py`
- `run_phase37a_vessel_falsification_cohort.py`
- `run_phase37b_vessel_falsification_uncertainty.py`
- `render_pmb_source_clean_figures.py`
- `run_high_history_paper_refresh.sh`

## Supporting modules most relevant to the manuscript workflow

- `bystander_multispecies_pde_solver.py`
- `bystander_pde_solver.py`
- `geometry_generators.py`
- `geometry_generalization_sets.py`
- `phase36_sink_falsification_utils.py`
- `phase37_sink_falsification_utils.py`

## What the rest of the folder is

Most remaining files are preserved for provenance:

- earlier benchmark and optimisation studies
- exploratory plotting and analysis scripts
- legacy figure-generation utilities
- shell wrappers used during local batch reruns

They are kept so the repository documents how the final submission evolved, but they are not required for a first-pass review of the paper workflow.
