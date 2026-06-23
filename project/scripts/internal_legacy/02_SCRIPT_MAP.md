# Script Map

This map separates the files needed for the manuscript from the larger historical script archive.

## Reviewer fast path

Use these files first:

- [`00_START_HERE.md`](./00_START_HERE.md)
- [`01_REPRODUCE_PMB.sh`](./01_REPRODUCE_PMB.sh)
- [`README.md`](./README.md)

## Core manuscript workflow

These are the main entry points behind the PMB submission:

| Purpose | Script |
| --- | --- |
| Synthetic cohort generation | `run_phase32_site_specific_template_phantoms.py` |
| TOPAS physical cohort | `run_phase33_phase32_topas_cohort.py` |
| Biology-aware reinterpretation | `run_phase34_phase32_bio_cohort.py` |
| Uncertainty repeat subset | `run_phase35_subset_repeat_uncertainty.py` |
| Stronger sink falsification cohort | `run_phase37a_vessel_falsification_cohort.py` |
| Stronger sink falsification uncertainty overlay | `run_phase37b_vessel_falsification_uncertainty.py` |
| Manuscript figure regeneration | `render_pmb_source_clean_figures.py` |
| One-command rerun wrapper | `run_high_history_paper_refresh.sh` |

## Supporting modules used by the core workflow

These are imported by the main workflow scripts and are useful when tracing implementation details:

- `bystander_multispecies_pde_solver.py`
- `bystander_pde_solver.py`
- `geometry_generators.py`
- `geometry_generalization_sets.py`
- `phase36_sink_falsification_utils.py`
- `phase37_sink_falsification_utils.py`

## Public-results reviewer mode

If you only want to validate the manuscript-facing outputs already shipped with the repository:

- run `01_REPRODUCE_PMB.sh --mode public`
- inspect `project/public_results/`
- inspect `figures/PMB_SFRT_publishable_source_clean/`

## Historical and provenance scripts

These files are kept for provenance but are not required for a first-pass review of the manuscript:

- `run_phase10_*.py` through `run_phase31_*.py`
- `generate_phase*.py`
- `analyze_*.py`
- `plot_*.py`
- exploratory shell launchers such as `*_1e5.sh`

They document the benchmark build-up, exploratory optimisation work, and earlier figure generation, but they are not the shortest path to the final paper results.

## Safe first-pass rule

If a file is not named in the **Core manuscript workflow** table above, it can usually be treated as secondary on first review.
