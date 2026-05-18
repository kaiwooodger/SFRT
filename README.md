# SFRT Submission Reproducibility Bundle

This desktop bundle gathers the source code, figure-generation scripts, manuscript-facing figure outputs, and the result folders used for the current SFRT submission package.

## Structure
- `project/scripts/` — project scripts, run wrappers, renderers, and analysis utilities.
- `project/topas/` — TOPAS templates.
- `project/data/` — supporting input data.
- `project/runs/` — linked benchmark, cohort, uncertainty, falsification, and manuscript asset outputs.
- `figures/PMB_SFRT_publishable_source_clean/` — cleaned manuscript figure bundle.
- `manuscript/SFRT_Submission.pdf` — current submission PDF.

## Important note
The large result folders and manuscript figure bundle are included as symbolic links to the original source locations on this machine. This keeps the desktop package lightweight while preserving a single entry folder that contains everything needed to inspect and reproduce the present submission on this workstation.

## Main reproduction entry points
- `project/scripts/run_phase30_phase28_topas_true_lattice_delivery.py`
- `project/scripts/run_phase33_phase32_topas_cohort.py`
- `project/scripts/run_phase34_phase32_bio_cohort.py`
- `project/scripts/run_phase35_subset_repeat_uncertainty.py`
- `project/scripts/run_phase36a_vessel_falsification_cohort.py`
- `project/scripts/run_phase36b_vessel_falsification_uncertainty.py`
- `project/scripts/run_phase37a_vessel_falsification_cohort.py`
- `project/scripts/run_phase37b_vessel_falsification_uncertainty.py`
- `project/scripts/render_pmb_source_clean_figures.py`
- `project/scripts/run_high_history_paper_refresh.sh`
