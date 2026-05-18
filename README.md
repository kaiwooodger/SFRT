# SFRT Biological Risk Analysis Reproducibility Package

This repository gathers the source code, figure-generation scripts, manuscript-facing figure outputs, and the curated result artifacts used for the current SFRT submission package.

## Structure
- `project/scripts/` — project scripts, run wrappers, renderers, and analysis utilities.
- `project/topas/` — TOPAS templates.
- `project/data/` — supporting input data.
- `project/public_results/` — copied manuscript-facing summary tables, quick assessments, and selected result assets for public sharing.
- `figures/PMB_SFRT_publishable_source_clean/` — cleaned manuscript figure bundle.
- `manuscript/SFRT_Submission.pdf` — current submission PDF.

## Important note
This is a portable public-clean package. Machine-specific symbolic links and oversized raw transport intermediates have been removed. The repository keeps the code, final figures, manuscript PDF, and manuscript-facing summary outputs needed to inspect the study and regenerate the paper workflow, while excluding bulky case-level dose cubes, binary material-tag files, and large volume arrays that are not suitable for standard GitHub hosting.

The principal excluded raw artifacts were the largest transport-volume products and repeat-run intermediates, including the original `phase11c_assay_volumes.npz`, `phase12_mc_coupled_volumes.npz`, case-level combined physical dose grids, and repeated-subset raw dose exports. These can be regenerated from the included scripts.

The `project/public_results/` directory is intentionally curated rather than exhaustive. It contains the high-signal manuscript-facing summaries from the benchmark, cohort, uncertainty, and falsification analyses, while leaving raw transport scratch space out of the public repository.

Current TOPAS defaults for the benchmark and cohort transport scripts are `1×10^6` histories for the background/base component and `2×10^6` histories for the vertex component. Historical result tables preserved under `project/public_results/` retain the metadata from the original archived runs and should be interpreted as fixed records rather than regenerated outputs at the current defaults.

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
