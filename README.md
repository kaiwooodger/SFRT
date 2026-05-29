# SFRT Biological Risk Analysis Reproducibility Package

This repository gathers the source code, figure-generation scripts, manuscript-facing figure outputs, and the curated result artifacts used for the current SFRT submission package.

This repository is provided for research reproducibility. The biological outputs are model-derived effective-dose-like scores and relative assay-like proxies. They are not delivered absorbed dose, clinical toxicity predictions, or treatment-planning recommendations.

## Structure
- `configs/` — reviewer-facing YAML snapshots of the locked biology, vascular-sink, TOPAS, and uncertainty settings used in the PMB submission.
- `examples/` — small runnable examples that demonstrate biological reinterpretation, vascular-sink comparison, and endpoint extraction from the archived public result tables.
- `project/scripts/` — project scripts, run wrappers, renderers, and analysis utilities.
- `project/topas/` — TOPAS templates.
- `project/data/` — supporting input data.
- `project/public_results/` — copied manuscript-facing summary tables, quick assessments, and selected result assets for public sharing.
- `manuscript/parameter_table.csv` — consolidated reviewer-facing parameter table for the PMB submission.
- `manuscript/pmb_reproducibility_guide.md` — direct rerun guide for the manuscript workflows and outputs.
- `manuscript/figure_manifest.md` — manifest for the submission figures and their generator/source assets.
- `manuscript/endpoint_manifest.csv` — definitions for the primary, supplemental, and assay-like outputs reported in the paper.
- `figures/PMB_SFRT_publishable_source_clean/` — cleaned manuscript figure bundle.
- `manuscript/SFRT_Submission.pdf` — current submission PDF.

## Important note
This is a portable public-clean package. Machine-specific symbolic links and oversized raw transport intermediates have been removed. The repository keeps the code, final figures, manuscript PDF, and manuscript-facing summary outputs needed to inspect the study and regenerate the paper workflow, while excluding bulky case-level dose cubes, binary material-tag files, and large volume arrays that are not suitable for standard GitHub hosting.

The principal excluded raw artifacts were the largest transport-volume products and repeat-run intermediates, including the original `phase11c_assay_volumes.npz`, `phase12_mc_coupled_volumes.npz`, case-level combined physical dose grids, and repeated-subset raw dose exports. These can be regenerated from the included scripts.

The `project/public_results/` directory is intentionally curated rather than exhaustive. It contains the high-signal manuscript-facing summaries from the benchmark, cohort, uncertainty, and falsification analyses, while leaving raw transport scratch space out of the public repository.

Current TOPAS defaults for the benchmark and cohort transport scripts are `1×10^6` histories for the background/base component and `2×10^6` histories for the vertex component.

## Reproducing the PMB submission
This repository supports the manuscript:

> **Biology-informed reinterpretation of lattice radiotherapy using non-local bystander signalling and anatomy-aware vascular sink modelling**

The public PMB bundle is **script-first**. Manuscript reproduction does not depend on a legacy package CLI. The central submission workflows in this repository cover:

1. synthetic phantom generation,
2. lattice candidate and retained-vertex generation,
3. TOPAS input and template generation,
4. biological reaction-diffusion post-processing,
5. physical and biological endpoint extraction,
6. vascular-sink falsification controls, and
7. figure and table regeneration for the manuscript.

### Minimal environment setup
```bash
git clone https://github.com/kair98-boop/SFRT.git
cd SFRT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
```

### Core manuscript rerun sequence
```bash
python project/scripts/run_phase32_site_specific_template_phantoms.py
python project/scripts/run_phase33_phase32_topas_cohort.py --topas-bin /path/to/topas --g4-data-dir /path/to/GEANT4
python project/scripts/run_phase34_phase32_bio_cohort.py
python project/scripts/run_phase35_subset_repeat_uncertainty.py
python project/scripts/run_phase37a_vessel_falsification_cohort.py
python project/scripts/run_phase37b_vessel_falsification_uncertainty.py
python project/scripts/render_pmb_source_clean_figures.py
```

### Reviewer-facing guides and manifests
- `manuscript/pmb_reproducibility_guide.md`
- `manuscript/parameter_table.csv`
- `manuscript/figure_manifest.md`
- `manuscript/endpoint_manifest.csv`
- `configs/biology_parameters.yaml`
- `configs/vascular_sink_parameters.yaml`
- `configs/topas_case_parameters.yaml`
- `configs/uncertainty_parameters.yaml`

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
