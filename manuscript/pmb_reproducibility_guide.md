# PMB reproducibility guide

This guide documents the direct rerun path for the manuscript:

> **Biology-informed reinterpretation of lattice radiotherapy using non-local bystander signalling and anatomy-aware vascular sink modelling**

## Scope
The public bundle contains the scripts, TOPAS templates, supporting input data, cleaned manuscript figures, and manuscript-facing result summaries needed to:

1. regenerate the site-specific synthetic cohort,
2. rerun the benchmark and cohort TOPAS dose workflows,
3. rerun biology-aware post-processing and endpoint extraction,
4. reproduce uncertainty and falsification analyses, and
5. regenerate the publication figure bundle.

## Environment
- Python 3.11 or later
- TOPAS executable available locally
- Geant4 data directory available locally

Setup:

```bash
git clone https://github.com/kair98-boop/SFRT.git
cd SFRT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
```

## Primary rerun sequence

### 1. Generate the 10-case site-specific synthetic cohort
```bash
python project/scripts/run_phase32_site_specific_template_phantoms.py
```

Main output:
- `project/runs/phase32_site_specific_template_phantoms/`

### 2. Generate the physical TOPAS cohort
```bash
python project/scripts/run_phase33_phase32_topas_cohort.py \
  --topas-bin /path/to/topas \
  --g4-data-dir /path/to/GEANT4
```

Main output:
- `project/runs/phase33_phase32_topas_cohort/`

### 3. Apply biology-aware risk analysis
```bash
python project/scripts/run_phase34_phase32_bio_cohort.py
```

Main output:
- `project/runs/phase33_phase32_topas_cohort/phase34_bio_cohort/`

### 4. Repeat the uncertainty subset
```bash
python project/scripts/run_phase35_subset_repeat_uncertainty.py
```

### 5. Run vascular-sink falsification analyses
```bash
python project/scripts/run_phase37a_vessel_falsification_cohort.py
python project/scripts/run_phase37b_vessel_falsification_uncertainty.py
```

### 6. Regenerate the cleaned manuscript figures
```bash
python project/scripts/render_pmb_source_clean_figures.py
```

Main output:
- `figures/PMB_SFRT_publishable_source_clean/`

## Fast reviewer inspection path
If a full TOPAS rerun is not required, the fastest review path is:

1. inspect `manuscript/parameter_table.csv`,
2. inspect `manuscript/endpoint_manifest.csv`,
3. inspect `manuscript/figure_manifest.md`,
4. inspect `project/public_results/`,
5. inspect `figures/PMB_SFRT_publishable_source_clean/`, and
6. inspect `manuscript/SFRT_Submission.pdf`.

## Core output tables used by the paper
- `project/public_results/phase33_34_cohort/phase33_physical_endpoint_table.csv`
- `project/public_results/phase33_34_cohort/phase34_endpoint_table.csv`
- `project/public_results/phase33_34_cohort/phase34_rank_shift_table.csv`
- `project/public_results/phase35_uncertainty/phase35_sink_delta_noise_table.csv`
- `project/public_results/phase37a_vessel_falsification_cohort/phase37a_falsification_endpoint_summary.csv`
- `project/public_results/phase37b_vessel_falsification_uncertainty/phase37b_true_vs_surrogate_noise_table.csv`

## Notes
- The public repository is a curated reproducibility bundle rather than a full raw-data archive.
- Large binary dose cubes and intermediate arrays were excluded from GitHub hosting but can be regenerated from the included scripts.
- The script-based workflow is the authoritative PMB rerun path for this repository.
