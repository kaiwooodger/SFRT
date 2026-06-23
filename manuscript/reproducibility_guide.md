# PMB reproducibility guide

This clean GitHub bundle is the public reproducibility package for:

> **Biology-informed reinterpretation of lattice radiotherapy using non-local bystander signalling and anatomy-aware vascular sink modelling**

## Fastest public validation

From the repository root:

```bash
bash project/scripts/reproduce_manuscript.sh --mode public
```

This validates the bundled manuscript-facing outputs and prints the frozen headline numbers.

## Full rerun path

```bash
TOPAS_BIN=/path/to/topas \
G4_DATA_DIR=/path/to/GEANT4 \
bash project/scripts/reproduce_manuscript.sh --mode full
```

The clean wrappers call the preserved full implementation stored in `project/scripts/internal_legacy/`.

## Main public result locations

- `project/public_results/cohort_summary/`
- `project/public_results/repeat_run_uncertainty/`
- `project/public_results/sink_falsification/`
- `project/public_results/biology_parameter_robustness/`
- `project/public_results/smoothing_kernel_sensitivity/`
- `figures/manuscript_clean/`
- `manuscript/SFRT_Submission.pdf`

## Important note

This repository is a clean public bundle. The exposed entry points and result directories are descriptive and stable for GitHub review. The original implementation scripts are preserved only for provenance and full reruns.
