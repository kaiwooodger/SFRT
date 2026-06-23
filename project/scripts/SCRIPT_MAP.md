# Script Map

## Clean public entry points

| Purpose | Script |
| --- | --- |
| Synthetic cohort generation | `generate_synthetic_cohort.py` |
| TOPAS physical cohort | `run_topas_cohort.py` |
| Biology-informed reinterpretation | `apply_biological_reinterpretation.py` |
| Repeated-run uncertainty | `run_repeat_uncertainty.py` |
| Sink falsification cohort | `run_sink_falsification_cohort.py` |
| Sink falsification uncertainty overlay | `run_sink_falsification_uncertainty.py` |
| Biology-parameter robustness | `run_biology_parameter_robustness.py` |
| Smoothing-kernel sensitivity | `run_smoothing_kernel_sensitivity.py` |
| Manuscript artifact verification | `verify_public_bundle.py` |
| One-command public/full rerun wrapper | `reproduce_manuscript.sh` |

## Preserved implementation layer

- `internal_legacy/` contains the preserved implementation and provenance scripts.
- That layer keeps the original working names so the scientific dependency chain does not break.
- The clean public layer is what should be linked from the GitHub README and used for first-pass review.
