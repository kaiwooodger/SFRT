# SFRT biological risk-analysis reproducibility bundle

This repository is the clean GitHub-facing reproducibility package for the PMB revision workflow.

It exposes descriptive public entry points, descriptive manuscript-facing result directories, and the current publication-ready figure set, while preserving the original implementation scripts in a provenance layer so the scientific workflow remains reproducible.

## Repository structure

- `project/scripts/` — clean public wrappers and verification scripts
- `project/scripts/internal_legacy/` — preserved original implementation scripts
- `project/public_results/` — clean manuscript-facing result tables and summaries
- `project/data/` — small supporting input files
- `project/topas/` — TOPAS templates
- `figures/manuscript_clean/` — publication-ready figures
- `manuscript/` — manuscript PDF plus reviewer-facing tables and guides
- `bundle_manifest.tsv` — source-to-bundle mapping for this package

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
bash project/scripts/reproduce_manuscript.sh --mode public
```

## Full rerun

```bash
TOPAS_BIN=/path/to/topas \
G4_DATA_DIR=/path/to/GEANT4 \
bash project/scripts/reproduce_manuscript.sh --mode full
```

## Reproducibility note

The clean public layer avoids internal stage labels in the main review surface. Where historical implementation names were necessary for script stability, they were preserved inside `project/scripts/internal_legacy/` and kept out of the first-pass review path.
