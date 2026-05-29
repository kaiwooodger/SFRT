# Start Here

This folder contains the full script history of the project, so it is broader than what a reviewer needs on first pass.

If you are reviewing the PMB submission, use this short path:

1. Read [`02_SCRIPT_MAP.md`](./02_SCRIPT_MAP.md).
2. Run [`01_REPRODUCE_PMB.sh`](./01_REPRODUCE_PMB.sh).
3. Inspect the outputs in:
   - `project/public_results/`
   - `figures/PMB_SFRT_publishable_source_clean/`
   - `manuscript/SFRT_Submission.pdf`

## Two review modes

- **Public-results mode**
  - Uses the archived manuscript-facing result tables already included in the repository.
  - Does **not** require TOPAS.
  - Best choice for a fast reproducibility check.

- **Full mode**
  - Rebuilds the main manuscript workflow from synthetic phantom generation through transport, biology, uncertainty, falsification, and figure rendering.
  - Requires a working local TOPAS installation.

## Fastest command

From the repository root:

```bash
bash project/scripts/01_REPRODUCE_PMB.sh --mode public
```

For the full transport-aware rerun:

```bash
TOPAS_BIN=/path/to/topas \
G4_DATA_DIR=/path/to/GEANT4 \
bash project/scripts/01_REPRODUCE_PMB.sh --mode full
```

## Files that matter most

- `01_REPRODUCE_PMB.sh` — single entry point for manuscript reproduction
- `02_SCRIPT_MAP.md` — concise map of which scripts are core, supporting, or historical
- `README.md` — slightly broader index of the scripts folder
