#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/kw/Documents/Playground/vhee_topas"
RUN_ROOT="$REPO_ROOT/runs/paper_asym_250_seed33"
CASE_ID="E250_p14p30"
MANIFEST="$RUN_ROOT/manifest.json"
CSV_PATH="$RUN_ROOT/cases/$CASE_ID/dose.csv"
ANALYSIS_DIR="$RUN_ROOT/analysis"
PDD_DIR="$RUN_ROOT/analysis_pdd"
LOG_FILE="$RUN_ROOT/postprocess.log"

cd "$REPO_ROOT"

echo "[$(date)] postprocess watcher started" >> "$LOG_FILE"
echo "[$(date)] waiting for TOPAS run to finish: $RUN_ROOT" >> "$LOG_FILE"

# Wait until no TOPAS process is attached to this run-root invocation.
while pgrep -f "build_asymmetric_sweep.py --run-root runs/paper_asym_250_seed33" >/dev/null 2>&1; do
  sleep 20
done

echo "[$(date)] build_asymmetric_sweep process finished" >> "$LOG_FILE"

if [[ ! -s "$CSV_PATH" ]]; then
  echo "[$(date)] ERROR: dose.csv missing or empty: $CSV_PATH" >> "$LOG_FILE"
  exit 1
fi

echo "[$(date)] running analyze_topas_outputs.py" >> "$LOG_FILE"
/opt/anaconda3/bin/python scripts/analyze_topas_outputs.py \
  --manifest "$MANIFEST" \
  --outdir "$ANALYSIS_DIR" \
  --z-mode integrated_xy \
  --sigma-mode integrated_xy \
  --sigma-fit-mode gaussian_2d \
  --io-retries 10 \
  --io-retry-delay-sec 1.0 >> "$LOG_FILE" 2>&1

echo "[$(date)] building PDD artifacts" >> "$LOG_FILE"
/opt/anaconda3/bin/python - <<'PY' >> "$LOG_FILE" 2>&1
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

repo = Path("/Users/kw/Documents/Playground/vhee_topas")
sys.path.append(str(repo / "scripts"))
from analyze_topas_outputs import load_topas_grid

csv_path = repo / "runs/paper_asym_250_seed33/cases/E250_p14p30/dose.csv"
outdir = repo / "runs/paper_asym_250_seed33/analysis_pdd"
outdir.mkdir(parents=True, exist_ok=True)

grid, header = load_topas_grid(csv_path, retries=10, retry_delay_sec=1.0)
nx, ny, nz = grid.shape
cx, cy = nx // 2, ny // 2
dz = float(header["dz_cm"])
z_cm = (np.arange(nz, dtype=float) + 0.5) * dz

on_axis = grid[cx, cy, :].astype(float)
integrated = np.sum(grid, axis=(0, 1)).astype(float)
on_axis_max = float(np.max(on_axis))
integrated_max = float(np.max(integrated))

pdd_on_axis = 100.0 * on_axis / on_axis_max if on_axis_max > 0 else np.full_like(on_axis, np.nan)
pdd_integrated = (
    100.0 * integrated / integrated_max if integrated_max > 0 else np.full_like(integrated, np.nan)
)

peak_idx_on = int(np.nanargmax(pdd_on_axis)) if np.any(np.isfinite(pdd_on_axis)) else -1
peak_idx_int = int(np.nanargmax(pdd_integrated)) if np.any(np.isfinite(pdd_integrated)) else -1

out_csv = outdir / "pdd_curve_250MeV_q4_14p3_h1e6_seed33.csv"
out_png = outdir / "pdd_curve_250MeV_q4_14p3_h1e6_seed33.png"
out_meta = outdir / "pdd_curve_250MeV_q4_14p3_h1e6_seed33_meta.txt"

pd.DataFrame(
    {
        "z_cm": z_cm,
        "pdd_on_axis_pct": pdd_on_axis,
        "pdd_integrated_xy_pct": pdd_integrated,
        "dose_on_axis_raw": on_axis,
        "dose_integrated_xy_raw": integrated,
    }
).to_csv(out_csv, index=False)

plt.figure(figsize=(8.6, 5.2))
plt.plot(z_cm, pdd_on_axis, lw=2.2, label="PDD on-axis (normalized to on-axis max)")
plt.plot(z_cm, pdd_integrated, lw=1.8, ls="--", label="PDD integrated x-y (normalized to integrated max)")
if peak_idx_on >= 0:
    plt.axvline(z_cm[peak_idx_on], color="tab:blue", ls=":", lw=1.2, alpha=0.9)
if peak_idx_int >= 0:
    plt.axvline(z_cm[peak_idx_int], color="tab:orange", ls=":", lw=1.2, alpha=0.9)
plt.xlabel("Depth in water, z (cm)")
plt.ylabel("Percentage Depth Dose (%)")
plt.ylim(0, 105)
plt.xlim(float(z_cm[0]), float(z_cm[-1]))
plt.title("250 MeV PDD (Q4=14.3 T/m, 1,000,000 histories, seed 33, ideal_opposite)")
plt.grid(alpha=0.25)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()

out_meta.write_text(
    "\n".join(
        [
            f"input_csv={csv_path.resolve()}",
            f"nx={nx}",
            f"ny={ny}",
            f"nz={nz}",
            f"dz_cm={dz}",
            f"peak_on_axis_z_cm={z_cm[peak_idx_on] if peak_idx_on >= 0 else float('nan')}",
            f"peak_integrated_z_cm={z_cm[peak_idx_int] if peak_idx_int >= 0 else float('nan')}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(f"WROTE_CSV {out_csv}")
print(f"WROTE_PNG {out_png}")
print(f"WROTE_META {out_meta}")
print(f"PEAK_ON_AXIS_Z_CM {z_cm[peak_idx_on] if peak_idx_on >= 0 else float('nan')}")
print(f"PEAK_INTEGRATED_Z_CM {z_cm[peak_idx_int] if peak_idx_int >= 0 else float('nan')}")
PY

echo "[$(date)] postprocess complete" >> "$LOG_FILE"
