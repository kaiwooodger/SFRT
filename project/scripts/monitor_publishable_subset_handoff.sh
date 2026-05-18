#!/bin/zsh

set -u

REPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$REPO_ROOT/runs/publishable_subset"
TARGET_ANALYZED=12
CHECK_INTERVAL_SEC=120

RUN_CMD=(
  caffeinate -dimsu
  /opt/anaconda3/bin/python
  scripts/run_publishable_subset.py
  --case-metrics runs/analysis_paper2d/case_metrics.csv
  --manifest runs/manifest.json
  --run-root runs/publishable_subset
  --histories 1000000
  --seeds 11 22 33
  --threads 8
  --run-topas
  --skip-existing
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

count_analyzed() {
  find "$RUN_ROOT/seed_runs" -path "*/analysis/case_metrics.csv" -type f 2>/dev/null | wc -l | tr -d " "
}

launcher_running() {
  pgrep -f "scripts/run_publishable_subset.py" >/dev/null 2>&1
}

print_status() {
  local analyzed
  analyzed="$(count_analyzed)"
  local seed_dirs
  seed_dirs="$(find "$RUN_ROOT/seed_runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d " ")"
  echo "[$(timestamp)] analyzed=${analyzed}/${TARGET_ANALYZED} seed_dirs=${seed_dirs}"
}

validate_outputs() {
  local missing=0
  local expected=(
    "$RUN_ROOT/replicate_metrics.csv"
    "$RUN_ROOT/aggregate_seed_stats.csv"
    "$RUN_ROOT/best_by_energy_seedmean.csv"
    "$RUN_ROOT/publishable_subset_report.md"
  )

  for file in "${expected[@]}"; do
    if [ ! -s "$file" ]; then
      echo "[$(timestamp)] missing_or_empty: $file"
      missing=1
    fi
  done

  local plot_count=0
  if [ -d "$RUN_ROOT/plots" ]; then
    plot_count="$(find "$RUN_ROOT/plots" -type f -name "*.png" | wc -l | tr -d " ")"
  fi
  echo "[$(timestamp)] plot_count=${plot_count}"
  if [ "$plot_count" -lt 4 ]; then
    echo "[$(timestamp)] expected at least 4 plot PNG files in $RUN_ROOT/plots"
    missing=1
  fi

  local agg_rows=0
  if [ -f "$RUN_ROOT/aggregate_seed_stats.csv" ]; then
    agg_rows="$(($(wc -l < "$RUN_ROOT/aggregate_seed_stats.csv") - 1))"
  fi
  echo "[$(timestamp)] aggregate_seed_stats_rows=${agg_rows}"

  if [ "$missing" -eq 0 ]; then
    echo "[$(timestamp)] VALIDATION_STATUS=PASS"
    return 0
  fi
  echo "[$(timestamp)] VALIDATION_STATUS=FAIL"
  return 1
}

cd "$REPO_ROOT" || exit 1
echo "[$(timestamp)] monitor start"
print_status

while true; do
  local_analyzed="$(count_analyzed)"
  if launcher_running; then
    echo "[$(timestamp)] launcher_active analyzed=${local_analyzed}/${TARGET_ANALYZED}"
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  if [ "$local_analyzed" -lt "$TARGET_ANALYZED" ]; then
    echo "[$(timestamp)] launcher_not_running analyzed=${local_analyzed}/${TARGET_ANALYZED}; resuming exact handoff command"
    (
      cd "$REPO_ROOT" || exit 1
      "${RUN_CMD[@]}"
    )
    rc=$?
    echo "[$(timestamp)] resume_command_exit_code=${rc}"
    sleep 10
    continue
  fi

  echo "[$(timestamp)] analyzed target reached (${local_analyzed}/${TARGET_ANALYZED})"
  break
done

echo "[$(timestamp)] running post-completion validation"
if ! validate_outputs; then
  echo "[$(timestamp)] outputs incomplete; rerunning exact workflow command to regenerate aggregates/plots"
  (
    cd "$REPO_ROOT" || exit 1
    "${RUN_CMD[@]}"
  )
  rc=$?
  echo "[$(timestamp)] regenerate_command_exit_code=${rc}"
  validate_outputs
fi

print_status
echo "[$(timestamp)] monitor finished"
