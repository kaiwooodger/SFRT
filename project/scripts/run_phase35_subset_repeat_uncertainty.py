#!/usr/bin/env python3
"""Phase 35: repeat a representative Phase 32 subset under seed/history uncertainty.

This phase is a publication-facing robustness package. It reruns a representative
subset of the Phase 32 site-specific cohort across multiple TOPAS seeds and
history levels, propagates those repeated physical doses through the Phase 34
biology workflow, and then quantifies whether sink-driven endpoint and rank
changes are larger than a combined Monte Carlo plus uptake-sensitivity band.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from run_phase26_vascular_sink_ablation import PRIMARY_ENDPOINTS, assign_ranks, build_rank_shift_rows, endpoint_z_scores
from run_phase31_publication_package import (
    bootstrap_interval,
    build_biology_sigma_lookup,
    load_csv_rows,
    paired_cohens_dz,
    scalar_summary,
    write_csv,
    write_json,
)


DEFAULT_CASE_SUBSET: Tuple[str, ...] = (
    "case02",  # orbit-adjacent sinonasal crescent, sink rank change
    "case03",  # BOT central mass, biology-added brainstem failure
    "case04",  # elongated laryngo-hypopharynx geometry
    "case05",  # deep parapharyngeal case with sink + failure signal
    "case06",  # superficial cheek crescent, sink rank change
    "case07",  # oral tongue, best physical rank downgraded biologically
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase32-root",
        type=Path,
        default=root / "runs" / "phase32_site_specific_template_phantoms",
    )
    parser.add_argument(
        "--baseline-phase33-root",
        type=Path,
        default=root / "runs" / "phase33_phase32_topas_cohort",
    )
    parser.add_argument(
        "--baseline-phase34-root",
        type=Path,
        default=root / "runs" / "phase33_phase32_topas_cohort" / "phase34_bio_cohort",
    )
    parser.add_argument(
        "--phase25-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5",
    )
    parser.add_argument(
        "--phase26-run-root",
        type=Path,
        default=root
        / "runs"
        / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase25_safe_core_plan_library_1e5"
        / "phase26_vascular_sink_ablation",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "runs" / "phase35_subset_repeat_uncertainty",
    )
    parser.add_argument(
        "--only-case-ids",
        type=str,
        nargs="*",
        default=list(DEFAULT_CASE_SUBSET),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--history-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--histories-base", type=int, default=12000)
    parser.add_argument("--histories-spot", type=int, default=24000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dose-smoothing-mm", type=float, default=6.0)
    parser.add_argument("--history-interval", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=35)
    parser.add_argument("--rank-sim-samples", type=int, default=2000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--verbose-pde", action="store_true")
    return parser.parse_args()


def scale_label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def run_logged_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Phase 35 repeat command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"Stdout tail:\n{(result.stdout or '')[-2000:]}\n"
            f"Stderr tail:\n{(result.stderr or '')[-2000:]}"
        )


def run_repeat_combination(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    case_ids: Sequence[str],
    seed: int,
    history_scale: float,
) -> Dict[str, object]:
    repeat_id = f"seed{int(seed)}_hist{scale_label(float(history_scale))}"
    repeat_root = args.out_root.resolve() / "repeat_runs" / repeat_id
    phase33_root = repeat_root / "phase33"
    phase34_root = repeat_root / "phase34_bio_cohort"
    phase33_manifest = phase33_root / "phase33_case_manifest.csv"
    phase34_endpoint = phase34_root / "phase34_endpoint_table.csv"
    histories_base = max(100, int(round(int(args.histories_base) * float(history_scale))))
    histories_spot = max(100, int(round(int(args.histories_spot) * float(history_scale))))

    phase33_script = repo_root / "scripts" / "run_phase33_phase32_topas_cohort.py"
    phase34_script = repo_root / "scripts" / "run_phase34_phase32_bio_cohort.py"

    if not args.analysis_only:
        if not (bool(args.skip_existing) and phase33_manifest.exists()):
            cmd = [
                sys.executable,
                str(phase33_script),
                "--phase32-root",
                str(args.phase32_root.resolve()),
                "--out-root",
                str(phase33_root),
                "--seed",
                str(int(seed)),
                "--histories-base",
                str(histories_base),
                "--histories-spot",
                str(histories_spot),
                "--threads",
                str(int(args.threads)),
                "--dose-smoothing-mm",
                str(float(args.dose_smoothing_mm)),
                "--only-case-ids",
                *[str(case_id) for case_id in case_ids],
            ]
            if bool(args.skip_existing):
                cmd.append("--skip-existing")
            run_logged_command(
                cmd,
                cwd=repo_root,
                stdout_log=repeat_root / "phase35_phase33_stdout.log",
                stderr_log=repeat_root / "phase35_phase33_stderr.log",
            )
        if not (bool(args.skip_existing) and phase34_endpoint.exists()):
            cmd = [
                sys.executable,
                str(phase34_script),
                "--phase32-root",
                str(args.phase32_root.resolve()),
                "--phase33-root",
                str(phase33_root),
                "--phase25-run-root",
                str(args.phase25_run_root.resolve()),
                "--out-root",
                str(phase34_root),
                "--history-interval",
                str(int(args.history_interval)),
                "--progress-interval",
                str(int(args.progress_interval)),
                "--only-case-ids",
                *[str(case_id) for case_id in case_ids],
            ]
            if bool(args.verbose_pde):
                cmd.append("--verbose-pde")
            run_logged_command(
                cmd,
                cwd=repo_root,
                stdout_log=repeat_root / "phase35_phase34_stdout.log",
                stderr_log=repeat_root / "phase35_phase34_stderr.log",
            )

    if not phase33_manifest.exists():
        raise FileNotFoundError(f"Missing Phase 33 manifest for repeat {repeat_id}: {phase33_manifest}")
    if not phase34_endpoint.exists():
        raise FileNotFoundError(f"Missing Phase 34 endpoint table for repeat {repeat_id}: {phase34_endpoint}")

    return {
        "repeat_id": repeat_id,
        "seed": int(seed),
        "history_scale": float(history_scale),
        "histories_base": int(histories_base),
        "histories_spot": int(histories_spot),
        "repeat_root": str(repeat_root),
        "phase33_root": str(phase33_root),
        "phase34_root": str(phase34_root),
    }


def attach_repeat_metadata(
    rows: Sequence[Mapping[str, object]],
    *,
    repeat_meta: Mapping[str, object],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        merged = {
            "repeat_id": str(repeat_meta["repeat_id"]),
            "seed": int(repeat_meta["seed"]),
            "history_scale": float(repeat_meta["history_scale"]),
            "histories_base": int(repeat_meta["histories_base"]),
            "histories_spot": int(repeat_meta["histories_spot"]),
            "repeat_root": str(repeat_meta["repeat_root"]),
        }
        merged.update({str(key): value for key, value in row.items()})
        out.append(merged)
    return out


def load_repeat_outputs(
    repeat_meta: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    phase33_root = Path(str(repeat_meta["phase33_root"]))
    phase34_root = Path(str(repeat_meta["phase34_root"]))

    physical_rows = [
        row
        for row in load_csv_rows(phase33_root / "phase33_physical_endpoint_table.csv")
        if str(row["component"]) == "combined_processed_scaled"
    ]
    endpoint_rows = load_csv_rows(phase34_root / "phase34_endpoint_table.csv")
    assay_rows = load_csv_rows(phase34_root / "phase34_assay_proxy_table.csv")
    rank_rows = load_csv_rows(phase34_root / "phase34_rank_shift_table.csv")
    return (
        attach_repeat_metadata(physical_rows, repeat_meta=repeat_meta),
        attach_repeat_metadata(endpoint_rows, repeat_meta=repeat_meta),
        attach_repeat_metadata(assay_rows, repeat_meta=repeat_meta),
        attach_repeat_metadata(rank_rows, repeat_meta=repeat_meta),
    )


def build_physical_repeat_bands(
    repeat_physical_rows: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    metrics = (
        ("ptv_d95_gy", "PTV D95", "Gy"),
        ("gtv_d95_gy", "GTV D95", "Gy"),
        ("peak_mean_gy", "Peak mean", "Gy"),
        ("valley_mean_gy", "Valley mean", "Gy"),
        ("pvdr", "PVDR", "ratio"),
        ("spill_shell_0_5_mean_gy", "Peri-GTV 0-5 mm mean", "Gy"),
        ("spill_shell_5_15_mean_gy", "Peri-GTV 5-15 mm mean", "Gy"),
        ("cord_d2_gy", "Cord D2", "Gy"),
        ("brainstem_d2_gy", "Brainstem D2", "Gy"),
        ("parotid_r_mean_gy", "Parotid R mean", "Gy"),
        ("thyroid_mean_gy", "Thyroid mean", "Gy"),
        ("body_dmax_gy", "Body Dmax", "Gy"),
    )
    by_case: Dict[str, List[Mapping[str, object]]] = {}
    for row in repeat_physical_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    out: List[Dict[str, object]] = []
    for case_id, rows in sorted(by_case.items()):
        case_label = str(rows[0]["case_label"])
        for key, label, units in metrics:
            values = [float(row[key]) for row in rows]
            summary = scalar_summary(values)
            ci = bootstrap_interval(values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
            out.append(
                {
                    "case_id": case_id,
                    "case_label": case_label,
                    "metric": key,
                    "label": label,
                    "units": units,
                    "n_repeats": int(len(values)),
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "p2_5": summary["p2_5"],
                    "p97_5": summary["p97_5"],
                    "mean_ci_lower": ci[0],
                    "mean_ci_upper": ci[1],
                    "coefficient_of_variation_pct": float(100.0 * summary["std"] / max(abs(summary["mean"]), 1.0e-6)),
                }
            )
    return out


def build_case_mode_endpoint_sigma_lookup(
    repeat_endpoint_rows: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str, str], float]:
    lookup: Dict[Tuple[str, str, str], float] = {}
    by_key: Dict[Tuple[str, str, str], List[float]] = {}
    for row in repeat_endpoint_rows:
        plan_id = str(row["plan_id"])
        mode = str(row["mode"])
        for endpoint in PRIMARY_ENDPOINTS:
            by_key.setdefault((plan_id, mode, endpoint.key), []).append(float(row[endpoint.key]))
    for key, values in by_key.items():
        lookup[key] = float(np.std(np.asarray(values, dtype=np.float64), ddof=0))
    return lookup


def build_endpoint_repeat_bands(
    repeat_endpoint_rows: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str, str], List[float]] = {}
    label_lookup: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for row in repeat_endpoint_rows:
        plan_id = str(row["plan_id"])
        mode = str(row["mode"])
        case_label = str(row["case_label"])
        for endpoint in PRIMARY_ENDPOINTS:
            key = (plan_id, mode, endpoint.key)
            by_key.setdefault(key, []).append(float(row[endpoint.key]))
            label_lookup[key] = (case_label, endpoint.units)
    out: List[Dict[str, object]] = []
    for (plan_id, mode, endpoint_key), values in sorted(by_key.items()):
        endpoint_spec = next(spec for spec in PRIMARY_ENDPOINTS if spec.key == endpoint_key)
        case_label, units = label_lookup[(plan_id, mode, endpoint_key)]
        summary = scalar_summary(values)
        ci = bootstrap_interval(values, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
        out.append(
            {
                "plan_id": plan_id,
                "case_label": case_label,
                "mode": mode,
                "endpoint": endpoint_key,
                "label": endpoint_spec.label,
                "units": units,
                "n_repeats": int(len(values)),
                "mean": summary["mean"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
                "p2_5": summary["p2_5"],
                "p97_5": summary["p97_5"],
                "mean_ci_lower": ci[0],
                "mean_ci_upper": ci[1],
            }
        )
    return out


def build_sink_delta_noise_rows(
    repeat_endpoint_rows: Sequence[Mapping[str, object]],
    *,
    bio_sigma_lookup: Mapping[str, float],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> List[Dict[str, object]]:
    repeat_lookup: Dict[Tuple[str, str, str], Mapping[str, object]] = {
        (str(row["repeat_id"]), str(row["plan_id"]), str(row["mode"])): row for row in repeat_endpoint_rows
    }
    plan_ids = sorted({str(row["plan_id"]) for row in repeat_endpoint_rows})
    repeat_ids = sorted({str(row["repeat_id"]) for row in repeat_endpoint_rows})
    out: List[Dict[str, object]] = []
    for plan_id in plan_ids:
        case_label = next(str(row["case_label"]) for row in repeat_endpoint_rows if str(row["plan_id"]) == plan_id)
        for endpoint in PRIMARY_ENDPOINTS:
            deltas: List[float] = []
            for repeat_id in repeat_ids:
                no_sink = repeat_lookup.get((repeat_id, plan_id, "bystander_no_sink"))
                with_sink = repeat_lookup.get((repeat_id, plan_id, "bystander_with_sink"))
                if no_sink is None or with_sink is None:
                    continue
                deltas.append(float(with_sink[endpoint.key]) - float(no_sink[endpoint.key]))
            if not deltas:
                continue
            summary = scalar_summary(deltas)
            ci = bootstrap_interval(deltas, rng=rng, samples=int(bootstrap_samples), reducer=np.mean)
            mc_sigma = float(summary["std"])
            bio_sigma = math.sqrt(2.0) * float(bio_sigma_lookup.get(endpoint.key, 0.0))
            combined_sigma = math.sqrt(mc_sigma**2 + bio_sigma**2)
            combined_band_95 = 1.96 * combined_sigma
            mean_delta = float(summary["mean"])
            out.append(
                {
                    "plan_id": plan_id,
                    "case_label": case_label,
                    "endpoint": endpoint.key,
                    "label": endpoint.label,
                    "units": endpoint.units,
                    "n_repeats": int(len(deltas)),
                    "mean_with_sink_minus_no_sink": mean_delta,
                    "std_repeat_delta": mc_sigma,
                    "min_delta": summary["min"],
                    "max_delta": summary["max"],
                    "p2_5_delta": summary["p2_5"],
                    "p97_5_delta": summary["p97_5"],
                    "mean_delta_ci_lower": ci[0],
                    "mean_delta_ci_upper": ci[1],
                    "bio_sigma_for_delta": bio_sigma,
                    "combined_sigma": combined_sigma,
                    "combined_95pct_noise_band": combined_band_95,
                    "abs_mean_over_combined_sigma": float(abs(mean_delta) / max(combined_sigma, 1.0e-9)),
                    "exceeds_1sigma_noise_band": bool(abs(mean_delta) > combined_sigma),
                    "exceeds_95pct_noise_band": bool(abs(mean_delta) > combined_band_95),
                    "mean_delta_ci_excludes_zero": bool((ci[0] > 0.0) or (ci[1] < 0.0)),
                    "cohens_dz": float(paired_cohens_dz(deltas)),
                }
            )
    return out


def build_empirical_rank_repeat_rows(
    repeat_rank_rows: Sequence[Mapping[str, object]],
    *,
    baseline_rank_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    baseline_lookup = {str(row["plan_id"]): row for row in baseline_rank_rows}
    by_plan: Dict[str, List[Mapping[str, object]]] = {}
    for row in repeat_rank_rows:
        by_plan.setdefault(str(row["plan_id"]), []).append(row)
    out: List[Dict[str, object]] = []
    for plan_id, rows in sorted(by_plan.items()):
        baseline = baseline_lookup[plan_id]
        nominal_shift = int(baseline["with_sink_vs_no_sink_shift"])
        shifts = np.asarray([int(row["with_sink_vs_no_sink_shift"]) for row in rows], dtype=np.float64)
        risk_deltas = np.asarray(
            [float(row["with_sink_risk_score"]) - float(row["no_sink_risk_score"]) for row in rows],
            dtype=np.float64,
        )
        if nominal_shift > 0:
            same_direction = float(np.mean(shifts > 0.0))
        elif nominal_shift < 0:
            same_direction = float(np.mean(shifts < 0.0))
        else:
            same_direction = float(np.mean(shifts == 0.0))
        out.append(
            {
                "plan_id": plan_id,
                "nominal_no_sink_rank": int(baseline["no_sink_rank"]),
                "nominal_with_sink_rank": int(baseline["with_sink_rank"]),
                "nominal_with_sink_vs_no_sink_shift": int(nominal_shift),
                "n_repeats": int(len(rows)),
                "mean_empirical_shift": float(np.mean(shifts)),
                "sd_empirical_shift": float(np.std(shifts, ddof=0)),
                "fraction_negative_shift": float(np.mean(shifts < 0.0)),
                "fraction_zero_shift": float(np.mean(shifts == 0.0)),
                "fraction_positive_shift": float(np.mean(shifts > 0.0)),
                "probability_same_direction_as_nominal": same_direction,
                "mean_empirical_risk_delta": float(np.mean(risk_deltas)),
                "sd_empirical_risk_delta": float(np.std(risk_deltas, ddof=0)),
            }
        )
    return out


def simulate_sink_rank_shift_robustness(
    baseline_endpoint_rows: Sequence[Mapping[str, object]],
    *,
    mc_sigma_lookup: Mapping[Tuple[str, str, str], float],
    bio_sigma_lookup: Mapping[str, float],
    rng: np.random.Generator,
    samples: int,
) -> List[Dict[str, object]]:
    nominal_rows = [{str(key): value for key, value in row.items()} for row in baseline_endpoint_rows]
    assign_ranks(nominal_rows)
    nominal_rank_lookup = {str(row["plan_id"]): row for row in build_rank_shift_rows(nominal_rows)}

    rows_by_mode: Dict[str, List[Dict[str, object]]] = {}
    for row in nominal_rows:
        rows_by_mode.setdefault(str(row["mode"]), []).append(row)

    shift_samples: Dict[str, List[float]] = {str(plan_id): [] for plan_id in nominal_rank_lookup}
    risk_delta_samples: Dict[str, List[float]] = {str(plan_id): [] for plan_id in nominal_rank_lookup}

    for _ in range(int(samples)):
        sim_rows_by_mode: Dict[str, List[Dict[str, object]]] = {}
        for mode, rows in rows_by_mode.items():
            sim_rows: List[Dict[str, object]] = []
            for row in rows:
                plan_id = str(row["plan_id"])
                sim_row: Dict[str, object] = {
                    "plan_id": plan_id,
                    "mode": mode,
                }
                for endpoint in PRIMARY_ENDPOINTS:
                    sigma = float(mc_sigma_lookup.get((plan_id, mode, endpoint.key), 0.0))
                    if mode == "bystander_with_sink":
                        sigma = math.sqrt(sigma**2 + float(bio_sigma_lookup.get(endpoint.key, 0.0)) ** 2)
                    value = float(row[endpoint.key])
                    if sigma > 0.0:
                        value = max(0.0, float(rng.normal(value, sigma)))
                    sim_row[endpoint.key] = value
                sim_rows.append(sim_row)
            risk_scores = np.zeros(len(sim_rows), dtype=np.float64)
            for endpoint in PRIMARY_ENDPOINTS:
                values = [float(sim_row[endpoint.key]) for sim_row in sim_rows]
                risk_scores += endpoint_z_scores(values, higher_is_better=endpoint.higher_is_better)
            order = np.argsort(np.argsort(risk_scores)) + 1
            for idx, sim_row in enumerate(sim_rows):
                sim_row["risk_score"] = float(risk_scores[idx])
                sim_row["rank"] = int(order[idx])
            sim_rows_by_mode[mode] = sim_rows
        no_sink_lookup = {str(row["plan_id"]): row for row in sim_rows_by_mode["bystander_no_sink"]}
        with_sink_lookup = {str(row["plan_id"]): row for row in sim_rows_by_mode["bystander_with_sink"]}
        for plan_id in shift_samples:
            no_sink = no_sink_lookup[plan_id]
            with_sink = with_sink_lookup[plan_id]
            shift_samples[plan_id].append(float(int(with_sink["rank"]) - int(no_sink["rank"])))
            risk_delta_samples[plan_id].append(float(with_sink["risk_score"]) - float(no_sink["risk_score"]))

    out: List[Dict[str, object]] = []
    for plan_id in sorted(shift_samples):
        nominal = nominal_rank_lookup[plan_id]
        shifts = np.asarray(shift_samples[plan_id], dtype=np.float64)
        risk_deltas = np.asarray(risk_delta_samples[plan_id], dtype=np.float64)
        shift_ci = (float(np.percentile(shifts, 2.5)), float(np.percentile(shifts, 97.5)))
        risk_ci = (float(np.percentile(risk_deltas, 2.5)), float(np.percentile(risk_deltas, 97.5)))
        nominal_shift = int(nominal["with_sink_vs_no_sink_shift"])
        if nominal_shift > 0:
            same_direction = float(np.mean(shifts > 0.0))
        elif nominal_shift < 0:
            same_direction = float(np.mean(shifts < 0.0))
        else:
            same_direction = float(np.mean(shifts == 0.0))
        out.append(
            {
                "plan_id": plan_id,
                "nominal_no_sink_rank": int(nominal["no_sink_rank"]),
                "nominal_with_sink_rank": int(nominal["with_sink_rank"]),
                "nominal_with_sink_vs_no_sink_shift": int(nominal_shift),
                "simulated_mean_shift": float(np.mean(shifts)),
                "simulated_shift_sd": float(np.std(shifts, ddof=0)),
                "simulated_shift_ci_lower": shift_ci[0],
                "simulated_shift_ci_upper": shift_ci[1],
                "simulated_probability_same_direction_as_nominal": same_direction,
                "simulated_probability_nonzero_shift": float(np.mean(shifts != 0.0)),
                "simulated_mean_risk_delta": float(np.mean(risk_deltas)),
                "simulated_risk_delta_sd": float(np.std(risk_deltas, ddof=0)),
                "simulated_risk_delta_ci_lower": risk_ci[0],
                "simulated_risk_delta_ci_upper": risk_ci[1],
                "noise_qualified_rank_change": bool(
                    nominal_shift != 0
                    and ((shift_ci[0] > 0.0) or (shift_ci[1] < 0.0))
                    and same_direction >= 0.80
                ),
            }
        )
    return out


def build_reproducibility_manifest(
    *,
    args: argparse.Namespace,
    repeat_meta_rows: Sequence[Mapping[str, object]],
    case_ids: Sequence[str],
) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return {
        "phase": 35,
        "description": "Repeated subset uncertainty package for the Phase 32/33/34 site-specific cohort.",
        "script": str(Path(__file__).resolve()),
        "repo_root": str(root),
        "selected_case_ids": [str(case_id) for case_id in case_ids],
        "input_roots": {
            "phase32_root": str(args.phase32_root.resolve()),
            "baseline_phase33_root": str(args.baseline_phase33_root.resolve()),
            "baseline_phase34_root": str(args.baseline_phase34_root.resolve()),
            "phase25_run_root": str(args.phase25_run_root.resolve()),
            "phase26_run_root": str(args.phase26_run_root.resolve()),
        },
        "output_root": str(args.out_root.resolve()),
        "repeat_settings": {
            "seeds": [int(v) for v in args.seeds],
            "history_scales": [float(v) for v in args.history_scales],
            "histories_base": int(args.histories_base),
            "histories_spot": int(args.histories_spot),
            "threads": int(args.threads),
            "dose_smoothing_mm": float(args.dose_smoothing_mm),
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
        },
        "rank_simulation_samples": int(args.rank_sim_samples),
        "repeat_runs": [
            {
                "repeat_id": str(row["repeat_id"]),
                "seed": int(row["seed"]),
                "history_scale": float(row["history_scale"]),
                "histories_base": int(row["histories_base"]),
                "histories_spot": int(row["histories_spot"]),
                "phase33_root": str(row["phase33_root"]),
                "phase34_root": str(row["phase34_root"]),
            }
            for row in repeat_meta_rows
        ],
        "script_dependencies": [
            str(root / "scripts" / "run_phase33_phase32_topas_cohort.py"),
            str(root / "scripts" / "run_phase34_phase32_bio_cohort.py"),
            str(root / "scripts" / "run_phase35_subset_repeat_uncertainty.py"),
        ],
    }


def write_quick_assessment(
    out_file: Path,
    *,
    case_ids: Sequence[str],
    repeat_meta_rows: Sequence[Mapping[str, object]],
    sink_delta_rows: Sequence[Mapping[str, object]],
    empirical_rank_rows: Sequence[Mapping[str, object]],
    simulated_rank_rows: Sequence[Mapping[str, object]],
) -> None:
    endpoint_hits = sum(1 for row in sink_delta_rows if bool(row["exceeds_95pct_noise_band"]))
    robust_rank_hits = sum(1 for row in simulated_rank_rows if bool(row["noise_qualified_rank_change"]))
    lines = [
        "# Phase 35 quick assessment",
        "",
        f"- Repeated subset cases: `{', '.join(str(case_id) for case_id in case_ids)}`.",
        f"- Repeat combinations executed: `{len(repeat_meta_rows)}`.",
        f"- Case-endpoint sink deltas exceeding the 95% combined MC+uptake band: `{endpoint_hits}` / `{len(sink_delta_rows)}`.",
        f"- Cases with noise-qualified sink-driven rank changes: `{robust_rank_hits}` / `{len(simulated_rank_rows)}`.",
    ]
    if empirical_rank_rows:
        mean_direction = float(np.mean([float(row["probability_same_direction_as_nominal"]) for row in empirical_rank_rows]))
        lines.append(f"- Mean empirical probability of preserving the nominal sink-shift direction: `{mean_direction:.3f}`.")
    lines.append(
        "- Interpretation: this package is intended to test whether sink-driven reinterpretation remains visible after repeated physical transport and uptake-sensitivity uncertainty are both considered."
    )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))
    case_ids = [str(case_id) for case_id in args.only_case_ids]

    repeat_meta_rows: List[Dict[str, object]] = []
    repeat_physical_rows: List[Dict[str, object]] = []
    repeat_endpoint_rows: List[Dict[str, object]] = []
    repeat_assay_rows: List[Dict[str, object]] = []
    repeat_rank_rows: List[Dict[str, object]] = []

    for seed in [int(v) for v in args.seeds]:
        for scale in [float(v) for v in args.history_scales]:
            repeat_meta = run_repeat_combination(
                args,
                repo_root=repo_root,
                case_ids=case_ids,
                seed=seed,
                history_scale=scale,
            )
            repeat_meta_rows.append(repeat_meta)
            physical_rows, endpoint_rows, assay_rows, rank_rows = load_repeat_outputs(repeat_meta)
            repeat_physical_rows.extend(physical_rows)
            repeat_endpoint_rows.extend(endpoint_rows)
            repeat_assay_rows.extend(assay_rows)
            repeat_rank_rows.extend(rank_rows)

    baseline_endpoint_rows = [
        row
        for row in load_csv_rows(args.baseline_phase34_root / "phase34_endpoint_table.csv")
        if str(row["plan_id"]) in set(case_ids)
    ]
    baseline_rank_input_rows = [
        row
        for row in load_csv_rows(args.baseline_phase34_root / "phase34_rank_shift_table.csv")
        if str(row["plan_id"]) in set(case_ids)
    ]
    baseline_subset_rank_rows = [{str(key): value for key, value in row.items()} for row in baseline_endpoint_rows]
    assign_ranks(baseline_subset_rank_rows)
    baseline_rank_rows = build_rank_shift_rows(baseline_subset_rank_rows)

    write_csv(out_root / "phase35_repeat_manifest.csv", repeat_meta_rows)
    write_csv(out_root / "phase35_repeat_physical_rows.csv", repeat_physical_rows)
    write_csv(out_root / "phase35_repeat_endpoint_rows.csv", repeat_endpoint_rows)
    write_csv(out_root / "phase35_repeat_assay_rows.csv", repeat_assay_rows)
    write_csv(out_root / "phase35_repeat_rank_rows.csv", repeat_rank_rows)

    physical_bands = build_physical_repeat_bands(
        repeat_physical_rows,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    write_csv(out_root / "phase35_physical_repeat_bands.csv", physical_bands)

    endpoint_bands = build_endpoint_repeat_bands(
        repeat_endpoint_rows,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    write_csv(out_root / "phase35_endpoint_repeat_bands.csv", endpoint_bands)

    bio_sigma_lookup = build_biology_sigma_lookup(args.phase26_run_root.resolve())
    sink_delta_rows = build_sink_delta_noise_rows(
        repeat_endpoint_rows,
        bio_sigma_lookup=bio_sigma_lookup,
        rng=rng,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    write_csv(out_root / "phase35_sink_delta_noise_table.csv", sink_delta_rows)

    empirical_rank_rows = build_empirical_rank_repeat_rows(
        repeat_rank_rows,
        baseline_rank_rows=baseline_rank_rows,
    )
    write_csv(out_root / "phase35_empirical_rank_repeat_summary.csv", empirical_rank_rows)
    write_csv(out_root / "phase35_baseline_subset_rank_shifts.csv", baseline_rank_rows)
    write_csv(out_root / "phase35_baseline_fullcohort_rank_shifts_reference.csv", baseline_rank_input_rows)

    mc_sigma_lookup = build_case_mode_endpoint_sigma_lookup(repeat_endpoint_rows)
    simulated_rank_rows = simulate_sink_rank_shift_robustness(
        baseline_endpoint_rows,
        mc_sigma_lookup=mc_sigma_lookup,
        bio_sigma_lookup=bio_sigma_lookup,
        rng=rng,
        samples=int(args.rank_sim_samples),
    )
    write_csv(out_root / "phase35_sink_rank_noise_assessment.csv", simulated_rank_rows)

    manifest = build_reproducibility_manifest(
        args=args,
        repeat_meta_rows=repeat_meta_rows,
        case_ids=case_ids,
    )
    write_json(out_root / "phase35_reproducibility_manifest.json", manifest)

    write_quick_assessment(
        out_root / "phase35_quick_assessment.md",
        case_ids=case_ids,
        repeat_meta_rows=repeat_meta_rows,
        sink_delta_rows=sink_delta_rows,
        empirical_rank_rows=empirical_rank_rows,
        simulated_rank_rows=simulated_rank_rows,
    )

    print("=== PHASE 35 SUBSET REPEAT UNCERTAINTY COMPLETE ===")
    print(f"Output root: {out_root}")
    print(f"Repeat manifest: {out_root / 'phase35_repeat_manifest.csv'}")
    print(f"Physical repeat bands: {out_root / 'phase35_physical_repeat_bands.csv'}")
    print(f"Endpoint repeat bands: {out_root / 'phase35_endpoint_repeat_bands.csv'}")
    print(f"Sink delta noise table: {out_root / 'phase35_sink_delta_noise_table.csv'}")
    print(f"Sink rank noise assessment: {out_root / 'phase35_sink_rank_noise_assessment.csv'}")
    print(f"Quick assessment: {out_root / 'phase35_quick_assessment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
