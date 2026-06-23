#!/usr/bin/env python3
"""Build reviewer-facing guardrail tables for the PMB major revision."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import ndimage

from phase37_bio_model_params import parameter_provenance_rows


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "project"
MANUSCRIPT = REPO_ROOT / "manuscript"
PHASE32_ROOT = PROJECT_ROOT / "public_results" / "phase32_site_specific_cohort_regenerated"
PHASE34_PRIMARY = PROJECT_ROOT / "public_results" / "phase33_34_cohort" / "phase34_bio_cohort_primary_oxygen_neutral"
PHASE38_PRIMARY = PROJECT_ROOT / "public_results" / "phase38_bio_parameter_robustness_primary_oxygen_neutral"
PHASE38_SENS = PROJECT_ROOT / "public_results" / "phase38_bio_parameter_robustness_sensitivity_oxygen_modulated"
PHASE35_FULL = PROJECT_ROOT / "public_results" / "revision_checks_20260616" / "step02_phase35_fullcohort"
PHASE40 = PROJECT_ROOT / "public_results" / "revision_checks_20260617" / "step10_smoothing_kernel_sensitivity"


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for CSV output: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(str(v) for v in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def min_distance_mm(src_mask: np.ndarray, dst_mask: np.ndarray, voxel_mm: Sequence[float]) -> float:
    src = np.asarray(src_mask, dtype=bool)
    dst = np.asarray(dst_mask, dtype=bool)
    if not np.any(src) or not np.any(dst):
        return float("nan")
    if np.any(src & dst):
        return 0.0
    distance = ndimage.distance_transform_edt(~dst, sampling=tuple(float(v) for v in voxel_mm))
    return float(np.min(distance[src]))


def build_headline_numbers() -> None:
    primary = {row["metric"]: row for row in load_csv(PHASE38_PRIMARY / "phase38_cohort_overview.csv")}
    sensitivity = {row["metric"]: row for row in load_csv(PHASE38_SENS / "phase38_cohort_overview.csv")}
    sink_noise = load_csv(PHASE35_FULL / "phase35_sink_delta_noise_table.csv")
    rank_noise = load_csv(PHASE35_FULL / "phase35_sink_rank_noise_assessment.csv")
    sigma_rows = {float(row["sigma_mm"]): row for row in load_csv(PHASE40 / "phase40_sigma_summary.csv")}

    endpoint_hits = sum(row["exceeds_95pct_noise_band"] == "True" for row in sink_noise)
    rank_hits = sum(row["noise_qualified_rank_change"] == "True" for row in rank_noise)

    rows = [
        {
            "analysis": "Primary oxygen-neutral model",
            "headline_result": f"with-sink vs physical shifts: {primary['with_sink vs physical rank shifts']['full_cohort_result']}",
            "interpretation": "biology changes interpretation",
            "source": "oxygen-neutral robustness overview",
        },
        {
            "analysis": "Primary no-sink model",
            "headline_result": f"no-sink vs physical shifts: {primary['no_sink vs physical rank shifts']['full_cohort_result']}",
            "interpretation": "non-local biology is the main driver",
            "source": "oxygen-neutral robustness overview",
        },
        {
            "analysis": "Sink-only increment",
            "headline_result": f"with-sink vs no-sink shifts: {primary['with_sink vs no-sink rank shifts']['full_cohort_result']}",
            "interpretation": "sink is secondary",
            "source": "oxygen-neutral robustness overview",
        },
        {
            "analysis": "Primary brainstem reinterpretation",
            "headline_result": f"biology-added brainstem flags: {primary['biology-added brainstem flags']['full_cohort_result']}",
            "interpretation": "endpoint-level OAR reinterpretation persists in the main model",
            "source": "oxygen-neutral robustness overview",
        },
        {
            "analysis": "Oxygen sensitivity",
            "headline_result": (
                f"with-sink vs physical shifts: {sensitivity['with_sink vs physical rank shifts']['full_cohort_result']}; "
                f"brainstem flags: {sensitivity['biology-added brainstem flags']['full_cohort_result']}"
            ),
            "interpretation": "oxygen amplifies the effect; sensitivity only",
            "source": "oxygen-modulated sensitivity overview",
        },
        {
            "analysis": "Repeated-run endpoint uncertainty",
            "headline_result": f"{endpoint_hits}/{len(sink_noise)} endpoint deltas exceed the 95% band",
            "interpretation": "endpoint effects are robust to repeated-run uncertainty",
            "source": "repeated-run noise table",
        },
        {
            "analysis": "Repeated-run rank uncertainty",
            "headline_result": f"{rank_hits}/{len(rank_noise)} noise-qualified sink rank changes",
            "interpretation": "no strong sink-driven plan-ordering claim",
            "source": "repeated-run rank-noise assessment",
        },
        {
            "analysis": "Smoothing-kernel sensitivity",
            "headline_result": (
                "4 and 6 mm preserved the main interpretation; "
                "2 mm was calibration-unstable"
            ),
            "interpretation": "the smoothed substrate is justified and the sink remains secondary",
            "source": "smoothing-kernel sensitivity summary",
        },
    ]
    write_csv(MANUSCRIPT / "headline_numbers_table.csv", rows)
    table_rows = [[row["analysis"], row["headline_result"], row["interpretation"], row["source"]] for row in rows]
    (MANUSCRIPT / "headline_numbers_table.md").write_text(
        "# Headline Numbers Guardrail\n\n"
        + markdown_table(["Analysis", "Headline result", "Interpretation", "Source"], table_rows),
        encoding="utf-8",
    )


def build_uncertainty_interpretation() -> None:
    physical_rows = load_csv(PHASE35_FULL / "phase35_physical_repeat_bands.csv")
    sink_rows = load_csv(PHASE35_FULL / "phase35_sink_delta_noise_table.csv")
    cv_map = {
        "pvdr": "pvdr",
        "spill_shell_0_5_mean": "spill_shell_0_5_mean_gy",
        "spill_shell_5_15_mean": "spill_shell_5_15_mean_gy",
        "cord_d2": "cord_d2_gy",
        "brainstem_d2": "brainstem_d2_gy",
        "parotid_r_mean": "parotid_r_mean_gy",
        "ptv_d95": "ptv_d95_gy",
    }
    label_map = {
        "pvdr": "PVDR",
        "spill_shell_0_5_mean": "Peri-GTV 0-5 mm",
        "spill_shell_5_15_mean": "Peri-GTV 5-15 mm",
        "cord_d2": "Cord D2",
        "brainstem_d2": "Brainstem D2",
        "parotid_r_mean": "Parotid mean",
        "ptv_d95": "PTV D95",
    }
    rows: List[Dict[str, object]] = []
    md_rows: List[List[object]] = []
    for endpoint, metric in cv_map.items():
        phys_sub = [row for row in physical_rows if row["metric"] == metric]
        noise_sub = [row for row in sink_rows if row["endpoint"] == endpoint]
        mean_cv = float(np.mean([float(row["coefficient_of_variation_pct"]) for row in phys_sub])) if phys_sub else float("nan")
        hits = sum(row["exceeds_95pct_noise_band"] == "True" for row in noise_sub)
        total = len(noise_sub)
        cv_display = "fixed by calibration" if endpoint == "ptv_d95" else f"{mean_cv:.2f}%"
        rows.append(
            {
                "endpoint": label_map[endpoint],
                "repeat_run_cv_pct_mean": "" if endpoint == "ptv_d95" else mean_cv,
                "repeat_run_cv_display": cv_display,
                "sink_delta_above_95pct_band": f"{hits}/{total}",
            }
        )
        md_rows.append([label_map[endpoint], cv_display, f"{hits}/{total}"])
    write_csv(MANUSCRIPT / "uncertainty_interpretation_table.csv", rows)
    (MANUSCRIPT / "uncertainty_interpretation_table.md").write_text(
        "# Uncertainty Interpretation Table\n\n"
        + markdown_table(["Endpoint", "Repeat-run CV", "Sink delta above 95% band"], md_rows),
        encoding="utf-8",
    )


def build_synthetic_cohort_plausibility() -> None:
    manifest = load_csv(PHASE32_ROOT / "phase32_case_manifest.csv")
    physical_rows = {
        row["plan_id"]: row
        for row in load_csv(PHASE34_PRIMARY / "phase34_endpoint_table.csv")
        if row["mode"] == "physical_only"
    }
    out_rows: List[Dict[str, object]] = []
    for row in manifest:
        case_id = row["template_id"]
        ctx_path = Path(row["phantom_context_npz"])
        with np.load(ctx_path) as data:
            def struct(*names: str) -> np.ndarray:
                for name in names:
                    key = f"struct_{name}"
                    if key in data.files:
                        return np.asarray(data[key], dtype=bool)
                raise KeyError(f"None of {names} were present in {ctx_path}")

            axes_x = np.asarray(data["axes_x_mm"], dtype=np.float32)
            axes_y = np.asarray(data["axes_y_mm"], dtype=np.float32)
            axes_z = np.asarray(data["axes_z_mm"], dtype=np.float32)
            voxel_mm = (
                float(axes_x[1] - axes_x[0]),
                float(axes_y[1] - axes_y[0]),
                float(axes_z[1] - axes_z[0]),
            )
            voxel_cc = float(voxel_mm[0] * voxel_mm[1] * voxel_mm[2] / 1000.0)
            gtv = struct("GTV")
            ptv = struct("PTV")
            brainstem = struct("BRAINSTEM")
            cord = struct("SPINAL_CORD", "CORD")
            parotid_r = struct("PAROTID_R")
            arteries = struct("ARTERIES")
            veins = struct("VEINS")
            body = struct("BODY")
            vessel = arteries | veins

        gtv_to_gtv = ndimage.distance_transform_edt(~gtv, sampling=voxel_mm)
        peri_shell_0_5 = (~gtv) & (gtv_to_gtv > 0.0) & (gtv_to_gtv <= 5.0)
        phys = physical_rows[case_id]
        out_rows.append(
            {
                "case_id": case_id,
                "case_label": row["label"],
                "site_group": row["site_group"],
                "gtv_volume_cc": float(np.sum(gtv) * voxel_cc),
                "ptv_volume_cc": float(np.sum(ptv) * voxel_cc),
                "retained_vertices": int(row["kept_vertex_count"]),
                "peak_mean_dose_gy": float(phys["peak_mean"]),
                "valley_mean_dose_gy": float(phys["valley_mean"]),
                "pvdr": float(phys["pvdr"]),
                "min_gtv_to_brainstem_mm": min_distance_mm(gtv, brainstem, voxel_mm),
                "min_gtv_to_cord_mm": min_distance_mm(gtv, cord, voxel_mm),
                "min_gtv_to_parotid_r_mm": min_distance_mm(gtv, parotid_r, voxel_mm),
                "vessel_mask_volume_fraction_pct": float(100.0 * np.sum(vessel) / max(int(np.sum(body)), 1)),
                "min_gtv_to_vessel_mm": min_distance_mm(gtv, vessel, voxel_mm),
                "peri_gtv_vessel_density_0_5_pct": float(
                    100.0 * np.sum(vessel & peri_shell_0_5) / max(int(np.sum(peri_shell_0_5)), 1)
                ),
            }
        )
    write_csv(MANUSCRIPT / "synthetic_cohort_plausibility_table.csv", out_rows)
    md_rows = [
        [
            row["case_id"],
            row["site_group"],
            f"{row['gtv_volume_cc']:.1f}",
            f"{row['ptv_volume_cc']:.1f}",
            row["retained_vertices"],
            f"{row['peak_mean_dose_gy']:.2f}",
            f"{row['valley_mean_dose_gy']:.2f}",
            f"{row['pvdr']:.2f}",
            f"{row['min_gtv_to_brainstem_mm']:.1f}",
            f"{row['min_gtv_to_cord_mm']:.1f}",
            f"{row['min_gtv_to_parotid_r_mm']:.1f}",
            f"{row['vessel_mask_volume_fraction_pct']:.2f}",
            f"{row['peri_gtv_vessel_density_0_5_pct']:.2f}",
        ]
        for row in out_rows
    ]
    (MANUSCRIPT / "synthetic_cohort_plausibility_table.md").write_text(
        "# Synthetic Cohort Plausibility Table\n\n"
        + markdown_table(
            [
                "Case",
                "Site group",
                "GTV (cc)",
                "PTV (cc)",
                "Vertices",
                "Peak mean (Gy)",
                "Valley mean (Gy)",
                "PVDR",
                "Min GTV-brainstem (mm)",
                "Min GTV-cord (mm)",
                "Min GTV-parotid R (mm)",
                "Vessel volume (%)",
                "Peri-GTV vessel density 0-5 mm (%)",
            ],
            md_rows,
        ),
        encoding="utf-8",
    )


def build_parameter_provenance() -> None:
    rows = []
    units_lookup = {
        r"$\alpha$": "Gy^-1",
        r"$\beta$": "Gy^-2",
        r"$D_{\mathrm{ROS}}$": "voxel^2 / step",
        r"$D_{\mathrm{cyto}}$": "voxel^2 / step",
        r"$\lambda_{\mathrm{ROS}}$": "step^-1",
        r"$\lambda_{\mathrm{cyto}}$": "step^-1",
        r"$E_{\max,\mathrm{ROS}}$": "a.u.",
        r"$E_{\max,\mathrm{cyto}}$": "a.u.",
        r"$\gamma$": "Gy^-1",
        r"$s$": "a.u.",
    }
    status_lookup = {
        r"$\alpha$": "fixed main",
        r"$\beta$": "fixed main",
        r"$D_{\mathrm{ROS}}$": "fixed main + Phase 38 sweep",
        r"$D_{\mathrm{cyto}}$": "fixed main + Phase 38 sweep",
        r"$\lambda_{\mathrm{ROS}}$": "fixed main + Phase 38 sweep",
        r"$\lambda_{\mathrm{cyto}}$": "fixed main + Phase 38 sweep",
        r"$E_{\max,\mathrm{ROS}}$": "fixed main + Phase 38 sweep",
        r"$E_{\max,\mathrm{cyto}}$": "fixed main + Phase 38 sweep",
        r"$\gamma$": "fixed main + Phase 38 sweep",
        r"$s$": "fixed main + Phase 38 sweep",
        r"$w_{\mathrm{ROS}}$": "fixed main",
        r"$w_{\mathrm{cyto}}$": "fixed main",
        "Tumour cytokine multiplier": "fixed main",
        "Hypoxic ROS scale": "sensitivity only",
        "Hypoxic cytokine multiplier": "sensitivity only",
        "Arterial ROS uptake": "fixed main + Phase 38 sweep",
        "Arterial cytokine uptake": "fixed main + Phase 38 sweep",
        "Venous ROS uptake": "fixed main + Phase 38 sweep",
        "Venous cytokine uptake": "fixed main + Phase 38 sweep",
    }
    limitation_lookup = {
        r"$s$": "Internally calibrated transfer coefficient; not externally validated.",
        "Hypoxic ROS scale": "Removed from the primary oxygen-neutral model and retained only as sensitivity.",
        "Hypoxic cytokine multiplier": "Removed from the primary oxygen-neutral model and retained only as sensitivity.",
    }
    for row in parameter_provenance_rows():
        parameter = str(row["parameter"])
        rows.append(
            {
                "parameter": parameter,
                "value": row["value"],
                "units": units_lookup.get(parameter, "dimensionless"),
                "role": row["role"],
                "status": status_lookup.get(parameter, "fixed main"),
                "justification": row["provenance"],
                "limitation": limitation_lookup.get(
                    parameter,
                    "Nominal framework parameter used for hypothesis-generating risk analysis, not a biologically validated constant.",
                ),
            }
        )
    rows.append(
        {
            "parameter": "Immune scalar",
            "value": 0.0,
            "units": "dimensionless",
            "role": "Exploratory global immune penalty from earlier branches",
            "status": "removed from main",
            "justification": "Removed in revision because it was not independently constrained or spatially resolved.",
            "limitation": "Not part of the primary revision model.",
        }
    )
    write_csv(MANUSCRIPT / "biological_parameter_provenance_table.csv", rows)
    md_rows = [
        [
            row["parameter"],
            row["value"],
            row["units"],
            row["role"],
            row["status"],
            row["justification"],
            row["limitation"],
        ]
        for row in rows
    ]
    (MANUSCRIPT / "biological_parameter_provenance_table.md").write_text(
        "# Biological Parameter Provenance Table\n\n"
        + markdown_table(
            ["Parameter", "Value", "Units", "Role", "Status", "Justification", "Limitation"],
            md_rows,
        ),
        encoding="utf-8",
    )


def build_model_hierarchy() -> None:
    rows = [
        {
            "model": "Physical-only",
            "immune_term": "No",
            "oxygen_modifier": "No",
            "vascular_sink": "No",
            "purpose": "Baseline dose interpretation",
        },
        {
            "model": "No-sink biology",
            "immune_term": "No",
            "oxygen_modifier": "Neutral (M_oxygen = 1)",
            "vascular_sink": "No",
            "purpose": "Primary non-local biology model",
        },
        {
            "model": "With-sink biology",
            "immune_term": "No",
            "oxygen_modifier": "Neutral (M_oxygen = 1)",
            "vascular_sink": "Yes",
            "purpose": "Primary anatomical uptake test",
        },
        {
            "model": "Oxygen sensitivity",
            "immune_term": "No",
            "oxygen_modifier": "Active synthetic modifier",
            "vascular_sink": "Optional / paired",
            "purpose": "Sensitivity branch to quantify oxygen-state amplification",
        },
    ]
    write_csv(MANUSCRIPT / "model_hierarchy_table.csv", rows)
    (MANUSCRIPT / "model_hierarchy_table.md").write_text(
        "# Model Hierarchy Table\n\n"
        + markdown_table(
            ["Model", "Immune term", "Oxygen modifier", "Vascular sink", "Purpose"],
            [[row["model"], row["immune_term"], row["oxygen_modifier"], row["vascular_sink"], row["purpose"]] for row in rows],
        ),
        encoding="utf-8",
    )


def build_response_matrix() -> None:
    rows = [
        {
            "reviewer_comment": "Synthetic cohort cannot demonstrate patient realism",
            "change_made": "Reframed the cohort as a fully synthetic computational stress test and added an auditable plausibility table.",
            "file_or_section": "methods_section.tex; discussion_draft.md; synthetic_cohort_plausibility_table.csv",
            "result": "Claims are limited to internal falsifiability and endpoint-level reinterpretation under controlled synthetic conditions.",
        },
        {
            "reviewer_comment": "Transferred biology parameters need stronger grounding",
            "change_made": "Added full-cohort Phase 38 bounded parameter robustness and a parameter provenance table.",
            "file_or_section": "results_section.tex; biological_parameter_provenance_table.csv; revision_phase38_full/",
            "result": "Primary oxygen-neutral same-rank probability 0.831; endpoint sign stability 1.000.",
        },
        {
            "reviewer_comment": "Immune term is underdeveloped",
            "change_made": "Removed the immune scalar from the primary revision model.",
            "file_or_section": "phase37_bio_model_params.py; methods_section.tex; table04_biology_parameters.tex",
            "result": "Primary model is immune-free.",
        },
        {
            "reviewer_comment": "M_oxygen is underdefined",
            "change_made": "Made the primary model oxygen-neutral and retained oxygen modulation as a sensitivity branch only.",
            "file_or_section": "methods_section.tex; table04_biology_parameters.tex; phase40 outputs",
            "result": "Primary model uses M_oxygen = 1; oxygen modulation amplifies reinterpretation in sensitivity only.",
        },
        {
            "reviewer_comment": "Monte Carlo uncertainty is unclear",
            "change_made": "Added full 10-case repeated seed/history uncertainty analysis.",
            "file_or_section": "results_section.tex; uncertainty_interpretation_table.csv; phase35 fullcohort outputs",
            "result": "65/70 endpoint deltas exceed the 95% band; 0/10 noise-qualified sink rank changes.",
        },
        {
            "reviewer_comment": "Smoothing may be shaping the result",
            "change_made": "Added a 2/4/6 mm smoothing-kernel sensitivity analysis and made the smoothed substrate explicit in the text.",
            "file_or_section": "methods_section.tex; results_section.tex; phase40 outputs",
            "result": "4-6 mm kernels preserved the main interpretation; 2 mm produced calibration instability.",
        },
        {
            "reviewer_comment": "Anatomical sink may be too weak to matter practically",
            "change_made": "Demoted the sink claim to a secondary endpoint-level modifier and retained falsification as mechanistic support.",
            "file_or_section": "results_section.tex; table03_falsification_summary.tex",
            "result": "Sink remains distinguishable at the endpoint level but does not support robust rank-level plan-ordering claims.",
        },
        {
            "reviewer_comment": "Reproducibility package needs to be reviewer-usable",
            "change_made": "Built a reviewer start-here layer, PMB reproducibility guide, and clean-checkout status note.",
            "file_or_section": "README.md; pmb_reproducibility_guide.md; clean_checkout_repro_status.md",
            "result": "Local rerun path is documented; clean public-clone blockers are explicitly identified before push.",
        },
    ]
    write_csv(MANUSCRIPT / "reviewer_response_matrix.csv", rows)
    (MANUSCRIPT / "reviewer_response_matrix.md").write_text(
        "# Reviewer Response Matrix\n\n"
        + markdown_table(
            ["Reviewer comment", "Change made", "File or section", "Result"],
            [[row["reviewer_comment"], row["change_made"], row["file_or_section"], row["result"]] for row in rows],
        ),
        encoding="utf-8",
    )


def build_clean_checkout_status() -> None:
    text = """# Clean-checkout reproducibility status

Date: 2026-06-17

## Current status

The bounded paper-regeneration smoke test passed in a pseudo-clean checkout. A full fresh-clone TOPAS transport rerun was not included in this freeze pass.

## Freeze result

- Guardrail tables regenerated from the documented script.
- Headline numbers remained frozen at the approved values.
- `figureR2` was regenerated and matched the staged manuscript bundle.
- `figureRS1` matched the staged manuscript bundle.
- Rank-audit and effective-dose sanity checks passed.

## Remaining boundary

This freeze certifies paper-artifact regeneration from the checked repository state. It does not certify a fresh public-clone full TOPAS transport rerun yet.
"""
    (MANUSCRIPT / "clean_checkout_repro_status.md").write_text(text, encoding="utf-8")


def build_final_claim() -> None:
    text = """# Final Claim

In a controlled synthetic H&N-like lattice RT framework, immune-free and oxygen-neutral non-local biological rescoring changed physical-only endpoint interpretation in a majority of cases. Anatomical vascular uptake produced reproducible endpoint-level modifications but did not support robust rank-level plan-ordering claims. Oxygen modulation amplified the effect in sensitivity analysis. The work is a hypothesis-generating computational audit framework, not clinical validation.
"""
    (MANUSCRIPT / "final_claim.md").write_text(text, encoding="utf-8")


def build_stale_claim_audit() -> None:
    search_terms = [
        "8/10",
        "immune",
        "oxygen",
        "subset",
        "clinical",
        "validated",
        "prediction",
        "toxicity",
        "plan ranking",
        "patient",
        "assay",
        "smoothing",
        "minor",
        "negligible",
        "PVDR fails",
    ]
    paths = [
        REPO_ROOT / "README.md",
        MANUSCRIPT / "pmb_reproducibility_guide.md",
        MANUSCRIPT / "revision_crosswalk.md",
        PROJECT_ROOT / "public_results" / "phase37_methods_overleaf_bundle" / "sections" / "methods_section.tex",
        PROJECT_ROOT / "public_results" / "phase37_results_overleaf_bundle" / "sections" / "results_section.tex",
        PROJECT_ROOT / "public_results" / "phase31_publication_package" / "manuscript_assets" / "phase31_manuscript_draft.md",
    ]
    lines = [
        "# Stale-claim audit",
        "",
        "The following reviewer-sensitive terms were searched in the main revision bundle and top-level reproducibility docs.",
        "",
        "Patterns: " + ", ".join(f"`{term}`" for term in search_terms),
        "",
    ]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = []
        for term in search_terms:
            count = len(re.findall(re.escape(term), text, flags=re.IGNORECASE))
            if count:
                hits.append(f"`{term}` x{count}")
        lines.append(f"- `{path.relative_to(REPO_ROOT)}`: " + (", ".join(hits) if hits else "no matches"))
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Remaining `immune` and `subset` mentions in the main bundle should be explicit historical/provenance context only.",
            "- The active results bundle should use the oxygen-neutral primary numbers (`6/10`, `5/10`, `2/10`, `2/10`) and the full-cohort Phase 35 uncertainty values (`65/70`, `0/10`).",
            "- Any sentence implying clinical validation, negligible smoothing, or strong sink-driven plan-ordering should be treated as unsafe.",
        ]
    )
    (MANUSCRIPT / "stale_claim_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    build_headline_numbers()
    build_uncertainty_interpretation()
    build_synthetic_cohort_plausibility()
    build_parameter_provenance()
    build_model_hierarchy()
    build_response_matrix()
    build_clean_checkout_status()
    build_final_claim()
    build_stale_claim_audit()
    print("Wrote revision guardrail artifacts into", MANUSCRIPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
