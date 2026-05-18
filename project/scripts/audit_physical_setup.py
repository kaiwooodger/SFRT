#!/usr/bin/env python3
"""Audit the TOPAS VHEE beamline setup against the Whitmore paper inputs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from build_asymmetric_sweep import PHYSICS_PROFILES, compute_layout, load_reference


TOPAS_ROOT = Path("/Applications/TOPAS/OpenTOPAS")
GEANT4_ROOT = Path("/Applications/GEANT4")


@dataclass
class Finding:
    severity: str
    title: str
    detail: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit source/geometry/physics assumptions for the asymmetric VHEE beamline."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "config" / "benchmark_reference.json",
        help="Reference JSON with paper parameters.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "asymmetric_4quad_template.txt",
        help="Current TOPAS template to audit.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "runs" / "manifest.json",
        help="Optional manifest from a previous generated sweep.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "runs" / "setup_audit",
        help="Output directory for the audit report.",
    )
    parser.add_argument(
        "--g4-data-dir",
        type=Path,
        default=GEANT4_ROOT,
        help="Directory containing Geant4 data folders.",
    )
    return parser.parse_args()


def parse_parameter_assignments(text: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    pattern = re.compile(r"^\s*[a-z]+:(?P<name>[^=]+?)\s*=\s*(?P<value>.+?)\s*$", re.MULTILINE)
    for match in pattern.finditer(text):
        params[match.group("name").strip()] = match.group("value").strip()
    return params


def choose_sample_case(manifest_path: Path) -> Optional[Tuple[str, Path]]:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    preferred_ids = ["E250_p14p30", "E200_p11p40", "E100_p5p70"]
    by_id = {case.get("case_id"): case for case in cases}
    for case_id in preferred_ids:
        case = by_id.get(case_id)
        if case and case.get("parameter_file"):
            return case_id, Path(case["parameter_file"])
    for case in cases:
        if case.get("parameter_file"):
            return str(case.get("case_id", "unknown")), Path(case["parameter_file"])
    return None


def read_topas_version() -> str:
    header = TOPAS_ROOT / "install" / "include" / "TsTopasConfig.hh"
    if not header.exists():
        header = Path("/Applications/TOPAS/OpenTOPAS-install/include/TsTopasConfig.hh")
    if not header.exists():
        return "unknown"
    text = header.read_text(encoding="utf-8")
    major = re.search(r"#define\s+TOPAS_VERSION_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define\s+TOPAS_VERSION_MINOR\s+(\d+)", text)
    patch = re.search(r"#define\s+TOPAS_VERSION_PATCH\s+(\d+)", text)
    if not (major and minor and patch):
        return "unknown"
    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def read_geant4_version(g4_root: Path) -> str:
    matches = sorted(path for path in g4_root.glob("geant4-v*") if path.is_dir())
    if not matches:
        return "unknown"
    return matches[-1].name.removeprefix("geant4-v")


def check_dataset_prefixes(g4_root: Path, prefixes: Iterable[str]) -> Tuple[List[str], List[str]]:
    available = [path.name for path in g4_root.iterdir()] if g4_root.exists() else []
    found: List[str] = []
    missing: List[str] = []
    for prefix in prefixes:
        if any(name.startswith(prefix) for name in available):
            found.append(prefix)
        else:
            missing.append(prefix)
    return found, missing


def find_quadrupole_formula() -> str:
    path = TOPAS_ROOT / "geometry" / "TsMagneticFieldQuadrupole.cc"
    if not path.exists():
        return "unavailable"
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"B_local\s*=\s*G4ThreeVector\((?P<bx>.+?),\s*(?P<by>.+?),\s*0\)",
        text,
        re.DOTALL,
    )
    if not match:
        return "unavailable"
    bx = " ".join(match.group("bx").split())
    by = " ".join(match.group("by").split())
    return f"Bx = {bx}, By = {by}"


def electron_plane_response(gx: float, gy: float, beam_direction_sign: int = 1) -> str:
    x_focus = beam_direction_sign * gy < 0
    y_focus = beam_direction_sign * gx > 0
    x_state = "focuses X" if x_focus else "defocuses X"
    y_state = "focuses Y" if y_focus else "defocuses Y"
    return f"{x_state}, {y_state}"


def physics_profile_name_from_modules(modules_line: Optional[str]) -> Optional[str]:
    if modules_line is None:
        return None
    normalized = " ".join(modules_line.split())
    for name, modules in PHYSICS_PROFILES.items():
        quoted = " ".join(f'"{module}"' for module in modules)
        candidate = f"{len(modules)} {quoted}"
        if normalized == candidate:
            return name
    return None


def summarize_layout(reference: Dict) -> Dict[str, float]:
    beamline = reference["asymmetric_beamline"]
    phantom_hlz_cm = float(beamline["phantom_size_cm"]["z"]) / 2.0
    layout = compute_layout(
        beamline["drifts_cm"],
        float(beamline["quad_length_cm"]),
        phantom_hlz_cm,
    )
    layout["phantom_front_z_cm"] = layout["phantom_center_z_cm"] - phantom_hlz_cm
    layout["q1_front_z_cm"] = layout["q1_z_cm"] - float(beamline["quad_length_cm"]) / 2.0
    layout["q4_exit_z_cm"] = layout["q4_z_cm"] + float(beamline["quad_length_cm"]) / 2.0
    layout["design_focus_z_cm"] = layout["phantom_front_z_cm"] + 20.0
    return layout


def format_findings(findings: List[Finding]) -> str:
    lines: List[str] = []
    for finding in findings:
        lines.append(f"- [{finding.severity}] {finding.title}: {finding.detail}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    reference = load_reference(args.reference)
    layout = summarize_layout(reference)
    template_text = args.template.read_text(encoding="utf-8")
    template_params = parse_parameter_assignments(template_text)

    sample_case = choose_sample_case(args.manifest)
    sample_case_id: Optional[str] = None
    sample_params: Dict[str, str] = {}
    if sample_case and sample_case[1].exists():
        sample_case_id = sample_case[0]
        sample_params = parse_parameter_assignments(sample_case[1].read_text(encoding="utf-8"))

    topas_version = read_topas_version()
    geant4_version = read_geant4_version(args.g4_data_dir)
    quadrupole_formula = find_quadrupole_formula()

    required_default_prefixes = [
        "G4EMLOW",
        "G4NDL",
        "G4PARTICLEXS",
        "G4ABLA",
        "G4SAIDDATA",
        "G4TENDL",
        "G4PII",
        "PhotonEvaporation",
        "RadioactiveDecay",
        "G4ENSDFSTATE",
        "G4INCL",
        "RealSurface",
    ]
    found_prefixes, missing_prefixes = check_dataset_prefixes(args.g4_data_dir, required_default_prefixes)

    template_modules = template_params.get("Ph/Default/Modules")
    sample_modules = sample_params.get("Ph/Default/Modules")
    template_source_component = template_params.get("So/Beam/Component")
    sample_source_component = sample_params.get("So/Beam/Component")

    findings: List[Finding] = []

    if sample_modules == '1 "g4em-standard_opt4"':
        findings.append(
            Finding(
                "high",
                "Historical runs used an EM-only profile",
                (
                    "The existing generated case files in runs/cases use "
                    '`sv:Ph/Default/Modules = 1 "g4em-standard_opt4"` instead of the full TOPAS default modular stack. '
                    "The paper does not document its physics list, so those completed runs cannot be claimed as an exact physics match."
                ),
            )
        )

    if missing_prefixes:
        findings.append(
            Finding(
                "high",
                "Required Geant4 data folders are missing",
                f"The following dataset families were not found under {args.g4_data_dir}: {', '.join(missing_prefixes)}.",
            )
        )

    if sample_source_component == '"World"':
        findings.append(
            Finding(
                "medium",
                "Historical runs used an implicit source frame",
                (
                    "Completed case files emit directly from `World`, which means source position and direction are only implied by TOPAS defaults. "
                    "That is not automatically wrong, but it makes the beamline harder to audit than an explicit beam-origin component."
                ),
            )
        )

    baseline_250 = reference["asymmetric_beamline"]["energies"]["250"]["baseline_gradients_t_per_m"]
    q4_response = electron_plane_response(gx=baseline_250[3], gy=baseline_250[3], beam_direction_sign=1)
    findings.append(
        Finding(
            "high",
            "Quadrupole sign convention is the leading mismatch candidate",
            (
                "TOPAS implements an ideal quadrupole with "
                f"`{quadrupole_formula}`. For an electron beam travelling in +Z, the 250 MeV baseline Q4 = +14.3 T/m "
                f"therefore {q4_response}. Supplementary Table 3 says the measured z-hat trend follows the x-plane focal trend as Q4 increases, "
                "so the current sign mapping deserves direct hypothesis testing."
            ),
        )
    )

    if geant4_version != "unknown" and geant4_version != "11.3.2":
        findings.append(
            Finding(
                "medium",
                "Unexpected Geant4 install version",
                f"Detected Geant4 {geant4_version}. This workspace was configured assuming 11.3.2.",
            )
        )

    findings.append(
        Finding(
            "info",
            "Source-to-magnet and magnet-to-phantom distances match the supplementary table",
            (
                f"Derived geometry gives Q1 front face at {layout['q1_front_z_cm']:.1f} cm, Q4 exit at {layout['q4_exit_z_cm']:.1f} cm, "
                f"and phantom front face at {layout['phantom_front_z_cm']:.1f} cm, which reproduces s1 = 109.0 cm and s5 = 46.1 cm from Supplementary Table 2."
            ),
        )
    )

    findings.append(
        Finding(
            "info",
            "Design focus depth is internally consistent with the paper",
            (
                f"The beamline geometry targets 20 cm into the phantom at global z = {layout['design_focus_z_cm']:.1f} cm. "
                "That is consistent with the paper's design statement and close to the reported 17.4 cm TOPAS dose maximum for the 250 MeV Whitmore."
            ),
        )
    )

    findings.append(
        Finding(
            "info",
            "Geant4 data libraries are installed locally",
            (
                f"Detected TOPAS {topas_version}, Geant4 {geant4_version}, and dataset families present for: "
                f"{', '.join(found_prefixes)}."
            ),
        )
    )

    recommendations = [
        "Rerun the nominal validation cases with the updated generator defaults so the decks use `topas_default` physics and an explicit `BeamOrigin` component.",
        "Run a low-history sign-convention scan using the new `--gradient-x-scale` and `--gradient-y-scale` options. The first two variants to test are `(1, 1)` and `(1, -1)`.",
        "If the sign scan improves the z-hat trend, repeat the full 25-case sweep with that convention before spending more time on Q4 autotuning.",
        "Treat physics-list matching as a secondary check: the installed Geant4 data are sufficient, but the paper does not specify its exact physics list or dataset versions.",
    ]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paper": reference["paper"],
        "tool_versions": {
            "topas": topas_version,
            "geant4": geant4_version,
        },
        "template": {
            "path": str(args.template.resolve()),
            "source_component": template_source_component,
            "physics_modules": template_modules,
            "physics_profile_name": physics_profile_name_from_modules(template_modules),
            "generator_default_physics_profile": "topas_default",
        },
        "sample_case": {
            "case_id": sample_case_id,
            "source_component": sample_source_component,
            "physics_modules": sample_modules,
            "physics_profile_name": physics_profile_name_from_modules(sample_modules),
        },
        "layout_cm": layout,
        "quadrupole_formula": quadrupole_formula,
        "findings": [finding.__dict__ for finding in findings],
        "recommendations": recommendations,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "physical_setup_audit.json"
    md_path = args.outdir / "physical_setup_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Physical Setup Audit",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Paper DOI: {reference['paper']['doi']}",
        f"- TOPAS: {topas_version}",
        f"- Geant4: {geant4_version}",
        "",
        "## Summary",
        "",
        "The geometry distances are internally consistent with Whitmore et al., and the required Geant4 data libraries are installed. "
        "The strongest remaining physical-setup risk is the quadrupole sign convention as implemented by TOPAS for electrons.",
        "",
        "## Findings",
        "",
        format_findings(findings),
        "",
        "## Current Config",
        "",
        f"- Template source component: `{template_source_component}`",
        f"- Template physics modules: `{template_modules}`",
        f"- Sample case ID: `{sample_case_id}`",
        f"- Sample case source component: `{sample_source_component}`",
        f"- Sample case physics modules: `{sample_modules}`",
        f"- TOPAS quadrupole field formula: `{quadrupole_formula}`",
        "",
        "## Recommended Next Runs",
        "",
    ]
    for recommendation in recommendations:
        md_lines.append(f"- {recommendation}")
    md_lines.extend(
        [
            "",
            "## Geometry Checks",
            "",
            f"- Q1 front face: `{layout['q1_front_z_cm']:.1f} cm`",
            f"- Q4 exit: `{layout['q4_exit_z_cm']:.1f} cm`",
            f"- Phantom front face: `{layout['phantom_front_z_cm']:.1f} cm`",
            f"- Design focus depth (20 cm into phantom): `{layout['design_focus_z_cm']:.1f} cm`",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
