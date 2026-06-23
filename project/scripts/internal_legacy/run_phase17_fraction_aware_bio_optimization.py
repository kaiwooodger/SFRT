#!/usr/bin/env python3
"""Run a fraction-aware biology-guided SFRT optimization loop on the detailed phantom.

Phase 17 extends the single-fraction Phase 16 search into a cumulative planning
loop. Each next-fraction candidate is evaluated by how it changes the running
course-level physical dose, the cumulative biology-aware effective dose, peak-
valley preservation, hotspot burden, and inter-fraction consistency.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage

from build_asymmetric_sweep import write_text_with_retries
from generate_detailed_headneck_topas_phantom import MATERIAL_SPECS
from run_linac_6mv_polyenergetic_clinical_sfrt import load_spectrum
from run_phase14_detailed_headneck_voxel_lattice import build_detailed_plan_phantom
from run_phase15_detailed_headneck_bioaware import (
    build_anatomical_biology_tensors,
    build_args_from_summary,
    load_phase14_summary,
)
from run_phase16_bio_guided_lattice_optimization import (
    build_candidate_centers,
    build_plan_args,
    build_safe_candidate_centers,
    build_structure_points_mm,
    compute_vessel_distance_reward,
    evaluate_plan,
    min_distance_mm,
    point_from_index,
    sphere_fits,
)
from run_phase13_headneck_voxel_lattice import compute_structure_metrics
from analyze_fractionated_bio_vs_physical import compute_effective_dose, compute_metrics_for_domain


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a cumulative fraction-aware optimization loop using physical and "
            "biology-aware objectives on the heterogeneous head-and-neck phantom."
        )
    )
    parser.add_argument("--phase-number", type=int, default=17)
    parser.add_argument(
        "--phase-description",
        type=str,
        default="Fraction-aware biology-guided SFRT optimization using cumulative course-level scoring.",
    )
    parser.add_argument(
        "--baseline-run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_directplan",
        help="Existing direct-plan run used to seed the first accepted fraction.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "linac_6mv_headneck_detailed_voxel_lattice_sfrt_phase17_fraction_aware",
        help="Output root for phase 17.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "topas" / "headneck_detailed_material_phantom_template.txt",
        help="TOPAS ImageCube template.",
    )
    parser.add_argument(
        "--spectrum-csv",
        type=Path,
        default=root / "data" / "linac_6mv_representative_spectrum.csv",
        help="Representative 6 MV spectrum CSV.",
    )
    parser.add_argument("--topas-bin", type=str, default="/Users/kw/shellScripts/topas")
    parser.add_argument("--g4-data-dir", type=str, default="/Applications/GEANT4")
    parser.add_argument("--physics-profile", type=str, default="em_opt4_only")
    parser.add_argument("--histories", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--cut-gamma-mm", type=float, default=0.01)
    parser.add_argument("--cut-electron-mm", type=float, default=0.01)
    parser.add_argument("--cut-positron-mm", type=float, default=0.01)
    parser.add_argument("--prescription-gy", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--pde-steps", type=int, default=400)
    parser.add_argument("--pde-dt", type=float, default=0.12)
    parser.add_argument("--tumor-cytokine-multiplier", type=float, default=2.0)
    parser.add_argument("--hypoxic-ros-scale", type=float, default=0.12)
    parser.add_argument("--hypoxic-cytokine-multiplier", type=float, default=2.7)
    parser.add_argument("--artery-ros-uptake", type=float, default=0.05)
    parser.add_argument("--artery-cyto-uptake", type=float, default=0.70)
    parser.add_argument("--vein-ros-uptake", type=float, default=0.05)
    parser.add_argument("--vein-cyto-uptake", type=float, default=0.90)
    parser.add_argument("--fractions", type=int, default=5, help="Total accepted fractions including baseline.")
    parser.add_argument("--num-spots", type=int, default=4)
    parser.add_argument("--spot-radius-mm", type=float, default=8.0)
    parser.add_argument("--base-margin-mm", type=float, default=6.0)
    parser.add_argument(
        "--base-min-ap-radius-mm",
        type=float,
        default=0.0,
        help="Minimum AP base-field radius, used to keep a stronger coverage backbone.",
    )
    parser.add_argument(
        "--base-min-lateral-radius-mm",
        type=float,
        default=0.0,
        help="Minimum lateral broad-field radius when present in the delivery model.",
    )
    parser.add_argument("--base-history-fraction", type=float, default=0.95)
    parser.add_argument(
        "--spot-ap-weight-scale",
        type=float,
        default=1.0,
        help="Relative weight of AP spot beams compared with lateral spot beams.",
    )
    parser.add_argument(
        "--spot-lateral-weight-scale",
        type=float,
        default=1.0,
        help="Relative weight assigned to each lateral spot beam.",
    )
    parser.add_argument(
        "--superior-posterior-lateral-scale",
        type=float,
        default=1.0,
        help="Extra multiplicative reduction applied to lateral spot beams for superior/posterior vertices.",
    )
    parser.add_argument(
        "--superior-threshold-mm",
        type=float,
        default=0.0,
        help="Vertex is treated as superior if its y-position exceeds the PTV centroid by this amount.",
    )
    parser.add_argument(
        "--posterior-threshold-mm",
        type=float,
        default=0.0,
        help="Vertex is treated as posterior if its z-position exceeds the PTV centroid by this amount.",
    )
    parser.add_argument(
        "--lateral-radius-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to lateral spot-beam apertures.",
    )
    parser.add_argument(
        "--ap-spot-radius-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to AP spot-beam apertures.",
    )
    parser.add_argument("--candidate-step-mm", type=float, default=6.0)
    parser.add_argument("--min-spot-spacing-mm", type=float, default=18.0)
    parser.add_argument(
        "--course-strategy",
        choices=(
            "adaptive_nofly_weight_opt",
            "adaptive_hotspot_avoidance",
            "adaptive_mutation",
            "alternating_two_pattern",
            "clinical_gtv_core",
            "larger_pitch_sweep",
            "phase24_joint_horizon_diversity",
            "randomized_protocol_mix",
            "reduced_vertices_interlaced",
            "vascular_sink_hugging",
        ),
        default="adaptive_mutation",
        help="Course-level lattice strategy used to generate fraction candidates.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("course_balance", "outside_gtv_spill"),
        default="course_balance",
        help="Scoring mode for course evaluation.",
    )
    parser.add_argument(
        "--candidate-top-k-centers",
        type=int,
        default=18,
        help="Top scored centers considered when assembling candidate spot sets.",
    )
    parser.add_argument(
        "--candidate-plan-limit",
        type=int,
        default=5,
        help="Maximum candidate spot sets evaluated per fraction.",
    )
    parser.add_argument(
        "--target-effective-gy-per-fraction",
        type=float,
        default=20.0,
        help="Desired cumulative effective dose target used to shape voxel-wise undercoverage guidance.",
    )
    parser.add_argument(
        "--target-need-cap-gy",
        type=float,
        default=10.0,
        help="Cap on voxel-wise undercoverage reward to avoid runaway candidate scores.",
    )
    parser.add_argument(
        "--peak-radius-mm",
        type=float,
        default=8.0,
        help="Radius used for peak ROI construction in the cumulative peak-valley metric.",
    )
    parser.add_argument(
        "--valley-exclusion-radius-mm",
        type=float,
        default=14.0,
        help="Radius excluded around each delivered vertex when defining cumulative valley ROIs.",
    )
    parser.add_argument(
        "--spill-shell-1-mm",
        type=float,
        default=5.0,
        help="Outer radius of the inner peri-GTV shell used for outside-GTV spill assessment.",
    )
    parser.add_argument(
        "--spill-shell-2-mm",
        type=float,
        default=15.0,
        help="Outer radius of the middle peri-GTV shell used for outside-GTV spill assessment.",
    )
    parser.add_argument(
        "--spill-shell-3-mm",
        type=float,
        default=30.0,
        help="Outer radius of the outer peri-GTV shell used for outside-GTV spill assessment.",
    )
    parser.add_argument(
        "--spill-oar-adjacent-mm",
        type=float,
        default=15.0,
        help="Distance used to define the outside-GTV OAR-adjacent spill region.",
    )
    parser.add_argument(
        "--hard-min-dist-cord-mm",
        type=float,
        default=55.0,
        help="Hard exclusion radius for candidate centers relative to the spinal cord.",
    )
    parser.add_argument(
        "--hard-min-dist-brainstem-mm",
        type=float,
        default=50.0,
        help="Hard exclusion radius for candidate centers relative to the brainstem.",
    )
    parser.add_argument("--hard-cumulative-cord-d2-eff-gy", type=float, default=85.0)
    parser.add_argument("--hard-cumulative-brainstem-d2-eff-gy", type=float, default=30.0)
    parser.add_argument("--hard-cumulative-parotid-r-mean-eff-gy", type=float, default=60.0)
    parser.add_argument("--hard-cumulative-thyroid-mean-eff-gy", type=float, default=50.0)
    parser.add_argument("--hard-cumulative-body-dmax-phys-gy", type=float, default=400.0)
    parser.add_argument("--soft-cumulative-ptv-d2-eff-gy", type=float, default=90.0)
    parser.add_argument(
        "--disable-body-hotspot-hard-constraint",
        action="store_true",
        help="Do not use cumulative body Dmax as a hard accept/reject constraint; keep it as a soft optimization penalty instead.",
    )
    parser.add_argument(
        "--adaptive-spot-memory-radius-mm",
        type=float,
        default=18.0,
        help="Phase 21: minimum distance from previously delivered spot centers when adaptive hotspot avoidance is enabled.",
    )
    parser.add_argument(
        "--adaptive-hotspot-avoidance-distance-mm",
        type=float,
        default=15.0,
        help="Phase 21: avoidance distance around accumulated hotspot/spill memory points.",
    )
    parser.add_argument(
        "--adaptive-hotspot-physical-threshold-gy",
        type=float,
        default=120.0,
        help="Phase 21: cumulative physical dose threshold used to seed hotspot-memory regions outside the GTV.",
    )
    parser.add_argument(
        "--adaptive-spill-effective-threshold-gy",
        type=float,
        default=35.0,
        help="Phase 21: cumulative effective-dose threshold used to mark outside-GTV spill burden regions.",
    )
    parser.add_argument(
        "--adaptive-oar-adjacent-effective-threshold-gy",
        type=float,
        default=20.0,
        help="Phase 21: cumulative effective-dose threshold used in outside-GTV OAR-adjacent burden zones.",
    )
    parser.add_argument(
        "--adaptive-local-effective-cap-gy",
        type=float,
        default=45.0,
        help="Phase 21: cap used to normalize per-center cumulative effective-dose penalty.",
    )
    parser.add_argument(
        "--adaptive-local-physical-cap-gy",
        type=float,
        default=140.0,
        help="Phase 21: cap used to normalize per-center cumulative physical-dose penalty.",
    )
    parser.add_argument(
        "--adaptive-local-effective-weight",
        type=float,
        default=12.0,
        help="Phase 21: weight penalizing candidate centers that sit in already burdened effective-dose regions.",
    )
    parser.add_argument(
        "--adaptive-local-physical-weight",
        type=float,
        default=9.0,
        help="Phase 21: weight penalizing candidate centers that sit in already burdened physical-dose regions.",
    )
    parser.add_argument(
        "--adaptive-hotspot-proximity-weight",
        type=float,
        default=14.0,
        help="Phase 21: weight penalizing proximity to accumulated hotspot/spill memory regions.",
    )
    parser.add_argument(
        "--adaptive-spot-memory-weight",
        type=float,
        default=16.0,
        help="Phase 21: weight penalizing proximity to previously delivered lattice centers.",
    )
    parser.add_argument(
        "--adaptive-min-filter-pool-size",
        type=int,
        default=8,
        help="Minimum surviving center pool after adaptive avoidance filtering before the filter is relaxed.",
    )
    parser.add_argument(
        "--phase22-no-fly-distance-mm",
        type=float,
        default=15.0,
        help="Phase 22: hard no-fly exclusion distance around accumulated spill/hotspot memory points.",
    )
    parser.add_argument(
        "--phase22-no-fly-physical-threshold-gy",
        type=float,
        default=100.0,
        help="Phase 22: cumulative physical-dose threshold used to seed hard no-fly hotspot regions outside the GTV.",
    )
    parser.add_argument(
        "--phase22-no-fly-effective-threshold-gy",
        type=float,
        default=32.0,
        help="Phase 22: cumulative effective-dose threshold used to seed hard outside-GTV spill no-fly regions.",
    )
    parser.add_argument(
        "--phase22-no-fly-oar-adjacent-threshold-gy",
        type=float,
        default=18.0,
        help="Phase 22: cumulative effective-dose threshold used for hard OAR-adjacent outside-GTV no-fly regions.",
    )
    parser.add_argument(
        "--phase22-no-fly-valley-threshold-gy",
        type=float,
        default=32.0,
        help="Phase 22: cumulative effective-dose threshold used for hard outside-GTV valley no-fly regions.",
    )
    parser.add_argument(
        "--phase22-no-fly-min-filter-pool-size",
        type=int,
        default=8,
        help="Phase 22: minimum surviving center pool after hard no-fly filtering before the filter is relaxed.",
    )
    parser.add_argument(
        "--phase22-no-fly-min-keep-candidate-sets",
        type=int,
        default=3,
        help="Phase 22: minimum surviving candidate-set count after hard no-fly filtering before the filter is relaxed.",
    )
    parser.add_argument(
        "--phase22-brainstem-trigger-fraction",
        type=float,
        default=0.65,
        help="Phase 22: activate enlarged brainstem avoidance once cumulative brainstem burden exceeds this fraction of its hard limit.",
    )
    parser.add_argument(
        "--phase22-brainstem-hard-avoidance-mm",
        type=float,
        default=28.0,
        help="Phase 22: hard center/vertex avoidance distance from the brainstem when adaptive brainstem protection is active.",
    )
    parser.add_argument(
        "--phase22-enable-weight-optimization",
        action="store_true",
        help="Phase 22: generate additional delivery-weight variants for each legal candidate geometry.",
    )
    parser.add_argument(
        "--phase22-weightopt-base-history-fraction",
        type=float,
        default=0.985,
        help="Phase 22: stronger backbone history fraction used in weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-weightopt-base-margin-mm",
        type=float,
        default=12.0,
        help="Phase 22: stronger backbone base-field margin used in weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-weightopt-ap-weight-scale",
        type=float,
        default=1.15,
        help="Phase 22: AP spot-beam weight scale used in balanced weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-weightopt-lateral-weight-scale",
        type=float,
        default=0.80,
        help="Phase 22: lateral spot-beam weight scale used in balanced weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-weightopt-superior-posterior-scale",
        type=float,
        default=0.45,
        help="Phase 22: superior/posterior lateral suppression used in balanced weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-weightopt-lateral-radius-scale",
        type=float,
        default=0.95,
        help="Phase 22: lateral aperture scale used in balanced weight-optimized variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-base-history-fraction",
        type=float,
        default=0.99,
        help="Phase 22: stronger broad-field fraction used in hotspot-sparing weight variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-base-margin-mm",
        type=float,
        default=14.0,
        help="Phase 22: stronger broad-field margin used in hotspot-sparing weight variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-ap-weight-scale",
        type=float,
        default=1.10,
        help="Phase 22: AP spot-beam weight scale used in hotspot-sparing variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-lateral-weight-scale",
        type=float,
        default=0.78,
        help="Phase 22: lateral spot-beam weight scale used in hotspot-sparing variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-superior-posterior-scale",
        type=float,
        default=0.40,
        help="Phase 22: superior/posterior lateral suppression used in hotspot-sparing variants.",
    )
    parser.add_argument(
        "--phase22-hotspot-spot-radius-mm",
        type=float,
        default=8.4,
        help="Phase 22: spot radius used in hotspot-sparing weight variants.",
    )
    parser.add_argument(
        "--phase22-brainstem-base-history-fraction",
        type=float,
        default=0.992,
        help="Phase 22: stronger broad-field fraction used when adaptive brainstem protection is active.",
    )
    parser.add_argument(
        "--phase22-brainstem-base-margin-mm",
        type=float,
        default=14.0,
        help="Phase 22: stronger broad-field margin used when adaptive brainstem protection is active.",
    )
    parser.add_argument(
        "--phase22-brainstem-ap-weight-scale",
        type=float,
        default=1.20,
        help="Phase 22: AP spot-beam weight scale used in brainstem-sparing variants.",
    )
    parser.add_argument(
        "--phase22-brainstem-lateral-weight-scale",
        type=float,
        default=0.68,
        help="Phase 22: lateral spot-beam weight scale used in brainstem-sparing variants.",
    )
    parser.add_argument(
        "--phase22-brainstem-superior-posterior-scale",
        type=float,
        default=0.25,
        help="Phase 22: superior/posterior lateral suppression used in brainstem-sparing variants.",
    )
    parser.add_argument(
        "--adaptive-veto-relax-mode",
        choices=("full", "partial", "none"),
        default="full",
        help="How adaptive no-fly/brainstem veto filters relax when too few candidates survive.",
    )
    parser.add_argument(
        "--adaptive-veto-relax-fraction",
        type=float,
        default=0.5,
        help="When using partial relaxation, keep at least this fraction of the requested minimum pool size.",
    )
    parser.add_argument(
        "--robust-top-k-candidates",
        type=int,
        default=0,
        help="Repeat this many top candidates with multiple seeds/histories for uncertainty-aware ranking. Zero disables robust re-evaluation.",
    )
    parser.add_argument(
        "--robust-seeds",
        type=str,
        default="",
        help="Comma-separated RNG seeds used for robust re-evaluation.",
    )
    parser.add_argument(
        "--robust-histories",
        type=str,
        default="",
        help="Comma-separated TOPAS history counts used for robust re-evaluation.",
    )
    parser.add_argument(
        "--robust-z-score",
        type=float,
        default=1.0,
        help="Uncertainty multiplier used when forming the Monte Carlo + biology noise band.",
    )
    parser.add_argument(
        "--robust-min-feasible-fraction",
        type=float,
        default=1.0,
        help="Minimum fraction of robust re-evaluations that must satisfy hard constraints for a candidate to be trusted as robust-feasible.",
    )
    parser.add_argument(
        "--clinical-gtv-contraction-mm",
        type=float,
        default=5.0,
        help="Clinical lattice strategy: minimum center distance from the GTV edge (mm).",
    )
    parser.add_argument(
        "--clinical-oar-clearance-mm",
        type=float,
        default=15.0,
        help="Clinical lattice strategy: hard center-to-OAR clearance (mm).",
    )
    parser.add_argument(
        "--clinical-inplane-pitch-mm",
        type=float,
        default=60.0,
        help="Clinical lattice strategy: nominal in-plane center-to-center spacing (mm).",
    )
    parser.add_argument(
        "--clinical-layer-spacing-mm",
        type=float,
        default=30.0,
        help="Clinical lattice strategy: nominal superior-inferior layer spacing (mm).",
    )
    parser.add_argument(
        "--clinical-grid-max-snap-mm",
        type=float,
        default=18.0,
        help="Clinical lattice strategy: maximum snap distance from a geometric template site to a feasible center (mm).",
    )
    parser.add_argument(
        "--reuse-baseline-dose",
        action="store_true",
        help="Reuse the existing baseline dose CSV instead of recomputing the baseline fraction at the current histories.",
    )
    parser.add_argument(
        "--pitch-scale-factors",
        nargs="+",
        type=float,
        default=[1.10, 1.25, 1.40],
        help="Scale factors used for larger-pitch candidate generation.",
    )
    parser.add_argument(
        "--interlaced-vertices-per-fraction",
        type=int,
        default=2,
        help="Number of lattice vertices delivered per fraction in the reduced-vertices interlaced strategy.",
    )
    parser.add_argument(
        "--pattern-complement-min-distance-mm",
        type=float,
        default=8.0,
        help="Minimum preferred separation between alternating-pattern vertices and the anchor pattern.",
    )
    parser.add_argument(
        "--vascular-candidate-multiplier",
        type=int,
        default=2,
        help="Multiplier on candidate-center pool size when assembling vessel-seeking lattice candidates.",
    )
    parser.add_argument(
        "--protocol-mix-per-family",
        type=int,
        default=2,
        help="Number of protocol-family candidates retained before stochastic down-selection in randomized_protocol_mix.",
    )
    parser.add_argument(
        "--phase24-min-vertices",
        type=int,
        default=2,
        help="Phase 24: minimum number of vertices allowed in the mixed lattice template family.",
    )
    parser.add_argument(
        "--phase24-max-vertices",
        type=int,
        default=4,
        help="Phase 24: maximum number of vertices allowed in the mixed lattice template family.",
    )
    parser.add_argument(
        "--phase24-template-pitch-scales",
        nargs="+",
        type=float,
        default=[0.85, 1.0, 1.15],
        help="Phase 24: clinical-template pitch scales explored when building mixed candidate families.",
    )
    parser.add_argument(
        "--phase24-soft-min-spacing-mm",
        type=float,
        default=15.0,
        help="Phase 24: softened minimum spacing used when generating mixed-template candidates.",
    )
    parser.add_argument(
        "--phase24-diversity-weight",
        type=float,
        default=8.0,
        help="Phase 24: reward for fraction-to-fraction spatial diversity of accepted vertices.",
    )
    parser.add_argument(
        "--phase24-diversity-cap-mm",
        type=float,
        default=36.0,
        help="Phase 24: cap used to normalize diversity reward distances.",
    )
    parser.add_argument(
        "--phase24-lookahead-weight",
        type=float,
        default=18.0,
        help="Phase 24: weight for the short-horizon future-space reserve bonus.",
    )
    parser.add_argument(
        "--phase24-future-center-count",
        type=int,
        default=6,
        help="Phase 24: number of top future-scored centers used when estimating reserved space for later fractions.",
    )
    parser.add_argument(
        "--phase24-future-center-score-cap",
        type=float,
        default=12.0,
        help="Phase 24: cap used to normalize future-space center scores in the lookahead surrogate.",
    )
    parser.add_argument(
        "--phase24-future-candidate-target",
        type=int,
        default=4,
        help="Phase 24: target count of future candidate sets used to normalize the lookahead reserve bonus.",
    )
    parser.add_argument("--weight-gtv-d95", type=float, default=2.5)
    parser.add_argument("--weight-ptv-d95", type=float, default=2.0)
    parser.add_argument("--weight-ptv-v95", type=float, default=0.1)
    parser.add_argument("--weight-pvdr-eff", type=float, default=6.0)
    parser.add_argument("--weight-pvdr-phys", type=float, default=2.0)
    parser.add_argument("--weight-hotspot", type=float, default=1.25)
    parser.add_argument("--weight-consistency", type=float, default=18.0)
    parser.add_argument("--weight-spill-shell-0-5-mean", type=float, default=2.2)
    parser.add_argument("--weight-spill-shell-5-15-mean", type=float, default=1.6)
    parser.add_argument("--weight-spill-shell-15-30-mean", type=float, default=0.9)
    parser.add_argument("--weight-spill-outside-gtv-d2", type=float, default=0.18)
    parser.add_argument("--weight-spill-ptv-valley-mean", type=float, default=1.2)
    parser.add_argument("--weight-spill-oar-adjacent-mean", type=float, default=1.8)
    parser.add_argument("--weight-target-floor", type=float, default=12.0)
    parser.add_argument("--min-gtv-d95-eff-per-fraction-gy", type=float, default=1.5)
    parser.add_argument("--min-ptv-d95-eff-per-fraction-gy", type=float, default=1.25)
    parser.add_argument("--min-ptv-v95-eff-pct", type=float, default=35.0)
    parser.add_argument(
        "--allow-infeasible-fallback",
        action="store_true",
        help="If no candidate satisfies the hard cumulative constraints, keep the best objective candidate anyway.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def round_spot(spot_mm: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(round(float(v), 1) for v in spot_mm)


def placement_key(spots_mm: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(sorted(round_spot(spot) for spot in spots_mm))


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= int(max_points):
        return points
    step = int(np.ceil(len(points) / float(max_points)))
    return points[::step]


def mask_to_points_mm(mask: np.ndarray, axes_mm: Dict[str, np.ndarray], max_points: int = 6000) -> np.ndarray:
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    idx = subsample_points(idx, max_points=max_points)
    return np.column_stack(
        [
            axes_mm["x"][idx[:, 0]],
            axes_mm["y"][idx[:, 1]],
            axes_mm["z"][idx[:, 2]],
        ]
    ).astype(np.float32)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for markdown output: {path}")
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    write_text_with_retries(path, "\n".join(lines) + "\n")


def parse_csv_int_list(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    items: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return items


def voxel_size_mm_from_axes(axes_mm: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    return (
        float(axes_mm["x"][1] - axes_mm["x"][0]),
        float(axes_mm["y"][1] - axes_mm["y"][0]),
        float(axes_mm["z"][1] - axes_mm["z"][0]),
    )


def contract_mask(mask: np.ndarray, axes_mm: Dict[str, np.ndarray], contraction_mm: float) -> np.ndarray:
    if float(contraction_mm) <= 0.0:
        return np.asarray(mask, dtype=bool)
    return ndimage.distance_transform_edt(mask, sampling=voxel_size_mm_from_axes(axes_mm)) > float(contraction_mm)


def build_clinical_gtv_core_candidate_centers(
    *,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    spot_radius_mm: float,
    candidate_step_mm: float,
    contraction_mm: float,
    oar_clearance_mm: float,
    min_pool_size: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, object]]:
    gtv_mask = np.asarray(structures["GTV"], dtype=bool)
    contracted_gtv = contract_mask(gtv_mask, axes_mm, float(contraction_mm))
    voxel_size_mm = voxel_size_mm_from_axes(axes_mm)
    step_vox = max(1, int(round(float(candidate_step_mm) / voxel_size_mm[0])))
    radius_vox = max(1, int(round(float(spot_radius_mm) / voxel_size_mm[0])))

    hard_oar_names = [
        "SPINAL_CORD",
        "BRAINSTEM",
        "PAROTID_R",
        "PAROTID_L",
        "THYROID",
        "PARATHYROIDS",
        "MANDIBLE",
    ]
    structure_points_mm = build_structure_points_mm(structures, axes_mm, hard_oar_names)

    stages = [
        {
            "name": "strict_contracted_shell_all_oars",
            "fit_mask": contracted_gtv,
            "hard_oars": hard_oar_names,
        },
        {
            "name": "center_in_contracted_gtv_all_oars",
            "fit_mask": gtv_mask,
            "hard_oars": hard_oar_names,
        },
        {
            "name": "center_in_contracted_gtv_soft_mandible",
            "fit_mask": gtv_mask,
            "hard_oars": [name for name in hard_oar_names if name != "MANDIBLE"],
        },
    ]

    stage_debug: List[Dict[str, object]] = []
    selected_candidates: List[Tuple[int, int, int]] = []
    selected_stage = stages[-1]
    for stage in stages:
        stage_candidates: List[Tuple[int, int, int]] = []
        for ix in range(0, gtv_mask.shape[0], step_vox):
            for iy in range(0, gtv_mask.shape[1], step_vox):
                for iz in range(0, gtv_mask.shape[2], step_vox):
                    if not contracted_gtv[ix, iy, iz]:
                        continue
                    if not sphere_fits(stage["fit_mask"], (ix, iy, iz), radius_vox):
                        continue
                    point_mm = point_from_index((ix, iy, iz), axes_mm)
                    if any(
                        min_distance_mm(point_mm, structure_points_mm[name]) < float(oar_clearance_mm)
                        for name in stage["hard_oars"]
                    ):
                        continue
                    stage_candidates.append((ix, iy, iz))
        stage_debug.append(
            {
                "stage": stage["name"],
                "hard_oars": list(stage["hard_oars"]),
                "candidate_count": int(len(stage_candidates)),
            }
        )
        if len(stage_candidates) >= int(min_pool_size):
            selected_candidates = stage_candidates
            selected_stage = stage
            break
        if stage_candidates and not selected_candidates:
            selected_candidates = stage_candidates
            selected_stage = stage

    if not selected_candidates:
        raise RuntimeError(
            "Clinical GTV-core lattice strategy found no feasible centers after GTV contraction and OAR-clearance filtering."
        )

    return selected_candidates, {
        "candidate_count": int(len(selected_candidates)),
        "selected_stage": str(selected_stage["name"]),
        "selected_hard_oars": list(selected_stage["hard_oars"]),
        "contraction_mm": float(contraction_mm),
        "oar_clearance_mm": float(oar_clearance_mm),
        "spot_radius_mm": float(spot_radius_mm),
        "candidate_step_mm": float(candidate_step_mm),
        "stage_debug": stage_debug,
    }


def compute_guidance_oar_weights(
    cumulative_effective_metrics: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], Dict[str, Tuple[str, float, float, float]]]:
    rules = {
        "SPINAL_CORD": ("d2_gy", float(args.hard_cumulative_cord_d2_eff_gy), 1.30),
        "BRAINSTEM": ("d2_gy", float(args.hard_cumulative_brainstem_d2_eff_gy), 1.15),
        "PAROTID_R": ("mean_gy", float(args.hard_cumulative_parotid_r_mean_eff_gy), 1.00),
        "PAROTID_L": ("mean_gy", 15.0, 0.30),
        "THYROID": ("mean_gy", float(args.hard_cumulative_thyroid_mean_eff_gy), 0.95),
        "PARATHYROIDS": ("mean_gy", 45.0, 0.45),
        "BRAIN": ("mean_gy", 20.0, 0.70),
        "BLOOD_BRAIN_BARRIER": ("mean_gy", 20.0, 0.35),
        "MANDIBLE": ("mean_gy", 50.0, 0.40),
    }
    weights: Dict[str, float] = {}
    details: Dict[str, Tuple[str, float, float, float]] = {}
    for structure, (metric, threshold, base_weight) in rules.items():
        if structure not in cumulative_effective_metrics or metric not in cumulative_effective_metrics[structure]:
            continue
        value = float(cumulative_effective_metrics[structure][metric])
        exceed = max(0.0, (value - threshold) / threshold)
        weight = float(base_weight * (1.0 + 3.0 * exceed))
        weights[structure] = weight
        details[structure] = (metric, value, threshold, exceed)
    return weights, details


def score_candidate_centers(
    *,
    current_cumulative_effective_dose: np.ndarray,
    current_cumulative_physical_dose: np.ndarray | None,
    current_fraction_idx: int,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    uptake_tensor: np.ndarray,
    candidate_indices: Sequence[Tuple[int, int, int]],
    target_effective_gy_per_fraction: float,
    target_need_cap_gy: float,
    oar_weights: Mapping[str, float],
    structure_points_mm: Mapping[str, np.ndarray],
    vessel_coords_mm: np.ndarray,
    history_counts: Mapping[Tuple[float, float, float], int],
    adaptive_avoidance: Mapping[str, object] | None = None,
) -> List[Tuple[float, Tuple[int, int, int], Dict[str, object]]]:
    desired_cumulative = float(current_fraction_idx) * float(target_effective_gy_per_fraction)
    tumour_need = np.clip(desired_cumulative - current_cumulative_effective_dose, a_min=0.0, a_max=None)
    hypoxia_mask = structures["HYPOXIA"]
    cyto_uptake = uptake_tensor[1]
    distance_weight_map = {
        "SPINAL_CORD": 0.90,
        "BRAINSTEM": 0.75,
        "PAROTID_R": 1.20,
        "PAROTID_L": 0.35,
        "THYROID": 0.90,
        "PARATHYROIDS": 0.55,
        "BRAIN": 0.80,
        "BLOOD_BRAIN_BARRIER": 0.45,
        "MANDIBLE": 0.40,
    }

    scored: List[Tuple[float, Tuple[int, int, int], Dict[str, object]]] = []
    for cand in candidate_indices:
        ix, iy, iz = cand
        point_mm = point_from_index(cand, axes_mm)
        need_score = float(min(tumour_need[ix, iy, iz], float(target_need_cap_gy)))
        hypoxia_bonus = 3.0 if bool(hypoxia_mask[ix, iy, iz]) else 0.0
        vessel_bonus = 6.0 * compute_vessel_distance_reward(cand, vessel_coords_mm, axes_mm)
        sink_bonus = 8.0 * float(cyto_uptake[ix, iy, iz])
        distance_score = 0.0
        dominant_structure = ""
        dominant_weight = -1.0
        for structure, weight in oar_weights.items():
            dist = min_distance_mm(point_mm, structure_points_mm[structure])
            weighted_distance = float(weight * distance_weight_map.get(structure, 0.25) * min(dist, 50.0) / 10.0)
            distance_score += weighted_distance
            if float(weight) > dominant_weight:
                dominant_weight = float(weight)
                dominant_structure = structure
        history_penalty = 2.0 * float(history_counts.get(round_spot(point_mm.tolist()), 0))
        adaptive_local_effective_penalty = 0.0
        adaptive_local_physical_penalty = 0.0
        adaptive_hotspot_penalty = 0.0
        adaptive_spot_memory_penalty = 0.0
        if adaptive_avoidance:
            eff_cap = max(1.0, float(adaptive_avoidance.get("local_effective_cap_gy", 45.0)))
            phys_cap = max(1.0, float(adaptive_avoidance.get("local_physical_cap_gy", 140.0)))
            adaptive_local_effective_penalty = float(adaptive_avoidance.get("local_effective_weight", 0.0)) * min(
                float(current_cumulative_effective_dose[ix, iy, iz]),
                eff_cap,
            ) / eff_cap
            if current_cumulative_physical_dose is not None:
                adaptive_local_physical_penalty = float(adaptive_avoidance.get("local_physical_weight", 0.0)) * min(
                    float(current_cumulative_physical_dose[ix, iy, iz]),
                    phys_cap,
                ) / phys_cap
            hotspot_points_mm = np.asarray(adaptive_avoidance.get("hotspot_points_mm", np.empty((0, 3), dtype=np.float32)))
            hotspot_avoidance_distance_mm = float(adaptive_avoidance.get("hotspot_avoidance_distance_mm", 0.0))
            if hotspot_points_mm.size and hotspot_avoidance_distance_mm > 0.0:
                hotspot_dist = min_distance_mm(point_mm, hotspot_points_mm)
                if hotspot_dist < hotspot_avoidance_distance_mm:
                    adaptive_hotspot_penalty = float(adaptive_avoidance.get("hotspot_proximity_weight", 0.0)) * (
                        (hotspot_avoidance_distance_mm - hotspot_dist) / hotspot_avoidance_distance_mm
                    )
            delivered_spots_mm = [tuple(float(v) for v in spot) for spot in adaptive_avoidance.get("delivered_spots_mm", [])]
            spot_memory_radius_mm = float(adaptive_avoidance.get("spot_memory_radius_mm", 0.0))
            if delivered_spots_mm and spot_memory_radius_mm > 0.0:
                memory_dist = min(math.dist(tuple(float(v) for v in point_mm.tolist()), spot) for spot in delivered_spots_mm)
                if memory_dist < spot_memory_radius_mm:
                    adaptive_spot_memory_penalty = float(adaptive_avoidance.get("spot_memory_weight", 0.0)) * (
                        (spot_memory_radius_mm - memory_dist) / spot_memory_radius_mm
                    )
        score = (
            1.6 * need_score
            + hypoxia_bonus
            + vessel_bonus
            + sink_bonus
            + distance_score
            - history_penalty
            - adaptive_local_effective_penalty
            - adaptive_local_physical_penalty
            - adaptive_hotspot_penalty
            - adaptive_spot_memory_penalty
        )
        scored.append(
            (
                float(score),
                cand,
                {
                    "center_mm": [float(v) for v in point_mm.tolist()],
                    "need_score": float(need_score),
                    "hypoxia_bonus": float(hypoxia_bonus),
                    "vessel_bonus": float(vessel_bonus),
                    "sink_bonus": float(sink_bonus),
                    "distance_score": float(distance_score),
                    "history_penalty": float(history_penalty),
                    "adaptive_local_effective_penalty": float(adaptive_local_effective_penalty),
                    "adaptive_local_physical_penalty": float(adaptive_local_physical_penalty),
                    "adaptive_hotspot_penalty": float(adaptive_hotspot_penalty),
                    "adaptive_spot_memory_penalty": float(adaptive_spot_memory_penalty),
                    "dominant_structure": dominant_structure,
                },
            )
        )
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored


def build_adaptive_hotspot_avoidance_state(
    *,
    args: argparse.Namespace,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    course_summary: Dict[str, object],
    accepted_sequence: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    delivered_spots_mm = [tuple(float(v) for v in spot) for result in accepted_sequence for spot in result["spot_centers_mm"]]
    peak_mask, _ = build_peak_valley_rois(
        structures,
        axes_mm,
        delivered_spots_mm,
        peak_radius_mm=float(args.peak_radius_mm),
        valley_exclusion_radius_mm=float(args.valley_exclusion_radius_mm),
    )
    spill_masks = build_spill_region_masks(
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        shell_1_mm=float(args.spill_shell_1_mm),
        shell_2_mm=float(args.spill_shell_2_mm),
        shell_3_mm=float(args.spill_shell_3_mm),
        oar_adjacent_mm=float(args.spill_oar_adjacent_mm),
    )
    cumulative_physical_dose = np.asarray(course_summary["cumulative_physical_dose"], dtype=np.float32)
    cumulative_effective_dose = np.asarray(course_summary["cumulative_effective_dose"], dtype=np.float32)
    outside_gtv_hotspot_mask = np.asarray(spill_masks["OUTSIDE_GTV"], dtype=bool) & (
        cumulative_physical_dose >= float(args.adaptive_hotspot_physical_threshold_gy)
    )
    outside_gtv_spill_mask = (
        (np.asarray(spill_masks["SPILL_SHELL_0_5"], dtype=bool) | np.asarray(spill_masks["SPILL_SHELL_5_15"], dtype=bool))
        & (cumulative_effective_dose >= float(args.adaptive_spill_effective_threshold_gy))
    )
    oar_adjacent_spill_mask = np.asarray(spill_masks["OAR_ADJACENT_OUTSIDE_GTV"], dtype=bool) & (
        cumulative_effective_dose >= float(args.adaptive_oar_adjacent_effective_threshold_gy)
    )
    combined_mask = outside_gtv_hotspot_mask | outside_gtv_spill_mask | oar_adjacent_spill_mask
    hotspot_points_mm = mask_to_points_mm(combined_mask, axes_mm, max_points=6000)
    return {
        "delivered_spots_mm": [[float(a), float(b), float(c)] for a, b, c in delivered_spots_mm],
        "spot_memory_radius_mm": float(args.adaptive_spot_memory_radius_mm),
        "hotspot_avoidance_distance_mm": float(args.adaptive_hotspot_avoidance_distance_mm),
        "local_effective_cap_gy": float(args.adaptive_local_effective_cap_gy),
        "local_physical_cap_gy": float(args.adaptive_local_physical_cap_gy),
        "local_effective_weight": float(args.adaptive_local_effective_weight),
        "local_physical_weight": float(args.adaptive_local_physical_weight),
        "hotspot_proximity_weight": float(args.adaptive_hotspot_proximity_weight),
        "spot_memory_weight": float(args.adaptive_spot_memory_weight),
        "hotspot_points_mm": hotspot_points_mm,
        "hotspot_point_count": int(hotspot_points_mm.shape[0]),
        "outside_gtv_hotspot_voxels": int(np.count_nonzero(outside_gtv_hotspot_mask)),
        "outside_gtv_spill_voxels": int(np.count_nonzero(outside_gtv_spill_mask)),
        "oar_adjacent_spill_voxels": int(np.count_nonzero(oar_adjacent_spill_mask)),
    }


def filter_candidate_indices_by_adaptive_avoidance(
    *,
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    adaptive_state: Mapping[str, object],
    min_pool_size: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, object]]:
    delivered_spots_mm = [tuple(float(v) for v in spot) for spot in adaptive_state.get("delivered_spots_mm", [])]
    hotspot_points_mm = np.asarray(adaptive_state.get("hotspot_points_mm", np.empty((0, 3), dtype=np.float32)))
    spot_memory_radius_mm = float(adaptive_state.get("spot_memory_radius_mm", 0.0))
    hotspot_avoidance_distance_mm = float(adaptive_state.get("hotspot_avoidance_distance_mm", 0.0))

    kept: List[Tuple[int, int, int]] = []
    rejected_spot_memory = 0
    rejected_hotspot = 0
    for cand in candidate_indices:
        point_mm = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
        if delivered_spots_mm and spot_memory_radius_mm > 0.0:
            dist_to_spot = min(math.dist(point_mm, spot) for spot in delivered_spots_mm)
            if dist_to_spot < spot_memory_radius_mm:
                rejected_spot_memory += 1
                continue
        if hotspot_points_mm.size and hotspot_avoidance_distance_mm > 0.0:
            dist_to_hotspot = min_distance_mm(np.asarray(point_mm, dtype=np.float32), hotspot_points_mm)
            if dist_to_hotspot < hotspot_avoidance_distance_mm:
                rejected_hotspot += 1
                continue
        kept.append(cand)

    relaxed = False
    if len(kept) < int(min_pool_size):
        kept = list(candidate_indices)
        relaxed = True
    return kept, {
        "initial_candidate_count": int(len(candidate_indices)),
        "kept_candidate_count": int(len(kept)),
        "rejected_spot_memory": int(rejected_spot_memory),
        "rejected_hotspot_proximity": int(rejected_hotspot),
        "relaxed_filter": bool(relaxed),
        "spot_memory_radius_mm": float(spot_memory_radius_mm),
        "hotspot_avoidance_distance_mm": float(hotspot_avoidance_distance_mm),
    }


def filter_candidate_sets_by_adaptive_memory(
    candidate_sets: Sequence[Dict[str, object]],
    *,
    reference_spots_mm: Sequence[Tuple[float, float, float]],
    spot_memory_radius_mm: float,
    min_keep: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not reference_spots_mm or float(spot_memory_radius_mm) <= 0.0:
        return list(candidate_sets), {
            "initial_candidate_set_count": int(len(candidate_sets)),
            "kept_candidate_set_count": int(len(candidate_sets)),
            "rejected_for_repeat_memory": 0,
            "relaxed_filter": False,
        }
    kept: List[Dict[str, object]] = []
    rejected = 0
    repeat_limit = max(2, int(math.ceil(0.5 * max(1, len(reference_spots_mm)))))
    for row in candidate_sets:
        spots = [tuple(float(v) for v in spot) for spot in row["spots_mm"]]
        repeated_count = 0
        for spot in spots:
            if min(math.dist(spot, ref) for ref in reference_spots_mm) < float(spot_memory_radius_mm):
                repeated_count += 1
        if repeated_count >= repeat_limit:
            rejected += 1
            continue
        kept.append(row)
    relaxed = False
    if len(kept) < int(min_keep):
        kept = list(candidate_sets)
        relaxed = True
    return kept, {
        "initial_candidate_set_count": int(len(candidate_sets)),
        "kept_candidate_set_count": int(len(kept)),
        "rejected_for_repeat_memory": int(rejected),
        "repeat_limit": int(repeat_limit),
        "relaxed_filter": bool(relaxed),
    }


def build_phase22_adaptive_state(
    *,
    args: argparse.Namespace,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    structure_points_mm: Mapping[str, np.ndarray],
    course_summary: Dict[str, object],
    accepted_sequence: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    adaptive_state = build_adaptive_hotspot_avoidance_state(
        args=args,
        structures=structures,
        axes_mm=axes_mm,
        course_summary=course_summary,
        accepted_sequence=accepted_sequence,
    )
    delivered_spots_mm = [tuple(float(v) for v in spot) for result in accepted_sequence for spot in result["spot_centers_mm"]]
    peak_mask, _ = build_peak_valley_rois(
        structures,
        axes_mm,
        delivered_spots_mm,
        peak_radius_mm=float(args.peak_radius_mm),
        valley_exclusion_radius_mm=float(args.valley_exclusion_radius_mm),
    )
    spill_masks = build_spill_region_masks(
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        shell_1_mm=float(args.spill_shell_1_mm),
        shell_2_mm=float(args.spill_shell_2_mm),
        shell_3_mm=float(args.spill_shell_3_mm),
        oar_adjacent_mm=float(args.spill_oar_adjacent_mm),
    )
    cumulative_physical_dose = np.asarray(course_summary["cumulative_physical_dose"], dtype=np.float32)
    cumulative_effective_dose = np.asarray(course_summary["cumulative_effective_dose"], dtype=np.float32)

    hard_hotspot_mask = np.asarray(spill_masks["OUTSIDE_GTV"], dtype=bool) & (
        cumulative_physical_dose >= float(args.phase22_no_fly_physical_threshold_gy)
    )
    hard_spill_mask = np.asarray(spill_masks["OUTSIDE_GTV"], dtype=bool) & (
        cumulative_effective_dose >= float(args.phase22_no_fly_effective_threshold_gy)
    )
    hard_oar_adjacent_mask = np.asarray(spill_masks["OAR_ADJACENT_OUTSIDE_GTV"], dtype=bool) & (
        cumulative_effective_dose >= float(args.phase22_no_fly_oar_adjacent_threshold_gy)
    )
    hard_valley_mask = np.asarray(spill_masks["PTV_VALLEY_OUTSIDE_GTV"], dtype=bool) & (
        cumulative_effective_dose >= float(args.phase22_no_fly_valley_threshold_gy)
    )
    hard_no_fly_mask = hard_hotspot_mask | hard_spill_mask | hard_oar_adjacent_mask | hard_valley_mask
    no_fly_points_mm = mask_to_points_mm(hard_no_fly_mask, axes_mm, max_points=7000)

    brainstem_limit = max(1.0, float(args.hard_cumulative_brainstem_d2_eff_gy))
    brainstem_trigger = float(args.phase22_brainstem_trigger_fraction) * brainstem_limit
    brainstem_d2 = float(course_summary["cumulative_effective_metrics"]["BRAINSTEM"]["d2_gy"])
    brainstem_guard_active = bool(brainstem_d2 >= brainstem_trigger)
    brainstem_points_mm = np.asarray(
        structure_points_mm.get("BRAINSTEM", np.empty((0, 3), dtype=np.float32)),
        dtype=np.float32,
    )
    if brainstem_points_mm.size:
        brainstem_points_mm = brainstem_points_mm[: min(len(brainstem_points_mm), 4000)]

    adaptive_state.update(
        {
            "no_fly_distance_mm": float(args.phase22_no_fly_distance_mm),
            "no_fly_points_mm": no_fly_points_mm,
            "hard_no_fly_voxel_count": int(np.count_nonzero(hard_no_fly_mask)),
            "hard_no_fly_hotspot_voxels": int(np.count_nonzero(hard_hotspot_mask)),
            "hard_no_fly_spill_voxels": int(np.count_nonzero(hard_spill_mask)),
            "hard_no_fly_oar_adjacent_voxels": int(np.count_nonzero(hard_oar_adjacent_mask)),
            "hard_no_fly_valley_voxels": int(np.count_nonzero(hard_valley_mask)),
            "brainstem_guard_active": bool(brainstem_guard_active),
            "brainstem_d2_eff_gy": float(brainstem_d2),
            "brainstem_trigger_gy": float(brainstem_trigger),
            "brainstem_points_mm": brainstem_points_mm,
            "brainstem_avoidance_distance_mm": (
                float(args.phase22_brainstem_hard_avoidance_mm) if brainstem_guard_active else 0.0
            ),
        }
    )
    return adaptive_state


def filter_candidate_indices_by_phase22_veto(
    *,
    args: argparse.Namespace,
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    adaptive_state: Mapping[str, object],
    min_pool_size: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, object]]:
    delivered_spots_mm = [tuple(float(v) for v in spot) for spot in adaptive_state.get("delivered_spots_mm", [])]
    spot_memory_radius_mm = float(adaptive_state.get("spot_memory_radius_mm", 0.0))
    no_fly_points_mm = np.asarray(adaptive_state.get("no_fly_points_mm", np.empty((0, 3), dtype=np.float32)))
    no_fly_distance_mm = float(adaptive_state.get("no_fly_distance_mm", 0.0))
    brainstem_points_mm = np.asarray(adaptive_state.get("brainstem_points_mm", np.empty((0, 3), dtype=np.float32)))
    brainstem_avoidance_distance_mm = float(adaptive_state.get("brainstem_avoidance_distance_mm", 0.0))

    kept: List[Tuple[int, int, int]] = []
    rejected_spot_memory = 0
    rejected_no_fly = 0
    rejected_brainstem = 0
    rejected_rows: List[Tuple[float, Tuple[int, int, int]]] = []
    for cand in candidate_indices:
        point_mm = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
        clearance_terms: List[float] = []
        if delivered_spots_mm and spot_memory_radius_mm > 0.0:
            spot_clearance = min(math.dist(point_mm, spot) for spot in delivered_spots_mm) - spot_memory_radius_mm
            clearance_terms.append(float(spot_clearance))
            if spot_clearance < 0.0:
                rejected_spot_memory += 1
                rejected_rows.append((float(min(clearance_terms)), cand))
                continue
        if no_fly_points_mm.size and no_fly_distance_mm > 0.0:
            no_fly_clearance = min_distance_mm(np.asarray(point_mm, dtype=np.float32), no_fly_points_mm) - no_fly_distance_mm
            clearance_terms.append(float(no_fly_clearance))
            if no_fly_clearance < 0.0:
                rejected_no_fly += 1
                rejected_rows.append((float(min(clearance_terms)), cand))
                continue
        if brainstem_points_mm.size and brainstem_avoidance_distance_mm > 0.0:
            brainstem_clearance = min_distance_mm(np.asarray(point_mm, dtype=np.float32), brainstem_points_mm) - brainstem_avoidance_distance_mm
            clearance_terms.append(float(brainstem_clearance))
            if brainstem_clearance < 0.0:
                rejected_brainstem += 1
                rejected_rows.append((float(min(clearance_terms)), cand))
                continue
        kept.append(cand)

    relaxed = False
    relax_mode = str(getattr(args, "adaptive_veto_relax_mode", "full"))
    kept_before_relax = len(kept)
    if len(kept) < int(min_pool_size):
        if relax_mode == "full":
            kept = list(candidate_indices)
            relaxed = True
        elif relax_mode == "partial":
            target_keep = max(
                len(kept),
                int(math.ceil(float(min_pool_size) * max(0.0, min(1.0, float(getattr(args, "adaptive_veto_relax_fraction", 0.5)))))),
            )
            seen = set(kept)
            for _, cand in sorted(rejected_rows, key=lambda row: row[0], reverse=True):
                if cand in seen:
                    continue
                kept.append(cand)
                seen.add(cand)
                if len(kept) >= target_keep:
                    break
            relaxed = len(kept) > kept_before_relax
    return kept, {
        "initial_candidate_count": int(len(candidate_indices)),
        "kept_candidate_count": int(len(kept)),
        "rejected_spot_memory": int(rejected_spot_memory),
        "rejected_hard_no_fly": int(rejected_no_fly),
        "rejected_brainstem_guard": int(rejected_brainstem),
        "relaxed_filter": bool(relaxed),
        "relax_mode": relax_mode,
        "kept_before_relax": int(kept_before_relax),
        "no_fly_distance_mm": float(no_fly_distance_mm),
        "brainstem_avoidance_distance_mm": float(brainstem_avoidance_distance_mm),
    }


def filter_candidate_sets_by_phase22_veto(
    candidate_sets: Sequence[Dict[str, object]],
    *,
    args: argparse.Namespace,
    adaptive_state: Mapping[str, object],
    min_keep: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    delivered_spots_mm = [tuple(float(v) for v in spot) for spot in adaptive_state.get("delivered_spots_mm", [])]
    spot_memory_radius_mm = float(adaptive_state.get("spot_memory_radius_mm", 0.0))
    no_fly_points_mm = np.asarray(adaptive_state.get("no_fly_points_mm", np.empty((0, 3), dtype=np.float32)))
    no_fly_distance_mm = float(adaptive_state.get("no_fly_distance_mm", 0.0))
    brainstem_points_mm = np.asarray(adaptive_state.get("brainstem_points_mm", np.empty((0, 3), dtype=np.float32)))
    brainstem_avoidance_distance_mm = float(adaptive_state.get("brainstem_avoidance_distance_mm", 0.0))

    kept: List[Dict[str, object]] = []
    rejected_spot_memory = 0
    rejected_no_fly = 0
    rejected_brainstem = 0
    rejected_rows: List[Tuple[float, Dict[str, object]]] = []
    for row in candidate_sets:
        spots = [tuple(float(v) for v in spot) for spot in row["spots_mm"]]
        reject_reason = None
        clearance_candidates: List[float] = []
        if delivered_spots_mm and spot_memory_radius_mm > 0.0:
            memory_clearance = min(
                min(math.dist(spot, ref) for ref in delivered_spots_mm) - spot_memory_radius_mm
                for spot in spots
            )
            clearance_candidates.append(float(memory_clearance))
            if memory_clearance < 0.0:
                reject_reason = "spot_memory"
        if reject_reason is None and no_fly_points_mm.size and no_fly_distance_mm > 0.0:
            no_fly_clearance = min(
                min_distance_mm(np.asarray(spot, dtype=np.float32), no_fly_points_mm) - no_fly_distance_mm
                for spot in spots
            )
            clearance_candidates.append(float(no_fly_clearance))
            if no_fly_clearance < 0.0:
                reject_reason = "hard_no_fly"
        if reject_reason is None and brainstem_points_mm.size and brainstem_avoidance_distance_mm > 0.0:
            brainstem_clearance = min(
                min_distance_mm(np.asarray(spot, dtype=np.float32), brainstem_points_mm) - brainstem_avoidance_distance_mm
                for spot in spots
            )
            clearance_candidates.append(float(brainstem_clearance))
            if brainstem_clearance < 0.0:
                reject_reason = "brainstem_guard"

        if reject_reason == "spot_memory":
            rejected_spot_memory += 1
            rejected_rows.append((float(min(clearance_candidates) if clearance_candidates else -1.0), row))
            continue
        if reject_reason == "hard_no_fly":
            rejected_no_fly += 1
            rejected_rows.append((float(min(clearance_candidates) if clearance_candidates else -1.0), row))
            continue
        if reject_reason == "brainstem_guard":
            rejected_brainstem += 1
            rejected_rows.append((float(min(clearance_candidates) if clearance_candidates else -1.0), row))
            continue
        kept.append(row)

    relaxed = False
    relax_mode = str(getattr(args, "adaptive_veto_relax_mode", "full"))
    kept_before_relax = len(kept)
    if len(kept) < int(min_keep):
        if relax_mode == "full":
            kept = list(candidate_sets)
            relaxed = True
        elif relax_mode == "partial":
            target_keep = max(
                len(kept),
                int(math.ceil(float(min_keep) * max(0.0, min(1.0, float(getattr(args, "adaptive_veto_relax_fraction", 0.5)))))),
            )
            seen = {
                (placement_key([tuple(float(v) for v in spot) for spot in row["spots_mm"]]), plan_override_key(row.get("plan_overrides")))
                for row in kept
            }
            for _, row in sorted(rejected_rows, key=lambda item: item[0], reverse=True):
                key = (
                    placement_key([tuple(float(v) for v in spot) for spot in row["spots_mm"]]),
                    plan_override_key(row.get("plan_overrides")),
                )
                if key in seen:
                    continue
                kept.append(row)
                seen.add(key)
                if len(kept) >= target_keep:
                    break
            relaxed = len(kept) > kept_before_relax
    return kept, {
        "initial_candidate_set_count": int(len(candidate_sets)),
        "kept_candidate_set_count": int(len(kept)),
        "rejected_spot_memory": int(rejected_spot_memory),
        "rejected_hard_no_fly": int(rejected_no_fly),
        "rejected_brainstem_guard": int(rejected_brainstem),
        "relaxed_filter": bool(relaxed),
        "relax_mode": relax_mode,
        "kept_before_relax": int(kept_before_relax),
    }


def expand_candidate_sets_with_phase22_weight_optimization(
    candidate_sets: Sequence[Dict[str, object]],
    *,
    args: argparse.Namespace,
    adaptive_state: Mapping[str, object] | None,
) -> List[Dict[str, object]]:
    if not bool(args.phase22_enable_weight_optimization):
        return list(candidate_sets)

    brainstem_guard_active = bool((adaptive_state or {}).get("brainstem_guard_active", False))
    expanded: List[Dict[str, object]] = []
    for row in candidate_sets:
        expanded.append(dict(row))
        base_overrides = {str(k): float(v) for k, v in (row.get("plan_overrides") or {}).items()}

        def current_value(name: str, default: float) -> float:
            return float(base_overrides.get(name, default))

        def add_variant(origin_suffix: str, heuristic_bonus: float, overrides: Mapping[str, float]) -> None:
            merged = dict(base_overrides)
            for key, value in overrides.items():
                merged[str(key)] = float(value)
            updated = dict(row)
            updated["candidate_origin"] = f"{str(row.get('candidate_origin', 'candidate'))}_{origin_suffix}"
            updated["heuristic_score"] = float(updated.get("heuristic_score", 0.0)) + float(heuristic_bonus)
            layout_debug = dict(updated.get("layout_debug") or {})
            layout_debug["delivery_weight_variant"] = str(origin_suffix)
            layout_debug["delivery_weight_overrides"] = {str(k): float(v) for k, v in merged.items()}
            updated["layout_debug"] = layout_debug
            updated["plan_overrides"] = merged
            expanded.append(updated)

        add_variant(
            "weightopt_balanced",
            0.18,
            {
                "base_history_fraction": max(current_value("base_history_fraction", float(args.base_history_fraction)), float(args.phase22_weightopt_base_history_fraction)),
                "base_margin_mm": max(current_value("base_margin_mm", float(args.base_margin_mm)), float(args.phase22_weightopt_base_margin_mm)),
                "spot_ap_weight_scale": max(current_value("spot_ap_weight_scale", float(args.spot_ap_weight_scale)), float(args.phase22_weightopt_ap_weight_scale)),
                "spot_lateral_weight_scale": min(current_value("spot_lateral_weight_scale", float(args.spot_lateral_weight_scale)), float(args.phase22_weightopt_lateral_weight_scale)),
                "superior_posterior_lateral_scale": min(current_value("superior_posterior_lateral_scale", float(args.superior_posterior_lateral_scale)), float(args.phase22_weightopt_superior_posterior_scale)),
                "lateral_radius_scale": min(current_value("lateral_radius_scale", float(args.lateral_radius_scale)), float(args.phase22_weightopt_lateral_radius_scale)),
            },
        )
        add_variant(
            "weightopt_hotspot",
            0.24,
            {
                "base_history_fraction": max(current_value("base_history_fraction", float(args.base_history_fraction)), float(args.phase22_hotspot_base_history_fraction)),
                "base_margin_mm": max(current_value("base_margin_mm", float(args.base_margin_mm)), float(args.phase22_hotspot_base_margin_mm)),
                "spot_ap_weight_scale": max(current_value("spot_ap_weight_scale", float(args.spot_ap_weight_scale)), float(args.phase22_hotspot_ap_weight_scale)),
                "spot_lateral_weight_scale": min(current_value("spot_lateral_weight_scale", float(args.spot_lateral_weight_scale)), float(args.phase22_hotspot_lateral_weight_scale)),
                "superior_posterior_lateral_scale": min(current_value("superior_posterior_lateral_scale", float(args.superior_posterior_lateral_scale)), float(args.phase22_hotspot_superior_posterior_scale)),
                "spot_radius_mm": max(current_value("spot_radius_mm", float(args.spot_radius_mm)), float(args.phase22_hotspot_spot_radius_mm)),
            },
        )
        if brainstem_guard_active:
            add_variant(
                "weightopt_brainstem",
                0.30,
                {
                    "base_history_fraction": max(current_value("base_history_fraction", float(args.base_history_fraction)), float(args.phase22_brainstem_base_history_fraction)),
                    "base_margin_mm": max(current_value("base_margin_mm", float(args.base_margin_mm)), float(args.phase22_brainstem_base_margin_mm)),
                    "spot_ap_weight_scale": max(current_value("spot_ap_weight_scale", float(args.spot_ap_weight_scale)), float(args.phase22_brainstem_ap_weight_scale)),
                    "spot_lateral_weight_scale": min(current_value("spot_lateral_weight_scale", float(args.spot_lateral_weight_scale)), float(args.phase22_brainstem_lateral_weight_scale)),
                    "superior_posterior_lateral_scale": min(current_value("superior_posterior_lateral_scale", float(args.superior_posterior_lateral_scale)), float(args.phase22_brainstem_superior_posterior_scale)),
                },
            )

    return dedupe_candidate_sets(expanded)


def augment_candidate_pool_with_reference_neighborhood(
    *,
    base_candidate_indices: Sequence[Tuple[int, int, int]],
    all_candidate_indices: Sequence[Tuple[int, int, int]],
    reference_spots_mm: Sequence[Tuple[float, float, float]],
    axes_mm: Dict[str, np.ndarray],
    radius_mm: float,
    per_spot_limit: int = 12,
) -> List[Tuple[int, int, int]]:
    augmented = list(base_candidate_indices)
    seen = set(base_candidate_indices)
    for ref_spot in reference_spots_mm:
        local_rows: List[Tuple[float, Tuple[int, int, int]]] = []
        for cand in all_candidate_indices:
            point_mm = point_from_index(cand, axes_mm)
            dist = math.dist(tuple(float(v) for v in point_mm.tolist()), ref_spot)
            if dist > float(radius_mm):
                continue
            local_rows.append((float(dist), cand))
        local_rows.sort(key=lambda row: row[0])
        kept = 0
        for _, cand in local_rows:
            if cand in seen:
                continue
            augmented.append(cand)
            seen.add(cand)
            kept += 1
            if kept >= int(per_spot_limit):
                break
    return augmented


def is_spacing_valid(points: Sequence[Tuple[float, float, float]], spacing_limit_mm: float) -> Tuple[bool, float, float]:
    pairwise_sum = 0.0
    min_pairwise = float("inf")
    for a, b in itertools.combinations(range(len(points)), 2):
        dist = math.dist(points[a], points[b])
        min_pairwise = min(min_pairwise, dist)
        if dist < float(spacing_limit_mm):
            return False, 0.0, float(dist)
        pairwise_sum += dist
    if not math.isfinite(min_pairwise):
        min_pairwise = 0.0
    return True, float(pairwise_sum), float(min_pairwise)


def compute_layout_heuristic(
    points: Sequence[Tuple[float, float, float]],
    *,
    center_score: float,
    min_spacing_mm: float,
    reference_spots_mm: Sequence[Tuple[float, float, float]] | None,
    pairwise_sum: float,
    min_pairwise: float,
    origin_bonus: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    coords = np.asarray(points, dtype=np.float32)
    spreads = np.ptp(coords, axis=0) if coords.shape[0] >= 2 else np.zeros(3, dtype=np.float32)
    spread_sorted = np.sort(spreads)[::-1]
    second_axis_spread = float(spread_sorted[1]) if spread_sorted.size >= 2 else 0.0
    third_axis_spread = float(spread_sorted[2]) if spread_sorted.size >= 3 else 0.0
    planar_target = max(10.0, float(min_spacing_mm) * 0.60)
    line_penalty = 8.0 * max(0.0, (planar_target - second_axis_spread) / planar_target)
    centroid_shift = 0.0
    retained_count = 0.0
    retention_bonus = 0.0
    centroid_penalty = 0.0
    if reference_spots_mm:
        ref = np.asarray(reference_spots_mm, dtype=np.float32)
        centroid_shift = float(np.linalg.norm(coords.mean(axis=0) - ref.mean(axis=0)))
        ref_set = {round_spot(point) for point in reference_spots_mm}
        retained_count = float(sum(1 for point in points if round_spot(point) in ref_set))
        retention_bonus = 1.25 * retained_count
        centroid_penalty = 0.22 * max(0.0, centroid_shift - float(min_spacing_mm) * 0.65)
    geometry_bonus = 0.02 * float(pairwise_sum) + 0.30 * second_axis_spread + 0.08 * third_axis_spread
    heuristic = float(center_score + geometry_bonus + retention_bonus + float(origin_bonus) - line_penalty - centroid_penalty)
    return heuristic, {
        "pairwise_sum": float(pairwise_sum),
        "min_pairwise": float(min_pairwise),
        "second_axis_spread": float(second_axis_spread),
        "third_axis_spread": float(third_axis_spread),
        "line_penalty": float(line_penalty),
        "centroid_shift": float(centroid_shift),
        "centroid_penalty": float(centroid_penalty),
        "retained_count": float(retained_count),
        "retention_bonus": float(retention_bonus),
        "origin_bonus": float(origin_bonus),
    }


def build_candidate_spot_sets(
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
    *,
    axes_mm: Dict[str, np.ndarray],
    num_spots: int,
    min_spacing_mm: float,
    top_k_centers: int,
    candidate_plan_limit: int,
    reference_spots_mm: Sequence[Tuple[float, float, float]] | None = None,
) -> List[Dict[str, object]]:
    subset = list(scored_centers[: max(int(top_k_centers), int(num_spots), 24)])
    mutation_subset = list(scored_centers[: max(int(top_k_centers) * 4, int(num_spots) * 10, 48)])
    score_lookup = {
        round_spot(point_from_index(cand, axes_mm).tolist()): float(score)
        for score, cand, _ in mutation_subset
    }
    default_center_score = float(np.median([float(row[0]) for row in subset])) if subset else 0.0
    reference_points = [tuple(float(v) for v in point) for point in reference_spots_mm] if reference_spots_mm else []

    def maybe_add_candidate(
        candidate_sets: List[Dict[str, object]],
        seen: set[Tuple[Tuple[float, float, float], ...]],
        *,
        points: Sequence[Tuple[float, float, float]],
        center_debug: Sequence[Dict[str, object]],
        base_center_score: float,
        spacing_limit: float,
        origin: str,
        origin_bonus: float,
    ) -> None:
        valid, pairwise_sum, min_pairwise = is_spacing_valid(points, spacing_limit)
        if not valid:
            return
        key = placement_key(points)
        if key in seen:
            return
        seen.add(key)
        heuristic, layout_debug = compute_layout_heuristic(
            points,
            center_score=float(base_center_score),
            min_spacing_mm=float(min_spacing_mm),
            reference_spots_mm=reference_points if reference_points else None,
            pairwise_sum=float(pairwise_sum),
            min_pairwise=float(min_pairwise),
            origin_bonus=float(origin_bonus),
        )
        candidate_sets.append(
            {
                "spots_mm": [tuple(float(v) for v in point) for point in points],
                "heuristic_score": float(heuristic),
                "center_debug": list(center_debug),
                "spacing_limit_mm": float(spacing_limit),
                "candidate_origin": origin,
                "layout_debug": layout_debug,
            }
        )

    relax_spacings = [
        float(min_spacing_mm),
        max(14.0, float(min_spacing_mm) * 0.85),
        12.0,
        10.0,
        8.0,
    ]
    for spacing_limit in relax_spacings:
        candidate_sets: List[Dict[str, object]] = []
        seen: set[Tuple[Tuple[float, float, float], ...]] = set()

        if reference_points:
            local_radius_mm = max(18.0, float(min_spacing_mm) * 1.35)
            extended_radius_mm = max(local_radius_mm * 1.5, local_radius_mm + 6.0)
            local_choices: List[List[Tuple[float, Tuple[float, float, float], Dict[str, object]]]] = []
            for ref_spot in reference_points:
                choices: List[Tuple[float, Tuple[float, float, float], Dict[str, object]]] = []
                for score, cand, debug in mutation_subset:
                    point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
                    dist_to_ref = math.dist(point, ref_spot)
                    if dist_to_ref < 1e-3:
                        continue
                    if dist_to_ref > extended_radius_mm:
                        continue
                    distance_penalty = 0.35 * max(0.0, dist_to_ref - float(min_spacing_mm) * 0.35)
                    adjusted_score = float(score) - float(distance_penalty)
                    local_debug = dict(debug)
                    local_debug["candidate_mm"] = [float(v) for v in point]
                    local_debug["distance_to_reference_mm"] = float(dist_to_ref)
                    local_debug["adjusted_local_score"] = float(adjusted_score)
                    if dist_to_ref <= local_radius_mm:
                        adjusted_score += 1.0
                    choices.append((float(adjusted_score), point, local_debug))
                choices.sort(key=lambda row: row[0], reverse=True)
                unique_choices: List[Tuple[float, Tuple[float, float, float], Dict[str, object]]] = []
                seen_local: set[Tuple[float, float, float]] = set()
                for item in choices:
                    point = round_spot(item[1])
                    if point in seen_local:
                        continue
                    seen_local.add(point)
                    unique_choices.append(item)
                    if len(unique_choices) >= 6:
                        break
                local_choices.append(unique_choices)

            reference_center_score = float(sum(score_lookup.get(round_spot(point), default_center_score) for point in reference_points))
            for spot_idx, choices in enumerate(local_choices):
                for adjusted_score, point, debug in choices:
                    mutated = list(reference_points)
                    mutated[spot_idx] = point
                    center_debug = []
                    for idx, original in enumerate(reference_points):
                        if idx == spot_idx:
                            center_debug.append(debug)
                        else:
                            center_debug.append(
                                {
                                    "candidate_mm": [float(v) for v in original],
                                    "retained_from_previous_fraction": True,
                                }
                            )
                    maybe_add_candidate(
                        candidate_sets,
                        seen,
                        points=mutated,
                        center_debug=center_debug,
                        base_center_score=reference_center_score - score_lookup.get(round_spot(reference_points[spot_idx]), default_center_score) + adjusted_score,
                        spacing_limit=float(spacing_limit),
                        origin="single_mutation",
                        origin_bonus=3.0,
                    )

            for left_idx, right_idx in itertools.combinations(range(len(reference_points)), 2):
                left_choices = local_choices[left_idx][:2]
                right_choices = local_choices[right_idx][:2]
                for (left_score, left_point, left_debug), (right_score, right_point, right_debug) in itertools.product(left_choices, right_choices):
                    if round_spot(left_point) == round_spot(right_point):
                        continue
                    mutated = list(reference_points)
                    mutated[left_idx] = left_point
                    mutated[right_idx] = right_point
                    center_debug = []
                    for idx, original in enumerate(reference_points):
                        if idx == left_idx:
                            center_debug.append(left_debug)
                        elif idx == right_idx:
                            center_debug.append(right_debug)
                        else:
                            center_debug.append(
                                {
                                    "candidate_mm": [float(v) for v in original],
                                    "retained_from_previous_fraction": True,
                                }
                            )
                    base_score = reference_center_score
                    base_score -= score_lookup.get(round_spot(reference_points[left_idx]), default_center_score)
                    base_score -= score_lookup.get(round_spot(reference_points[right_idx]), default_center_score)
                    base_score += float(left_score + right_score)
                    maybe_add_candidate(
                        candidate_sets,
                        seen,
                        points=mutated,
                        center_debug=center_debug,
                        base_center_score=float(base_score),
                        spacing_limit=float(spacing_limit),
                        origin="double_mutation",
                        origin_bonus=1.5,
                    )

        for combo in itertools.combinations(range(len(subset)), int(num_spots)):
            points = [tuple(float(v) for v in point_from_index(subset[idx][1], axes_mm).tolist()) for idx in combo]
            center_score = float(sum(subset[idx][0] for idx in combo))
            maybe_add_candidate(
                candidate_sets,
                seen,
                points=points,
                center_debug=[subset[idx][2] for idx in combo],
                base_center_score=float(center_score),
                spacing_limit=float(spacing_limit),
                origin="global_combo",
                origin_bonus=0.0,
            )
        if candidate_sets:
            candidate_sets.sort(key=lambda row: float(row["heuristic_score"]), reverse=True)
            return candidate_sets[: int(candidate_plan_limit)]

    # Final fallback: keep the best unique centers even if spacing has to be violated.
    fallback_points: List[Tuple[float, float, float]] = []
    fallback_debug: List[Dict[str, object]] = []
    for score, cand, debug in subset:
        point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
        if point in fallback_points:
            continue
        fallback_points.append(point)
        fallback_debug.append(debug)
        if len(fallback_points) >= int(num_spots):
            return [
                {
                    "spots_mm": fallback_points,
                    "heuristic_score": float(sum(float(sc[0]) for sc in subset[: len(fallback_points)])),
                    "center_debug": fallback_debug,
                    "spacing_limit_mm": 0.0,
                    "spacing_violated": True,
                }
            ]
    return []


def build_explicit_candidate(
    spots_mm: Sequence[Tuple[float, float, float]],
    *,
    origin: str,
    heuristic_score: float = 0.0,
    center_debug: Sequence[Dict[str, object]] | None = None,
    layout_debug: Dict[str, object] | None = None,
    plan_overrides: Dict[str, float] | None = None,
) -> Dict[str, object]:
    return {
        "spots_mm": [tuple(float(v) for v in spot) for spot in spots_mm],
        "heuristic_score": float(heuristic_score),
        "center_debug": list(center_debug or []),
        "spacing_limit_mm": 0.0,
        "candidate_origin": str(origin),
        "layout_debug": layout_debug,
        "plan_overrides": {str(k): float(v) for k, v in (plan_overrides or {}).items()},
    }


def plan_override_key(plan_overrides: Mapping[str, object] | None) -> Tuple[Tuple[str, float], ...]:
    if not plan_overrides:
        return ()
    items: List[Tuple[str, float]] = []
    for key, value in plan_overrides.items():
        try:
            items.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(items))


def evaluation_cache_key(
    spots_mm: Sequence[Tuple[float, float, float]],
    plan_overrides: Mapping[str, object] | None = None,
) -> Tuple[Tuple[float, float, float], ...] | Tuple[Tuple[float, float, float], ...]:
    return placement_key(spots_mm), plan_override_key(plan_overrides)


def robust_evaluation_cache_key(
    spots_mm: Sequence[Tuple[float, float, float]],
    plan_overrides: Mapping[str, object] | None,
    seed: int,
    histories: int,
) -> Tuple[Tuple[Tuple[float, float, float], ...] | Tuple[Tuple[float, float, float], ...], int, int]:
    return evaluation_cache_key(spots_mm, plan_overrides), int(seed), int(histories)


def dedupe_candidate_sets(candidate_sets: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    seen: set[Tuple[Tuple[float, float, float], ...] | Tuple[Tuple[Tuple[float, float, float], ...], Tuple[Tuple[str, float], ...]]] = set()
    for row in candidate_sets:
        key = (
            placement_key([tuple(float(v) for v in spot) for spot in row["spots_mm"]]),
            plan_override_key(row.get("plan_overrides")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def tag_candidate_family(candidate_sets: Sequence[Dict[str, object]], family: str) -> List[Dict[str, object]]:
    tagged: List[Dict[str, object]] = []
    for row in candidate_sets:
        updated = dict(row)
        origin = str(updated.get("candidate_origin", "candidate"))
        updated["candidate_origin"] = f"{family}__{origin}"
        layout_debug = dict(updated.get("layout_debug") or {})
        layout_debug["protocol_family"] = str(family)
        updated["layout_debug"] = layout_debug
        tagged.append(updated)
    return tagged


def weighted_sample_candidate_sets(
    candidate_sets: Sequence[Dict[str, object]],
    *,
    rng: np.random.Generator,
    keep_total: int,
) -> List[Dict[str, object]]:
    rows = list(candidate_sets)
    if len(rows) <= int(keep_total):
        return sorted(rows, key=lambda row: float(row.get("heuristic_score", 0.0)), reverse=True)
    rows.sort(key=lambda row: float(row.get("heuristic_score", 0.0)), reverse=True)
    selected = [rows[0]]
    remaining = list(rows[1:])
    while remaining and len(selected) < int(keep_total):
        scores = np.asarray([float(row.get("heuristic_score", 0.0)) for row in remaining], dtype=np.float64)
        scores -= float(np.min(scores))
        weights = scores + 1.0e-3
        weights /= float(np.sum(weights))
        chosen_idx = int(rng.choice(len(remaining), p=weights))
        selected.append(remaining.pop(chosen_idx))
    selected.sort(key=lambda row: float(row.get("heuristic_score", 0.0)), reverse=True)
    return selected


def select_protocol_family_subset(
    candidate_sets: Sequence[Dict[str, object]],
    *,
    family_name: str,
    per_family_limit: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    tagged = tag_candidate_family(dedupe_candidate_sets(candidate_sets), family_name)
    if not tagged:
        return []
    tagged.sort(key=lambda row: float(row.get("heuristic_score", 0.0)), reverse=True)
    keep_limit = max(1, int(per_family_limit))
    if len(tagged) <= keep_limit:
        return tagged
    chosen = [tagged[0]]
    remaining = list(tagged[1:])
    while remaining and len(chosen) < keep_limit:
        scores = np.asarray([float(row.get("heuristic_score", 0.0)) for row in remaining], dtype=np.float64)
        scores -= float(np.min(scores))
        weights = scores + 1.0e-3
        weights /= float(np.sum(weights))
        idx = int(rng.choice(len(remaining), p=weights))
        chosen.append(remaining.pop(idx))
    chosen.sort(key=lambda row: float(row.get("heuristic_score", 0.0)), reverse=True)
    return chosen


def choose_point_farthest_from_set(
    remaining: Sequence[Tuple[float, float, float]],
    chosen: Sequence[Tuple[float, float, float]],
    centroid: np.ndarray,
) -> Tuple[float, float, float]:
    if not chosen:
        return max(
            remaining,
            key=lambda point: float(np.linalg.norm(np.asarray(point, dtype=np.float32) - centroid)),
        )
    return max(
        remaining,
        key=lambda point: min(math.dist(point, other) for other in chosen),
    )


def build_interlaced_groups(
    reference_spots_mm: Sequence[Tuple[float, float, float]],
    vertices_per_fraction: int,
) -> List[List[Tuple[float, float, float]]]:
    spots = [tuple(float(v) for v in spot) for spot in reference_spots_mm]
    if not spots:
        return []
    if int(vertices_per_fraction) >= len(spots):
        return [spots]
    centroid = np.mean(np.asarray(spots, dtype=np.float32), axis=0)
    remaining = list(spots)
    groups: List[List[Tuple[float, float, float]]] = []
    while remaining:
        group: List[Tuple[float, float, float]] = []
        while remaining and len(group) < int(vertices_per_fraction):
            point = choose_point_farthest_from_set(remaining, group, centroid)
            group.append(point)
            remaining.remove(point)
        groups.append(group)
    return groups


def select_farthest_point_subset(
    points: Sequence[Tuple[float, float, float]],
    *,
    max_points: int,
) -> List[Tuple[float, float, float]]:
    if len(points) <= int(max_points):
        return [tuple(float(v) for v in point) for point in points]
    centroid = np.mean(np.asarray(points, dtype=np.float32), axis=0)
    remaining = [tuple(float(v) for v in point) for point in points]
    chosen: List[Tuple[float, float, float]] = []
    while remaining and len(chosen) < int(max_points):
        point = choose_point_farthest_from_set(remaining, chosen, centroid)
        chosen.append(point)
        remaining.remove(point)
    return chosen


def snap_targets_to_candidates(
    targets_mm: Sequence[Tuple[float, float, float]],
    *,
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    min_spacing_mm: float,
    max_snap_mm: float,
) -> Tuple[List[Tuple[float, float, float]], List[Dict[str, object]]] | None:
    chosen: List[Tuple[float, float, float]] = []
    chosen_indices: set[Tuple[int, int, int]] = set()
    debug: List[Dict[str, object]] = []
    for target in targets_mm:
        ranked: List[Tuple[float, Tuple[int, int, int], Tuple[float, float, float]]] = []
        for cand in candidate_indices:
            point = tuple(float(v) for v in point_from_index(cand, axes_mm).tolist())
            dist = math.dist(target, point)
            if dist > float(max_snap_mm):
                continue
            ranked.append((float(dist), cand, point))
        ranked.sort(key=lambda row: row[0])
        accepted = False
        for snap_dist, cand, point in ranked:
            if cand in chosen_indices:
                continue
            if chosen and min(math.dist(point, other) for other in chosen) < float(min_spacing_mm):
                continue
            chosen.append(point)
            chosen_indices.add(cand)
            debug.append(
                {
                    "target_mm": [float(v) for v in target],
                    "candidate_mm": [float(v) for v in point],
                    "snap_distance_mm": float(snap_dist),
                }
            )
            accepted = True
            break
        if not accepted:
            return None
    return chosen, debug


def build_scaled_pitch_candidates(
    *,
    reference_spots_mm: Sequence[Tuple[float, float, float]],
    pitch_scale_factors: Sequence[float],
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    min_spacing_mm: float,
    complement_anchor_mm: Sequence[Tuple[float, float, float]] | None = None,
    candidate_origin_prefix: str = "pitch_scale",
    base_spot_radius_mm: float | None = None,
    base_margin_mm: float | None = None,
    base_history_fraction: float | None = None,
) -> List[Dict[str, object]]:
    if not reference_spots_mm:
        return []
    centroid = np.mean(np.asarray(reference_spots_mm, dtype=np.float32), axis=0)
    candidates: List[Dict[str, object]] = []
    for scale in pitch_scale_factors:
        if float(scale) <= 0.0:
            continue
        targets: List[Tuple[float, float, float]] = []
        for point in reference_spots_mm:
            vec = np.asarray(point, dtype=np.float32) - centroid
            target = centroid + float(scale) * vec
            targets.append(tuple(float(v) for v in target.tolist()))
        snapped = snap_targets_to_candidates(
            targets,
            candidate_indices=candidate_indices,
            axes_mm=axes_mm,
            min_spacing_mm=float(min_spacing_mm),
            max_snap_mm=max(18.0, float(min_spacing_mm) * 1.5),
        )
        if snapped is None:
            continue
        points, debug = snapped
        complement_bonus = 0.0
        if complement_anchor_mm:
            min_dists = [
                min(math.dist(point, anchor) for anchor in complement_anchor_mm)
                for point in points
            ]
            complement_bonus = 0.15 * float(np.mean(min_dists))
        heuristic = float(5.0 * (float(scale) - 1.0) + complement_bonus)
        candidates.append(
            build_explicit_candidate(
                points,
                origin=f"{candidate_origin_prefix}_{float(scale):.2f}",
                heuristic_score=heuristic,
                center_debug=debug,
                layout_debug={
                    "scale_factor": float(scale),
                    "complement_bonus": float(complement_bonus),
                },
            )
        )
        if (
            float(scale) > 1.0
            and base_spot_radius_mm is not None
            and base_margin_mm is not None
            and base_history_fraction is not None
        ):
            coverage_boost = {
                "spot_radius_mm": float(base_spot_radius_mm) * (1.0 + 0.04 * (float(scale) - 1.0)),
                "base_margin_mm": float(base_margin_mm) + 4.0 * (float(scale) - 1.0),
                "base_history_fraction": min(
                    0.985,
                    float(base_history_fraction) + 0.03 + 0.10 * (float(scale) - 1.0),
                ),
            }
            candidates.append(
                build_explicit_candidate(
                    points,
                    origin=f"{candidate_origin_prefix}_{float(scale):.2f}_covboost",
                    heuristic_score=heuristic + 0.15,
                    center_debug=debug,
                    layout_debug={
                        "scale_factor": float(scale),
                        "complement_bonus": float(complement_bonus),
                        "coverage_boost": coverage_boost,
                    },
                    plan_overrides=coverage_boost,
                )
            )
    return dedupe_candidate_sets(candidates)


def build_clinical_grid_targets(
    *,
    centroid_mm: np.ndarray,
    bbox_min_mm: np.ndarray,
    bbox_max_mm: np.ndarray,
    inplane_pitch_mm: float,
    layer_spacing_mm: float,
    x_shift_mm: float,
    y_shift_mm: float,
    z_shift_mm: float,
    margin_mm: float,
) -> List[Tuple[float, float, float]]:
    x_pitch = float(inplane_pitch_mm)
    y_pitch = float(inplane_pitch_mm)
    z_pitch = float(layer_spacing_mm)
    x_min = float(bbox_min_mm[0] - margin_mm)
    x_max = float(bbox_max_mm[0] + margin_mm)
    y_min = float(bbox_min_mm[1] - margin_mm)
    y_max = float(bbox_max_mm[1] + margin_mm)
    z_min = float(bbox_min_mm[2] - margin_mm)
    z_max = float(bbox_max_mm[2] + margin_mm)

    x_base = float(centroid_mm[0] + x_shift_mm)
    y_base = float(centroid_mm[1] + y_shift_mm)
    z_base = float(centroid_mm[2] + z_shift_mm)

    x_indices = range(
        int(math.floor((x_min - x_base) / x_pitch)) - 1,
        int(math.ceil((x_max - x_base) / x_pitch)) + 2,
    )
    y_indices = range(
        int(math.floor((y_min - y_base) / y_pitch)) - 1,
        int(math.ceil((y_max - y_base) / y_pitch)) + 2,
    )
    z_indices = range(
        int(math.floor((z_min - z_base) / z_pitch)) - 1,
        int(math.ceil((z_max - z_base) / z_pitch)) + 2,
    )

    targets: List[Tuple[float, float, float]] = []
    for layer_idx in z_indices:
        z_mm = z_base + float(layer_idx) * z_pitch
        if z_mm < z_min or z_mm > z_max:
            continue
        layer_offset = 0.5 * float(inplane_pitch_mm) if abs(int(layer_idx)) % 2 == 1 else 0.0
        for ix in x_indices:
            x_mm = x_base + float(ix) * x_pitch + layer_offset
            if x_mm < x_min or x_mm > x_max:
                continue
            for iy in y_indices:
                y_mm = y_base + float(iy) * y_pitch + layer_offset
                if y_mm < y_min or y_mm > y_max:
                    continue
                targets.append((float(x_mm), float(y_mm), float(z_mm)))
    return targets


def build_clinical_gtv_core_candidates(
    *,
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    reference_spots_mm: Sequence[Tuple[float, float, float]] | None,
    inplane_pitch_mm: float,
    layer_spacing_mm: float,
    max_snap_mm: float,
    candidate_plan_limit: int,
    requested_num_spots: int,
) -> List[Dict[str, object]]:
    if not candidate_indices:
        return []
    candidate_points = np.asarray(
        [point_from_index(cand, axes_mm) for cand in candidate_indices],
        dtype=np.float32,
    )
    bbox_min = candidate_points.min(axis=0)
    bbox_max = candidate_points.max(axis=0)
    centroid = candidate_points.mean(axis=0)
    reference_points = [tuple(float(v) for v in spot) for spot in reference_spots_mm] if reference_spots_mm else []

    x_shifts = [0.0, float(inplane_pitch_mm) * 0.5]
    y_shifts = [0.0, float(inplane_pitch_mm) * 0.5]
    z_shifts = [0.0, float(layer_spacing_mm) * 0.5]
    candidate_sets: List[Dict[str, object]] = []
    for x_shift in x_shifts:
        for y_shift in y_shifts:
            for z_shift in z_shifts:
                targets = build_clinical_grid_targets(
                    centroid_mm=centroid,
                    bbox_min_mm=bbox_min,
                    bbox_max_mm=bbox_max,
                    inplane_pitch_mm=float(inplane_pitch_mm),
                    layer_spacing_mm=float(layer_spacing_mm),
                    x_shift_mm=float(x_shift),
                    y_shift_mm=float(y_shift),
                    z_shift_mm=float(z_shift),
                    margin_mm=float(max_snap_mm),
                )
                if not targets:
                    continue
                snapped = snap_targets_to_candidates(
                    targets,
                    candidate_indices=candidate_indices,
                    axes_mm=axes_mm,
                    min_spacing_mm=max(18.0, float(layer_spacing_mm) * 0.75),
                    max_snap_mm=float(max_snap_mm),
                )
                if snapped is None:
                    continue
                points, debug = snapped
                unique_points = []
                seen_points: set[Tuple[float, float, float]] = set()
                for point in points:
                    rounded = round_spot(point)
                    if rounded in seen_points:
                        continue
                    seen_points.add(rounded)
                    unique_points.append(tuple(float(v) for v in point))
                if len(unique_points) < 2:
                    continue
                chosen_points = select_farthest_point_subset(
                    unique_points,
                    max_points=max(2, int(requested_num_spots)),
                )
                valid, pairwise_sum, min_pairwise = is_spacing_valid(
                    chosen_points,
                    spacing_limit_mm=max(18.0, float(layer_spacing_mm) * 0.65),
                )
                if not valid and len(chosen_points) < 2:
                    continue
                min_target_dist = min(
                    min(math.dist(point, target) for target in targets)
                    for point in chosen_points
                )
                retained_count = 0
                if reference_points:
                    ref_set = {round_spot(point) for point in reference_points}
                    retained_count = sum(1 for point in chosen_points if round_spot(point) in ref_set)
                heuristic = (
                    18.0 * float(len(chosen_points))
                    + 0.10 * float(pairwise_sum)
                    + 0.25 * float(min_pairwise)
                    + 1.00 * float(retained_count)
                    - 0.40 * float(min_target_dist)
                )
                candidate_sets.append(
                    build_explicit_candidate(
                        chosen_points,
                        origin=f"clinical_grid_x{x_shift:.0f}_y{y_shift:.0f}_z{z_shift:.0f}",
                        heuristic_score=float(heuristic),
                        center_debug=debug,
                        layout_debug={
                            "strategy": "clinical_gtv_core",
                            "target_count": int(len(targets)),
                            "snapped_count": int(len(unique_points)),
                            "selected_count": int(len(chosen_points)),
                            "pairwise_sum": float(pairwise_sum),
                            "min_pairwise": float(min_pairwise),
                            "x_shift_mm": float(x_shift),
                            "y_shift_mm": float(y_shift),
                            "z_shift_mm": float(z_shift),
                        },
                    )
                )
    if not candidate_sets:
        reference_set = {round_spot(point) for point in reference_points}
        candidate_point_list = [tuple(float(v) for v in point.tolist()) for point in candidate_points]
        combo_rows: List[Dict[str, object]] = []
        max_points = max(2, int(requested_num_spots))
        for combo_size in range(max_points, 1, -1):
            for combo in itertools.combinations(candidate_point_list, combo_size):
                valid, pairwise_sum, min_pairwise = is_spacing_valid(combo, spacing_limit_mm=18.0)
                if not valid:
                    continue
                same_layer_match = 0.0
                adjacent_layer_match = 0.0
                retained_count = float(sum(1 for point in combo if round_spot(point) in reference_set))
                z_values = np.asarray([point[2] for point in combo], dtype=np.float32)
                layer_count = int(len({round(float(z) / float(layer_spacing_mm), 1) for z in z_values}))
                for left, right in itertools.combinations(combo, 2):
                    dx = float(left[0] - right[0])
                    dy = float(left[1] - right[1])
                    dz = abs(float(left[2] - right[2]))
                    inplane = math.hypot(dx, dy)
                    if dz <= float(layer_spacing_mm) * 0.40:
                        same_layer_match += math.exp(-((inplane - float(inplane_pitch_mm)) / 18.0) ** 2)
                    else:
                        adjacent_layer_match += (
                            math.exp(-((dz - float(layer_spacing_mm)) / 10.0) ** 2)
                            * math.exp(-((inplane - float(inplane_pitch_mm) / math.sqrt(2.0)) / 20.0) ** 2)
                        )
                heuristic = (
                    16.0 * float(combo_size)
                    + 0.12 * float(pairwise_sum)
                    + 0.40 * float(min_pairwise)
                    + 8.0 * float(same_layer_match)
                    + 6.0 * float(adjacent_layer_match)
                    + 1.5 * float(layer_count)
                    + 1.0 * float(retained_count)
                )
                combo_rows.append(
                    build_explicit_candidate(
                        combo,
                        origin=f"clinical_best_feasible_{combo_size}v",
                        heuristic_score=float(heuristic),
                        center_debug=[{"candidate_mm": [float(a), float(b), float(c)]} for a, b, c in combo],
                        layout_debug={
                            "strategy": "clinical_gtv_core",
                            "mode": "best_feasible_combo",
                            "combo_size": int(combo_size),
                            "pairwise_sum": float(pairwise_sum),
                            "min_pairwise": float(min_pairwise),
                            "same_layer_match": float(same_layer_match),
                            "adjacent_layer_match": float(adjacent_layer_match),
                            "layer_count": int(layer_count),
                            "retained_count": float(retained_count),
                        },
                    )
                )
        combo_rows.sort(key=lambda row: float(row["heuristic_score"]), reverse=True)
        candidate_sets = combo_rows[: max(1, int(candidate_plan_limit))]
    boosted_sets: List[Dict[str, object]] = []
    for row in candidate_sets:
        boosted_sets.append(row)
        boosted_sets.append(
            build_explicit_candidate(
                row["spots_mm"],
                origin=f"{row['candidate_origin']}_covboost",
                heuristic_score=float(row["heuristic_score"]) + 0.2,
                center_debug=row.get("center_debug"),
                layout_debug={
                    **dict(row.get("layout_debug") or {}),
                    "coverage_boost": {
                        "base_margin_mm": 12.0,
                        "base_history_fraction": 0.985,
                        "spot_radius_mm_scale": 1.05,
                    },
                },
                plan_overrides={
                    "base_margin_mm": 12.0,
                    "base_history_fraction": 0.985,
                    "spot_radius_mm": 7.9,
                },
            )
        )
    return dedupe_candidate_sets(boosted_sets)[: max(1, int(candidate_plan_limit))]


def rerank_scored_centers_for_vascular_sink(
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
) -> List[Tuple[float, Tuple[int, int, int], Dict[str, object]]]:
    reranked: List[Tuple[float, Tuple[int, int, int], Dict[str, object]]] = []
    for _, cand, debug in scored_centers:
        need_score = float(debug.get("need_score", 0.0))
        hypoxia_bonus = float(debug.get("hypoxia_bonus", 0.0))
        vessel_bonus = float(debug.get("vessel_bonus", 0.0))
        sink_bonus = float(debug.get("sink_bonus", 0.0))
        distance_score = float(debug.get("distance_score", 0.0))
        history_penalty = float(debug.get("history_penalty", 0.0))
        vascular_score = (
            0.90 * need_score
            + 1.10 * hypoxia_bonus
            + 2.80 * vessel_bonus
            + 3.20 * sink_bonus
            + 0.40 * distance_score
            - 0.75 * history_penalty
        )
        reranked_debug = dict(debug)
        reranked_debug["vascular_strategy_score"] = float(vascular_score)
        reranked.append((float(vascular_score), cand, reranked_debug))
    reranked.sort(key=lambda row: row[0], reverse=True)
    return reranked


def build_vascular_sink_candidate_sets(
    *,
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
    axes_mm: Dict[str, np.ndarray],
    num_spots: int,
    min_spacing_mm: float,
    top_k_centers: int,
    candidate_plan_limit: int,
    reference_spots_mm: Sequence[Tuple[float, float, float]],
    candidate_multiplier: int,
) -> List[Dict[str, object]]:
    reranked = rerank_scored_centers_for_vascular_sink(scored_centers)
    vessel_candidates = build_candidate_spot_sets(
        reranked,
        axes_mm=axes_mm,
        num_spots=int(num_spots),
        min_spacing_mm=float(min_spacing_mm),
        top_k_centers=max(int(top_k_centers), int(top_k_centers) * max(1, int(candidate_multiplier))),
        candidate_plan_limit=max(int(candidate_plan_limit), int(candidate_plan_limit) * max(1, int(candidate_multiplier))),
        reference_spots_mm=reference_spots_mm,
    )
    prefixed: List[Dict[str, object]] = []
    for row in vessel_candidates:
        updated = dict(row)
        updated["candidate_origin"] = f"vascular_{str(row.get('candidate_origin', 'candidate'))}"
        layout_debug = dict(updated.get("layout_debug") or {})
        layout_debug["strategy"] = "vascular_sink_hugging"
        updated["layout_debug"] = layout_debug
        prefixed.append(updated)
    return dedupe_candidate_sets(prefixed)[: max(int(candidate_plan_limit), 1)]


def build_phase24_joint_template_candidates(
    *,
    args: argparse.Namespace,
    fraction_idx: int,
    baseline_spots: Sequence[Tuple[float, float, float]],
    current_repeat_spots: Sequence[Tuple[float, float, float]],
    strategy_state: Dict[str, object],
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
    axes_mm: Dict[str, np.ndarray],
    candidate_indices: Sequence[Tuple[int, int, int]],
    min_spacing_mm: float,
    top_k_centers: int,
    candidate_plan_limit: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(args.seed) + 7001 * int(fraction_idx))
    reference_spots = list(current_repeat_spots) if current_repeat_spots else list(baseline_spots)
    per_family_limit = max(1, int(args.protocol_mix_per_family))
    min_vertices = max(2, int(args.phase24_min_vertices))
    max_vertices = max(min_vertices, int(args.phase24_max_vertices))
    spacing_mm = min(float(min_spacing_mm), float(args.phase24_soft_min_spacing_mm))
    pitch_scales = [float(scale) for scale in args.phase24_template_pitch_scales]
    if not pitch_scales:
        pitch_scales = [1.0]

    family_rows: List[Dict[str, object]] = []
    for vertex_count in range(min_vertices, max_vertices + 1):
        adaptive_rows = build_candidate_spot_sets(
            scored_centers,
            axes_mm=axes_mm,
            num_spots=int(vertex_count),
            min_spacing_mm=float(spacing_mm),
            top_k_centers=max(int(top_k_centers), 12),
            candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 3),
            reference_spots_mm=reference_spots,
        )
        family_rows.extend(
            select_protocol_family_subset(
                adaptive_rows,
                family_name=f"phase24_adaptive_v{int(vertex_count)}",
                per_family_limit=per_family_limit,
                rng=rng,
            )
        )

        for pitch_scale in pitch_scales:
            clinical_rows = build_clinical_gtv_core_candidates(
                candidate_indices=candidate_indices,
                axes_mm=axes_mm,
                reference_spots_mm=reference_spots,
                inplane_pitch_mm=float(args.clinical_inplane_pitch_mm) * float(pitch_scale),
                layer_spacing_mm=float(args.clinical_layer_spacing_mm) * float(max(0.85, pitch_scale)),
                max_snap_mm=float(args.clinical_grid_max_snap_mm) * 1.15,
                candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 3),
                requested_num_spots=int(vertex_count),
            )
            family_rows.extend(
                select_protocol_family_subset(
                    clinical_rows,
                    family_name=f"phase24_clinical_v{int(vertex_count)}_p{float(pitch_scale):.2f}",
                    per_family_limit=max(1, min(per_family_limit, 2)),
                    rng=rng,
                )
            )

    mixed_rows = dedupe_candidate_sets(family_rows)
    if not mixed_rows:
        return []
    return weighted_sample_candidate_sets(
        mixed_rows,
        rng=rng,
        keep_total=max(1, int(candidate_plan_limit)),
    )


def build_randomized_protocol_mix_candidates(
    *,
    args: argparse.Namespace,
    fraction_idx: int,
    baseline_spots: Sequence[Tuple[float, float, float]],
    current_repeat_spots: Sequence[Tuple[float, float, float]],
    strategy_state: Dict[str, object],
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
    axes_mm: Dict[str, np.ndarray],
    candidate_indices: Sequence[Tuple[int, int, int]],
    num_spots: int,
    min_spacing_mm: float,
    top_k_centers: int,
    candidate_plan_limit: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(args.seed) + 1009 * int(fraction_idx))
    reference_spots = list(current_repeat_spots) if current_repeat_spots else list(baseline_spots)
    per_family_limit = max(1, int(args.protocol_mix_per_family))
    family_rows: List[Dict[str, object]] = []

    adaptive_rows = build_candidate_spot_sets(
        scored_centers,
        axes_mm=axes_mm,
        num_spots=int(num_spots),
        min_spacing_mm=float(min_spacing_mm),
        top_k_centers=int(top_k_centers),
        candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 2),
        reference_spots_mm=reference_spots,
    )
    family_rows.extend(
        select_protocol_family_subset(
            adaptive_rows,
            family_name="adaptive_mutation",
            per_family_limit=per_family_limit,
            rng=rng,
        )
    )

    clinical_rows = build_clinical_gtv_core_candidates(
        candidate_indices=candidate_indices,
        axes_mm=axes_mm,
        reference_spots_mm=reference_spots,
        inplane_pitch_mm=float(args.clinical_inplane_pitch_mm),
        layer_spacing_mm=float(args.clinical_layer_spacing_mm),
        max_snap_mm=float(args.clinical_grid_max_snap_mm),
        candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 2),
        requested_num_spots=int(num_spots),
    )
    family_rows.extend(
        select_protocol_family_subset(
            clinical_rows,
            family_name="clinical_gtv_core",
            per_family_limit=per_family_limit,
            rng=rng,
        )
    )

    if reference_spots:
        pitch_rows = build_scaled_pitch_candidates(
            reference_spots_mm=reference_spots,
            pitch_scale_factors=[1.0] + [float(factor) for factor in args.pitch_scale_factors],
            candidate_indices=candidate_indices,
            axes_mm=axes_mm,
            min_spacing_mm=float(min_spacing_mm),
            candidate_origin_prefix="pitch_sweep",
            base_spot_radius_mm=float(args.spot_radius_mm),
            base_margin_mm=float(args.base_margin_mm),
            base_history_fraction=float(args.base_history_fraction),
        )
        family_rows.extend(
            select_protocol_family_subset(
                pitch_rows,
                family_name="larger_pitch_sweep",
                per_family_limit=per_family_limit,
                rng=rng,
            )
        )

        alternating_pattern = [tuple(float(v) for v in spot) for spot in strategy_state.get("pattern_a", reference_spots)]
        alternating_rows = dedupe_candidate_sets(
            build_candidate_spot_sets(
                scored_centers,
                axes_mm=axes_mm,
                num_spots=int(num_spots),
                min_spacing_mm=float(min_spacing_mm),
                top_k_centers=int(top_k_centers),
                candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 2),
                reference_spots_mm=alternating_pattern,
            )
            + build_scaled_pitch_candidates(
                reference_spots_mm=alternating_pattern,
                pitch_scale_factors=[factor for factor in args.pitch_scale_factors if float(factor) > 1.0],
                candidate_indices=candidate_indices,
                axes_mm=axes_mm,
                min_spacing_mm=float(min_spacing_mm),
                complement_anchor_mm=alternating_pattern,
                candidate_origin_prefix="alternate_pitch",
                base_spot_radius_mm=float(args.spot_radius_mm),
                base_margin_mm=float(args.base_margin_mm),
                base_history_fraction=float(args.base_history_fraction),
            )
        )
        family_rows.extend(
            select_protocol_family_subset(
                alternating_rows,
                family_name="alternating_two_pattern",
                per_family_limit=per_family_limit,
                rng=rng,
            )
        )

    vascular_rows = build_vascular_sink_candidate_sets(
        scored_centers=scored_centers,
        axes_mm=axes_mm,
        num_spots=int(num_spots),
        min_spacing_mm=float(min_spacing_mm),
        top_k_centers=int(top_k_centers),
        candidate_plan_limit=max(int(candidate_plan_limit), per_family_limit * 2),
        reference_spots_mm=reference_spots,
        candidate_multiplier=int(args.vascular_candidate_multiplier),
    )
    family_rows.extend(
        select_protocol_family_subset(
            vascular_rows,
            family_name="vascular_sink_hugging",
            per_family_limit=per_family_limit,
            rng=rng,
        )
    )

    interlaced_groups = strategy_state.get("interlaced_groups")
    if not interlaced_groups:
        interlaced_groups = build_interlaced_groups(baseline_spots if baseline_spots else reference_spots, int(args.interlaced_vertices_per_fraction))
        strategy_state["interlaced_groups"] = interlaced_groups
    interlaced_rows: List[Dict[str, object]] = []
    for group_idx, group in enumerate(interlaced_groups or [], start=1):
        interlaced_rows.append(
            build_explicit_candidate(
                [tuple(float(v) for v in spot) for spot in group],
                origin=f"interlaced_group_{group_idx}",
                heuristic_score=1.0 + 0.1 * float(len(group)),
                center_debug=[
                    {
                        "note": "protocol-mix reduced-vertices interlaced subset",
                        "group_index": int(group_idx),
                        "group_size": int(len(group)),
                    }
                ],
                layout_debug={
                    "strategy": "reduced_vertices_interlaced",
                    "group_index": int(group_idx),
                    "group_size": int(len(group)),
                },
            )
        )
    family_rows.extend(
        select_protocol_family_subset(
            interlaced_rows,
            family_name="reduced_vertices_interlaced",
            per_family_limit=per_family_limit,
            rng=rng,
        )
    )

    mixed_rows = dedupe_candidate_sets(family_rows)
    if not mixed_rows:
        return []
    return weighted_sample_candidate_sets(
        mixed_rows,
        rng=rng,
        keep_total=max(1, int(candidate_plan_limit)),
    )


def build_strategy_candidate_sets(
    *,
    args: argparse.Namespace,
    fraction_idx: int,
    baseline_spots: Sequence[Tuple[float, float, float]],
    current_repeat_spots: Sequence[Tuple[float, float, float]],
    strategy_state: Dict[str, object],
    scored_centers: Sequence[Tuple[float, Tuple[int, int, int], Dict[str, object]]],
    adaptive_state: Mapping[str, object] | None,
    axes_mm: Dict[str, np.ndarray],
    candidate_indices: Sequence[Tuple[int, int, int]],
    num_spots: int,
    min_spacing_mm: float,
    top_k_centers: int,
    candidate_plan_limit: int,
) -> List[Dict[str, object]]:
    strategy = str(args.course_strategy)
    if strategy == "phase24_joint_horizon_diversity":
        mixed_rows = build_phase24_joint_template_candidates(
            args=args,
            fraction_idx=int(fraction_idx),
            baseline_spots=baseline_spots,
            current_repeat_spots=current_repeat_spots,
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            axes_mm=axes_mm,
            candidate_indices=candidate_indices,
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=max(int(candidate_plan_limit), int(candidate_plan_limit) * 2),
        )
        if adaptive_state is not None:
            mixed_rows, _ = filter_candidate_sets_by_phase22_veto(
                mixed_rows,
                args=args,
                adaptive_state=adaptive_state,
                min_keep=max(2, int(args.phase22_no_fly_min_keep_candidate_sets)),
            )
        mixed_rows = expand_candidate_sets_with_phase22_weight_optimization(
            mixed_rows,
            args=args,
            adaptive_state=adaptive_state,
        )
        return weighted_sample_candidate_sets(
            dedupe_candidate_sets(mixed_rows),
            rng=np.random.default_rng(int(args.seed) + 6007 * int(fraction_idx)),
            keep_total=max(1, int(candidate_plan_limit)),
        )

    if strategy == "adaptive_nofly_weight_opt":
        mixed_rows = build_randomized_protocol_mix_candidates(
            args=args,
            fraction_idx=int(fraction_idx),
            baseline_spots=baseline_spots,
            current_repeat_spots=current_repeat_spots,
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            axes_mm=axes_mm,
            candidate_indices=candidate_indices,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=max(int(candidate_plan_limit), int(candidate_plan_limit) * 2),
        )
        if adaptive_state is not None:
            mixed_rows, _ = filter_candidate_sets_by_phase22_veto(
                mixed_rows,
                args=args,
                adaptive_state=adaptive_state,
                min_keep=max(2, int(args.phase22_no_fly_min_keep_candidate_sets)),
            )
        mixed_rows = expand_candidate_sets_with_phase22_weight_optimization(
            mixed_rows,
            args=args,
            adaptive_state=adaptive_state,
        )
        return weighted_sample_candidate_sets(
            dedupe_candidate_sets(mixed_rows),
            rng=np.random.default_rng(int(args.seed) + 5003 * int(fraction_idx)),
            keep_total=max(1, int(candidate_plan_limit)),
        )

    if strategy == "adaptive_hotspot_avoidance":
        adaptive_rows = build_randomized_protocol_mix_candidates(
            args=args,
            fraction_idx=int(fraction_idx),
            baseline_spots=baseline_spots,
            current_repeat_spots=current_repeat_spots,
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            axes_mm=axes_mm,
            candidate_indices=candidate_indices,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=int(candidate_plan_limit),
        )
        filtered_rows, _ = filter_candidate_sets_by_adaptive_memory(
            adaptive_rows,
            reference_spots_mm=current_repeat_spots,
            spot_memory_radius_mm=float(args.adaptive_spot_memory_radius_mm),
            min_keep=max(2, min(int(candidate_plan_limit), 4)),
        )
        return filtered_rows

    if strategy == "adaptive_mutation":
        return build_candidate_spot_sets(
            scored_centers,
            axes_mm=axes_mm,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=int(candidate_plan_limit),
            reference_spots_mm=current_repeat_spots,
        )

    if strategy == "alternating_two_pattern":
        pattern_a = [tuple(float(v) for v in spot) for spot in strategy_state.get("pattern_a", baseline_spots)]
        pattern_b = strategy_state.get("pattern_b")
        if int(fraction_idx) % 2 == 1:
            return [
                build_explicit_candidate(
                    pattern_a,
                    origin="alternate_pattern_a",
                    heuristic_score=2.0,
                    center_debug=[{"note": "fixed alternating pattern A"}],
                )
            ]
        if pattern_b is not None:
            return [
                build_explicit_candidate(
                    [tuple(float(v) for v in spot) for spot in pattern_b],
                    origin="alternate_pattern_b_locked",
                    heuristic_score=2.5,
                    center_debug=[{"note": "locked alternating pattern B"}],
                )
            ]
        adaptive = build_candidate_spot_sets(
            scored_centers,
            axes_mm=axes_mm,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=max(int(candidate_plan_limit), 6),
            reference_spots_mm=pattern_a,
        )
        pitch = build_scaled_pitch_candidates(
            reference_spots_mm=pattern_a,
            pitch_scale_factors=[factor for factor in args.pitch_scale_factors if float(factor) > 1.0],
            candidate_indices=candidate_indices,
            axes_mm=axes_mm,
            min_spacing_mm=float(min_spacing_mm),
            complement_anchor_mm=pattern_a,
            candidate_origin_prefix="alternate_pitch",
            base_spot_radius_mm=float(args.spot_radius_mm),
            base_margin_mm=float(args.base_margin_mm),
            base_history_fraction=float(args.base_history_fraction),
        )
        return dedupe_candidate_sets(adaptive + pitch)[: int(candidate_plan_limit)]

    if strategy == "clinical_gtv_core":
        return build_clinical_gtv_core_candidates(
            candidate_indices=candidate_indices,
            axes_mm=axes_mm,
            reference_spots_mm=current_repeat_spots if current_repeat_spots else baseline_spots,
            inplane_pitch_mm=float(args.clinical_inplane_pitch_mm),
            layer_spacing_mm=float(args.clinical_layer_spacing_mm),
            max_snap_mm=float(args.clinical_grid_max_snap_mm),
            candidate_plan_limit=int(candidate_plan_limit),
            requested_num_spots=int(args.num_spots),
        )

    if strategy == "larger_pitch_sweep":
        return build_scaled_pitch_candidates(
            reference_spots_mm=current_repeat_spots,
            pitch_scale_factors=[1.0] + [float(factor) for factor in args.pitch_scale_factors],
            candidate_indices=candidate_indices,
            axes_mm=axes_mm,
            min_spacing_mm=float(min_spacing_mm),
            candidate_origin_prefix="pitch_sweep",
            base_spot_radius_mm=float(args.spot_radius_mm),
            base_margin_mm=float(args.base_margin_mm),
            base_history_fraction=float(args.base_history_fraction),
        )[: int(candidate_plan_limit)]

    if strategy == "randomized_protocol_mix":
        return build_randomized_protocol_mix_candidates(
            args=args,
            fraction_idx=int(fraction_idx),
            baseline_spots=baseline_spots,
            current_repeat_spots=current_repeat_spots,
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            axes_mm=axes_mm,
            candidate_indices=candidate_indices,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=int(candidate_plan_limit),
        )

    if strategy == "reduced_vertices_interlaced":
        groups = strategy_state.get("interlaced_groups")
        if not groups:
            groups = build_interlaced_groups(baseline_spots, int(args.interlaced_vertices_per_fraction))
            strategy_state["interlaced_groups"] = groups
        if not groups:
            return []
        group_idx = (int(fraction_idx) - 1) % len(groups)
        return [
            build_explicit_candidate(
                [tuple(float(v) for v in spot) for spot in groups[group_idx]],
                origin=f"interlaced_group_{group_idx + 1}",
                heuristic_score=1.0,
                center_debug=[
                    {
                        "note": "reduced-vertices interlaced subset",
                        "group_index": int(group_idx + 1),
                        "group_size": int(len(groups[group_idx])),
                    }
                ],
            )
        ]

    if strategy == "vascular_sink_hugging":
        return build_vascular_sink_candidate_sets(
            scored_centers=scored_centers,
            axes_mm=axes_mm,
            num_spots=int(num_spots),
            min_spacing_mm=float(min_spacing_mm),
            top_k_centers=int(top_k_centers),
            candidate_plan_limit=int(candidate_plan_limit),
            reference_spots_mm=current_repeat_spots,
            candidate_multiplier=int(args.vascular_candidate_multiplier),
        )

    raise ValueError(f"Unsupported course strategy: {strategy}")


def compute_phase24_diversity_bonus(
    *,
    args: argparse.Namespace,
    candidate_spots_mm: Sequence[Tuple[float, float, float]],
    accepted_sequence: Sequence[Dict[str, object]],
) -> Tuple[float, Dict[str, float]]:
    delivered_spots_mm = [tuple(float(v) for v in spot) for result in accepted_sequence for spot in result["spot_centers_mm"]]
    if not delivered_spots_mm or not candidate_spots_mm:
        return 0.0, {
            "mean_distance_mm": 0.0,
            "normalized_distance": 0.0,
            "num_prior_spots": int(len(delivered_spots_mm)),
        }
    nearest_distances = [
        min(math.dist(tuple(float(v) for v in spot), prior) for prior in delivered_spots_mm)
        for spot in candidate_spots_mm
    ]
    mean_distance = float(np.mean(np.asarray(nearest_distances, dtype=np.float64)))
    normalized_distance = min(1.0, mean_distance / max(1.0, float(args.phase24_diversity_cap_mm)))
    bonus = float(args.phase24_diversity_weight) * normalized_distance
    return bonus, {
        "mean_distance_mm": float(mean_distance),
        "normalized_distance": float(normalized_distance),
        "num_prior_spots": int(len(delivered_spots_mm)),
    }


def estimate_phase24_future_space_bonus(
    *,
    args: argparse.Namespace,
    trial_summary: Dict[str, object],
    trial_sequence: Sequence[Dict[str, object]],
    trial_spots_mm: Sequence[Tuple[float, float, float]],
    baseline_spots: Sequence[Tuple[float, float, float]],
    strategy_state: Dict[str, object],
    safe_candidate_indices: Sequence[Tuple[int, int, int]],
    candidate_indices: Sequence[Tuple[int, int, int]],
    axes_mm: Dict[str, np.ndarray],
    structures: Dict[str, np.ndarray],
    uptake_tensor: np.ndarray,
    structure_points_mm: Mapping[str, np.ndarray],
    vessel_coords_mm: np.ndarray,
    history_counts: Mapping[Tuple[float, float, float], int],
) -> Tuple[float, Dict[str, object]]:
    remaining_fractions = max(0, int(args.fractions) - int(len(trial_sequence)))
    if remaining_fractions <= 0:
        return 0.0, {
            "remaining_fractions": 0,
            "future_center_count": 0,
            "future_candidate_count": 0,
            "normalized_center_score": 0.0,
            "normalized_candidate_count": 0.0,
        }

    next_fraction_idx = int(len(trial_sequence) + 1)
    temp_history_counts: Dict[Tuple[float, float, float], int] = dict(history_counts)
    for spot in trial_spots_mm:
        rounded = round_spot(spot)
        temp_history_counts[rounded] = temp_history_counts.get(rounded, 0) + 1

    next_candidate_indices = augment_candidate_pool_with_reference_neighborhood(
        base_candidate_indices=safe_candidate_indices,
        all_candidate_indices=candidate_indices,
        reference_spots_mm=trial_spots_mm,
        axes_mm=axes_mm,
        radius_mm=max(float(args.min_spot_spacing_mm) * 1.2, 18.0),
        per_spot_limit=10,
    )
    adaptive_state: Dict[str, object] | None = build_phase22_adaptive_state(
        args=args,
        structures=structures,
        axes_mm=axes_mm,
        structure_points_mm=structure_points_mm,
        course_summary=trial_summary,
        accepted_sequence=trial_sequence,
    )
    next_candidate_indices, filter_debug = filter_candidate_indices_by_phase22_veto(
        args=args,
        candidate_indices=next_candidate_indices,
        axes_mm=axes_mm,
        adaptive_state=adaptive_state,
        min_pool_size=max(int(args.phase22_no_fly_min_filter_pool_size), int(args.phase24_min_vertices) * 2),
    )
    if not next_candidate_indices:
        return 0.0, {
            "remaining_fractions": int(remaining_fractions),
            "future_center_count": 0,
            "future_candidate_count": 0,
            "normalized_center_score": 0.0,
            "normalized_candidate_count": 0.0,
            "adaptive_filter_debug": filter_debug,
        }

    oar_weights, _ = compute_guidance_oar_weights(trial_summary["cumulative_effective_metrics"], args)
    next_scored_centers = score_candidate_centers(
        current_cumulative_effective_dose=trial_summary["cumulative_effective_dose"],
        current_cumulative_physical_dose=trial_summary["cumulative_physical_dose"],
        current_fraction_idx=next_fraction_idx,
        structures=structures,
        axes_mm=axes_mm,
        uptake_tensor=uptake_tensor,
        candidate_indices=next_candidate_indices,
        target_effective_gy_per_fraction=float(args.target_effective_gy_per_fraction),
        target_need_cap_gy=float(args.target_need_cap_gy),
        oar_weights=oar_weights,
        structure_points_mm=structure_points_mm,
        vessel_coords_mm=vessel_coords_mm,
        history_counts=temp_history_counts,
        adaptive_avoidance=adaptive_state,
    )
    top_scores = [max(0.0, float(score)) for score, _, _ in next_scored_centers[: max(1, int(args.phase24_future_center_count))]]
    mean_top_score = float(np.mean(np.asarray(top_scores, dtype=np.float64))) if top_scores else 0.0
    normalized_center_score = min(1.0, mean_top_score / max(1.0, float(args.phase24_future_center_score_cap)))

    next_candidate_sets = build_phase24_joint_template_candidates(
        args=args,
        fraction_idx=next_fraction_idx,
        baseline_spots=baseline_spots,
        current_repeat_spots=trial_spots_mm,
        strategy_state=dict(strategy_state),
        scored_centers=next_scored_centers,
        axes_mm=axes_mm,
        candidate_indices=next_candidate_indices,
        min_spacing_mm=float(args.phase24_soft_min_spacing_mm),
        top_k_centers=int(args.candidate_top_k_centers),
        candidate_plan_limit=max(2, min(int(args.candidate_plan_limit), int(args.phase24_future_candidate_target))),
    )
    if adaptive_state is not None and next_candidate_sets:
        next_candidate_sets, _ = filter_candidate_sets_by_phase22_veto(
            next_candidate_sets,
            args=args,
            adaptive_state=adaptive_state,
            min_keep=max(2, int(args.phase22_no_fly_min_keep_candidate_sets)),
        )
    normalized_candidate_count = min(
        1.0,
        float(len(next_candidate_sets)) / max(1.0, float(args.phase24_future_candidate_target)),
    )
    remaining_ratio = float(remaining_fractions) / max(1.0, float(max(1, int(args.fractions) - 1)))
    bonus = float(args.phase24_lookahead_weight) * remaining_ratio * (
        0.65 * normalized_center_score + 0.35 * normalized_candidate_count
    )
    return bonus, {
        "remaining_fractions": int(remaining_fractions),
        "future_center_count": int(len(next_scored_centers)),
        "future_candidate_count": int(len(next_candidate_sets)),
        "mean_top_future_center_score": float(mean_top_score),
        "normalized_center_score": float(normalized_center_score),
        "normalized_candidate_count": float(normalized_candidate_count),
        "remaining_ratio": float(remaining_ratio),
        "adaptive_filter_debug": filter_debug,
    }


def spherical_union_mask(
    axes_mm: Dict[str, np.ndarray],
    centers_mm: Sequence[Tuple[float, float, float]],
    radius_mm: float,
) -> np.ndarray:
    x_mm = axes_mm["x"]
    y_mm = axes_mm["y"]
    z_mm = axes_mm["z"]
    mask = np.zeros((x_mm.size, y_mm.size, z_mm.size), dtype=bool)
    margin = float(radius_mm) + max(float(x_mm[1] - x_mm[0]), float(y_mm[1] - y_mm[0]), float(z_mm[1] - z_mm[0]))
    radius2 = float(radius_mm) ** 2
    for cx, cy, cz in centers_mm:
        ix = np.flatnonzero((x_mm >= float(cx) - margin) & (x_mm <= float(cx) + margin))
        iy = np.flatnonzero((y_mm >= float(cy) - margin) & (y_mm <= float(cy) + margin))
        iz = np.flatnonzero((z_mm >= float(cz) - margin) & (z_mm <= float(cz) + margin))
        if ix.size == 0 or iy.size == 0 or iz.size == 0:
            continue
        xx = x_mm[ix][:, None, None]
        yy = y_mm[iy][None, :, None]
        zz = z_mm[iz][None, None, :]
        local = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 + (zz - float(cz)) ** 2) <= radius2
        mask[np.ix_(ix, iy, iz)] |= local
    return mask


def build_peak_valley_rois(
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    delivered_spots_mm: Sequence[Tuple[float, float, float]],
    *,
    peak_radius_mm: float,
    valley_exclusion_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ptv_mask = structures["PTV"]
    peak_mask = spherical_union_mask(axes_mm, delivered_spots_mm, float(peak_radius_mm)) & ptv_mask
    valley_mask = ptv_mask & ~spherical_union_mask(axes_mm, delivered_spots_mm, float(valley_exclusion_radius_mm))
    if int(np.count_nonzero(valley_mask)) == 0:
        relaxed_radius = max(float(peak_radius_mm) * 1.1, float(peak_radius_mm) + 1.0)
        valley_mask = ptv_mask & ~spherical_union_mask(axes_mm, delivered_spots_mm, relaxed_radius)
    if int(np.count_nonzero(valley_mask)) == 0:
        valley_mask = ptv_mask & ~peak_mask
    if int(np.count_nonzero(peak_mask)) == 0 or int(np.count_nonzero(valley_mask)) == 0:
        raise RuntimeError("Peak or valley ROI became empty while computing the cumulative PVDR metric.")
    return peak_mask, valley_mask


def build_spill_region_masks(
    *,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    peak_mask: np.ndarray,
    shell_1_mm: float,
    shell_2_mm: float,
    shell_3_mm: float,
    oar_adjacent_mm: float,
) -> Dict[str, np.ndarray]:
    gtv_mask = np.asarray(structures["GTV"], dtype=bool)
    ptv_mask = np.asarray(structures["PTV"], dtype=bool)
    body_mask = np.asarray(structures["BODY"], dtype=bool)
    outside_gtv = body_mask & ~gtv_mask

    distance_to_gtv_mm = ndimage.distance_transform_edt(~gtv_mask, sampling=voxel_size_mm_from_axes(axes_mm))
    shell_0_5 = outside_gtv & (distance_to_gtv_mm > 0.0) & (distance_to_gtv_mm <= float(shell_1_mm))
    shell_5_15 = outside_gtv & (distance_to_gtv_mm > float(shell_1_mm)) & (distance_to_gtv_mm <= float(shell_2_mm))
    shell_15_30 = outside_gtv & (distance_to_gtv_mm > float(shell_2_mm)) & (distance_to_gtv_mm <= float(shell_3_mm))

    critical_oar_union = (
        np.asarray(structures["SPINAL_CORD"], dtype=bool)
        | np.asarray(structures["BRAINSTEM"], dtype=bool)
        | np.asarray(structures["PAROTID_R"], dtype=bool)
        | np.asarray(structures["PAROTID_L"], dtype=bool)
        | np.asarray(structures["THYROID"], dtype=bool)
        | np.asarray(structures["PARATHYROIDS"], dtype=bool)
    )
    distance_to_oar_mm = ndimage.distance_transform_edt(~critical_oar_union, sampling=voxel_size_mm_from_axes(axes_mm))
    oar_adjacent_outside_gtv = outside_gtv & (distance_to_oar_mm <= float(oar_adjacent_mm))

    ptv_valley_outside_gtv = (ptv_mask & ~gtv_mask) & ~np.asarray(peak_mask, dtype=bool)
    return {
        "SPILL_SHELL_0_5": shell_0_5,
        "SPILL_SHELL_5_15": shell_5_15,
        "SPILL_SHELL_15_30": shell_15_30,
        "OUTSIDE_GTV": outside_gtv,
        "OAR_ADJACENT_OUTSIDE_GTV": oar_adjacent_outside_gtv,
        "PTV_VALLEY_OUTSIDE_GTV": ptv_valley_outside_gtv,
    }


def compute_region_metrics(
    dose: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
    *,
    voxel_volume_cc: float,
    volume_thresholds_gy: Sequence[float] = (2.0, 5.0, 10.0, 20.0),
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, mask in region_masks.items():
        if int(np.count_nonzero(mask)) == 0:
            continue
        metrics[name] = compute_structure_metrics(
            dose,
            np.asarray(mask, dtype=bool),
            prescription_gy=None,
            voxel_volume_cc=float(voxel_volume_cc),
            volume_thresholds_gy=volume_thresholds_gy,
        )
    return metrics


def compute_peak_valley_metrics(dose: np.ndarray, peak_mask: np.ndarray, valley_mask: np.ndarray) -> Dict[str, float]:
    peak_values = np.asarray(dose[peak_mask], dtype=np.float64)
    valley_values = np.asarray(dose[valley_mask], dtype=np.float64)
    peak_mean = float(np.mean(peak_values))
    valley_mean = float(np.mean(valley_values))
    peak_p90 = float(np.percentile(peak_values, 90.0))
    valley_p10 = float(np.percentile(valley_values, 10.0))
    pvdr = float(peak_mean / max(valley_mean, 1e-6))
    return {
        "peak_mean_gy": peak_mean,
        "valley_mean_gy": valley_mean,
        "peak_p90_gy": peak_p90,
        "valley_p10_gy": valley_p10,
        "pvdr": pvdr,
        "peak_voxels": int(peak_values.size),
        "valley_voxels": int(valley_values.size),
    }


def compute_fraction_consistency_penalty(sequence_results: Sequence[Dict[str, object]]) -> Tuple[float, Dict[str, float]]:
    tracked = {
        "PTV_D95_eff_cv": [float(row["effective_metrics"]["PTV"]["d95_gy"]) for row in sequence_results],
        "CORD_D2_eff_cv": [float(row["effective_metrics"]["SPINAL_CORD"]["d2_gy"]) for row in sequence_results],
        "PAROTID_R_mean_eff_cv": [float(row["effective_metrics"]["PAROTID_R"]["mean_gy"]) for row in sequence_results],
        "THYROID_mean_eff_cv": [float(row["effective_metrics"]["THYROID"]["mean_gy"]) for row in sequence_results],
    }
    cv_terms: Dict[str, float] = {}
    penalty = 0.0
    for name, values in tracked.items():
        values_arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(values_arr))
        std = float(np.std(values_arr, ddof=1)) if values_arr.size > 1 else 0.0
        cv = float(std / mean) if abs(mean) > 1e-9 else 0.0
        cv_terms[name] = cv
        penalty += cv
    return float(penalty), cv_terms


def compute_cumulative_course_summary(
    *,
    args: argparse.Namespace,
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    voxel_volume_cc: float,
    voxel_size_mm: Tuple[float, float, float],
    cumulative_physical_dose: np.ndarray,
    accepted_sequence: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    bio_args = SimpleNamespace(
        tumor_cytokine_multiplier=float(args.tumor_cytokine_multiplier),
        hypoxic_ros_scale=float(args.hypoxic_ros_scale),
        hypoxic_cytokine_multiplier=float(args.hypoxic_cytokine_multiplier),
        artery_ros_uptake=float(args.artery_ros_uptake),
        artery_cyto_uptake=float(args.artery_cyto_uptake),
        vein_ros_uptake=float(args.vein_ros_uptake),
        vein_cyto_uptake=float(args.vein_cyto_uptake),
    )
    cumulative_effective_dose = compute_effective_dose(
        physical_dose=cumulative_physical_dose,
        voxel_size_mm=voxel_size_mm,
        structures=structures,
        bio_args=bio_args,
        alpha=float(args.alpha),
        beta=float(args.beta),
        steps=int(args.pde_steps),
        dt=float(args.pde_dt),
    ).astype(np.float32)
    cumulative_physical_metrics = compute_metrics_for_domain(
        cumulative_physical_dose,
        structures=structures,
        voxel_volume_cc=voxel_volume_cc,
        prescription_gy=float(args.prescription_gy),
    )
    cumulative_effective_metrics = compute_metrics_for_domain(
        cumulative_effective_dose,
        structures=structures,
        voxel_volume_cc=voxel_volume_cc,
        prescription_gy=float(args.prescription_gy),
    )
    delivered_spots = [tuple(float(v) for v in spot) for result in accepted_sequence for spot in result["spot_centers_mm"]]
    peak_mask, valley_mask = build_peak_valley_rois(
        structures,
        axes_mm,
        delivered_spots,
        peak_radius_mm=float(args.peak_radius_mm),
        valley_exclusion_radius_mm=float(args.valley_exclusion_radius_mm),
    )
    pvdr_physical = compute_peak_valley_metrics(cumulative_physical_dose, peak_mask, valley_mask)
    pvdr_effective = compute_peak_valley_metrics(cumulative_effective_dose, peak_mask, valley_mask)
    spill_masks = build_spill_region_masks(
        structures=structures,
        axes_mm=axes_mm,
        peak_mask=peak_mask,
        shell_1_mm=float(args.spill_shell_1_mm),
        shell_2_mm=float(args.spill_shell_2_mm),
        shell_3_mm=float(args.spill_shell_3_mm),
        oar_adjacent_mm=float(args.spill_oar_adjacent_mm),
    )
    spill_physical_metrics = compute_region_metrics(
        cumulative_physical_dose,
        spill_masks,
        voxel_volume_cc=float(voxel_volume_cc),
    )
    spill_effective_metrics = compute_region_metrics(
        cumulative_effective_dose,
        spill_masks,
        voxel_volume_cc=float(voxel_volume_cc),
    )
    consistency_penalty, consistency_terms = compute_fraction_consistency_penalty(accepted_sequence)
    return {
        "cumulative_physical_dose": cumulative_physical_dose,
        "cumulative_effective_dose": cumulative_effective_dose,
        "cumulative_physical_metrics": cumulative_physical_metrics,
        "cumulative_effective_metrics": cumulative_effective_metrics,
        "pvdr_physical": pvdr_physical,
        "pvdr_effective": pvdr_effective,
        "spill_physical_metrics": spill_physical_metrics,
        "spill_effective_metrics": spill_effective_metrics,
        "consistency_penalty_raw": float(consistency_penalty),
        "consistency_terms": consistency_terms,
        "num_accepted_fractions": int(len(accepted_sequence)),
        "delivered_spots_mm": [[float(a), float(b), float(c)] for a, b, c in delivered_spots],
    }


def check_cumulative_constraints(
    args: argparse.Namespace,
    course_summary: Dict[str, object],
) -> Tuple[bool, Dict[str, Dict[str, object]]]:
    phys = course_summary["cumulative_physical_metrics"]
    eff = course_summary["cumulative_effective_metrics"]
    checks = {
        "cord_d2_eff": {
            "value": float(eff["SPINAL_CORD"]["d2_gy"]),
            "limit": float(args.hard_cumulative_cord_d2_eff_gy),
        },
        "brainstem_d2_eff": {
            "value": float(eff["BRAINSTEM"]["d2_gy"]),
            "limit": float(args.hard_cumulative_brainstem_d2_eff_gy),
        },
        "parotid_r_mean_eff": {
            "value": float(eff["PAROTID_R"]["mean_gy"]),
            "limit": float(args.hard_cumulative_parotid_r_mean_eff_gy),
        },
        "thyroid_mean_eff": {
            "value": float(eff["THYROID"]["mean_gy"]),
            "limit": float(args.hard_cumulative_thyroid_mean_eff_gy),
        },
        "body_dmax_phys": {
            "value": float(phys["BODY"]["dmax_gy"]),
            "limit": float(args.hard_cumulative_body_dmax_phys_gy),
        },
    }
    ok = True
    for name, detail in checks.items():
        if str(name) == "body_dmax_phys" and bool(args.disable_body_hotspot_hard_constraint):
            detail["passed"] = True
            detail["enabled"] = False
            continue
        passed = bool(float(detail["value"]) <= float(detail["limit"]))
        detail["passed"] = passed
        detail["enabled"] = True
        ok &= passed
    return bool(ok), checks


def compute_course_objective(
    args: argparse.Namespace,
    course_summary: Dict[str, object],
) -> Tuple[float, Dict[str, float]]:
    phys = course_summary["cumulative_physical_metrics"]
    eff = course_summary["cumulative_effective_metrics"]
    pv_phys = course_summary["pvdr_physical"]
    pv_eff = course_summary["pvdr_effective"]
    spill_eff = course_summary["spill_effective_metrics"]
    consistency_penalty_raw = float(course_summary["consistency_penalty_raw"])
    num_accepted_fractions = int(course_summary.get("num_accepted_fractions", 1))

    if str(args.objective_mode) == "outside_gtv_spill":
        target_floor_gtv = float(args.min_gtv_d95_eff_per_fraction_gy) * float(num_accepted_fractions)
        target_floor_ptv = float(args.min_ptv_d95_eff_per_fraction_gy) * float(num_accepted_fractions)
        ptv_v95_floor = float(args.min_ptv_v95_eff_pct)

        target_margin_reward = 0.35 * max(0.0, float(eff["GTV"]["d95_gy"]) - target_floor_gtv) + 0.25 * max(
            0.0,
            float(eff["PTV"]["d95_gy"]) - target_floor_ptv,
        )
        target_floor_penalties = {
            "gtv_d95_floor_shortfall": max(0.0, target_floor_gtv - float(eff["GTV"]["d95_gy"])),
            "ptv_d95_floor_shortfall": max(0.0, target_floor_ptv - float(eff["PTV"]["d95_gy"])),
            "ptv_v95_floor_shortfall_pct": max(0.0, ptv_v95_floor - float(eff["PTV"].get("v95_pct", 0.0))),
        }
        target_floor_penalty = float(args.weight_target_floor) * (
            target_floor_penalties["gtv_d95_floor_shortfall"]
            + target_floor_penalties["ptv_d95_floor_shortfall"]
            + 0.06 * target_floor_penalties["ptv_v95_floor_shortfall_pct"]
        )

        spill_penalties = {
            "spill_shell_0_5_mean_eff": float(args.weight_spill_shell_0_5_mean)
            * float(spill_eff.get("SPILL_SHELL_0_5", {}).get("mean_gy", 0.0)),
            "spill_shell_5_15_mean_eff": float(args.weight_spill_shell_5_15_mean)
            * float(spill_eff.get("SPILL_SHELL_5_15", {}).get("mean_gy", 0.0)),
            "spill_shell_15_30_mean_eff": float(args.weight_spill_shell_15_30_mean)
            * float(spill_eff.get("SPILL_SHELL_15_30", {}).get("mean_gy", 0.0)),
            "outside_gtv_d2_eff": float(args.weight_spill_outside_gtv_d2)
            * float(spill_eff.get("OUTSIDE_GTV", {}).get("d2_gy", 0.0)),
            "ptv_valley_outside_gtv_mean_eff": float(args.weight_spill_ptv_valley_mean)
            * float(spill_eff.get("PTV_VALLEY_OUTSIDE_GTV", {}).get("mean_gy", 0.0)),
            "oar_adjacent_outside_gtv_mean_eff": float(args.weight_spill_oar_adjacent_mean)
            * float(spill_eff.get("OAR_ADJACENT_OUTSIDE_GTV", {}).get("mean_gy", 0.0)),
        }
        spill_penalty = float(sum(spill_penalties.values()))

        pv_reward = (
            float(args.weight_pvdr_eff) * math.log1p(float(pv_eff["pvdr"]))
            + 0.50 * float(args.weight_pvdr_phys) * math.log1p(float(pv_phys["pvdr"]))
        )
        oar_penalties = {
            "cord_d2_eff": 1.35 * max(0.0, float(eff["SPINAL_CORD"]["d2_gy"]) - float(args.hard_cumulative_cord_d2_eff_gy)),
            "brainstem_d2_eff": 1.20 * max(0.0, float(eff["BRAINSTEM"]["d2_gy"]) - float(args.hard_cumulative_brainstem_d2_eff_gy)),
            "parotid_r_mean_eff": 1.00 * max(0.0, float(eff["PAROTID_R"]["mean_gy"]) - float(args.hard_cumulative_parotid_r_mean_eff_gy)),
            "thyroid_mean_eff": 0.90 * max(0.0, float(eff["THYROID"]["mean_gy"]) - float(args.hard_cumulative_thyroid_mean_eff_gy)),
            "brain_mean_eff": 0.65 * max(0.0, float(eff["BRAIN"]["mean_gy"]) - 20.0),
            "parotid_l_mean_eff": 0.25 * max(0.0, float(eff["PAROTID_L"]["mean_gy"]) - 15.0),
        }
        if "PARATHYROIDS" in eff and "mean_gy" in eff["PARATHYROIDS"]:
            oar_penalties["parathyroid_mean_eff"] = 0.45 * max(0.0, float(eff["PARATHYROIDS"]["mean_gy"]) - 45.0)
        if "BLOOD_BRAIN_BARRIER" in eff and "mean_gy" in eff["BLOOD_BRAIN_BARRIER"]:
            oar_penalties["bbb_mean_eff"] = 0.30 * max(0.0, float(eff["BLOOD_BRAIN_BARRIER"]["mean_gy"]) - 20.0)

        hotspot_penalty = float(args.weight_hotspot) * (
            max(0.0, float(phys["BODY"]["dmax_gy"]) - float(args.hard_cumulative_body_dmax_phys_gy))
            + 0.6 * max(0.0, float(eff["PTV"]["d2_gy"]) - float(args.soft_cumulative_ptv_d2_eff_gy))
        )
        consistency_penalty = float(args.weight_consistency) * consistency_penalty_raw
        total_oar_penalty = float(sum(oar_penalties.values()))
        objective = target_margin_reward + pv_reward - target_floor_penalty - spill_penalty - total_oar_penalty - hotspot_penalty - consistency_penalty
        details = {
            "objective_mode": "outside_gtv_spill",
            "target_margin_reward": float(target_margin_reward),
            "target_floor_penalty": float(target_floor_penalty),
            "spill_penalty": float(spill_penalty),
            "pv_reward": float(pv_reward),
            "oar_penalty": float(total_oar_penalty),
            "hotspot_penalty": float(hotspot_penalty),
            "consistency_penalty": float(consistency_penalty),
            "target_floor_gtv_d95_eff_gy": float(target_floor_gtv),
            "target_floor_ptv_d95_eff_gy": float(target_floor_ptv),
            "target_floor_ptv_v95_eff_pct": float(ptv_v95_floor),
            **{f"target_{k}": float(v) for k, v in target_floor_penalties.items()},
            **{f"spill_{k}": float(v) for k, v in spill_penalties.items()},
            **{f"oar_{k}": float(v) for k, v in oar_penalties.items()},
        }
        return float(objective), details

    target_reward = (
        float(args.weight_gtv_d95) * float(eff["GTV"]["d95_gy"])
        + float(args.weight_ptv_d95) * float(eff["PTV"]["d95_gy"])
        + float(args.weight_ptv_v95) * float(eff["PTV"].get("v95_pct", 0.0))
    )
    pv_reward = (
        float(args.weight_pvdr_eff) * math.log1p(float(pv_eff["pvdr"]))
        + float(args.weight_pvdr_phys) * math.log1p(float(pv_phys["pvdr"]))
    )

    oar_penalties = {
        "cord_d2_eff": 1.25 * max(0.0, float(eff["SPINAL_CORD"]["d2_gy"]) - float(args.hard_cumulative_cord_d2_eff_gy)),
        "brainstem_d2_eff": 1.10 * max(0.0, float(eff["BRAINSTEM"]["d2_gy"]) - float(args.hard_cumulative_brainstem_d2_eff_gy)),
        "parotid_r_mean_eff": 1.00 * max(0.0, float(eff["PAROTID_R"]["mean_gy"]) - float(args.hard_cumulative_parotid_r_mean_eff_gy)),
        "thyroid_mean_eff": 0.85 * max(0.0, float(eff["THYROID"]["mean_gy"]) - float(args.hard_cumulative_thyroid_mean_eff_gy)),
        "brain_mean_eff": 0.60 * max(0.0, float(eff["BRAIN"]["mean_gy"]) - 20.0),
        "parotid_l_mean_eff": 0.25 * max(0.0, float(eff["PAROTID_L"]["mean_gy"]) - 15.0),
    }
    if "PARATHYROIDS" in eff and "mean_gy" in eff["PARATHYROIDS"]:
        oar_penalties["parathyroid_mean_eff"] = 0.45 * max(0.0, float(eff["PARATHYROIDS"]["mean_gy"]) - 45.0)
    if "BLOOD_BRAIN_BARRIER" in eff and "mean_gy" in eff["BLOOD_BRAIN_BARRIER"]:
        oar_penalties["bbb_mean_eff"] = 0.30 * max(0.0, float(eff["BLOOD_BRAIN_BARRIER"]["mean_gy"]) - 20.0)
    hotspot_penalty = float(args.weight_hotspot) * (
        max(0.0, float(phys["BODY"]["dmax_gy"]) - float(args.hard_cumulative_body_dmax_phys_gy))
        + 0.6 * max(0.0, float(eff["PTV"]["d2_gy"]) - float(args.soft_cumulative_ptv_d2_eff_gy))
    )
    consistency_penalty = float(args.weight_consistency) * consistency_penalty_raw
    total_oar_penalty = float(sum(oar_penalties.values()))
    objective = target_reward + pv_reward - total_oar_penalty - hotspot_penalty - consistency_penalty
    details = {
        "objective_mode": "course_balance",
        "target_reward": float(target_reward),
        "pv_reward": float(pv_reward),
        "oar_penalty": float(total_oar_penalty),
        "hotspot_penalty": float(hotspot_penalty),
        "consistency_penalty": float(consistency_penalty),
        **{f"oar_{k}": float(v) for k, v in oar_penalties.items()},
    }
    return float(objective), details


def summarize_fraction_row(
    fraction_idx: int,
    accepted_result: Dict[str, object],
    course_summary: Dict[str, object],
    course_objective: float,
    hard_constraints_ok: bool,
) -> Dict[str, object]:
    eff = course_summary["cumulative_effective_metrics"]
    phys = course_summary["cumulative_physical_metrics"]
    pv_eff = course_summary["pvdr_effective"]
    spill_eff = course_summary.get("spill_effective_metrics", {})
    return {
        "fraction": int(fraction_idx),
        "placement_name": str(accepted_result["placement_name"]),
        "accepted_spots_mm": "; ".join(
            f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in accepted_result["spot_centers_mm"]
        ),
        "course_objective": f"{float(course_objective):.4f}",
        "hard_constraints_ok": "yes" if bool(hard_constraints_ok) else "no",
        "cum_ptv_d95_eff_gy": f"{float(eff['PTV']['d95_gy']):.4f}",
        "cum_gtv_d95_eff_gy": f"{float(eff['GTV']['d95_gy']):.4f}",
        "cum_pvdr_eff": f"{float(pv_eff['pvdr']):.4f}",
        "cum_cord_d2_eff_gy": f"{float(eff['SPINAL_CORD']['d2_gy']):.4f}",
        "cum_brainstem_d2_eff_gy": f"{float(eff['BRAINSTEM']['d2_gy']):.4f}",
        "cum_parotid_r_mean_eff_gy": f"{float(eff['PAROTID_R']['mean_gy']):.4f}",
        "cum_thyroid_mean_eff_gy": f"{float(eff['THYROID']['mean_gy']):.4f}",
        "cum_spill_shell_0_5_mean_eff_gy": f"{float(spill_eff.get('SPILL_SHELL_0_5', {}).get('mean_gy', 0.0)):.4f}",
        "cum_spill_shell_5_15_mean_eff_gy": f"{float(spill_eff.get('SPILL_SHELL_5_15', {}).get('mean_gy', 0.0)):.4f}",
        "cum_spill_shell_15_30_mean_eff_gy": f"{float(spill_eff.get('SPILL_SHELL_15_30', {}).get('mean_gy', 0.0)):.4f}",
        "cum_oar_adjacent_outside_gtv_mean_eff_gy": f"{float(spill_eff.get('OAR_ADJACENT_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
        "cum_ptv_valley_outside_gtv_mean_eff_gy": f"{float(spill_eff.get('PTV_VALLEY_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
        "cum_body_dmax_phys_gy": f"{float(phys['BODY']['dmax_gy']):.4f}",
    }


def candidate_row(
    fraction_idx: int,
    candidate_rank: int,
    spot_set: Sequence[Tuple[float, float, float]],
    *,
    candidate_origin: str,
    plan_overrides: Mapping[str, object] | None,
    heuristic_score: float,
    course_objective: float,
    constraints_ok: bool,
    eff_metrics: Dict[str, Dict[str, float]],
    pvdr_effective: Dict[str, float],
    spill_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    return {
        "fraction": int(fraction_idx),
        "candidate_rank": int(candidate_rank),
        "candidate_origin": str(candidate_origin),
        "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in spot_set),
        "plan_overrides": json.dumps({str(k): float(v) for k, v in (plan_overrides or {}).items()}, sort_keys=True),
        "heuristic_score": f"{float(heuristic_score):.4f}",
        "course_objective": f"{float(course_objective):.4f}",
        "constraints_ok": "yes" if bool(constraints_ok) else "no",
        "cum_ptv_d95_eff_gy": f"{float(eff_metrics['PTV']['d95_gy']):.4f}",
        "cum_gtv_d95_eff_gy": f"{float(eff_metrics['GTV']['d95_gy']):.4f}",
        "cum_pvdr_eff": f"{float(pvdr_effective['pvdr']):.4f}",
        "cum_cord_d2_eff_gy": f"{float(eff_metrics['SPINAL_CORD']['d2_gy']):.4f}",
        "cum_parotid_r_mean_eff_gy": f"{float(eff_metrics['PAROTID_R']['mean_gy']):.4f}",
        "cum_thyroid_mean_eff_gy": f"{float(eff_metrics['THYROID']['mean_gy']):.4f}",
        "cum_spill_shell_0_5_mean_eff_gy": f"{float(spill_metrics.get('SPILL_SHELL_0_5', {}).get('mean_gy', 0.0)):.4f}",
        "cum_spill_shell_5_15_mean_eff_gy": f"{float(spill_metrics.get('SPILL_SHELL_5_15', {}).get('mean_gy', 0.0)):.4f}",
        "cum_spill_shell_15_30_mean_eff_gy": f"{float(spill_metrics.get('SPILL_SHELL_15_30', {}).get('mean_gy', 0.0)):.4f}",
        "cum_oar_adjacent_outside_gtv_mean_eff_gy": f"{float(spill_metrics.get('OAR_ADJACENT_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
        "cum_ptv_valley_outside_gtv_mean_eff_gy": f"{float(spill_metrics.get('PTV_VALLEY_OUTSIDE_GTV', {}).get('mean_gy', 0.0)):.4f}",
    }


def candidate_label_slug(candidate_origin: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(candidate_origin))
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:48] or "candidate"


def build_robust_eval_specs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    seed_values = parse_csv_int_list(str(getattr(args, "robust_seeds", "")))
    history_values = parse_csv_int_list(str(getattr(args, "robust_histories", "")))
    if not seed_values:
        seed_values = [int(args.seed)]
    if not history_values:
        history_values = [int(args.histories)]
    specs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    ordered_pairs = [(int(args.seed), int(args.histories))] + list(itertools.product(seed_values, history_values))
    for seed, histories in ordered_pairs:
        pair = (int(seed), int(histories))
        if pair in seen:
            continue
        seen.add(pair)
        specs.append(pair)
    return specs


def rerank_top_candidates_with_uncertainty(
    *,
    args: argparse.Namespace,
    plan_args: SimpleNamespace,
    phantom: Dict[str, object],
    spectrum_energies: Sequence[float],
    spectrum_weights: Sequence[float],
    fraction_idx: int,
    current_cumulative_physical: np.ndarray,
    accepted_sequence: Sequence[Dict[str, object]],
    voxel_volume_cc: float,
    voxel_size_mm: Tuple[float, float, float],
    structures: Dict[str, np.ndarray],
    axes_mm: Dict[str, np.ndarray],
    evaluated_candidates: Sequence[Dict[str, object]],
    candidate_rows: Sequence[Dict[str, object]],
    placement_counter: int,
    robust_evaluation_cache: Dict[Tuple[object, int, int], Dict[str, object]],
) -> Tuple[Dict[str, object] | None, int, Dict[str, object]]:
    robust_top_k = max(0, int(getattr(args, "robust_top_k_candidates", 0)))
    robust_specs = list(getattr(args, "_robust_eval_specs", []))
    if robust_top_k <= 0 or not robust_specs:
        return None, int(placement_counter), {"enabled": False}

    feasible_candidates = [row for row in evaluated_candidates if bool(row["constraints_ok"])]
    base_pool = feasible_candidates if feasible_candidates else list(evaluated_candidates)
    if not base_pool:
        return None, int(placement_counter), {"enabled": False, "reason": "no_candidates"}

    top_candidates = sorted(base_pool, key=lambda row: float(row["course_objective"]), reverse=True)[:robust_top_k]
    row_lookup = {
        (int(row.get("fraction", -1)), int(row.get("candidate_rank", -1))): row
        for row in candidate_rows
        if str(row.get("course_objective", "")) != "nan"
    }
    robust_debug_rows: List[Dict[str, object]] = []
    robust_z = float(getattr(args, "robust_z_score", 1.0))
    min_feasible_fraction = max(0.0, min(1.0, float(getattr(args, "robust_min_feasible_fraction", 1.0))))

    for candidate in top_candidates:
        candidate_rank = int(candidate["candidate_rank"])
        candidate_origin = str(candidate.get("candidate_origin", "unknown"))
        spots_mm = [tuple(float(v) for v in spot) for spot in candidate["single_fraction_result"]["spot_centers_mm"]]
        plan_overrides = dict(candidate.get("plan_overrides") or {})
        sample_rows: List[Dict[str, object]] = [
            {
                "seed": int(args.seed),
                "histories": int(args.histories),
                "course_objective": float(candidate["course_objective"]),
                "constraints_ok": bool(candidate["constraints_ok"]),
            }
        ]
        robust_errors: List[Dict[str, object]] = []
        for seed_value, history_value in robust_specs:
            if int(seed_value) == int(args.seed) and int(history_value) == int(args.histories):
                continue
            cache_key = robust_evaluation_cache_key(spots_mm, plan_overrides, int(seed_value), int(history_value))
            if cache_key in robust_evaluation_cache:
                robust_trial = robust_evaluation_cache[cache_key]
            else:
                robust_args = argparse.Namespace(**vars(args))
                robust_args.seed = int(seed_value)
                robust_args.histories = int(history_value)
                robust_plan_args = SimpleNamespace(**vars(plan_args))
                for override_key, override_value in plan_overrides.items():
                    setattr(robust_plan_args, str(override_key), float(override_value))
                try:
                    robust_result = evaluate_plan(
                        args=robust_args,
                        plan_args=robust_plan_args,
                        phantom=phantom,
                        spectrum_energies=spectrum_energies,
                        spectrum_weights=spectrum_weights,
                        placement_id=placement_counter,
                        placement_name=(
                            f"fraction{fraction_idx:02d}_robust{candidate_rank:02d}_"
                            f"s{int(seed_value)}_h{int(history_value)}_{candidate_label_slug(candidate_origin)}"
                        ),
                        spot_centers_mm=spots_mm,
                        reuse_existing_dose_csv=None,
                    )
                except RuntimeError as exc:
                    robust_errors.append(
                        {
                            "seed": int(seed_value),
                            "histories": int(history_value),
                            "error": str(exc),
                        }
                    )
                    placement_counter += 1
                    continue
                placement_counter += 1
                trial_sequence = list(accepted_sequence) + [robust_result]
                trial_cumulative_physical = np.asarray(current_cumulative_physical, dtype=np.float32) + np.asarray(
                    robust_result["physical_dose"], dtype=np.float32
                )
                trial_summary = compute_cumulative_course_summary(
                    args=args,
                    structures=structures,
                    axes_mm=axes_mm,
                    voxel_volume_cc=voxel_volume_cc,
                    voxel_size_mm=voxel_size_mm,
                    cumulative_physical_dose=trial_cumulative_physical,
                    accepted_sequence=trial_sequence,
                )
                constraints_ok, constraint_details = check_cumulative_constraints(args, trial_summary)
                trial_objective, trial_objective_details = compute_course_objective(args, trial_summary)
                robust_trial = {
                    "single_fraction_result": robust_result,
                    "course_summary": trial_summary,
                    "course_objective": float(trial_objective),
                    "constraints_ok": bool(constraints_ok),
                    "constraint_details": constraint_details,
                    "objective_details": trial_objective_details,
                }
                robust_evaluation_cache[cache_key] = robust_trial
            sample_rows.append(
                {
                    "seed": int(seed_value),
                    "histories": int(history_value),
                    "course_objective": float(robust_trial["course_objective"]),
                    "constraints_ok": bool(robust_trial["constraints_ok"]),
                }
            )

        sample_objectives = np.asarray([float(row["course_objective"]) for row in sample_rows], dtype=np.float64)
        sample_feasible = np.asarray([1.0 if bool(row["constraints_ok"]) else 0.0 for row in sample_rows], dtype=np.float64)
        mean_objective = float(sample_objectives.mean()) if sample_objectives.size else float(candidate["course_objective"])
        std_objective = float(sample_objectives.std(ddof=0)) if sample_objectives.size > 1 else 0.0
        lower_bound = float(mean_objective - robust_z * std_objective)
        upper_bound = float(mean_objective + robust_z * std_objective)
        feasible_fraction = float(sample_feasible.mean()) if sample_feasible.size else (1.0 if bool(candidate["constraints_ok"]) else 0.0)
        robust_feasible = bool(feasible_fraction >= min_feasible_fraction)
        robust_stats = {
            "sample_count": int(len(sample_rows)),
            "mean_objective": float(mean_objective),
            "std_objective": float(std_objective),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "feasible_fraction": float(feasible_fraction),
            "robust_feasible": bool(robust_feasible),
            "samples": sample_rows,
            "errors": robust_errors,
        }
        candidate["robust_stats"] = robust_stats
        csv_row = row_lookup.get((int(fraction_idx), int(candidate_rank)))
        if csv_row is not None:
            csv_row.update(
                {
                    "robust_sample_count": int(len(sample_rows)),
                    "robust_mean_objective": f"{mean_objective:.4f}",
                    "robust_std_objective": f"{std_objective:.4f}",
                    "robust_lower_bound": f"{lower_bound:.4f}",
                    "robust_upper_bound": f"{upper_bound:.4f}",
                    "robust_feasible_fraction": f"{feasible_fraction:.4f}",
                    "robust_feasible": "yes" if robust_feasible else "no",
                    "robust_errors": int(len(robust_errors)),
                }
            )
        robust_debug_rows.append(
            {
                "candidate_rank": int(candidate_rank),
                "candidate_origin": candidate_origin,
                "mean_objective": float(mean_objective),
                "std_objective": float(std_objective),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "feasible_fraction": float(feasible_fraction),
                "robust_feasible": bool(robust_feasible),
                "sample_count": int(len(sample_rows)),
                "error_count": int(len(robust_errors)),
            }
        )

    robust_ranked = [row for row in top_candidates if "robust_stats" in row]
    if not robust_ranked:
        return None, int(placement_counter), {"enabled": True, "reason": "no_robust_samples"}

    robust_feasible_candidates = [row for row in robust_ranked if bool(row["robust_stats"]["robust_feasible"])]
    ranking_pool = robust_feasible_candidates if robust_feasible_candidates else robust_ranked
    ranked_by_mean = sorted(ranking_pool, key=lambda row: float(row["robust_stats"]["mean_objective"]), reverse=True)
    top_row = ranked_by_mean[0]
    top_stats = dict(top_row["robust_stats"])
    tie_pool: List[Dict[str, object]] = []
    for row in ranked_by_mean:
        row_stats = dict(row["robust_stats"])
        combined_band = float(robust_z * math.sqrt(row_stats["std_objective"] ** 2 + top_stats["std_objective"] ** 2))
        within_band = bool((top_stats["mean_objective"] - row_stats["mean_objective"]) <= (combined_band + 1e-9))
        row["robust_stats"]["combined_noise_band_to_top"] = float(combined_band)
        row["robust_stats"]["within_top_noise_band"] = bool(within_band)
        csv_row = row_lookup.get((int(fraction_idx), int(row["candidate_rank"])))
        if csv_row is not None:
            csv_row.update(
                {
                    "robust_combined_noise_band_to_top": f"{combined_band:.4f}",
                    "robust_within_top_noise_band": "yes" if within_band else "no",
                }
            )
        if within_band:
            tie_pool.append(row)

    robust_choice = max(
        tie_pool,
        key=lambda row: (
            float(row["robust_stats"]["lower_bound"]),
            float(row["robust_stats"]["feasible_fraction"]),
            float(row["course_objective"]),
        ),
    )
    trusted_ranking = bool(len(tie_pool) == 1)
    robust_choice["robust_stats"]["trusted_ranking"] = bool(trusted_ranking)
    choice_csv_row = row_lookup.get((int(fraction_idx), int(robust_choice["candidate_rank"])))
    if choice_csv_row is not None:
        choice_csv_row["robust_selected"] = "yes"
        choice_csv_row["robust_trusted_ranking"] = "yes" if trusted_ranking else "no"

    return robust_choice, int(placement_counter), {
        "enabled": True,
        "top_k_requested": int(robust_top_k),
        "eval_specs": [{"seed": int(seed), "histories": int(histories)} for seed, histories in robust_specs],
        "ranking_trusted": bool(trusted_ranking),
        "candidate_summaries": robust_debug_rows,
        "selected_candidate_rank": int(robust_choice["candidate_rank"]),
        "selected_candidate_origin": str(robust_choice.get("candidate_origin", "unknown")),
        "selected_lower_bound": float(robust_choice["robust_stats"]["lower_bound"]),
    }


def main() -> int:
    args = parse_args()
    args._cached_template_text = Path(args.template).read_text(encoding="utf-8")
    args._robust_eval_specs = build_robust_eval_specs(args)
    args.run_root.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_phase14_summary(args.baseline_run_root.resolve())
    phantom_args = build_args_from_summary(baseline_summary)
    args.size_x_cm = float(baseline_summary["phantom"]["size_cm"][0])
    args.size_y_cm = float(baseline_summary["phantom"]["size_cm"][1])
    args.size_z_cm = float(baseline_summary["phantom"]["size_cm"][2])
    args.material_specs = MATERIAL_SPECS

    phantom = build_detailed_plan_phantom(phantom_args)
    structures = phantom["structures"]
    axes_mm = phantom["axes_mm"]
    phantom_meta = phantom["meta"]
    voxel_volume_cc = float(phantom_meta["voxel_volume_cc"])
    voxel_size_mm = tuple(float(v) for v in phantom_meta["voxel_size_mm"])

    plan_args = build_plan_args(args, phantom_meta)
    spectrum_energies, spectrum_weights = load_spectrum(args.spectrum_csv)

    candidate_indices = build_candidate_centers(
        structures,
        axes_mm,
        spot_radius_mm=float(args.spot_radius_mm),
        candidate_step_mm=float(args.candidate_step_mm),
    )
    structure_points_mm = build_structure_points_mm(
        structures,
        axes_mm,
        [
            "SPINAL_CORD",
            "BRAINSTEM",
            "PAROTID_R",
            "PAROTID_L",
            "THYROID",
            "PARATHYROIDS",
            "BRAIN",
            "BLOOD_BRAIN_BARRIER",
            "MANDIBLE",
            "ARTERIES",
            "VEINS",
        ],
    )
    candidate_pool_debug: Dict[str, object] = {}
    if str(args.course_strategy) in {"clinical_gtv_core", "randomized_protocol_mix", "adaptive_hotspot_avoidance", "adaptive_nofly_weight_opt", "phase24_joint_horizon_diversity"}:
        safe_candidate_indices, candidate_pool_debug = build_clinical_gtv_core_candidate_centers(
            structures=structures,
            axes_mm=axes_mm,
            spot_radius_mm=float(args.spot_radius_mm),
            candidate_step_mm=float(args.candidate_step_mm),
            contraction_mm=float(args.clinical_gtv_contraction_mm),
            oar_clearance_mm=float(args.clinical_oar_clearance_mm),
            min_pool_size=max(8, int(args.num_spots) * 2),
        )
    else:
        safe_candidate_indices = build_safe_candidate_centers(
            candidate_indices,
            axes_mm,
            structure_points_mm,
            hard_min_dist_cord_mm=float(args.hard_min_dist_cord_mm),
            hard_min_dist_brainstem_mm=float(args.hard_min_dist_brainstem_mm),
        )
    vessel_coords = np.argwhere(structures["ARTERIES"] | structures["VEINS"])
    vessel_coords_mm = np.column_stack(
        [
            axes_mm["x"][vessel_coords[:, 0]],
            axes_mm["y"][vessel_coords[:, 1]],
            axes_mm["z"][vessel_coords[:, 2]],
        ]
    ).astype(np.float32)
    uptake_tensor, _, _, _ = build_anatomical_biology_tensors(args, structures)

    accepted_sequence: List[Dict[str, object]] = []
    fraction_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    history_counts: Dict[Tuple[float, float, float], int] = {}
    evaluation_cache: Dict[Tuple[Tuple[float, float, float], ...] | Tuple[Tuple[Tuple[float, float, float], ...], Tuple[Tuple[str, float], ...]], Dict[str, object]] = {}
    robust_evaluation_cache: Dict[Tuple[object, int, int], Dict[str, object]] = {}
    placement_counter = 1
    strategy_state: Dict[str, object] = {}
    adaptive_avoidance_state: Dict[str, object] | None = None
    adaptive_filter_debug: Dict[str, object] | None = None

    if str(args.course_strategy) in {"clinical_gtv_core", "randomized_protocol_mix", "adaptive_hotspot_avoidance", "adaptive_nofly_weight_opt", "phase24_joint_horizon_diversity"}:
        zero_course = np.zeros(structures["BODY"].shape, dtype=np.float32)
        zero_effective = np.zeros_like(zero_course)
        oar_weights, weight_details = compute_guidance_oar_weights(
            {
                "SPINAL_CORD": {"d2_gy": 0.0},
                "BRAINSTEM": {"d2_gy": 0.0},
                "PAROTID_R": {"mean_gy": 0.0},
                "PAROTID_L": {"mean_gy": 0.0},
                "THYROID": {"mean_gy": 0.0},
                "PARATHYROIDS": {"mean_gy": 0.0},
                "BRAIN": {"mean_gy": 0.0},
                "BLOOD_BRAIN_BARRIER": {"mean_gy": 0.0},
                "MANDIBLE": {"mean_gy": 0.0},
            },
            args,
        )
        scored_centers = score_candidate_centers(
            current_cumulative_effective_dose=zero_effective,
            current_cumulative_physical_dose=zero_course,
            current_fraction_idx=1,
            structures=structures,
            axes_mm=axes_mm,
            uptake_tensor=uptake_tensor,
            candidate_indices=safe_candidate_indices,
            target_effective_gy_per_fraction=float(args.target_effective_gy_per_fraction),
            target_need_cap_gy=float(args.target_need_cap_gy),
            oar_weights=oar_weights,
            structure_points_mm=structure_points_mm,
            vessel_coords_mm=vessel_coords_mm,
            history_counts=history_counts,
            adaptive_avoidance=None,
        )
        baseline_candidates = build_strategy_candidate_sets(
            args=args,
            fraction_idx=1,
            baseline_spots=[],
            current_repeat_spots=[],
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            adaptive_state=None,
            axes_mm=axes_mm,
            candidate_indices=safe_candidate_indices,
            num_spots=int(args.num_spots),
            min_spacing_mm=float(args.min_spot_spacing_mm),
            top_k_centers=int(args.candidate_top_k_centers),
            candidate_plan_limit=int(args.candidate_plan_limit),
        )
        if not baseline_candidates:
            raise RuntimeError("Clinical GTV-core strategy could not build any baseline candidate spot sets.")

        evaluated_baseline_candidates: List[Dict[str, object]] = []
        for candidate_rank, candidate in enumerate(baseline_candidates, start=1):
            spots_mm = [tuple(float(v) for v in spot) for spot in candidate["spots_mm"]]
            candidate_plan_overrides = candidate.get("plan_overrides") or {}
            candidate_plan_args = SimpleNamespace(**vars(plan_args))
            for override_key, override_value in candidate_plan_overrides.items():
                setattr(candidate_plan_args, str(override_key), float(override_value))
            try:
                single_fraction_result = evaluate_plan(
                    args=args,
                    plan_args=candidate_plan_args,
                    phantom=phantom,
                    spectrum_energies=spectrum_energies,
                    spectrum_weights=spectrum_weights,
                    placement_id=placement_counter,
                    placement_name=f"fraction01_candidate{candidate_rank:02d}_{candidate_label_slug(candidate.get('candidate_origin', 'clinical'))}",
                    spot_centers_mm=spots_mm,
                    reuse_existing_dose_csv=None,
                )
            except RuntimeError as exc:
                candidate_rows.append(
                    {
                        "fraction": 1,
                        "candidate_rank": int(candidate_rank),
                        "candidate_origin": str(candidate.get("candidate_origin", "clinical_grid")),
                        "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in spots_mm),
                        "plan_overrides": json.dumps({str(k): float(v) for k, v in candidate_plan_overrides.items()}, sort_keys=True),
                        "heuristic_score": f"{float(candidate['heuristic_score']):.4f}",
                        "course_objective": "nan",
                        "constraints_ok": "no",
                        "cum_ptv_d95_eff_gy": "nan",
                        "cum_gtv_d95_eff_gy": "nan",
                        "cum_pvdr_eff": "nan",
                        "cum_cord_d2_eff_gy": "nan",
                        "cum_parotid_r_mean_eff_gy": "nan",
                        "cum_thyroid_mean_eff_gy": "nan",
                        "evaluation_error": str(exc),
                    }
                )
                placement_counter += 1
                continue
            evaluation_cache[evaluation_cache_key(spots_mm, candidate_plan_overrides)] = single_fraction_result
            placement_counter += 1
            trial_summary = compute_cumulative_course_summary(
                args=args,
                structures=structures,
                axes_mm=axes_mm,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                cumulative_physical_dose=np.asarray(single_fraction_result["physical_dose"], dtype=np.float32),
                accepted_sequence=[single_fraction_result],
            )
            constraints_ok, constraint_details = check_cumulative_constraints(args, trial_summary)
            trial_objective, trial_objective_details = compute_course_objective(args, trial_summary)
            phase24_diversity_bonus = 0.0
            phase24_diversity_details: Dict[str, object] | None = None
            phase24_lookahead_bonus = 0.0
            phase24_lookahead_details: Dict[str, object] | None = None
            selection_score = float(trial_objective)
            if str(args.course_strategy) == "phase24_joint_horizon_diversity":
                phase24_diversity_bonus, phase24_diversity_details = compute_phase24_diversity_bonus(
                    args=args,
                    candidate_spots_mm=spots_mm,
                    accepted_sequence=[],
                )
                phase24_lookahead_bonus, phase24_lookahead_details = estimate_phase24_future_space_bonus(
                    args=args,
                    trial_summary=trial_summary,
                    trial_sequence=[single_fraction_result],
                    trial_spots_mm=spots_mm,
                    baseline_spots=spots_mm,
                    strategy_state=strategy_state,
                    safe_candidate_indices=safe_candidate_indices,
                    candidate_indices=candidate_indices,
                    axes_mm=axes_mm,
                    structures=structures,
                    uptake_tensor=uptake_tensor,
                    structure_points_mm=structure_points_mm,
                    vessel_coords_mm=vessel_coords_mm,
                    history_counts=history_counts,
                )
                selection_score = float(trial_objective + phase24_diversity_bonus + phase24_lookahead_bonus)
            evaluated_baseline_candidates.append(
                {
                    "candidate_rank": int(candidate_rank),
                    "candidate_origin": str(candidate.get("candidate_origin", "clinical_grid")),
                    "spots_mm": spots_mm,
                    "plan_overrides": {str(k): float(v) for k, v in candidate_plan_overrides.items()},
                    "single_fraction_result": single_fraction_result,
                    "course_summary": trial_summary,
                    "course_objective": float(trial_objective),
                    "selection_score": float(selection_score),
                    "phase24_diversity_bonus": float(phase24_diversity_bonus),
                    "phase24_lookahead_bonus": float(phase24_lookahead_bonus),
                    "phase24_diversity_details": phase24_diversity_details,
                    "phase24_lookahead_details": phase24_lookahead_details,
                    "constraints_ok": bool(constraints_ok),
                    "constraint_details": constraint_details,
                    "objective_details": trial_objective_details,
                    "heuristic_score": float(candidate["heuristic_score"]),
                }
            )
            row = candidate_row(
                1,
                candidate_rank,
                spots_mm,
                candidate_origin=str(candidate.get("candidate_origin", "clinical_grid")),
                plan_overrides=candidate_plan_overrides,
                heuristic_score=float(candidate["heuristic_score"]),
                course_objective=float(trial_objective),
                constraints_ok=bool(constraints_ok),
                eff_metrics=trial_summary["cumulative_effective_metrics"],
                pvdr_effective=trial_summary["pvdr_effective"],
                spill_metrics=trial_summary["spill_effective_metrics"],
            )
            row.update(
                {
                    "selection_score": f"{float(selection_score):.4f}",
                    "phase24_diversity_bonus": f"{float(phase24_diversity_bonus):.4f}",
                    "phase24_lookahead_bonus": f"{float(phase24_lookahead_bonus):.4f}",
                }
            )
            candidate_rows.append(row)

        feasible_baseline_candidates = [row for row in evaluated_baseline_candidates if bool(row["constraints_ok"])]
        if not evaluated_baseline_candidates:
            raise RuntimeError("Clinical GTV-core strategy baseline evaluation failed for every candidate.")
        chosen_baseline = max(
            feasible_baseline_candidates if feasible_baseline_candidates else evaluated_baseline_candidates,
            key=lambda row: float(row.get("selection_score", row["course_objective"])),
        )
        baseline_result = chosen_baseline["single_fraction_result"]
        baseline_spots = [tuple(float(v) for v in spot) for spot in baseline_result["spot_centers_mm"]]
        strategy_state["pattern_a"] = baseline_spots
        accepted_sequence.append(baseline_result)
        for spot in baseline_spots:
            key = round_spot(spot)
            history_counts[key] = history_counts.get(key, 0) + 1
        course_summary = chosen_baseline["course_summary"]
        course_constraints_ok = bool(chosen_baseline["constraints_ok"])
        course_constraint_details = chosen_baseline["constraint_details"]
        course_objective = float(chosen_baseline["course_objective"])
        course_objective_details = chosen_baseline["objective_details"]
        fraction_rows.append(
            summarize_fraction_row(
                1,
                baseline_result,
                course_summary,
                course_objective,
                course_constraints_ok,
            )
        )
        write_text_with_retries(
            args.run_root / "fraction01_selection_debug.json",
            json.dumps(
                {
                    "fraction": 1,
                    "candidate_pool_debug": candidate_pool_debug,
                    "guidance_weights": {key: float(value) for key, value in oar_weights.items()},
                    "guidance_weight_details": {
                        name: {
                            "metric": info[0],
                            "value": float(info[1]),
                            "threshold": float(info[2]),
                            "relative_exceedance": float(info[3]),
                        }
                        for name, info in weight_details.items()
                    },
                    "adaptive_avoidance_state": {
                        key: value
                        for key, value in (adaptive_avoidance_state or {}).items()
                        if not str(key).endswith("_points_mm")
                    },
                    "adaptive_filter_debug": adaptive_filter_debug,
                    "candidates": [
                        {
                            "candidate_rank": int(row["candidate_rank"]),
                            "candidate_origin": str(row["candidate_origin"]),
                            "spots_mm": [[float(a), float(b), float(c)] for a, b, c in row["single_fraction_result"]["spot_centers_mm"]],
                            "course_objective": float(row["course_objective"]),
                            "selection_score": float(row.get("selection_score", row["course_objective"])),
                            "constraints_ok": bool(row["constraints_ok"]),
                            "constraint_details": row["constraint_details"],
                            "objective_details": row["objective_details"],
                            "phase24_diversity_bonus": float(row.get("phase24_diversity_bonus", 0.0)),
                            "phase24_lookahead_bonus": float(row.get("phase24_lookahead_bonus", 0.0)),
                            "phase24_diversity_details": row.get("phase24_diversity_details"),
                            "phase24_lookahead_details": row.get("phase24_lookahead_details"),
                        }
                        for row in evaluated_baseline_candidates
                    ],
                    "selected_candidate_rank": int(chosen_baseline["candidate_rank"]),
                    "selected_candidate_origin": str(chosen_baseline["candidate_origin"]),
                    "selected_constraint_details": course_constraint_details,
                    "selected_objective_details": course_objective_details,
                },
                indent=2,
            ),
        )
    else:
        baseline_spots = [tuple(map(float, row)) for row in baseline_summary["plan"]["spot_centers_mm"]]
        strategy_state["pattern_a"] = baseline_spots
        if str(args.course_strategy) == "reduced_vertices_interlaced":
            interlaced_groups = build_interlaced_groups(baseline_spots, int(args.interlaced_vertices_per_fraction))
            strategy_state["interlaced_groups"] = interlaced_groups
            if not interlaced_groups:
                raise RuntimeError("Reduced-vertices interlaced strategy could not build any lattice subsets.")
            baseline_spots = [tuple(float(v) for v in spot) for spot in interlaced_groups[0]]
        baseline_key = evaluation_cache_key(baseline_spots, None)
        baseline_result = evaluate_plan(
            args=args,
            plan_args=plan_args,
            phantom=phantom,
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            placement_id=placement_counter,
            placement_name="fraction01_baseline",
            spot_centers_mm=baseline_spots,
            reuse_existing_dose_csv=(
                (args.baseline_run_root / "case" / "dosedata.csv")
                if bool(args.reuse_baseline_dose) and str(args.course_strategy) != "reduced_vertices_interlaced" and placement_key(baseline_spots) == placement_key(strategy_state["pattern_a"])
                else None
            ),
        )
        evaluation_cache[baseline_key] = baseline_result
        accepted_sequence.append(baseline_result)
        for spot in baseline_spots:
            key = round_spot(spot)
            history_counts[key] = history_counts.get(key, 0) + 1
        course_summary = compute_cumulative_course_summary(
            args=args,
            structures=structures,
            axes_mm=axes_mm,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            cumulative_physical_dose=baseline_result["physical_dose"],
            accepted_sequence=accepted_sequence,
        )
        course_constraints_ok, course_constraint_details = check_cumulative_constraints(args, course_summary)
        course_objective, course_objective_details = compute_course_objective(args, course_summary)
        fraction_rows.append(
            summarize_fraction_row(
                1,
                baseline_result,
                course_summary,
                course_objective,
                course_constraints_ok,
            )
        )
        placement_counter += 1

    for fraction_idx in range(2, int(args.fractions) + 1):
        oar_weights, weight_details = compute_guidance_oar_weights(course_summary["cumulative_effective_metrics"], args)
        current_repeat_spots = [tuple(float(v) for v in spot) for spot in accepted_sequence[-1]["spot_centers_mm"]]
        fraction_candidate_indices = augment_candidate_pool_with_reference_neighborhood(
            base_candidate_indices=safe_candidate_indices,
            all_candidate_indices=candidate_indices,
            reference_spots_mm=current_repeat_spots,
            axes_mm=axes_mm,
            radius_mm=max(float(args.min_spot_spacing_mm) * 1.2, 18.0),
            per_spot_limit=10,
        )
        adaptive_avoidance_state = None
        adaptive_filter_debug: Dict[str, object] | None = None
        if str(args.course_strategy) == "adaptive_hotspot_avoidance":
            adaptive_avoidance_state = build_adaptive_hotspot_avoidance_state(
                args=args,
                structures=structures,
                axes_mm=axes_mm,
                course_summary=course_summary,
                accepted_sequence=accepted_sequence,
            )
            fraction_candidate_indices, adaptive_filter_debug = filter_candidate_indices_by_adaptive_avoidance(
                candidate_indices=fraction_candidate_indices,
                axes_mm=axes_mm,
                adaptive_state=adaptive_avoidance_state,
                min_pool_size=max(int(args.adaptive_min_filter_pool_size), int(args.num_spots) * 2),
            )
        elif str(args.course_strategy) in {"adaptive_nofly_weight_opt", "phase24_joint_horizon_diversity"}:
            adaptive_avoidance_state = build_phase22_adaptive_state(
                args=args,
                structures=structures,
                axes_mm=axes_mm,
                structure_points_mm=structure_points_mm,
                course_summary=course_summary,
                accepted_sequence=accepted_sequence,
            )
            fraction_candidate_indices, adaptive_filter_debug = filter_candidate_indices_by_phase22_veto(
                args=args,
                candidate_indices=fraction_candidate_indices,
                axes_mm=axes_mm,
                adaptive_state=adaptive_avoidance_state,
                min_pool_size=max(int(args.phase22_no_fly_min_filter_pool_size), int(args.num_spots) * 2),
            )
        scored_centers = score_candidate_centers(
            current_cumulative_effective_dose=course_summary["cumulative_effective_dose"],
            current_cumulative_physical_dose=course_summary["cumulative_physical_dose"],
            current_fraction_idx=fraction_idx,
            structures=structures,
            axes_mm=axes_mm,
            uptake_tensor=uptake_tensor,
            candidate_indices=fraction_candidate_indices,
            target_effective_gy_per_fraction=float(args.target_effective_gy_per_fraction),
            target_need_cap_gy=float(args.target_need_cap_gy),
            oar_weights=oar_weights,
            structure_points_mm=structure_points_mm,
            vessel_coords_mm=vessel_coords_mm,
            history_counts=history_counts,
            adaptive_avoidance=adaptive_avoidance_state,
        )
        candidate_sets = build_strategy_candidate_sets(
            args=args,
            fraction_idx=int(fraction_idx),
            baseline_spots=baseline_spots,
            current_repeat_spots=current_repeat_spots,
            strategy_state=strategy_state,
            scored_centers=scored_centers,
            adaptive_state=adaptive_avoidance_state,
            axes_mm=axes_mm,
            candidate_indices=fraction_candidate_indices,
            num_spots=int(args.num_spots),
            min_spacing_mm=float(args.min_spot_spacing_mm),
            top_k_centers=int(args.candidate_top_k_centers),
            candidate_plan_limit=int(args.candidate_plan_limit),
        )
        allow_repeat_fallback = (
            str(args.course_strategy) in {"adaptive_mutation", "clinical_gtv_core", "larger_pitch_sweep", "randomized_protocol_mix", "vascular_sink_hugging"}
            or (str(args.course_strategy) == "alternating_two_pattern" and "pattern_b" not in strategy_state)
        )
        if allow_repeat_fallback and evaluation_cache_key(current_repeat_spots, None) not in {
            evaluation_cache_key(row["spots_mm"], row.get("plan_overrides")) for row in candidate_sets
        }:
            candidate_sets.insert(
                0,
                {
                    "spots_mm": current_repeat_spots,
                    "heuristic_score": -1.0,
                    "center_debug": [{"note": "repeat_current_accepted_fraction"}],
                    "spacing_limit_mm": float(args.min_spot_spacing_mm),
                    "repeat_fraction_candidate": True,
                    "candidate_origin": "repeat_previous_fraction",
                    "plan_overrides": {},
                },
            )
        if not candidate_sets:
            raise RuntimeError(f"Fraction {fraction_idx}: no candidate spot sets could be assembled from the safe center pool.")

        evaluated_candidates: List[Dict[str, object]] = []
        current_cumulative_physical = np.asarray(course_summary["cumulative_physical_dose"], dtype=np.float32)
        for candidate_rank, candidate in enumerate(candidate_sets, start=1):
            spots_mm = [tuple(float(v) for v in spot) for spot in candidate["spots_mm"]]
            candidate_plan_overrides = candidate.get("plan_overrides") or {}
            key = evaluation_cache_key(spots_mm, candidate_plan_overrides)
            if key in evaluation_cache:
                single_fraction_result = evaluation_cache[key]
            else:
                candidate_plan_args = SimpleNamespace(**vars(plan_args))
                for override_key, override_value in candidate_plan_overrides.items():
                    setattr(candidate_plan_args, str(override_key), float(override_value))
                try:
                    single_fraction_result = evaluate_plan(
                        args=args,
                        plan_args=candidate_plan_args,
                        phantom=phantom,
                        spectrum_energies=spectrum_energies,
                        spectrum_weights=spectrum_weights,
                        placement_id=placement_counter,
                        placement_name=(
                            f"fraction{fraction_idx:02d}_candidate{candidate_rank:02d}_"
                            f"{candidate_label_slug(candidate.get('candidate_origin', 'unknown'))}"
                        ),
                        spot_centers_mm=spots_mm,
                        reuse_existing_dose_csv=None,
                    )
                except RuntimeError as exc:
                    candidate_rows.append(
                        {
                            "fraction": int(fraction_idx),
                            "candidate_rank": int(candidate_rank),
                            "candidate_origin": str(candidate.get("candidate_origin", "unknown")),
                            "spots_mm": "; ".join(f"({x:.1f},{y:.1f},{z:.1f})" for x, y, z in spots_mm),
                            "plan_overrides": json.dumps({str(k): float(v) for k, v in candidate_plan_overrides.items()}, sort_keys=True),
                            "heuristic_score": f"{float(candidate['heuristic_score']):.4f}",
                            "course_objective": "nan",
                            "constraints_ok": "no",
                            "cum_ptv_d95_eff_gy": "nan",
                            "cum_gtv_d95_eff_gy": "nan",
                            "cum_pvdr_eff": "nan",
                            "cum_cord_d2_eff_gy": "nan",
                            "cum_parotid_r_mean_eff_gy": "nan",
                            "cum_thyroid_mean_eff_gy": "nan",
                            "evaluation_error": str(exc),
                        }
                    )
                    placement_counter += 1
                    continue
                evaluation_cache[key] = single_fraction_result
                placement_counter += 1

            trial_sequence = accepted_sequence + [single_fraction_result]
            trial_cumulative_physical = current_cumulative_physical + np.asarray(single_fraction_result["physical_dose"], dtype=np.float32)
            trial_summary = compute_cumulative_course_summary(
                args=args,
                structures=structures,
                axes_mm=axes_mm,
                voxel_volume_cc=voxel_volume_cc,
                voxel_size_mm=voxel_size_mm,
                cumulative_physical_dose=trial_cumulative_physical,
                accepted_sequence=trial_sequence,
            )
            constraints_ok, constraint_details = check_cumulative_constraints(args, trial_summary)
            trial_objective, trial_objective_details = compute_course_objective(args, trial_summary)
            phase24_diversity_bonus = 0.0
            phase24_diversity_details: Dict[str, object] | None = None
            phase24_lookahead_bonus = 0.0
            phase24_lookahead_details: Dict[str, object] | None = None
            selection_score = float(trial_objective)
            if str(args.course_strategy) == "phase24_joint_horizon_diversity":
                phase24_diversity_bonus, phase24_diversity_details = compute_phase24_diversity_bonus(
                    args=args,
                    candidate_spots_mm=spots_mm,
                    accepted_sequence=accepted_sequence,
                )
                phase24_lookahead_bonus, phase24_lookahead_details = estimate_phase24_future_space_bonus(
                    args=args,
                    trial_summary=trial_summary,
                    trial_sequence=trial_sequence,
                    trial_spots_mm=spots_mm,
                    baseline_spots=baseline_spots,
                    strategy_state=strategy_state,
                    safe_candidate_indices=safe_candidate_indices,
                    candidate_indices=candidate_indices,
                    axes_mm=axes_mm,
                    structures=structures,
                    uptake_tensor=uptake_tensor,
                    structure_points_mm=structure_points_mm,
                    vessel_coords_mm=vessel_coords_mm,
                    history_counts=history_counts,
                )
                selection_score = float(trial_objective + phase24_diversity_bonus + phase24_lookahead_bonus)
            evaluated_candidates.append(
                {
                    "candidate_rank": int(candidate_rank),
                    "candidate_origin": str(candidate.get("candidate_origin", "unknown")),
                    "spots_mm": spots_mm,
                    "plan_overrides": {str(k): float(v) for k, v in candidate_plan_overrides.items()},
                    "heuristic_score": float(candidate["heuristic_score"]),
                    "center_debug": candidate["center_debug"],
                    "layout_debug": candidate.get("layout_debug"),
                    "single_fraction_result": single_fraction_result,
                    "course_summary": trial_summary,
                    "course_objective": float(trial_objective),
                    "selection_score": float(selection_score),
                    "phase24_diversity_bonus": float(phase24_diversity_bonus),
                    "phase24_lookahead_bonus": float(phase24_lookahead_bonus),
                    "phase24_diversity_details": phase24_diversity_details,
                    "phase24_lookahead_details": phase24_lookahead_details,
                    "objective_details": trial_objective_details,
                    "constraints_ok": bool(constraints_ok),
                    "constraint_details": constraint_details,
                }
            )
            row = candidate_row(
                fraction_idx,
                candidate_rank,
                spots_mm,
                candidate_origin=str(candidate.get("candidate_origin", "unknown")),
                plan_overrides=candidate_plan_overrides,
                heuristic_score=float(candidate["heuristic_score"]),
                course_objective=float(trial_objective),
                constraints_ok=bool(constraints_ok),
                eff_metrics=trial_summary["cumulative_effective_metrics"],
                pvdr_effective=trial_summary["pvdr_effective"],
                spill_metrics=trial_summary["spill_effective_metrics"],
            )
            row.update(
                {
                    "selection_score": f"{float(selection_score):.4f}",
                    "phase24_diversity_bonus": f"{float(phase24_diversity_bonus):.4f}",
                    "phase24_lookahead_bonus": f"{float(phase24_lookahead_bonus):.4f}",
                }
            )
            candidate_rows.append(row)

        if not evaluated_candidates:
            raise RuntimeError(f"Fraction {fraction_idx}: all candidate evaluations failed before cumulative scoring.")

        feasible_candidates = [row for row in evaluated_candidates if bool(row["constraints_ok"])]
        robust_choice, placement_counter, robust_selection_debug = rerank_top_candidates_with_uncertainty(
            args=args,
            plan_args=plan_args,
            phantom=phantom,
            spectrum_energies=spectrum_energies,
            spectrum_weights=spectrum_weights,
            fraction_idx=int(fraction_idx),
            current_cumulative_physical=current_cumulative_physical,
            accepted_sequence=accepted_sequence,
            voxel_volume_cc=voxel_volume_cc,
            voxel_size_mm=voxel_size_mm,
            structures=structures,
            axes_mm=axes_mm,
            evaluated_candidates=evaluated_candidates,
            candidate_rows=candidate_rows,
            placement_counter=placement_counter,
            robust_evaluation_cache=robust_evaluation_cache,
        )
        if feasible_candidates:
            chosen = robust_choice if robust_choice is not None and bool(robust_choice["constraints_ok"]) else max(
                feasible_candidates, key=lambda row: float(row.get("selection_score", row["course_objective"]))
            )
        else:
            if not bool(args.allow_infeasible_fallback):
                break
            chosen = robust_choice if robust_choice is not None else max(
                evaluated_candidates, key=lambda row: float(row.get("selection_score", row["course_objective"]))
            )

        accepted_result = chosen["single_fraction_result"]
        accepted_sequence.append(accepted_result)
        if str(args.course_strategy) == "alternating_two_pattern" and int(fraction_idx) % 2 == 0:
            if str(chosen.get("candidate_origin", "")).startswith("alternate_") or str(chosen.get("candidate_origin", "")).startswith("single_mutation") or str(chosen.get("candidate_origin", "")).startswith("double_mutation") or str(chosen.get("candidate_origin", "")).startswith("global_combo"):
                if placement_key(accepted_result["spot_centers_mm"]) != placement_key(strategy_state.get("pattern_a", baseline_spots)):
                    strategy_state["pattern_b"] = [tuple(float(v) for v in spot) for spot in accepted_result["spot_centers_mm"]]
        for spot in accepted_result["spot_centers_mm"]:
            key = round_spot(spot)
            history_counts[key] = history_counts.get(key, 0) + 1

        course_summary = chosen["course_summary"]
        course_objective = float(chosen["course_objective"])
        course_constraints_ok = bool(chosen["constraints_ok"])
        course_constraint_details = chosen["constraint_details"]
        course_objective_details = chosen["objective_details"]
        fraction_rows.append(
            summarize_fraction_row(
                fraction_idx,
                accepted_result,
                course_summary,
                course_objective,
                course_constraints_ok,
            )
        )

        write_text_with_retries(
            args.run_root / f"fraction{fraction_idx:02d}_selection_debug.json",
            json.dumps(
                {
                    "fraction": int(fraction_idx),
                    "guidance_weights": {key: float(value) for key, value in oar_weights.items()},
                    "guidance_weight_details": {
                        name: {
                            "metric": info[0],
                            "value": float(info[1]),
                            "threshold": float(info[2]),
                            "relative_exceedance": float(info[3]),
                        }
                        for name, info in weight_details.items()
                    },
                    "adaptive_avoidance_state": {
                        key: value
                        for key, value in (adaptive_avoidance_state or {}).items()
                        if not str(key).endswith("_points_mm")
                    },
                    "adaptive_filter_debug": adaptive_filter_debug,
                    "robust_selection_debug": robust_selection_debug,
                    "candidates": [
                        {
                            "candidate_rank": int(row["candidate_rank"]),
                            "candidate_origin": str(row.get("candidate_origin", "unknown")),
                            "spots_mm": [[float(a), float(b), float(c)] for a, b, c in row["single_fraction_result"]["spot_centers_mm"]],
                            "heuristic_score": float(row["heuristic_score"]),
                            "course_objective": float(row["course_objective"]),
                            "selection_score": float(row.get("selection_score", row["course_objective"])),
                            "constraints_ok": bool(row["constraints_ok"]),
                            "objective_details": row["objective_details"],
                            "constraint_details": row["constraint_details"],
                            "layout_debug": row.get("layout_debug"),
                            "robust_stats": row.get("robust_stats"),
                            "phase24_diversity_bonus": float(row.get("phase24_diversity_bonus", 0.0)),
                            "phase24_lookahead_bonus": float(row.get("phase24_lookahead_bonus", 0.0)),
                            "phase24_diversity_details": row.get("phase24_diversity_details"),
                            "phase24_lookahead_details": row.get("phase24_lookahead_details"),
                        }
                        for row in evaluated_candidates
                    ],
                    "selected_candidate_rank": int(chosen["candidate_rank"]),
                    "selected_constraints_ok": bool(course_constraints_ok),
                    "selected_objective": float(course_objective),
                    "selected_constraint_details": course_constraint_details,
                    "selected_objective_details": course_objective_details,
                },
                indent=2,
            ),
        )

    summary = {
        "phase": int(args.phase_number),
        "description": str(args.phase_description),
        "course_strategy": str(args.course_strategy),
        "objective_mode": str(args.objective_mode),
        "baseline_run_root": str(args.baseline_run_root),
        "run_root": str(args.run_root),
        "fractions_requested": int(args.fractions),
        "fractions_accepted": int(len(accepted_sequence)),
        "adaptive_veto_relax_mode": str(args.adaptive_veto_relax_mode),
        "adaptive_veto_relax_fraction": float(args.adaptive_veto_relax_fraction),
        "robust_top_k_candidates": int(args.robust_top_k_candidates),
        "robust_eval_specs": [
            {"seed": int(seed), "histories": int(histories)}
            for seed, histories in list(getattr(args, "_robust_eval_specs", []))
        ],
        "robust_z_score": float(args.robust_z_score),
        "robust_min_feasible_fraction": float(args.robust_min_feasible_fraction),
        "phase24_min_vertices": int(args.phase24_min_vertices),
        "phase24_max_vertices": int(args.phase24_max_vertices),
        "phase24_template_pitch_scales": [float(v) for v in args.phase24_template_pitch_scales],
        "phase24_soft_min_spacing_mm": float(args.phase24_soft_min_spacing_mm),
        "phase24_diversity_weight": float(args.phase24_diversity_weight),
        "phase24_lookahead_weight": float(args.phase24_lookahead_weight),
        "strategy_state": {
            key: value
            for key, value in strategy_state.items()
        },
        "candidate_pool_debug": candidate_pool_debug,
        "fraction_rows": fraction_rows,
        "final_course_objective": float(course_objective),
        "final_constraints_ok": bool(course_constraints_ok),
        "final_constraint_details": course_constraint_details,
        "final_objective_details": course_objective_details,
        "final_cumulative_physical_metrics": course_summary["cumulative_physical_metrics"],
        "final_cumulative_effective_metrics": course_summary["cumulative_effective_metrics"],
        "final_pvdr_physical": course_summary["pvdr_physical"],
        "final_pvdr_effective": course_summary["pvdr_effective"],
        "final_spill_physical_metrics": course_summary["spill_physical_metrics"],
        "final_spill_effective_metrics": course_summary["spill_effective_metrics"],
        "consistency_terms": course_summary["consistency_terms"],
        "accepted_sequence": [
            {
                "fraction": idx + 1,
                "placement_name": row["placement_name"],
                "spot_centers_mm": [[float(a), float(b), float(c)] for a, b, c in accepted_sequence[idx]["spot_centers_mm"]],
                "single_fraction_effective_metrics": accepted_sequence[idx]["effective_metrics"],
            }
            for idx, row in enumerate(fraction_rows)
        ],
    }

    write_csv(args.run_root / "fraction_sequence_summary.csv", fraction_rows)
    write_markdown_table(args.run_root / "fraction_sequence_summary.md", fraction_rows)
    if candidate_rows:
        write_csv(args.run_root / "fraction_candidate_evaluations.csv", candidate_rows)
        write_markdown_table(args.run_root / "fraction_candidate_evaluations.md", candidate_rows)
    write_text_with_retries(args.run_root / "phase17_fraction_aware_summary.json", json.dumps(summary, indent=2))

    print(f"=== PHASE {int(args.phase_number)} FRACTION-AWARE BIO-GUIDED OPTIMIZATION COMPLETE ===")
    print(f"Fractions accepted: {len(accepted_sequence)} / {int(args.fractions)}")
    print(f"Final course objective: {float(course_objective):.4f}")
    print(f"Final cumulative PTV D95(eq): {float(course_summary['cumulative_effective_metrics']['PTV']['d95_gy']):.4f} Gy")
    print(f"Final cumulative cord D2(eq): {float(course_summary['cumulative_effective_metrics']['SPINAL_CORD']['d2_gy']):.4f} Gy")
    print(f"Final cumulative PVDR(eq): {float(course_summary['pvdr_effective']['pvdr']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
