#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


DESKTOP = Path("/Users/kw/Desktop")

DEFAULT_HALF_FIELD_CANDIDATES = [
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization_tuned_tail"
    / "half_field_profile.csv",
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization"
    / "half_field_profile.csv",
    DESKTOP
    / "SFRT_repo_main_update"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization_tuned_tail"
    / "half_field_profile.csv",
    DESKTOP
    / "SFRT_repo_main_update"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization"
    / "half_field_profile.csv",
]

DEFAULT_STRIPE_CANDIDATES = [
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization_tuned_tail"
    / "stripe_validation_profiles.csv",
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization"
    / "stripe_validation_profiles.csv",
    DESKTOP
    / "SFRT_repo_main_update"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization_tuned_tail"
    / "stripe_validation_profiles.csv",
    DESKTOP
    / "SFRT_repo_main_update"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase8_geometry_generalization"
    / "stripe_validation_profiles.csv",
]

DEFAULT_HOLDOUT_CANDIDATES = [
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase9_holdout_3d_lattice"
    / "phase9_holdout_3d_lattice_metrics.csv",
    DESKTOP
    / "SFRT_repo_main_update"
    / "project"
    / "runs"
    / "linac_6mv_polyenergetic_clinical_sfrt"
    / "analysis_phase9_holdout_3d_lattice"
    / "phase9_holdout_3d_lattice_metrics.csv",
]

SYNC_PNG_TARGETS = [
    DESKTOP / "fig02_calibration_transfer.png",
    DESKTOP / "PMB_revised" / "PMB_SFRT_publishable_source_clean" / "figures" / "fig02_calibration_transfer.png",
    DESKTOP
    / "PMB_revised_conservative"
    / "PMB_SFRT_publishable_source_clean"
    / "figures"
    / "fig02_calibration_transfer.png",
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "figures"
    / "PMB_SFRT_publishable_source_clean"
    / "figures"
    / "fig02_calibration_transfer.png",
    DESKTOP / "PMB_Overleaf_Master_Figures_20260622" / "figures" / "fig02_calibration_transfer.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PMB Fig. 02 from the source CSVs as SVG, then convert to PNG/PDF."
    )
    parser.add_argument("--half-field-csv", type=Path, help="Override half-field calibration CSV.")
    parser.add_argument("--stripe-csv", type=Path, help="Override stripe transfer CSV.")
    parser.add_argument("--holdout-csv", type=Path, help="Override 3D holdout CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DESKTOP / "PMB_revised_conservative" / "PMB_SFRT_publishable_source_clean" / "figures",
        help="Directory to write the regenerated figure files.",
    )
    parser.add_argument(
        "--basename",
        default="fig02_calibration_transfer",
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not copy the regenerated PNG into the standard Desktop bundle paths.",
    )
    return parser.parse_args()


def resolve_existing_path(explicit: Path | None, candidates: list[Path], label: str) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"{label} not found: {explicit}")
        return explicit
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to resolve {label}. Tried:\n{tried}")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_pitch_rows(rows: list[dict[str, str]], pitch_mm: float) -> list[dict[str, str]]:
    result = [row for row in rows if abs(float(row["pitch_mm"]) - pitch_mm) < 1e-6]
    result.sort(key=lambda row: float(row["x_mm"]))
    return result


def nearest_row(rows: list[dict[str, str]], x_mm: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(float(row["x_mm"]) - x_mm))


@dataclass
class Chart:
    x: float
    y: float
    width: float
    height: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def plot_x(self) -> float:
        return self.x + self.margin_left

    @property
    def plot_y(self) -> float:
        return self.y + self.margin_top

    @property
    def plot_width(self) -> float:
        return self.width - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> float:
        return self.height - self.margin_top - self.margin_bottom

    def sx(self, x_value: float) -> float:
        fraction = (x_value - self.x_min) / (self.x_max - self.x_min)
        return self.plot_x + fraction * self.plot_width

    def sy(self, y_value: float) -> float:
        fraction = (y_value - self.y_min) / (self.y_max - self.y_min)
        return self.plot_y + (1.0 - fraction) * self.plot_height


class SvgBuilder:
    def __init__(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.defs: list[str] = []
        self.body: list[str] = []

    def add(self, element: str) -> None:
        self.body.append(element)

    def add_clip(self, clip_id: str, chart: Chart) -> None:
        self.defs.append(
            (
                f'<clipPath id="{escape(clip_id)}">'
                f'<rect x="{chart.plot_x:.2f}" y="{chart.plot_y:.2f}" '
                f'width="{chart.plot_width:.2f}" height="{chart.plot_height:.2f}" />'
                f"</clipPath>"
            )
        )

    def to_svg(self) -> str:
        defs_block = ""
        if self.defs:
            defs_block = "<defs>\n" + "\n".join(self.defs) + "\n</defs>\n"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="{self.width}" height="{self.height}" fill="white" />\n'
            f"{defs_block}"
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


def fmt_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.1f}"


def polyline_points(chart: Chart, x_values: list[float], y_values: list[float]) -> str:
    points = []
    for x_value, y_value in zip(x_values, y_values):
        points.append(f"{chart.sx(x_value):.2f},{chart.sy(y_value):.2f}")
    return " ".join(points)


def draw_chart_frame(
    svg: SvgBuilder,
    chart: Chart,
    *,
    clip_id: str,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str | None,
    y_label: str | None,
    show_x_tick_labels: bool,
    panel_label: str | None = None,
    panel_label_outside: bool = False,
) -> None:
    svg.add_clip(clip_id, chart)

    grid_color = "#d1d1d1"
    border_color = "#c9c9c9"
    tick_color = "#555555"

    for x_tick in x_ticks:
        x_px = chart.sx(x_tick)
        svg.add(
            f'<line x1="{x_px:.2f}" y1="{chart.plot_y:.2f}" x2="{x_px:.2f}" '
            f'y2="{chart.plot_y + chart.plot_height:.2f}" stroke="{grid_color}" stroke-width="1" />'
        )
    for y_tick in y_ticks:
        y_px = chart.sy(y_tick)
        svg.add(
            f'<line x1="{chart.plot_x:.2f}" y1="{y_px:.2f}" x2="{chart.plot_x + chart.plot_width:.2f}" '
            f'y2="{y_px:.2f}" stroke="{grid_color}" stroke-width="1" />'
        )

    svg.add(
        f'<rect x="{chart.plot_x:.2f}" y="{chart.plot_y:.2f}" width="{chart.plot_width:.2f}" '
        f'height="{chart.plot_height:.2f}" fill="none" stroke="{border_color}" stroke-width="2" />'
    )

    for y_tick in y_ticks:
        y_px = chart.sy(y_tick)
        svg.add(
            f'<text x="{chart.plot_x - 10:.2f}" y="{y_px + 6:.2f}" font-family="Arial" font-size="16" '
            f'fill="{tick_color}" text-anchor="end">{escape(fmt_tick(y_tick))}</text>'
        )

    if show_x_tick_labels:
        for x_tick in x_ticks:
            x_px = chart.sx(x_tick)
            svg.add(
                f'<text x="{x_px:.2f}" y="{chart.plot_y + chart.plot_height + 30:.2f}" font-family="Arial" '
                f'font-size="16" fill="{tick_color}" text-anchor="middle">{escape(fmt_tick(x_tick))}</text>'
            )

    if x_label:
        svg.add(
            f'<text x="{chart.plot_x + chart.plot_width / 2.0:.2f}" y="{chart.y + chart.height - 8:.2f}" '
            f'font-family="Arial" font-size="20" fill="black" text-anchor="middle">{escape(x_label)}</text>'
        )

    if y_label:
        x_pos = chart.x + 22
        y_pos = chart.plot_y + chart.plot_height / 2.0
        svg.add(
            f'<text x="{x_pos:.2f}" y="{y_pos:.2f}" font-family="Arial" font-size="20" fill="black" '
            f'text-anchor="middle" transform="rotate(-90 {x_pos:.2f} {y_pos:.2f})">{escape(y_label)}</text>'
        )

    if panel_label:
        if panel_label_outside:
            box_x = chart.plot_x - 2
            box_y = chart.plot_y - 4
        else:
            box_x = chart.plot_x + 8
            box_y = chart.plot_y + 8
        box_w = 52
        box_h = 46
        text_x = box_x + box_w / 2.0
        text_y = box_y + 31
        svg.add(
            f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_w:.2f}" height="{box_h:.2f}" '
            f'fill="white" stroke="black" stroke-width="1.2" />'
        )
        svg.add(
            f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Arial" font-size="20" font-weight="700" '
            f'fill="black" text-anchor="middle">{escape(panel_label)}</text>'
        )


def draw_polyline(
    svg: SvgBuilder,
    chart: Chart,
    *,
    clip_id: str,
    x_values: list[float],
    y_values: list[float],
    stroke: str,
    stroke_width: float,
    dasharray: str | None = None,
) -> None:
    attributes = [
        f'points="{polyline_points(chart, x_values, y_values)}"',
        f'fill="none"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
        f'clip-path="url(#{escape(clip_id)})"',
        'stroke-linejoin="round"',
        'stroke-linecap="round"',
    ]
    if dasharray:
        attributes.append(f'stroke-dasharray="{dasharray}"')
    svg.add(f"<polyline {' '.join(attributes)} />")


def draw_vertical_line(
    svg: SvgBuilder,
    chart: Chart,
    *,
    x_value: float,
    stroke: str,
    stroke_width: float,
    dasharray: str | None = None,
) -> None:
    x_px = chart.sx(x_value)
    attributes = [
        f'x1="{x_px:.2f}"',
        f'y1="{chart.plot_y:.2f}"',
        f'x2="{x_px:.2f}"',
        f'y2="{chart.plot_y + chart.plot_height:.2f}"',
        f'stroke="{stroke}"',
        f'stroke-width="{stroke_width}"',
    ]
    if dasharray:
        attributes.append(f'stroke-dasharray="{dasharray}"')
    svg.add(f"<line {' '.join(attributes)} />")


def draw_circle(svg: SvgBuilder, chart: Chart, *, x_value: float, y_value: float, radius: float, fill: str) -> None:
    svg.add(
        f'<circle cx="{chart.sx(x_value):.2f}" cy="{chart.sy(y_value):.2f}" r="{radius:.2f}" fill="{fill}" />'
    )


def draw_text(
    svg: SvgBuilder,
    *,
    x: float,
    y: float,
    text: str,
    size: int = 17,
    color: str = "#444444",
    anchor: str = "start",
) -> None:
    svg.add(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def convert_with_sips(source: Path, target: Path, fmt: str) -> None:
    subprocess.run(
        ["sips", "-s", "format", fmt, str(source), "--out", str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sync_png(source_png: Path) -> list[Path]:
    copied: list[Path] = []
    for target in SYNC_PNG_TARGETS:
        if target.resolve() == source_png.resolve():
            copied.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_png, target)
        copied.append(target)
    return copied


def main() -> int:
    args = parse_args()
    half_csv = resolve_existing_path(args.half_field_csv, DEFAULT_HALF_FIELD_CANDIDATES, "half-field CSV")
    stripe_csv = resolve_existing_path(args.stripe_csv, DEFAULT_STRIPE_CANDIDATES, "stripe CSV")
    holdout_csv = resolve_existing_path(args.holdout_csv, DEFAULT_HOLDOUT_CANDIDATES, "3D holdout CSV")

    half_rows = load_rows(half_csv)
    half_rows.sort(key=lambda row: float(row["x_mm"]))
    stripe_rows = load_rows(stripe_csv)
    holdout_rows = load_rows(holdout_csv)
    holdout_rows.sort(key=lambda row: float(row["pitch_mm"]))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{args.basename}.svg"
    png_path = out_dir / f"{args.basename}.png"
    pdf_path = out_dir / f"{args.basename}.pdf"

    width = 1900
    height = 1200
    svg = SvgBuilder(width=width, height=height)

    chart_a1 = Chart(40, 20, 960, 280, 78, 18, 10, 20, -88, 88, -0.5, 10.3)
    chart_a2 = Chart(40, 310, 960, 320, 78, 18, 10, 58, -88, 88, 0.0, 1.02)
    chart_b1 = Chart(1030, 20, 850, 160, 58, 14, 10, 20, -88, 88, 0.0, 1.05)
    chart_b2 = Chart(1030, 220, 850, 160, 58, 14, 10, 20, -88, 88, 0.0, 1.05)
    chart_b3 = Chart(1030, 420, 850, 210, 58, 14, 10, 58, -88, 88, 0.0, 1.05)
    chart_c = Chart(40, 670, 1840, 500, 88, 18, 14, 68, 19.0, 41.0, 1.8, 5.95)

    x_ticks_ab = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
    y_ticks_a1 = [0, 2, 4, 6, 8, 10]
    y_ticks_a2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_ticks_b = [0.0, 0.5, 1.0]
    x_ticks_c = [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
    y_ticks_c = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    draw_chart_frame(
        svg,
        chart_a1,
        clip_id="clip_a1",
        x_ticks=x_ticks_ab,
        y_ticks=y_ticks_a1,
        x_label=None,
        y_label="Dose (Gy)",
        show_x_tick_labels=False,
        panel_label="(a)",
    )
    draw_chart_frame(
        svg,
        chart_a2,
        clip_id="clip_a2",
        x_ticks=x_ticks_ab,
        y_ticks=y_ticks_a2,
        x_label="Distance from field edge x (mm)",
        y_label="Cell survival",
        show_x_tick_labels=True,
    )
    draw_chart_frame(
        svg,
        chart_b1,
        clip_id="clip_b1",
        x_ticks=x_ticks_ab,
        y_ticks=y_ticks_b,
        x_label=None,
        y_label="Survival",
        show_x_tick_labels=False,
        panel_label="(b)",
    )
    draw_chart_frame(
        svg,
        chart_b2,
        clip_id="clip_b2",
        x_ticks=x_ticks_ab,
        y_ticks=y_ticks_b,
        x_label=None,
        y_label="Survival",
        show_x_tick_labels=False,
    )
    draw_chart_frame(
        svg,
        chart_b3,
        clip_id="clip_b3",
        x_ticks=x_ticks_ab,
        y_ticks=y_ticks_b,
        x_label="Lateral position x (mm)",
        y_label="Survival",
        show_x_tick_labels=True,
    )
    draw_chart_frame(
        svg,
        chart_c,
        clip_id="clip_c",
        x_ticks=x_ticks_c,
        y_ticks=y_ticks_c,
        x_label="Lattice pitch (mm)",
        y_label="Effective dose (Gy)",
        show_x_tick_labels=True,
        panel_label="(c)",
        panel_label_outside=True,
    )

    half_x = [float(row["x_mm"]) for row in half_rows]
    half_dose = [float(row["dose_gy"]) for row in half_rows]
    half_lq = [float(row["survival_lq"]) for row in half_rows]
    half_total = [float(row["survival_total"]) for row in half_rows]

    draw_polyline(svg, chart_a1, clip_id="clip_a1", x_values=half_x, y_values=half_dose, stroke="#34495e", stroke_width=3.0)
    draw_vertical_line(svg, chart_a1, x_value=0.0, stroke="#7f8c8d", stroke_width=1.4, dasharray="8 6")

    draw_polyline(svg, chart_a2, clip_id="clip_a2", x_values=half_x, y_values=half_lq, stroke="#1f77b4", stroke_width=2.6)
    draw_polyline(svg, chart_a2, clip_id="clip_a2", x_values=half_x, y_values=half_total, stroke="#d62728", stroke_width=3.0)
    draw_vertical_line(svg, chart_a2, x_value=0.0, stroke="#7f8c8d", stroke_width=1.4, dasharray="8 6")

    anchor_2 = nearest_row(half_rows, 2.0)
    anchor_10 = nearest_row(half_rows, 10.0)
    anchor_2_y = float(anchor_2["survival_total"])
    anchor_10_y = float(anchor_10["survival_total"])
    draw_circle(svg, chart_a2, x_value=2.0, y_value=anchor_2_y, radius=5.5, fill="#d62728")
    draw_circle(svg, chart_a2, x_value=10.0, y_value=anchor_10_y, radius=5.5, fill="#d62728")
    draw_text(
        svg,
        x=chart_a2.sx(2.0) + 24,
        y=chart_a2.sy(anchor_2_y) - 12,
        text=f"{anchor_2_y:.2f}",
        size=22,
    )
    draw_text(
        svg,
        x=chart_a2.sx(10.0) + 20,
        y=chart_a2.sy(anchor_10_y) - 10,
        text=f"{anchor_10_y:.2f}",
        size=22,
    )

    for pitch_mm, chart in ((20.0, chart_b1), (30.0, chart_b2), (40.0, chart_b3)):
        pitch_rows = filter_pitch_rows(stripe_rows, pitch_mm)
        x_values = [float(row["x_mm"]) for row in pitch_rows]
        dose_norm = [float(row["dose_gy"]) / 10.0 for row in pitch_rows]
        survival = [float(row["survival_total"]) for row in pitch_rows]
        clip_id = f"clip_b{int((pitch_mm - 10) / 10)}"
        draw_polyline(svg, chart, clip_id=clip_id, x_values=x_values, y_values=dose_norm, stroke="#1f77b4", stroke_width=2.6)
        draw_polyline(svg, chart, clip_id=clip_id, x_values=x_values, y_values=survival, stroke="#d62728", stroke_width=3.0)
        draw_text(
            svg,
            x=chart.plot_x + chart.plot_width - 10,
            y=chart.plot_y + 32,
            text=f"{int(pitch_mm)} mm",
            size=22,
            anchor="end",
        )

    holdout_pitch = [float(row["pitch_mm"]) for row in holdout_rows]
    holdout_deff = [float(row["valley_effective_dose_gy"]) for row in holdout_rows]
    holdout_sf = [float(row["valley_survival_total"]) for row in holdout_rows]
    draw_polyline(svg, chart_c, clip_id="clip_c", x_values=holdout_pitch, y_values=holdout_deff, stroke="#d62728", stroke_width=3.4)
    for pitch_mm, deff_gy in zip(holdout_pitch, holdout_deff):
        draw_circle(svg, chart_c, x_value=pitch_mm, y_value=deff_gy, radius=6.0, fill="#d62728")
    for pitch_mm, deff_gy, sf_value in zip(holdout_pitch, holdout_deff, holdout_sf):
        if math.isclose(pitch_mm, 20.0):
            x_px = chart_c.sx(pitch_mm) - 6
            y_px = chart_c.sy(deff_gy) - 22
            anchor = "start"
        elif math.isclose(pitch_mm, 30.0):
            x_px = chart_c.sx(pitch_mm)
            y_px = chart_c.sy(deff_gy) - 20
            anchor = "middle"
        else:
            x_px = chart_c.sx(pitch_mm) - 40
            y_px = chart_c.sy(deff_gy) - 16
            anchor = "start"
        draw_text(svg, x=x_px, y=y_px, text=f"SF={sf_value:.3f}", size=22, anchor=anchor)

    svg_path.write_text(svg.to_svg(), encoding="utf-8")
    convert_with_sips(svg_path, png_path, "png")
    convert_with_sips(svg_path, pdf_path, "pdf")

    print(f"HALF_FIELD_CSV: {half_csv}")
    print(f"STRIPE_CSV: {stripe_csv}")
    print(f"HOLDOUT_CSV: {holdout_csv}")
    print(f"OUTPUT_SVG: {svg_path}")
    print(f"OUTPUT_PNG: {png_path}")
    print(f"OUTPUT_PDF: {pdf_path}")

    if not args.no_sync:
        for target in sync_png(png_path):
            print(f"SYNCED_PNG: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
