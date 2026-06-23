#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


DESKTOP = Path("/Users/kw/Desktop")

DEFAULT_TABLE = (
    DESKTOP
    / "SFRT_repo_docs_fix"
    / "project"
    / "public_results"
    / "revision_checks_20260616"
    / "step06_falsification_final_table.csv"
)

SYNC_PNG_TARGETS = [
    DESKTOP / "PMB_revised" / "PMB_SFRT_publishable_source_clean" / "figures" / "fig05b_sink_falsification.png",
    DESKTOP / "PMB_revised_conservative" / "PMB_SFRT_publishable_source_clean" / "figures" / "fig05b_sink_falsification.png",
    DESKTOP / "PMB_Overleaf_Master_Figures_20260622" / "figures" / "fig05b_sink_falsification.png",
    DESKTOP
    / "SFRT_Submission_reproducibility_bundle_ready"
    / "figures"
    / "PMB_SFRT_publishable_source_clean"
    / "figures"
    / "fig05b_sink_falsification.png",
]

SYNC_SVG_PDF_TARGETS = [
    DESKTOP / "PMB_revised" / "PMB_SFRT_publishable_source_clean" / "figures",
    DESKTOP / "PMB_revised_conservative" / "PMB_SFRT_publishable_source_clean" / "figures",
    DESKTOP / "PMB_Overleaf_Master_Figures_20260622" / "figures",
]

COMPARATOR_ORDER = [
    "no_sink",
    "uniform_body_sink_mass_matched",
    "local_dropout_sink_20mm",
    "si_flip_sink",
    "ap_flip_sink",
    "randomized_displacement_sink",
]

LABELS = {
    "no_sink": "No sink",
    "uniform_body_sink_mass_matched": "Uniform\nsink",
    "local_dropout_sink_20mm": "Local\ndropout",
    "si_flip_sink": "SI-flip",
    "ap_flip_sink": "AP-flip",
    "randomized_displacement_sink": "Random\nshift",
}

COLORS = {
    "no_sink": "#4b6179",
    "uniform_body_sink_mass_matched": "#d95f02",
    "local_dropout_sink_20mm": "#2ca089",
    "si_flip_sink": "#8e44ad",
    "ap_flip_sink": "#3f86b9",
    "randomized_displacement_sink": "#9aa8ab",
}


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

    def sy(self, value: float) -> float:
        frac = (value - self.y_min) / (self.y_max - self.y_min)
        return self.plot_y + (1.0 - frac) * self.plot_height


class SvgBuilder:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.body: list[str] = []

    def add(self, text: str) -> None:
        self.body.append(text)

    def to_svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="{self.width}" height="{self.height}" fill="white" />\n'
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fig05b_sink_falsification from the updated revision-check falsification table."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Updated falsification summary table.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DESKTOP / "PMB_revised" / "PMB_SFRT_publishable_source_clean" / "figures",
        help="Output directory.",
    )
    parser.add_argument("--basename", default="fig05b_sink_falsification", help="Output file stem.")
    parser.add_argument("--no-sync", action="store_true", help="Do not sync the generated assets to bundle paths.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_observed(observed: str) -> tuple[int | None, int | None, int | None, int | None, int | None, int | None]:
    endpoint_match = re.search(r"(\d+)/(\d+)\s+endpoint summaries excluded zero", observed)
    noise_match = re.search(r"(\d+)/(\d+)\s+case-endpoint rows exceeded 95% noise band", observed)
    rank_match = re.search(r"(\d+)/(\d+)\s+noise-qualified rank changes", observed)
    return (
        int(endpoint_match.group(1)) if endpoint_match else None,
        int(endpoint_match.group(2)) if endpoint_match else None,
        int(noise_match.group(1)) if noise_match else None,
        int(noise_match.group(2)) if noise_match else None,
        int(rank_match.group(1)) if rank_match else None,
        int(rank_match.group(2)) if rank_match else None,
    )


def draw_text(svg: SvgBuilder, *, x: float, y: float, text: str, size: int, anchor: str = "start", weight: int = 400) -> None:
    svg.add(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial" font-size="{size}" font-weight="{weight}" '
        f'fill="black" text-anchor="{anchor}">{escape(text)}</text>'
    )


def draw_multiline_text(
    svg: SvgBuilder,
    *,
    x: float,
    y: float,
    text: str,
    size: int,
    anchor: str = "middle",
    line_height: float = 1.08,
    weight: int = 400,
) -> None:
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        dy = idx * size * line_height
        draw_text(svg, x=x, y=y + dy, text=line, size=size, anchor=anchor, weight=weight)


def draw_panel_label(svg: SvgBuilder, *, chart: Chart, label: str) -> None:
    box_w = 62
    box_h = 50
    box_x = chart.plot_x + chart.plot_width - box_w + 6
    box_y = chart.plot_y + 8
    svg.add(
        f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_w:.2f}" height="{box_h:.2f}" '
        f'fill="white" stroke="black" stroke-width="1.4" />'
    )
    draw_text(svg, x=box_x + box_w / 2.0, y=box_y + 34, text=label, size=22, anchor="middle", weight=700)


def draw_chart(
    svg: SvgBuilder,
    *,
    chart: Chart,
    values: list[float],
    labels: list[str],
    colors: list[str],
    y_ticks: list[float],
    y_label: str,
    panel_label: str,
) -> None:
    grid_color = "#d6d6d6"
    border_color = "#c9c9c9"
    tick_color = "#4d4d4d"

    for y_tick in y_ticks:
        y_px = chart.sy(y_tick)
        svg.add(
            f'<line x1="{chart.plot_x:.2f}" y1="{y_px:.2f}" x2="{chart.plot_x + chart.plot_width:.2f}" y2="{y_px:.2f}" '
            f'stroke="{grid_color}" stroke-width="1" />'
        )
        draw_text(svg, x=chart.plot_x - 12, y=y_px + 6, text=f"{y_tick:.1f}" if y_tick % 1 else f"{int(y_tick)}", size=22, anchor="end")

    svg.add(
        f'<rect x="{chart.plot_x:.2f}" y="{chart.plot_y:.2f}" width="{chart.plot_width:.2f}" height="{chart.plot_height:.2f}" '
        f'fill="none" stroke="{border_color}" stroke-width="1.8" />'
    )

    n = len(values)
    slot = chart.plot_width / n
    bar_w = slot * 0.78
    for idx, (value, label, color) in enumerate(zip(values, labels, colors)):
        x_left = chart.plot_x + idx * slot + (slot - bar_w) / 2.0
        y_top = chart.sy(value)
        bar_h = chart.plot_y + chart.plot_height - y_top
        svg.add(
            f'<rect x="{x_left:.2f}" y="{y_top:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" />'
        )
        draw_multiline_text(
            svg,
            x=x_left + bar_w / 2.0,
            y=chart.plot_y + chart.plot_height + 36,
            text=label,
            size=21,
            anchor="middle",
            line_height=1.0,
        )

    x_label_x = chart.plot_x + chart.plot_width / 2.0
    draw_text(svg, x=x_label_x, y=chart.y + chart.height - 8, text="Comparator control", size=20, anchor="middle")

    y_label_x = chart.x + 20
    y_label_y = chart.plot_y + chart.plot_height / 2.0
    svg.add(
        f'<text x="{y_label_x:.2f}" y="{y_label_y:.2f}" font-family="Arial" font-size="20" fill="{tick_color}" '
        f'text-anchor="middle" transform="rotate(-90 {y_label_x:.2f} {y_label_y:.2f})">{escape(y_label)}</text>'
    )

    draw_panel_label(svg, chart=chart, label=panel_label)


def convert_with_sips(source: Path, target: Path, fmt: str) -> None:
    subprocess.run(
        ["sips", "-s", "format", fmt, str(source), "--out", str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sync_assets(png_path: Path, svg_path: Path, pdf_path: Path) -> None:
    for target in SYNC_PNG_TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != png_path.resolve():
            shutil.copy2(png_path, target)
    for target_dir in SYNC_SVG_PDF_TARGETS:
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in (svg_path, pdf_path):
            target = target_dir / source.name
            if target.resolve() != source.resolve():
                shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.table)
    lookup = {row["mode"]: row for row in rows}

    endpoint_values: list[float] = []
    noise_fraction_values: list[float] = []
    rank_values: list[float] = []
    label_values: list[str] = []
    color_values: list[str] = []

    for comparator in COMPARATOR_ORDER:
        row = lookup[comparator]
        ep_num, ep_den, noise_num, noise_den, rank_num, rank_den = parse_observed(row["observed"])
        if ep_num is None or noise_num is None or rank_num is None:
            raise ValueError(f"Unable to parse falsification summary for comparator '{comparator}': {row['observed']}")
        endpoint_values.append(float(ep_num))
        noise_fraction_values.append(float(noise_num) / float(noise_den) if noise_den else 0.0)
        rank_values.append(float(rank_num))
        label_values.append(LABELS[comparator])
        color_values.append(COLORS[comparator])

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{args.basename}.svg"
    png_path = out_dir / f"{args.basename}.png"
    pdf_path = out_dir / f"{args.basename}.pdf"

    width = 2200
    height = 970
    svg = SvgBuilder(width=width, height=height)

    chart_a = Chart(40, 28, 690, 860, 88, 18, 18, 150, 0.0, 8.6)
    chart_b = Chart(755, 28, 690, 860, 88, 18, 18, 150, 0.0, 1.04)
    chart_c = Chart(1470, 28, 690, 860, 88, 18, 18, 150, 0.0, 2.3)

    draw_chart(
        svg,
        chart=chart_a,
        values=endpoint_values,
        labels=label_values,
        colors=color_values,
        y_ticks=[0, 2, 4, 6, 8],
        y_label="Endpoint summaries excl. zero CI",
        panel_label="(a)",
    )
    draw_chart(
        svg,
        chart=chart_b,
        values=noise_fraction_values,
        labels=label_values,
        colors=color_values,
        y_ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
        y_label="Fraction above 95% band",
        panel_label="(b)",
    )
    draw_chart(
        svg,
        chart=chart_c,
        values=rank_values,
        labels=label_values,
        colors=color_values,
        y_ticks=[0, 1, 2],
        y_label="Noise-qualified rank changes",
        panel_label="(c)",
    )

    svg_path.write_text(svg.to_svg(), encoding="utf-8")
    convert_with_sips(svg_path, png_path, "png")
    convert_with_sips(svg_path, pdf_path, "pdf")

    print(f"DATA_TABLE: {args.table}")
    for comparator, ep, frac, rank in zip(COMPARATOR_ORDER, endpoint_values, noise_fraction_values, rank_values):
        print(f"VALUE {comparator}: endpoint_summaries={int(ep)} noise_fraction={frac:.6f} rank_changes={int(rank)}")
    print(f"OUTPUT_SVG: {svg_path}")
    print(f"OUTPUT_PNG: {png_path}")
    print(f"OUTPUT_PDF: {pdf_path}")

    if not args.no_sync:
        sync_assets(png_path, svg_path, pdf_path)
        for target in SYNC_PNG_TARGETS:
            print(f"SYNCED_PNG: {target}")
        for target_dir in SYNC_SVG_PDF_TARGETS:
            print(f"SYNCED_VECTOR_DIR: {target_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
