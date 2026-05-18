#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/kw/Documents/Playground/vhee_topas")
DESKTOP_ROOT = Path("/Users/kw/Desktop/PMB_SFRT_figures_for_overleaf")
SRC_DIR = DESKTOP_ROOT / "figures"

plt.style.use("seaborn-v0_8-whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DESKTOP_ROOT / "publishable_regenerated",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVu Sans"
    if bold:
        family = "DejaVu Sans Bold"
    path = font_manager.findfont(family)
    return ImageFont.truetype(path, size=size)


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def add_panel_box(ax: plt.Axes, label: str) -> None:
    ax.add_patch(
        Rectangle(
            (0.012, 0.925),
            0.07,
            0.075,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=20,
        )
    )
    ax.text(
        0.047,
        0.962,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        zorder=21,
    )


def mask(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    draw.rectangle(rect, fill="white")


def draw_line_legend(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    entries: list[tuple[str, str, str]],
    *,
    font: ImageFont.FreeTypeFont,
    box_pad: int = 18,
    row_gap: int = 16,
    line_len: int = 68,
) -> None:
    x0, y0 = origin
    text_width = max(draw.textbbox((0, 0), label, font=font)[2] for label, _, _ in entries)
    box_w = box_pad * 2 + line_len + 16 + text_width
    row_h = max(draw.textbbox((0, 0), label, font=font)[3] for label, _, _ in entries)
    box_h = box_pad * 2 + len(entries) * row_h + (len(entries) - 1) * row_gap
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=14,
        fill="white",
        outline="#cfcfcf",
    )
    y = y0 + box_pad + row_h // 2
    for label, color, style in entries:
        if style == "dashed":
            for x in range(x0 + box_pad, x0 + box_pad + line_len, 14):
                draw.line((x, y, min(x + 8, x0 + box_pad + line_len), y), fill=color, width=4)
        else:
            draw.line((x0 + box_pad, y, x0 + box_pad + line_len, y), fill=color, width=4)
        draw.text((x0 + box_pad + line_len + 16, y - row_h // 2), label, fill="black", font=font)
        y += row_h + row_gap


def draw_panel_label_pil(
    draw: ImageDraw.ImageDraw,
    label: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), label, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    pad_x = 18
    pad_y = 10
    draw.rectangle((x, y, x + w + 2 * pad_x, y + h + 2 * pad_y), fill="white", outline="black", width=2)
    draw.text((x + pad_x, y + pad_y - 2), label, fill="black", font=font)


def save_image(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_workflow(dst: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = {
        "geometry": (0.06, 0.70, 0.21, 0.13, "Synthetic anatomy and\nlattice geometry"),
        "transport": (0.37, 0.70, 0.23, 0.13, "Monte Carlo photon transport\nand physical dose $D(x)$"),
        "source": (0.70, 0.70, 0.23, 0.13, "Dose-driven source terms\nROS-like and cytokine-like"),
        "cal": (0.06, 0.40, 0.21, 0.13, "One-dimensional calibration\nwith locked parameters"),
        "bio": (0.37, 0.40, 0.23, 0.13, "Reaction-diffusion transport\nand temporal hazard"),
        "sink": (0.70, 0.40, 0.23, 0.13, "Anatomy-aware vascular uptake\nand falsification"),
        "reinterpret": (0.23, 0.11, 0.23, 0.13, "Survival and effective-dose\nreinterpretation"),
        "endpoints": (0.57, 0.11, 0.24, 0.13, "Endpoint extraction:\nPVDR, spill, OAR burden,\nassay-like readouts"),
    }

    for x, y, w, h, text in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.6,
                edgecolor="black",
                facecolor="#f6f6f6",
            )
        )
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=14)

    arrows = [
        ((0.27, 0.765), (0.37, 0.765)),
        ((0.60, 0.765), (0.70, 0.765)),
        ((0.165, 0.70), (0.165, 0.53)),
        ((0.485, 0.70), (0.485, 0.53)),
        ((0.815, 0.70), (0.815, 0.53)),
        ((0.27, 0.465), (0.37, 0.465)),
        ((0.60, 0.465), (0.70, 0.465)),
        ((0.48, 0.40), (0.34, 0.24)),
        ((0.81, 0.40), (0.69, 0.24)),
        ((0.46, 0.175), (0.57, 0.175)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="Simple,head_length=12,head_width=12,tail_width=1.2",
                mutation_scale=1.0,
                color="black",
            )
        )

    save_image(fig, dst, dpi=dpi)


def clean_calibration_transfer(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    title_font = pil_font(28, bold=True)
    legend_font = pil_font(24)

    for rect in [
        (170, 0, 1810, 118),
        (1980, 0, 3565, 118),
        (240, 1660, 2050, 1805),
    ]:
        mask(draw, rect)

    mask(draw, (1245, 1010, 1765, 1165))
    draw_line_legend(
        draw,
        (1265, 1020),
        [
            ("Standard LQ", "#1f77b4", "solid"),
            ("Locked calibration", "#d62728", "solid"),
            ("Field edge", "#7f7f7f", "dashed"),
        ],
        font=legend_font,
    )

    mask(draw, (2000, 505, 2825, 640))
    draw_line_legend(
        draw,
        (2020, 516),
        [
            ("Standard LQ", "#1f77b4", "solid"),
            ("Locked calibration", "#d62728", "solid"),
            ("Dose / max", "#7f7f7f", "dashed"),
        ],
        font=legend_font,
    )

    mask(draw, (2500, 960, 3600, 1180))

    img.save(dst)


def build_calibration_transfer(dst: Path) -> None:
    src = Image.open(SRC_DIR / "fig02_calibration_transfer.png").convert("RGB")
    font = pil_font(24)
    panel_font = pil_font(34, bold=True)

    panel_a = src.crop((60, 120, 1770, 1680))
    draw_a = ImageDraw.Draw(panel_a)
    mask(draw_a, (0, 0, 120, 70))
    mask(draw_a, (1180, 930, 1710, 1095))
    draw_line_legend(
        draw_a,
        (1195, 940),
        [
            ("Standard LQ", "#1f77b4", "solid"),
            ("Locked calibration", "#d62728", "solid"),
            ("Field edge", "#7f7f7f", "dashed"),
        ],
        font=font,
    )

    panel_b = src.crop((1940, 120, 3560, 1680))
    draw_b = ImageDraw.Draw(panel_b)
    mask(draw_b, (0, 0, 120, 70))
    mask(draw_b, (60, 420, 760, 600))
    draw_line_legend(
        draw_b,
        (75, 435),
        [
            ("Standard LQ", "#1f77b4", "solid"),
            ("Locked calibration", "#d62728", "solid"),
            ("Dose / max", "#7f7f7f", "dashed"),
        ],
        font=font,
    )

    panel_c = src.crop((60, 1840, 1770, 2820))
    draw_c = ImageDraw.Draw(panel_c)
    mask(draw_c, (0, 0, 120, 70))
    mask(draw_c, (0, 0, 900, 58))

    canvas = Image.new("RGB", (3560, 2700), "white")
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (1860, 0))
    canvas.paste(panel_c, (0, 1700))
    draw = ImageDraw.Draw(canvas)
    draw_panel_label_pil(draw, "(a)", (20, 20), font=panel_font)
    draw_panel_label_pil(draw, "(b)", (1820, 20), font=panel_font)
    draw_panel_label_pil(draw, "(c)", (20, 1680), font=panel_font)
    canvas.save(dst)


def build_synthetic_cohort(dst: Path, *, dpi: int) -> None:
    img = Image.open(SRC_DIR / "fig03_synthetic_cohort.png").convert("RGB")
    draw = ImageDraw.Draw(img)
    for rect in [
        (255, 14, 2165, 108),
        (785, 1350, 2035, 1445),
    ]:
        mask(draw, rect)
    img.save(dst)


def clean_cohort_reinterpretation(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    mask(draw, (0, 0, img.width, 72))
    img.save(dst)


def clean_uncertainty_sensitivity(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    legend_font = pil_font(24)

    for rect in [
        (170, 0, 1885, 112),
        (2040, 0, 3505, 112),
        (160, 1080, 2600, 1215),
    ]:
        mask(draw, rect)

    mask(draw, (2580, 185, 3490, 250))
    mask(draw, (1405, 610, 1810, 760))
    mask(draw, (2490, 790, 3600, 1085))
    draw_line_legend(
        draw,
        (1425, 625),
        [
            ("Local uncertainty band", "#f4b6b6", "solid"),
            ("Baseline", "#1f77b4", "solid"),
        ],
        font=legend_font,
    )
    img.save(dst)


def build_uncertainty_sensitivity(dst: Path) -> None:
    src = Image.open(SRC_DIR / "fig05a_uncertainty_sensitivity.png").convert("RGB")
    font = pil_font(24)
    panel_font = pil_font(34, bold=True)

    def fit_to_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        scale = min(max_w / img.width, max_h / img.height)
        out_w = max(1, int(round(img.width * scale)))
        out_h = max(1, int(round(img.height * scale)))
        return img.resize((out_w, out_h), Image.Resampling.LANCZOS)

    def trim_white(img: Image.Image, *, pad: int = 8, threshold: int = 245) -> Image.Image:
        arr = np.asarray(img)
        keep = np.any(arr < threshold, axis=2)
        ys, xs = np.where(keep)
        if len(xs) == 0 or len(ys) == 0:
            return img
        left = max(0, int(xs.min()) - pad)
        right = min(img.width, int(xs.max()) + pad + 1)
        top = max(0, int(ys.min()) - pad)
        bottom = min(img.height, int(ys.max()) + pad + 1)
        return img.crop((left, top, right, bottom))

    def make_panel(
        img: Image.Image,
        label: str,
        *,
        cell_w: int,
        cell_h: int,
        legend_entries: list[tuple[str, str, str]] | None = None,
    ) -> Image.Image:
        label_pad = 86
        bottom_pad = 102 if legend_entries else 24
        side_pad = 20
        top_pad = 14
        inner = fit_to_box(img, cell_w - 2 * side_pad, cell_h - label_pad - top_pad - bottom_pad)
        panel = Image.new("RGB", (cell_w, cell_h), "white")
        x = (cell_w - inner.width) // 2
        y = label_pad + top_pad
        panel.paste(inner, (x, y))
        draw_panel = ImageDraw.Draw(panel)
        draw_panel_label_pil(draw_panel, label, (20, 18), font=panel_font)
        if legend_entries:
            legend_y = cell_h - 76
            draw_line_legend(
                draw_panel,
                (cell_w // 2 - 210, legend_y),
                legend_entries,
                font=font,
                box_pad=12,
                row_gap=8,
                line_len=52,
            )
        return panel

    panel_a = src.crop((150, 140, 1805, 945))
    draw_a = ImageDraw.Draw(panel_a)
    mask(draw_a, (0, 0, 80, 70))
    mask(draw_a, (1180, 525, 1645, 785))
    panel_a = trim_white(panel_a)

    panel_b = src.crop((2100, 150, 3555, 920))
    draw_b = ImageDraw.Draw(panel_b)
    mask(draw_b, (0, 0, 80, 70))
    panel_b = trim_white(panel_b)

    panel_c = src.crop((155, 1325, 1810, 1848))
    draw_c = ImageDraw.Draw(panel_c)
    mask(draw_c, (0, 0, 80, 70))
    panel_c = trim_white(panel_c)

    cell_w = 1660
    cell_h = 740
    gap = 26
    canvas = Image.new("RGB", (cell_w * 2 + gap * 3, cell_h * 2 + gap * 3), "white")

    panel_a_box = make_panel(
        panel_a,
        "(a)",
        cell_w=cell_w,
        cell_h=cell_h,
        legend_entries=[
            ("Local uncertainty band", "#f4b6b6", "solid"),
            ("Baseline", "#1f77b4", "solid"),
        ],
    )
    panel_b_box = make_panel(panel_b, "(b)", cell_w=cell_w, cell_h=cell_h)
    panel_c_box = make_panel(panel_c, "(c)", cell_w=cell_w, cell_h=cell_h)

    canvas.paste(panel_a_box, (gap, gap))
    canvas.paste(panel_b_box, (gap * 2 + cell_w, gap))
    canvas.paste(panel_c_box, ((canvas.width - cell_w) // 2, gap * 2 + cell_h))
    canvas.save(dst)


def clean_sink_falsification(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    mask(draw, (0, 0, img.width, 72))
    img.save(dst)


def clean_assay_readouts(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    for rect in [
        (170, 0, 1640, 112),
        (2040, 0, 3490, 112),
        (240, 920, 1820, 1070),
        (1890, 920, 3495, 1070),
        (1860, 1070, 3560, 1175),
    ]:
        mask(draw, rect)
    img.save(dst)


def build_assay_readouts(dst: Path) -> None:
    src = Image.open(SRC_DIR / "fig06_assay_readouts.png").convert("RGB")
    panel_font = pil_font(34, bold=True)
    panel_a = src.crop((60, 110, 1770, 790))
    draw_a = ImageDraw.Draw(panel_a)
    mask(draw_a, (0, 0, 120, 70))
    panel_b = src.crop((1940, 110, 3560, 790))
    draw_b = ImageDraw.Draw(panel_b)
    mask(draw_b, (0, 0, 120, 70))
    panel_c = src.crop((60, 1140, 1820, 1970))
    draw_c = ImageDraw.Draw(panel_c)
    mask(draw_c, (0, 0, 120, 70))
    panel_d = src.crop((1930, 1140, 3560, 1970))
    draw_d = ImageDraw.Draw(panel_d)
    mask(draw_d, (0, 0, 120, 70))

    canvas = Image.new("RGB", (3560, 1990), "white")
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (1910, 0))
    canvas.paste(panel_c, (0, 1030))
    canvas.paste(panel_d, (1910, 1030))
    draw = ImageDraw.Draw(canvas)
    draw_panel_label_pil(draw, "(a)", (20, 20), font=panel_font)
    draw_panel_label_pil(draw, "(b)", (1880, 20), font=panel_font)
    draw_panel_label_pil(draw, "(c)", (20, 1000), font=panel_font)
    draw_panel_label_pil(draw, "(d)", (1880, 1000), font=panel_font)
    canvas.save(dst)


def clean_complex_surrogate(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    for rect in [
        (170, 0, 1660, 112),
        (1950, 0, 3280, 112),
        (250, 1220, 1765, 1360),
    ]:
        mask(draw, rect)
    img.save(dst)


def build_complex_surrogate(dst: Path) -> None:
    src = Image.open(SRC_DIR / "supp_complex_surrogate.png").convert("RGB")
    panel_font = pil_font(34, bold=True)
    panel_a = src.crop((60, 120, 1830, 1450))
    draw_a = ImageDraw.Draw(panel_a)
    mask(draw_a, (0, 0, 120, 70))
    panel_b = src.crop((2000, 120, 3560, 770))
    draw_b = ImageDraw.Draw(panel_b)
    mask(draw_b, (0, 0, 120, 70))
    panel_c = src.crop((60, 1450, 1830, 2239))
    draw_c = ImageDraw.Draw(panel_c)
    mask(draw_c, (0, 0, 120, 70))

    canvas = Image.new("RGB", (3560, 2260), "white")
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (1940, 0))
    canvas.paste(panel_c, (0, 1360))
    draw = ImageDraw.Draw(canvas)
    draw_panel_label_pil(draw, "(a)", (20, 20), font=panel_font)
    draw_panel_label_pil(draw, "(b)", (1900, 20), font=panel_font)
    draw_panel_label_pil(draw, "(c)", (20, 1330), font=panel_font)
    canvas.save(dst)


def clean_mc_stochasticity(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    for rect in [
        (170, 0, 1595, 112),
        (1950, 0, 3230, 112),
    ]:
        mask(draw, rect)
    img.save(dst)


def build_mc_stochasticity(dst: Path) -> None:
    src = Image.open(SRC_DIR / "supp_mc_stochasticity.png").convert("RGB")
    panel_font = pil_font(34, bold=True)
    panel_a = src.crop((60, 110, 1770, 874))
    draw_a = ImageDraw.Draw(panel_a)
    mask(draw_a, (0, 0, 120, 70))
    panel_b = src.crop((1940, 110, 3560, 874))
    draw_b = ImageDraw.Draw(panel_b)
    mask(draw_b, (0, 0, 120, 70))

    canvas = Image.new("RGB", (3560, 910), "white")
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (1910, 0))
    draw = ImageDraw.Draw(canvas)
    draw_panel_label_pil(draw, "(a)", (20, 20), font=panel_font)
    draw_panel_label_pil(draw, "(b)", (1880, 20), font=panel_font)
    canvas.save(dst)


def export_pdf(image_paths: list[Path], pdf_path: Path, *, dpi: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            h, w = arr.shape[:2]
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(arr)
            ax.axis("off")
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    build_workflow(fig_dir / "fig01_workflow.png", dpi=args.dpi)
    build_calibration_transfer(fig_dir / "fig02_calibration_transfer.png")
    build_synthetic_cohort(fig_dir / "fig03_synthetic_cohort.png", dpi=args.dpi)
    clean_cohort_reinterpretation(SRC_DIR / "fig04_cohort_reinterpretation.png", fig_dir / "fig04_cohort_reinterpretation.png")
    build_uncertainty_sensitivity(fig_dir / "fig05a_uncertainty_sensitivity.png")
    clean_sink_falsification(SRC_DIR / "fig05b_sink_falsification.png", fig_dir / "fig05b_sink_falsification.png")
    build_assay_readouts(fig_dir / "fig06_assay_readouts.png")
    build_complex_surrogate(fig_dir / "supp_complex_surrogate.png")
    build_mc_stochasticity(fig_dir / "supp_mc_stochasticity.png")

    order = [
        fig_dir / "fig01_workflow.png",
        fig_dir / "fig02_calibration_transfer.png",
        fig_dir / "fig03_synthetic_cohort.png",
        fig_dir / "fig04_cohort_reinterpretation.png",
        fig_dir / "fig05a_uncertainty_sensitivity.png",
        fig_dir / "fig05b_sink_falsification.png",
        fig_dir / "fig06_assay_readouts.png",
        fig_dir / "supp_complex_surrogate.png",
        fig_dir / "supp_mc_stochasticity.png",
    ]
    export_pdf(order, out_root / "PMB_SFRT_publishable_figures.pdf", dpi=args.dpi)
    print(f"Output root: {out_root}")
    print(f"Figures: {fig_dir}")
    print(f"Combined PDF: {out_root / 'PMB_SFRT_publishable_figures.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
