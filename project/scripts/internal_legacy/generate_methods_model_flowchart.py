#!/usr/bin/env python3
"""Generate a manuscript-ready methods flowchart as a landscape PNG."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, key, x, y, w, h, title, lines, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=2.0,
        edgecolor="#222222",
        facecolor=facecolor,
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(patch)

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=48) or [""])

    text = title + "\n" + "\n".join(wrapped)
    ax.text(
        x + 0.012,
        y + h - 0.014,
        text,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#111111",
        zorder=3,
        linespacing=1.20,
    )
    return {
        "key": key,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": x + w / 2.0,
        "cy": y + h / 2.0,
    }


def anchor(box, side, frac=0.5):
    if side == "top":
        return (box["x"] + frac * box["w"], box["y"] + box["h"])
    if side == "bottom":
        return (box["x"] + frac * box["w"], box["y"])
    if side == "left":
        return (box["x"], box["y"] + frac * box["h"])
    if side == "right":
        return (box["x"] + box["w"], box["y"] + frac * box["h"])
    raise ValueError(f"Unsupported side: {side}")


def draw_arrow(ax, start, end, color="#333333", connectionstyle="arc3,rad=0.0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.1,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=6,
        shrinkB=6,
        zorder=1,
    )
    ax.add_patch(patch)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "runs" / "methods_section_assets"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "methods_model_flowchart.png"

    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "physics": "#dbeafe",
        "equation": "#dcfce7",
        "state": "#ede9fe",
        "output": "#ffedd5",
        "immune": "#fee2e2",
    }

    boxes = {}
    boxes["mc"] = add_box(
        ax,
        "mc",
        0.04,
        0.83,
        0.36,
        0.13,
        "Phase 1. Monte Carlo Physical Foundation",
        [
            "TOPAS transport of a representative 6 MV polyenergetic direct-photon source surrogate in water.",
            "Monte Carlo outputs the absorbed dose kernel D(x) on a voxel grid.",
        ],
        colors["physics"],
    )
    boxes["sigma"] = add_box(
        ax,
        "sigma",
        0.62,
        0.835,
        0.32,
        0.12,
        "Phase 12. Deeper Monte Carlo Coupling",
        [
            "TOPAS Standard_Deviation is exported as a normalized stochasticity field sigma_D(x).",
            "This field optionally amplifies biological emission.",
        ],
        colors["state"],
    )
    boxes["geom"] = add_box(
        ax,
        "geom",
        0.19,
        0.64,
        0.62,
        0.13,
        "Geometry Synthesis and Plan Construction",
        [
            "The Monte Carlo single-beam kernel is shifted and superposed to build regular 3D LATTICE arrays (20, 30, 40 mm pitch) and the later irregular 10-peak surrogate plan.",
            "Transmission floor, vascular masks, tumor masks, and hypoxia masks are added on top of the physical geometry.",
        ],
        colors["physics"],
    )
    boxes["lq"] = add_box(
        ax,
        "lq",
        0.05,
        0.43,
        0.34,
        0.14,
        "Direct-Radiation Null Hypothesis (LQ)",
        [
            "S_LQ(x) = exp[-alpha D(x) - beta D(x)^2]",
            "This is the baseline survival model against which all nonlocal biological effects are measured.",
        ],
        colors["equation"],
    )
    boxes["emission"] = add_box(
        ax,
        "emission",
        0.54,
        0.41,
        0.40,
        0.18,
        "State-Dependent Emission Law",
        [
            "E_k(x) = E_max,k (1 - exp[-gamma D(x)]) (1 + kappa_sigma sigma_D(x)) M_type(x,k) M_oxygen(x,k)",
            "Dose drives saturating emission, while cell state, oxygenation, and optional Monte Carlo stochasticity modulate source strength.",
        ],
        colors["equation"],
    )
    boxes["pde"] = add_box(
        ax,
        "pde",
        0.25,
        0.22,
        0.50,
        0.15,
        "Phases 2–7. Multi-Species Reaction-Diffusion Transport",
        [
            "dC_k/dt = D_k nabla^2 C_k - lambda_k C_k - u_k(x) C_k + E_k(x)",
            "ROS and cytokines are propagated with species-specific diffusion, decay, and vascular sink terms.",
        ],
        colors["equation"],
    )
    boxes["immune"] = add_box(
        ax,
        "immune",
        0.04,
        0.16,
        0.18,
        0.11,
        "Phase 5. Systemic Immune Term",
        [
            "P_immune = I_max V_ICD / (V_ICD + V_half)",
            "Global penalty derived from volume above the ICD dose threshold.",
        ],
        colors["immune"],
    )
    boxes["hazard"] = add_box(
        ax,
        "hazard",
        0.29,
        0.03,
        0.42,
        0.13,
        "Phases 5–7. Temporal Hazard Integration",
        [
            "h(x,t) = w_ROS C_ROS(x,t) + w_cyto C_cyto(x,t)",
            "H(x) = integral h(x,t) dt",
            "The model accumulates biochemical stress over time instead of using only a final concentration snapshot.",
        ],
        colors["state"],
    )
    boxes["survival"] = add_box(
        ax,
        "survival",
        0.74,
        0.18,
        0.22,
        0.18,
        "Final Coupled Survival Model",
        [
            "SF(x) = S_LQ(x) exp[-s(H(x) + w_immune P_immune)]",
            "This combines direct physical kill, local nonlocal stress, and the global immune background penalty.",
        ],
        colors["equation"],
    )
    boxes["holdout"] = add_box(
        ax,
        "holdout",
        0.74,
        0.01,
        0.22,
        0.12,
        "Phases 8–9. Calibration and Holdouts",
        [
            "1D half-field calibration, then strict no-touch transfer to 2D stripes, 3D LATTICE, and complex-plan holdouts.",
        ],
        colors["output"],
    )
    boxes["outputs"] = add_box(
        ax,
        "outputs",
        0.04,
        0.01,
        0.21,
        0.11,
        "Phases 8–12. Readouts",
        [
            "D_eff inversion, valley survival maps, nonlocal EUD shift, uncertainty bands, sensitivity rankings, and in silico assay proxies.",
        ],
        colors["output"],
    )

    # Arrows arranged to avoid visual overlap.
    draw_arrow(ax, anchor(boxes["mc"], "bottom", 0.55), anchor(boxes["geom"], "top", 0.28))
    draw_arrow(ax, anchor(boxes["sigma"], "bottom", 0.28), anchor(boxes["emission"], "top", 0.78), connectionstyle="arc3,rad=-0.08")
    draw_arrow(ax, anchor(boxes["geom"], "bottom", 0.18), anchor(boxes["lq"], "top", 0.55))
    draw_arrow(ax, anchor(boxes["geom"], "bottom", 0.82), anchor(boxes["emission"], "top", 0.40))
    draw_arrow(ax, anchor(boxes["emission"], "bottom", 0.45), anchor(boxes["pde"], "top", 0.70))
    draw_arrow(ax, anchor(boxes["pde"], "left", 0.55), anchor(boxes["immune"], "right", 0.48), connectionstyle="arc3,rad=0.05")
    draw_arrow(ax, anchor(boxes["pde"], "bottom", 0.50), anchor(boxes["hazard"], "top", 0.52))
    draw_arrow(ax, anchor(boxes["lq"], "right", 0.35), anchor(boxes["survival"], "left", 0.72), connectionstyle="arc3,rad=-0.18")
    draw_arrow(ax, anchor(boxes["hazard"], "right", 0.52), anchor(boxes["survival"], "left", 0.35), connectionstyle="arc3,rad=-0.02")
    draw_arrow(ax, anchor(boxes["immune"], "right", 0.22), anchor(boxes["survival"], "left", 0.20), connectionstyle="arc3,rad=-0.24")
    draw_arrow(ax, anchor(boxes["survival"], "bottom", 0.50), anchor(boxes["holdout"], "top", 0.50))
    draw_arrow(ax, anchor(boxes["hazard"], "left", 0.26), anchor(boxes["outputs"], "right", 0.70), connectionstyle="arc3,rad=0.18")

    ax.text(
        0.5,
        0.985,
        "Chronological Development of the Multiscale SFRT Model",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.5,
        0.965,
        (
            "The framework begins with Monte Carlo physical dosimetry, builds synthetic and irregular lattice geometries, "
            "adds state-dependent multi-species spatial biology, integrates cumulative hazard over time, "
            "and outputs clinically and biologically interpretable readouts."
        ),
        ha="center",
        va="top",
        fontsize=11.5,
        color="#333333",
    )

    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(outfile)


if __name__ == "__main__":
    main()
