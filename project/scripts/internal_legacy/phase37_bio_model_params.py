#!/usr/bin/env python3
"""Shared biology-parameter definitions for the phase 34+ risk-analysis workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Mapping


@dataclass(frozen=True)
class BioModelParams:
    alpha: float = 0.03
    beta: float = 0.003
    diffusion_ros: float = 0.8
    diffusion_cyto: float = 1.2
    decay_ros: float = 0.2
    decay_cyto: float = 0.001
    emax_ros: float = 1.5
    emax_cyto: float = 0.8
    gamma: float = 0.35
    scaling_factor: float = 0.0029365813
    weight_ros: float = 0.4
    weight_cyto: float = 0.4
    weight_immune: float = 0.0
    tumor_cytokine_multiplier: float = 2.0
    hypoxic_ros_scale: float = 1.0
    hypoxic_cytokine_multiplier: float = 1.0
    artery_ros_uptake: float = 0.05
    artery_cyto_uptake: float = 0.70
    vein_ros_uptake: float = 0.05
    vein_cyto_uptake: float = 0.90
    pde_steps: int = 400
    pde_dt: float = 0.12

    def with_updates(self, **kwargs: float) -> "BioModelParams":
        return replace(self, **kwargs)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_BIO_MODEL_PARAMS = BioModelParams()


def bio_model_params_from_config(
    config: Mapping[str, object],
    *,
    uptake_scale: float = 1.0,
    include_immune: bool = False,
    overrides: Mapping[str, float] | None = None,
) -> BioModelParams:
    params = dict(config.get("bio_parameters", {}))
    resolved = DEFAULT_BIO_MODEL_PARAMS.with_updates(
        alpha=float(params.get("alpha", DEFAULT_BIO_MODEL_PARAMS.alpha)),
        beta=float(params.get("beta", DEFAULT_BIO_MODEL_PARAMS.beta)),
        pde_steps=int(params.get("pde_steps", DEFAULT_BIO_MODEL_PARAMS.pde_steps)),
        pde_dt=float(params.get("pde_dt", DEFAULT_BIO_MODEL_PARAMS.pde_dt)),
        tumor_cytokine_multiplier=float(
            params.get("tumor_cytokine_multiplier", DEFAULT_BIO_MODEL_PARAMS.tumor_cytokine_multiplier)
        ),
        hypoxic_ros_scale=float(params.get("hypoxic_ros_scale", DEFAULT_BIO_MODEL_PARAMS.hypoxic_ros_scale)),
        hypoxic_cytokine_multiplier=float(
            params.get("hypoxic_cytokine_multiplier", DEFAULT_BIO_MODEL_PARAMS.hypoxic_cytokine_multiplier)
        ),
        artery_ros_uptake=float(params.get("artery_ros_uptake", DEFAULT_BIO_MODEL_PARAMS.artery_ros_uptake))
        * float(uptake_scale),
        artery_cyto_uptake=float(params.get("artery_cyto_uptake", DEFAULT_BIO_MODEL_PARAMS.artery_cyto_uptake))
        * float(uptake_scale),
        vein_ros_uptake=float(params.get("vein_ros_uptake", DEFAULT_BIO_MODEL_PARAMS.vein_ros_uptake))
        * float(uptake_scale),
        vein_cyto_uptake=float(params.get("vein_cyto_uptake", DEFAULT_BIO_MODEL_PARAMS.vein_cyto_uptake))
        * float(uptake_scale),
        weight_immune=float(DEFAULT_BIO_MODEL_PARAMS.weight_immune if not include_immune else 0.2),
    )
    if overrides:
        resolved = resolved.with_updates(**{str(key): value for key, value in overrides.items()})
    return resolved


def parameter_provenance_rows() -> list[dict[str, object]]:
    """Return a compact provenance manifest for manuscript-facing parameter tables."""
    params = DEFAULT_BIO_MODEL_PARAMS
    return [
        {
            "parameter": r"$\alpha$",
            "value": params.alpha,
            "role": "Local LQ survival coefficient",
            "provenance": "Fixed reference setting",
        },
        {
            "parameter": r"$\beta$",
            "value": params.beta,
            "role": "Local LQ survival coefficient",
            "provenance": "Fixed reference setting",
        },
        {
            "parameter": r"$D_{\mathrm{ROS}}$",
            "value": params.diffusion_ros,
            "role": "ROS-like diffusion coefficient",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$D_{\mathrm{cyto}}$",
            "value": params.diffusion_cyto,
            "role": "Cytokine-like diffusion coefficient",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$\lambda_{\mathrm{ROS}}$",
            "value": params.decay_ros,
            "role": "ROS-like decay coefficient",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$\lambda_{\mathrm{cyto}}$",
            "value": params.decay_cyto,
            "role": "Cytokine-like decay coefficient",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$E_{\max,\mathrm{ROS}}$",
            "value": params.emax_ros,
            "role": "ROS-like emission ceiling",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$E_{\max,\mathrm{cyto}}$",
            "value": params.emax_cyto,
            "role": "Cytokine-like emission ceiling",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$\gamma$",
            "value": params.gamma,
            "role": "Dose-to-emission saturation coefficient",
            "provenance": "Inherited phenomenological coefficient",
        },
        {
            "parameter": r"$s$",
            "value": params.scaling_factor,
            "role": "Non-local transfer scaling coefficient",
            "provenance": "Internally calibrated transfer coefficient",
        },
        {
            "parameter": r"$w_{\mathrm{ROS}}$",
            "value": params.weight_ros,
            "role": "ROS-like hazard weight",
            "provenance": "Inherited phenomenological reference weight",
        },
        {
            "parameter": r"$w_{\mathrm{cyto}}$",
            "value": params.weight_cyto,
            "role": "Cytokine-like hazard weight",
            "provenance": "Inherited phenomenological reference weight",
        },
        {
            "parameter": "Tumour cytokine multiplier",
            "value": params.tumor_cytokine_multiplier,
            "role": "Tumour-state emission modifier",
            "provenance": "Anatomy/state-dependent modifier",
        },
        {
            "parameter": "Hypoxic ROS scale",
            "value": params.hypoxic_ros_scale,
            "role": r"$M_{\mathrm{oxygen}}$ modifier for ROS-like channel",
            "provenance": "Anatomy/state-dependent modifier",
        },
        {
            "parameter": "Hypoxic cytokine multiplier",
            "value": params.hypoxic_cytokine_multiplier,
            "role": r"$M_{\mathrm{oxygen}}$ modifier for cytokine-like channel",
            "provenance": "Anatomy/state-dependent modifier",
        },
        {
            "parameter": "Arterial ROS uptake",
            "value": params.artery_ros_uptake,
            "role": "Artery-localized first-order ROS loss",
            "provenance": "Anatomy-aware sink coefficient",
        },
        {
            "parameter": "Arterial cytokine uptake",
            "value": params.artery_cyto_uptake,
            "role": "Artery-localized first-order cytokine loss",
            "provenance": "Anatomy-aware sink coefficient",
        },
        {
            "parameter": "Venous ROS uptake",
            "value": params.vein_ros_uptake,
            "role": "Vein-localized first-order ROS loss",
            "provenance": "Anatomy-aware sink coefficient",
        },
        {
            "parameter": "Venous cytokine uptake",
            "value": params.vein_cyto_uptake,
            "role": "Vein-localized first-order cytokine loss",
            "provenance": "Anatomy-aware sink coefficient",
        },
    ]
