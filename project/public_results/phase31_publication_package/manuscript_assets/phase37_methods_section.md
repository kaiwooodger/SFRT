# Methods

## Study Design and Scope

This study was designed as a computational **biological risk-analysis framework** for lattice radiotherapy rather than as a clinical treatment-planning-system (TPS) replacement or inverse optimizer. The final analytic workflow reported here corresponds to the mature risk-analysis branch of the project, spanning **Phase 25 through Phase 37**, while earlier phases were used to develop the physical lattice geometry, heterogeneous phantom infrastructure, Monte Carlo transport coupling, reaction-diffusion biology engine, and exploratory optimization logic. The present paper therefore evaluates whether protocol-constrained lattice plans that appear acceptable under physical dose metrics retain the same interpretation after non-local biological burden, vascular washout, uncertainty, and falsification analyses are applied.

The computational workflow had four layers:  
1. generation of synthetic but anatomically structured head-and-neck (H&N) phantoms and benchmark cases;  
2. generation of physically calibrated lattice dose distributions using TOPAS Monte Carlo transport;  
3. biological reinterpretation of those dose distributions using a multispecies reaction-diffusion bystander model with optional vascular sink uptake; and  
4. uncertainty and falsification analyses to test whether the vascular sink term was distinguishable from simpler surrogate washout models.

The principal data-generating phases used in the present manuscript were:

- **Phase 25-26:** fixed-library biological risk analysis and vascular sink ablation  
- **Phase 28-30:** Yang-style benchmark construction and TOPAS-derived photon lattice anchor  
- **Phase 32-34:** site-specific 10-case synthetic H&N cohort with TOPAS transport and biology analysis  
- **Phase 35:** repeat-seed / repeat-history uncertainty analysis  
- **Phase 36-37:** sink falsification and stronger spatial surrogate comparisons

### Methodological novelty

The central methodological novelty was not the use of a single bystander term in isolation, but the integration of:

- a **site-specific synthetic H&N lattice cohort**
- **TOPAS-derived physical dose** rather than dose-only analytical scoring
- an **anatomy-aware vascular sink field** derived from explicit vessel masks
- a **fixed-endpoint biological reinterpretation framework**
- and **noise-qualified falsification testing** against surrogate washout models

## Terminology and Core Definitions

The following terms were used consistently throughout the study:

- **GTV:** gross tumour volume, defined as the voxelized tumour mask
- **CTVboost:** lattice peripheral boost volume, defined as `GTV + 5 mm`
- **PTV:** planning target volume used for physical dose calibration and target coverage reporting
- **Vertex:** spherical intratumoural lattice hotspot
- **Peak region:** union of spherical regions surrounding accepted vertices
- **Valley region:** intratumoural region excluding the peak support regions
- **Peri-GTV shell:** normal-tissue shell outside the GTV, used to quantify spill
- **Physical-only mode:** endpoint extraction from the physical dose grid alone
- **Bystander no-sink mode:** biological reinterpretation using diffusible signalling without vascular uptake
- **Bystander with-sink mode:** biological reinterpretation using the same signalling model plus explicit vessel uptake
- **Effective dose,** \(D_{\mathrm{eff}}\): linear-quadratic-equivalent dose obtained by inverting the final survival field
- **PVDR:** peak-to-valley dose ratio, reported as the ratio of peak mean to valley mean

## Overall Workflow and Model Use

The final study workflow proceeded as follows. First, lattice plans were generated under fixed protocol rules rather than free optimization. Second, the resulting physical dose distributions were calculated using TOPAS or, for earlier screening phases, loaded from previously generated plan packages. Third, each physical dose was converted into a biological burden field through the non-local reaction-diffusion model. Fourth, plan quality was summarized using a locked endpoint set and assay-like proxies. Finally, the interpretation of the vascular sink term was tested using repeat-run uncertainty and spatial falsification analyses.

This final workflow intentionally differed from the earlier optimization-oriented phases. In Phases 17-24, the biology model was embedded inside adaptive and fraction-aware optimization loops. Although those phases were useful for defining constraints and identifying failure modes, the biologically guided optimizer did not produce a sufficiently convincing clinical planning solution under the available search, constraint, and dose-model conditions. The study was therefore intentionally reframed in **Phase 25** as a biological risk-analysis problem: instead of asking the model to select a clinically optimal plan, the model was used to determine whether physically similar lattice plans carried different non-local biological liabilities.

## Synthetic Phantom Construction

### Foundational voxelized head-and-neck phantom

The physical and biological model infrastructure was first established on a synthetic heterogeneous voxelized H&N phantom. The original detailed phantom was discretized on a Cartesian grid with isotropic millimeter-scale voxels and included explicit body contour, skull, mandible, vertebral bone, airway, brain, brainstem, spinal cord, salivary glands, thyroid, parathyroids, vascular structures, and tumour compartments. Tissue masks were constructed algorithmically from ellipsoids, cylinders, tubular paths, and boolean combinations so that every phantom remained fully synthetic and reproducible while still preserving anatomical structure.

This early phantom served three purposes:

- it allowed development of the TOPAS `TsImageCube` material workflow
- it provided explicit artery and vein masks for the vascular sink model
- it established the planning rules later transferred to the final cohort study

### Site-specific synthetic benchmark cohort

To avoid over-interpreting results from a single anatomy, the final study used a **10-case site-specific synthetic H&N cohort** (Phase 32). Each case instantiated a separate phantom with a distinct tumour site, surrounding organs-at-risk (OARs), vessel geometry, and accepted lattice configuration. The cohort included:

1. right sinonasal / maxillary bulky ellipsoid
2. left maxillary crescent abutting orbit
3. base-of-tongue / oropharynx bilobed mass
4. laryngo-hypopharyngeal elongated mass
5. parapharyngeal / prestyloid deep-space mass
6. buccal mucosa / cheek superficial crescent
7. oral tongue / floor-of-mouth horseshoe mass
8. bulky nodal level II-IV conglomerate with central necrosis
9. deep parotid / infratemporal mass
10. composite oropharynx plus upper-neck very bulky mass

These cases spanned estimated tumour sizes from approximately `66 cc` to `326 cc` and represented the major site groups most relevant to crowded H&N lattice planning, including sinonasal, orbital-adjacent, deep neck, tongue-base, cheek, nodal, and composite bulky disease.

Each case was generated by placing the tumour inside a site-appropriate synthetic anatomy rather than by copying the same tumour into a fixed phantom. This distinction is important methodologically: the final cohort therefore tested how the model behaved across **different realized anatomies**, not just across different vertex layouts.

### Materialization for TOPAS transport

For Monte Carlo transport, each synthetic phantom was exported as a TOPAS `TsImageCube` phantom. Voxels were assigned explicit material tags and bulk densities representing air, soft tissue, brain, brainstem, spinal cord, bone, glandular tissue, vascular blood-like compartments, and tumour. These tags were written to binary ImageCube files together with case-specific material include files, thereby producing a transport-ready heterogeneous phantom while preserving the underlying anatomy and vessel masks for later biological analysis.

### Methodological novelty

The novelty at this stage lies in combining **site-specific synthetic tumour anatomy**, **explicit voxelized vascular structure**, and **transport-ready materialized phantoms** inside one reproducible lattice-analysis workflow. Many lattice planning studies use either simple analytical phantoms or clinical plans without explicit vessel-aware biology; the present framework was designed to bridge those two extremes.

## Planning Protocol and Lattice Geometry Rules

### Lattice prescription concept

The study adopted a Yang-style lattice boost concept as the geometric and prescription prior for benchmark construction. The common assumptions were:

- **vertex dose:** `15 Gy` to each intratumoural vertex
- **peripheral lattice dose:** `3.0-3.5 Gy` to the boost periphery
- **CTVboost definition:** `GTV + 5 mm`
- **default photon delivery concept:** two full VMAT-like arcs as a benchmark planning analogue

For the site-specific synthetic cohort, these prescription concepts were mapped into the TOPAS physical model through two-component photon dose generation and offline dose calibration, as described below.

### Vertex sizing and spacing

The planning protocol used the following geometric rules derived from the benchmark literature and the synthetic case templates:

- vertex diameters typically `0.8-1.2 cm`, with `1.0 cm` as the default Yang-style size and `1.2 cm` permitted in very large tumours
- candidate vertices required to be **fully intratumoural**
- vertices were rejected if they approached the tumour boundary excessively
- a minimum inter-vertex spacing was enforced to avoid geometric overlap and to preserve peak-valley separation
- if a candidate conflicted with nearby high-priority anatomy, it was **deleted rather than forced for symmetry**

This last rule was especially important in H&N cases. The protocol explicitly prioritized safety relative to the carotid and jugular territories, spinal cord, brainstem, optic pathway, cochlea, mandible, skull base, and skin. If the tumour geometry did not allow a symmetric array without violating these criteria, fewer vertices were accepted.

### Candidate generation and safe-core selection

In the Phase 25 fixed-library risk-analysis package, candidate points were generated inside a **contracted GTV** rather than directly on the boundary of the full tumour. The locked selection settings were:

- GTV contraction: `5 mm`
- OAR clearance: `15 mm`
- spot radius: `6 mm`
- candidate sampling step: `6 mm`
- minimum spot spacing: `18 mm`
- layer spacing: `30 mm` where layered patterns were used

The fixed-library analysis used either explicit pre-specified safe-core plans or protocol-constrained randomized candidate plans, depending on the phase. The final safe-core publication library used five fixed plans, five fractions, and a reproducible seed of `33`.

### Type of tumours and lattice intent

The tumour classes studied here were not generic spheres. They included compact sinonasal masses, crescentic orbital-adjacent lesions, bilobed base-of-tongue masses, elongated laryngo-hypopharyngeal lesions, deep-space parapharyngeal tumours, superficial cheek lesions, horseshoe oral-tongue volumes, bulky necrotic nodal conglomerates, deep parotid/infratemporal masses, and very large composite oropharyngeal-upper-neck volumes. This diversity was methodologically important because it forced the planning protocol to confront different crowding patterns, OAR relationships, and allowable vertex counts.

## TOPAS Physical Dose Calculation

### Role of TOPAS in the workflow

TOPAS was used as the **physical dose engine**, not as an inverse planner. The workflow did not attempt to reproduce a full clinical TPS optimizer, dynamic multileaf-collimator sequence, or patient-specific VMAT control-point set. Instead, TOPAS provided physically grounded photon transport through the voxelized phantom so that downstream biological analysis rested on heterogeneous Monte Carlo dose rather than purely analytical surrogates.

### Two-component photon lattice-delivery model

In the mature benchmark and cohort branches, each lattice plan was decomposed into two physical components:

1. a **broad base component** designed to cover the peripheral target volume
2. a **focused vertex component** designed to intensify dose at the accepted lattice vertices

Because absorbed dose is linear in incident fluence, the final combined physical dose was formed by a weighted sum of the base and vertex component dose grids:

\[
D_{\mathrm{phys}}(\mathbf{x}) = w_{\mathrm{base}} D_{\mathrm{base}}(\mathbf{x}) + w_{\mathrm{vert}} D_{\mathrm{vert}}(\mathbf{x}),
\]

where \(D_{\mathrm{base}}\) and \(D_{\mathrm{vert}}\) were obtained from separate TOPAS runs and \(w_{\mathrm{base}}\), \(w_{\mathrm{vert}}\) were chosen offline to match the target prescription goals.

The calibration targets were:

- **PTV \(D_{95}\)** approximately `3.5 Gy`
- **peak mean dose** approximately `15 Gy`

This decomposition was first demonstrated on the Yang-style benchmark phantom (Phase 30) and then extended to all 10 site-specific cohort cases (Phase 33).

### Monte Carlo settings and dose handling

TOPAS was driven through case-specific `TsImageCube` phantoms with multithreaded photon transport. The benchmark and cohort runs used modest but reproducible history counts per component, with default Phase 30/33 values of approximately:

- base component histories: `12,000`
- vertex component histories: `24,000`

Raw dose grids were smoothed using a body-aware Gaussian denoising step prior to component-weight calibration. This step was introduced to reduce Monte Carlo noise sufficiently for stable lattice metrics while preserving the broader peak-valley structure used in the downstream biological analysis.

### Why TOPAS was not used as a full screening engine

TOPAS was not run for every hypothetical lattice candidate generated during the earlier optimizer phases because the design objective of the final paper was not to claim exhaustive search over the geometric space. Instead, TOPAS was used at the points where physical credibility mattered most:

- to anchor the Yang-style benchmark
- to generate the full Phase 33 site-specific cohort
- and to quantify repeat-run Monte Carlo uncertainty in Phase 35

This hierarchical use of TOPAS allowed the paper to retain a transport-based physical foundation without falsely implying a clinical inverse-planning workflow.

### Methodological novelty

The novelty in the transport layer lies in pairing a **TOPAS-derived lattice-delivery surrogate** with a **biology-first interpretation framework**. The purpose of TOPAS here was not to prove clinical deliverability, but to provide a physically heterogeneous dose field to test whether biology altered the interpretation of otherwise protocol-constrained lattice plans.

## Multispecies Non-Local Biology Model

### Baseline local survival

Physical dose was first converted to a local linear-quadratic (LQ) survival field:

\[
S_{\mathrm{LQ}}(\mathbf{x}) = \exp\!\left[-\alpha D_{\mathrm{phys}}(\mathbf{x}) - \beta D_{\mathrm{phys}}(\mathbf{x})^2\right],
\]

with locked parameters \(\alpha = 0.03\) Gy\(^{-1}\) and \(\beta = 0.003\) Gy\(^{-2}\).

### Signalling species and source construction

Two diffusible signalling channels were modeled:

- a **ROS-like short-range species**
- a **cytokine-like longer-range species**

Dose-dependent source emission for species \(k\) was defined as

\[
S_k(\mathbf{x}) = E_{\max,k}\left[1 - \exp\!\left(-\gamma D_{\mathrm{phys}}(\mathbf{x})\right)\right] M_{\mathrm{type},k}(\mathbf{x}) M_{\mathrm{oxygen},k}(\mathbf{x}),
\]

where \(E_{\max,k}\) is the maximum emission strength, \(\gamma\) controls dose saturation, \(M_{\mathrm{type},k}\) is a tissue/tumour-specific emission modifier, and \(M_{\mathrm{oxygen},k}\) is an oxygen-state modifier.

The locked global biology constants used in the mature risk-analysis phases included:

- \(\gamma = 0.35\)
- \(D_{\mathrm{ROS}} = 0.8\)
- \(D_{\mathrm{cyto}} = 1.2\)
- \(\lambda_{\mathrm{ROS}} = 0.2\)
- \(\lambda_{\mathrm{cyto}} = 0.001\)
- \(E_{\max,\mathrm{ROS}} = 1.5\)
- \(E_{\max,\mathrm{cyto}} = 0.8\)

Within tumour voxels, the cytokine-emission multiplier was increased (`2.0×`). Within hypoxic subvolumes, ROS emission was reduced (`0.12×`) and cytokine emission was increased (`2.7×`) to model altered non-local signalling under oxygen limitation.

### Reaction-diffusion transport with vascular uptake

For each species \(k\), concentration \(c_k(\mathbf{x},t)\) was evolved using:

\[
\frac{\partial c_k}{\partial t} = D_k \nabla^2 c_k - \lambda_k c_k - u_k(\mathbf{x}) c_k + S_k(\mathbf{x}),
\]

where \(D_k\) is diffusion coefficient, \(\lambda_k\) is decay, and \(u_k(\mathbf{x})\) is the spatial uptake field. The uptake term is what we refer to as the **vascular sink term**: it represents removal or washout of diffusible biological signals in voxels designated as arteries or veins.

The uptake field took three forms depending on the analysis mode:

- **physical-only:** no bystander PDE was solved
- **no-sink:** the PDE was solved with \(u_k(\mathbf{x}) = 0\)
- **with-sink:** \(u_k(\mathbf{x})\) was populated using explicit arterial and venous masks

Arterial and venous uptake were species-specific. In the locked phantom branch, typical uptake coefficients were:

- arterial ROS uptake: `0.05`
- venous ROS uptake: `0.05`
- arterial cytokine uptake: `0.70`
- venous cytokine uptake: `0.90`

This formulation should be interpreted as a **static anatomy-aware washout field**, not a hemodynamic blood-flow simulation. The model does not represent pulsatility, convection, vessel permeability, or patient-specific perfusion. Instead, it tests whether spatially explicit vessel-adjacent signal removal changes plan interpretation compared with no-sink and surrogate-sink alternatives.

### Cumulative non-local hazard and final survival

At each time step, the model converted instantaneous concentrations into a weighted biological stress:

\[
h(\mathbf{x},t) = w_{\mathrm{ROS}} c_{\mathrm{ROS}}(\mathbf{x},t) + w_{\mathrm{cyto}} c_{\mathrm{cyto}}(\mathbf{x},t),
\]

with locked weights \(w_{\mathrm{ROS}} = 0.4\) and \(w_{\mathrm{cyto}} = 0.4\). The cumulative hazard field was then integrated over time:

\[
H(\mathbf{x}) = \int h(\mathbf{x},t)\,dt.
\]

An additional immune-associated scalar penalty was incorporated with weight \(w_{\mathrm{immune}} = 0.2\), giving a final non-local penalty

\[
\Pi(\mathbf{x}) = H(\mathbf{x}) + w_{\mathrm{immune}} I(\mathbf{x}),
\]

where \(I(\mathbf{x})\) is the immune-associated penalty field or scalar contribution derived from immunogenic-cell-death burden.

Final survival was then defined as

\[
S_{\mathrm{final}}(\mathbf{x}) = S_{\mathrm{LQ}}(\mathbf{x}) \exp\!\left[-s\,\Pi(\mathbf{x})\right],
\]

where \(s = 0.0029365813\) is the locked non-local scaling factor.

### Effective-dose transformation

To preserve familiar radiotherapy endpoint definitions, final survival was inverted to an LQ-equivalent effective dose:

\[
D_{\mathrm{eff}}(\mathbf{x}) =
\frac{-\alpha + \sqrt{\alpha^2 - 4\beta \ln S_{\mathrm{final}}(\mathbf{x})}}{2\beta}.
\]

All biological risk endpoints were extracted from either \(D_{\mathrm{eff}}\) or the time-dependent concentration histories while retaining exactly the same anatomical ROIs used for physical dose analysis.

### Methodological novelty

The key novelty of the biological model lies in the **combination** of non-local bystander transport, explicit vascular uptake, tumour/hypoxia emission modifiers, and translation back into **radiotherapy-like interpretable endpoints** rather than reporting only abstract concentration fields.

## Risk-Analysis Endpoints and Assay Proxies

### Locked endpoint set

The final risk-analysis framework used a locked endpoint set so that all comparisons across phases, cases, uncertainty runs, and falsification controls were made on the same metrics. The primary endpoints were:

- PTV \(D_{95}\)
- PVDR
- peri-GTV shell `0-5 mm` mean
- peri-GTV shell `5-15 mm` mean
- spinal cord \(D_2\)
- brainstem \(D_2\)
- right parotid mean

Additional supplemental endpoints included GTV \(D_{95}\), thyroid mean, body \(D_{\max}\), left parotid mean, `15-30 mm` spill shell mean, outside-GTV \(D_2\), OAR-adjacent outside-GTV mean, PTV-valley outside-GTV mean, peak mean, and valley mean.

The peak region was constructed from vertex-centred spherical support masks, while the valley region was defined by excluding a larger peak-neighbourhood radius to avoid direct contamination by the hotspots. Outside-GTV spill regions were quantified using concentric shells extending outward from the tumour surface.

### Assay-like outputs

To improve interpretability, the model also generated three classes of assay-like outputs:

- **\(\gamma\)H2AX-like DNA damage proxy**
- **TUNEL-like apoptosis proxy**
- **ELISA-like cytokine / AUC proxy**

The \(\gamma\)H2AX proxy combined physical-dose drive and ROS-like biological drive:

\[
\Gamma(\mathbf{x}) = 1 - \exp\!\left[-\left(\alpha D_{\mathrm{phys}}(\mathbf{x}) + \beta D_{\mathrm{phys}}(\mathbf{x})^2 + 0.45\,R^*(\mathbf{x})\right)\right],
\]

where \(R^*(\mathbf{x})\) is a normalized ROS-like concentration term. The TUNEL-like proxy was defined as

\[
T(\mathbf{x}) = 1 - S_{\mathrm{final}}(\mathbf{x}),
\]

and cytokine burden was summarized by temporal area-under-the-curve (AUC) in selected ROIs.

These proxies were not treated as experimental validation, but as translational readouts that mapped the model output into forms more closely aligned with laboratory assays.

## Cohort-Level Risk Analysis

### Phase 25 fixed-library analysis

The first fully risk-analysis-oriented package evaluated a fixed safe-core plan library over five fractions. The purpose of this stage was to determine whether a protocol-constrained plan set that was physically plausible under common H&N lattice rules remained distinguishable after biological reinterpretation. This phase established the locked endpoint set and the `physical_only`, `no_sink`, and `with_sink` comparison structure that remained in later analyses.

### Phase 28-30 benchmark anchoring

To anchor the workflow externally, a Yang-style sinonasal benchmark case was constructed as a synthetic but literature-matched geometry with two intratumoural vertices, a `15 Gy` peak concept, and `3.5 Gy` peripheral target concept. Phase 28 provided the synthetic benchmark geometry and biological interpretation. Phase 30 then generated a TOPAS-derived photon lattice delivery on that same benchmark, allowing the biological model to be applied directly to a physically transported dose field rather than only to an analytical surrogate.

### Phase 32-34 site-specific cohort analysis

The principal cohort analysis used the 10 instantiated site-specific H&N phantoms from Phase 32, their per-case TOPAS dose packages from Phase 33, and the corresponding biological reinterpretation package from Phase 34. The key question at this stage was whether the biology model changed plan interpretation across distinct anatomies, not merely within a single phantom.

## Uncertainty Analysis

### Repeat seed and history analysis

Phase 35 quantified whether sink-driven endpoint changes exceeded a practical combined noise band. A representative six-case subset (`case02-case07`) was rerun across:

- random seeds: `11`, `22`, `33`
- history scales: `0.5`, `1.0`, `2.0`

For each repeated physical dose package, the biological model was rerun and endpoint distributions were tabulated. This allowed empirical estimation of repeated-run variability attributable to Monte Carlo sampling combined with previously identified uptake-sensitivity variability.

The principal uncertainty criterion was whether a sink-driven endpoint delta exceeded the **95% combined Monte Carlo plus uptake-sensitivity noise band**. This endpoint-level framing was used because rank order is a much stricter quantity than raw endpoint direction, and the study aimed first to determine whether the sink effect was measurable at all before claiming stable plan reordering.

### Rank preservation analysis

In addition to raw endpoint deltas, nominal rank preservation and rank-change probabilities were estimated from repeated runs. This allowed the study to distinguish between:

- an effect that changes endpoint magnitude reproducibly
- and an effect that is strong enough to reverse comparative plan ranking robustly

This distinction became important in the interpretation of the vascular sink term, which proved more stable at the endpoint level than at the rank-flip level.

## Vessel Falsification and Surrogate-Control Analyses

### First falsification layer

Phase 36 tested whether the true anatomical sink was distinguishable from simpler surrogate washout models using the saved Phase 33 physical dose packages. Comparator sink fields included:

- no sink
- uniform body sink with matched total uptake mass
- blurred vessel sink
- peri-GTV shell sink
- distance-decay sink

The objective was to determine whether the effect attributed to the anatomical sink could be reproduced by a generic washout field without respect to realized vessel geometry.

### Stronger spatial falsification layer

Phase 37 replaced the less informative left-right mirror control with stronger spatial falsifications:

- anterior-posterior flip
- superior-inferior flip
- local vessel dropout within `20 mm` of the tumour
- randomized but anatomy-constrained vessel displacement

All surrogate sink fields were constructed to preserve the same underlying uptake coefficients and, where relevant, comparable total uptake mass. Thus, these tests isolated the role of **sink geometry** rather than simply changing the global washout strength.

### Novelty of the falsification design

This falsification layer is a key methodological contribution of the final study. Instead of only comparing `with_sink` and `without_sink`, the workflow tested whether the putative vascular effect:

- survived repeat-run uncertainty
- remained distinct from uniform washout
- and was sensitive to more local disruptions of the tumour-adjacent vascular field

That design allowed the sink term to be evaluated as a falsifiable spatial hypothesis rather than as a fixed phenomenological add-on.

## Suggested Methods Figures

The most useful figures for the Methods section are:

1. **Benchmark / phantom geometry:** [figure03_phase28_benchmark_geometry_and_biology.png](/Users/kw/Documents/Playground/vhee_topas/runs/phase25_to_phase37_publication_figure_atlas/figures/figure03_phase28_benchmark_geometry_and_biology.png)  
   Use to show the benchmark geometry, tumour, vertices, and the bridge between physical and biological interpretation.

2. **TOPAS delivery architecture:** [figure04_phase30_topas_delivery_decomposition.png](/Users/kw/Documents/Playground/vhee_topas/runs/phase25_to_phase37_publication_figure_atlas/figures/figure04_phase30_topas_delivery_decomposition.png)  
   Use to illustrate the two-component TOPAS delivery model: base field, vertex boost, and calibrated combination.

3. **Site-specific cohort anatomy:** [figure06_phase32_site_specific_phantom_montage.png](/Users/kw/Documents/Playground/vhee_topas/runs/phase25_to_phase37_publication_figure_atlas/figures/figure06_phase32_site_specific_phantom_montage.png)  
   Use to show that the final cohort comprises distinct anatomies rather than a single reused phantom.

4. **Optional historical phantom / vessel methods figures:**  
   - [figureM1_voxel_phantom_anatomy.png](/Users/kw/Documents/Playground/vhee_topas/runs/manuscript_voxel_phantom_methods_assets/figureM1_voxel_phantom_anatomy.png)  
   - [figureM3_topas_material_phantom.png](/Users/kw/Documents/Playground/vhee_topas/runs/manuscript_voxel_phantom_methods_assets/figureM3_topas_material_phantom.png)  
   - [figureM5_vascular_sink_field.png](/Users/kw/Documents/Playground/vhee_topas/runs/manuscript_voxel_phantom_methods_assets/figureM5_vascular_sink_field.png)  
   These are useful if the journal allows a longer Methods or Supplementary Methods section and you want to show phantom anatomy, materialization, and the explicit sink-field construction.

## Statistical and Reproducibility Notes

All synthetic cases, seeds, history schedules, and output roots were fixed and written to reproducibility manifests during the publication-package phases. Bootstrap summaries, repeated-run uncertainty bands, and surrogate-comparison noise tables were used to quantify whether the sink-driven effects exceeded practical computational uncertainty.

Because the study was entirely synthetic and computational, no patient-identifiable data, institutional review, or prospective clinical intervention was involved. The intended claim is therefore restricted to **computational risk analysis and hypothesis generation**, not clinical validation.

