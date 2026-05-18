# Phase 25-37 Publication Figure Atlas

## Figure 1. Safe-core biological endpoint heatmap

Phase: `Phase 25`

What it shows: Biological risk-analysis of the fixed safe-core plan library across the locked endpoint set after the project was first reframed away from adaptive optimization.

Caption: Heatmap summarizing how the Phase 25 biological risk-analysis model re-scored the safe-core lattice plan library. This figure introduced the core question of the later phases: whether physically similar lattice plans remain biologically distinguishable once non-local burden is included.

![Safe-core biological endpoint heatmap](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure01_phase25_safe_core_biological_heatmap.png)

## Figure 2. Vascular sink ablation deltas

Phase: `Phase 26`

What it shows: Endpoint changes induced by adding the anatomical vascular sink term relative to the no-sink biological model.

Caption: Ablation plot from Phase 26 showing that the vascular sink operates as a secondary modifier of biological burden rather than the dominant driver of plan reinterpretation. The direction and magnitude of the deltas motivated the later falsification analyses.

![Vascular sink ablation deltas](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure02_phase26_vascular_sink_ablation.png)

## Figure 3. Yang-style benchmark geometry and biological interpretation

Phase: `Phase 28`

What it shows: The synthetic Yang-style sinonasal benchmark phantom alongside the biological readout used to reinterpret the benchmarked lattice plan.

Caption: Phase 28 anchored the workflow to a Yang-style sinonasal benchmark case. The left panel shows the benchmark phantom geometry, and the right panel shows the biological predictions demonstrating that physical peak-valley contrast can be biologically filled in by non-local effects.

![Yang-style benchmark geometry and biological interpretation](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure03_phase28_benchmark_geometry_and_biology.png)

## Figure 4. TOPAS-derived photon lattice delivery decomposition

Phase: `Phase 30`

What it shows: How the broad base field and focused vertex boost combine to produce the final TOPAS-derived photon lattice dose used as a physical anchor for the biology model.

Caption: Phase 30 decomposed the benchmark photon delivery into a base component, a vertex-boost component, and their calibrated combination. The final combined plan achieved PTV D95 3.50 Gy, peak mean 15.23 Gy, PVDR 2.34, and brainstem D2 14.10 Gy.

![TOPAS-derived photon lattice delivery decomposition](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure04_phase30_topas_delivery_decomposition.png)

## Figure 5. Rank reinterpretation and robustness synthesis

Phase: `Phase 31`

What it shows: Manuscript-stage synthesis of how biological reinterpretation changed plan ordering and how stable those changes remained under repeat analyses.

Caption: Phase 31 consolidated the early biological risk-analysis workflow into a publication-oriented synthesis, emphasizing that the most important signal was plan reinterpretation rather than physical-plan optimization.

![Rank reinterpretation and robustness synthesis](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure05_phase31_rank_reinterpretation_and_robustness.png)

## Figure 6. Site-specific synthetic H&N cohort montage

Phase: `Phase 32`

What it shows: The full 10-case site-specific synthetic cohort, replacing the earlier single-phantom assumption with distinct anatomical sites and vertex-pruned geometries.

Caption: Phase 32 instantiated 10 separate site-specific synthetic phantoms spanning sinonasal, oropharyngeal, parapharyngeal, cheek, nodal, and composite bulky head-and-neck geometries. The median instantiated GTV was 112.3 cc, with 24 kept vertices across the cohort after anatomy-aware pruning.

![Site-specific synthetic H&N cohort montage](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure06_phase32_site_specific_phantom_montage.png)

## Figure 7. Physical TOPAS cohort summary

Phase: `Phase 33`

What it shows: Distribution of key physical endpoints across the 10 site-specific cases after TOPAS-based photon lattice delivery was generated for each phantom.

Caption: Phase 33 established the physical anchor for the synthetic cohort. Across 10 completed cases, the mean PTV D95 was 3.50 Gy and the median PVDR was 1.80, while substantial spread remained in spill and OAR-adjacent physical dose.

![Physical TOPAS cohort summary](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure07_phase33_physical_cohort_summary.png)

## Figure 8. Biological reinterpretation of the 10-case cohort

Phase: `Phase 34`

What it shows: How the same physical cohort was re-scored when the bystander model was applied with and without vascular sink uptake, including case-wise rank shifts.

Caption: Phase 34 marked the full cohort-level biological reinterpretation step. Biology changed the final ranking in 8/10 cases, and cohort-mean spill, parotid, and brainstem burden rose markedly relative to the physical-only view.

![Biological reinterpretation of the 10-case cohort](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure08_phase34_biological_reinterpretation.png)

## Figure 9. Repeated-subset uncertainty robustness

Phase: `Phase 35`

What it shows: Endpoint-level and rank-direction robustness of the sink effect after repeated transport and biology solves on the 6-case uncertainty subset.

Caption: Phase 35 showed that 40/42 case-endpoint sink deltas exceeded the combined Monte Carlo plus uptake-sensitivity 95% noise band, even though rank-direction stability was weaker than endpoint-level stability.

![Repeated-subset uncertainty robustness](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure09_phase35_uncertainty_robustness.png)

## Figure 10. First falsification controls for the vascular sink

Phase: `Phase 36`

What it shows: Comparison of the true anatomical sink against no sink, mirrored sink, and mass-matched uniform washout in both baseline and uncertainty-qualified analyses.

Caption: Phase 36 tested whether the sink effect could be explained away by simpler controls. The mirrored left-right sink remained essentially null, while the uniform body sink retained a strong repeated-subset signal (fraction above 95% noise band 0.88), highlighting that the sink was not reducible to a trivial mirror artefact.

![First falsification controls for the vascular sink](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure10_phase36_falsification_controls.png)

## Figure 11. Stronger spatial falsification controls

Phase: `Phase 37`

What it shows: Performance of the true anatomical sink against stronger spatial falsifications: AP flip, SI flip, local vessel dropout, randomized displacement, uniform washout, and no-sink control.

Caption: Phase 37 strengthened the novelty claim by showing that 222/288 repeated case-endpoint differences remained larger than the combined noise band when the true sink was compared with stronger spatial falsifications. Only 2 noise-qualified rank changes remained, indicating that the sink behaves mainly as a robust endpoint modulator rather than a broad rank-flipping driver.

![Stronger spatial falsification controls](PROJECT_ROOT/runs/phase25_to_phase37_publication_figure_atlas/figures/figure11_phase37_stronger_falsification_controls.png)
