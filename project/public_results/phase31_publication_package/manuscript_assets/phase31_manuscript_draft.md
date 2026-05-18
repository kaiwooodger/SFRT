# Phase 31 Manuscript Draft

## Abstract

Purpose: To convert a protocol-constrained lattice radiotherapy modelling workflow into a manuscript-ready biological risk-analysis package anchored to synthetic head-and-neck benchmarks and a TOPAS-derived sinonasal photon case.

Methods: We froze a supplementary library of 10 synthetic H&N Yang-like lattice templates (estimated GTV range 66-326 cc; 2-5 vertices; recommended diameters 0.8-1.2 cm; mean spacing 3.17 cm). We then summarized a fixed safe-core plan library under physical-only scoring, bystander signalling without vascular sink uptake, and bystander signalling with vascular sink uptake; derived manuscript endpoint and assay tables; and repeated the Phase 30 TOPAS Yang-style photon benchmark across multiple seeds and history levels before applying the biology model directly to that TOPAS dose.

Results: Biological modelling changed plan ordering in the safe-core library, with a mean absolute with-sink-versus-physical rank shift of 0.80 (95% bootstrap CI 0.20-1.40). Adding vascular sink uptake reduced peri-GTV 0-5 mm burden by 0.448 Gy and parotid mean by 0.495 Gy, while increasing PVDR by 0.028. In the TOPAS-derived Yang-style photon benchmark, physical PVDR was 1.945 and with-sink biological PVDR was 1.295, with with-sink cytokine valley AUC 1546.6. Repeated TOPAS runs yielded a PVDR coefficient of variation of 9.47% and a peri-GTV 0-5 mm spill coefficient of variation of 12.53%.

Conclusion: The resulting manuscript package supports a benchmark-anchored, hypothesis-generating biological risk-analysis framework for lattice RT. It does not establish TPS-equivalent optimization, but it does show that protocol-constrained lattice plans can be reinterpreted biologically in ways not captured by physical dose metrics alone.

## Results

### Synthetic benchmark library

The supplementary benchmark library comprised 10 synthetic head-and-neck lattice templates spanning an estimated GTV range of 66-326 cc (median 106 cc). Recommended lattice complexity ranged from 2 to 5 vertices with recommended vertex diameters of 0.8-1.2 cm. Mean centre-to-centre spacing remained close to the Yang-style geometric prior, averaging 3.17 cm (range 2.92-3.61 cm). Together, these templates provide a structured geometry supplement for bulky sinonasal, oral cavity, oropharyngeal, deep-space, and nodal lattice scenarios without claiming patient-level anatomy.

### Biological reinterpretation of the safe-core library

In the fixed safe-core library, biological modelling altered plan ranking relative to physical-only scoring. The mean absolute no-sink-versus-physical rank shift was 0.80, and the mean absolute with-sink-versus-physical rank shift was 0.80 (95% bootstrap CI 0.20-1.40). These rank shifts indicate that the model changes comparative plan interpretation rather than simply rescaling all plans uniformly.

Rank robustness under combined Monte Carlo and biology uncertainty was modest rather than dominant. Across modes, rank-retention probabilities typically fell between 0.12 and 0.47. For the nominally best physical plan, top-rank retention remained 0.06, while the nominally best with-sink plan (plan05) retained top rank with probability 0.12. This supports a hypothesis-generating interpretation rather than a claim of stable winner-take-all optimization.

### Vascular sink ablation and assay-proxy shifts

Adding anatomical vascular sink uptake consistently shifted off-target burden downward. Relative to the no-sink model, the with-sink model reduced peri-GTV 0-5 mm mean dose by 0.448 Gy (95% CI 0.447-0.448 Gy) and peri-GTV 5-15 mm mean dose by 0.223 Gy. Parotid mean fell by 0.495 Gy, while PVDR increased by 0.028. At the assay-proxy level, the largest mean change was seen for cytokine valley AUC (239.8 a.u. reduction), whereas peak gammaH2AX changed minimally. This pattern is consistent with vascular sink uptake acting mainly on diffusible non-local signalling rather than on direct peak injury.

### Direct TOPAS-derived benchmark reinterpretation

When the biology model was applied directly to the TOPAS-derived Yang-style photon case, physical PVDR was 1.945. This fell to 1.276 without sink uptake and to 1.295 with sink uptake, indicating biological valley fill-in even after Monte Carlo transport. Peri-GTV 0-5 mm mean increased from 7.24 Gy physically to 20.17 Gy biologically, and parotid mean increased from 4.38 Gy to 14.51 Gy(eq). With-sink cytokine valley AUC reached 1546.6, again supporting a non-local burden outside the physical hot spots.

### Repeated TOPAS uncertainty bands

Nine repeated Phase 30 TOPAS runs were generated across three random seeds and three history levels. Peripheral target D95 remained numerically stable (coefficient of variation 0.0000%), whereas PVDR showed a coefficient of variation of 9.47%. Peri-GTV 0-5 mm spill varied by 12.53% and parotid mean by 41.36%. These repeated-run bands provide a practical Monte Carlo noise scale against which downstream biological reinterpretation can be judged.

## Discussion

This study should be framed as a biological risk-analysis investigation rather than a clinical plan-optimization paper.
The strongest result is that fixed, protocol-constrained lattice plans can change rank once non-local bystander burden is considered, even before any claim is made about deliverable TPS optimization.

In the safe-core library, rank shifts persisted across modelling modes, with a mean absolute with-sink-versus-physical shift of `0.80` ranks (95% bootstrap CI `0.20` to `1.40`).
This supports the claim that the model changes plan interpretation rather than simply rescaling endpoint magnitudes.

The vascular sink should be described as a mechanistic modifier of diffusible non-local burden rather than a dominant driver of direct peak damage.
For example, the mean with-sink-minus-no-sink delta for peri-GTV 0-5 mm spill was `-0.448 Gy`, while the corresponding delta for peak gammaH2AX was `0.0001` a.u.
That pattern is consistent with sink uptake acting mainly on the propagated biological field.

The Yang-style benchmark remains important as an external anchor. The analytical benchmark already showed biological valley fill-in, and the direct TOPAS-derived photon case now supports the same qualitative conclusion.
In the TOPAS case, physical PVDR was `1.945` and the with-sink biological PVDR was `1.295`, with biological peri-GTV 0-5 mm mean `20.170 Gy`.
This strengthens the interpretation that the biological signal is not solely an artefact of the analytical surrogate.

The repeated Phase 30 TOPAS runs provide a practical uncertainty band for the physical benchmark. The repeated-run coefficient of variation for PVDR was `9.47%`, and for peri-GTV 0-5 mm spill it was `12.53%`.
These bands should be used to justify that downstream biological reinterpretation is being judged against a measured Monte Carlo noise scale rather than against an unrealistically exact physical baseline.

The Discussion should explicitly avoid claiming clinical VMAT equivalence, deliverable patient-specific planning, or experimental validation of the assay proxies.
Instead, it should claim that protocol-constrained lattice plans may carry hidden biological liability outside the tumour core, and that the current framework provides a structured way to quantify that liability and compare plans on that basis.

A fair concluding statement is that this is a benchmark-anchored, hypothesis-generating biological risk-analysis framework for lattice RT, suitable for retrospective plan reinterpretation and for prioritizing which candidate plans warrant deeper physics or experimental follow-up.

## Figure Legends

**Figure 1.** Synthetic head-and-neck benchmark lattice library. Left: estimated GTV versus mean centre-to-centre spacing for the 10 frozen synthetic benchmark templates; point colour indicates vertex count and point size scales with recommended vertex diameter. Right: template size distribution with overlaid lattice complexity.

**Figure 2.** Rank reinterpretation and uncertainty in the fixed safe-core plan library. Left: slopegraph of plan ranks across physical-only, bystander without vascular sink uptake, and bystander with anatomical vascular sink uptake. Right: heatmap of nominal-rank retention probability under combined Monte Carlo and biology uncertainty.

**Figure 3.** Mean with-sink minus no-sink effect sizes with bootstrap confidence intervals. Left: primary endpoint shifts. Right: assay-proxy shifts. Negative values indicate lower burden after adding anatomical vascular sink uptake.

**Figure 4.** Direct biological reinterpretation of the Phase 30 TOPAS-derived Yang-style photon benchmark. Bars show physical-only, no-sink, and with-sink values for physical/biological selectivity, spill, OAR, and assay-proxy endpoints.

**Figure 5.** Repeated TOPAS uncertainty bands for the Phase 30 photon benchmark. Left and middle: metric trajectories across history scale for each seed. Right: coefficient of variation for selected benchmark metrics.
