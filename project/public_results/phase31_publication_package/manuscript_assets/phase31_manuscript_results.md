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
