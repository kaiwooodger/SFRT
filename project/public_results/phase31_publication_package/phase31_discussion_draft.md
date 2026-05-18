# Phase 31 discussion draft

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
