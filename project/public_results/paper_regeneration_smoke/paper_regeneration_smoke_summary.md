# Paper-regeneration smoke test

- Status: `paper-regeneration passed; full TOPAS rerun not included`
- Regenerated `figureR2` matches staged bundle: `True`
- Verified `figureRS1` matches staged bundle: `True`
- Rank audit: max score diff `0.0`, max rank diff `0`
- Effective-dose sanity: max |Deff - Dphys| at H=0 is `1.91e-06`, monotonic=`True`
- Headline checks:
  - `primary_with_sink_vs_physical` = `6/10`
  - `primary_no_sink_vs_physical` = `5/10`
  - `with_sink_vs_no_sink` = `2/10`
  - `primary_brainstem_flags` = `2/10`
  - `endpoint_deltas_above_band` = `65/70`
  - `noise_qualified_rank_changes` = `0/10`
  - `stable_smoothing_range` = `4,6`
  - `unstable_smoothing_range` = `2`
