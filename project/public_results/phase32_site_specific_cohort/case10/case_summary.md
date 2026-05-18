# case10: Composite oropharynx + upper-neck very bulky tumour

- Site group: `composite_oropharynx_upper_neck`
- Site note: Very bulky composite oropharynx + upper-neck target extending across compartments.
- Tumour model: `giant composite volume`
- Estimated GTV: `326.0 cc`
- Instantiated GTV: `329.0 cc`
- Instantiated CTVboost: `409.8 cc`
- Proposed vertices: `5`
- Kept vertices: `4`
- Coordinate note: Coordinate system follows the synthetic benchmark convention: x = left (+) / right (-), y = anterior (+) / posterior (-), z = superior (+) / inferior (-).

## Nearby critical anatomy

- `ARTERIES`: `0.00 mm` from GTV
- `VEINS`: `0.00 mm` from GTV
- `SPINAL_CORD`: `0.00 mm` from GTV
- `BRAINSTEM`: `0.00 mm` from GTV
- `CHIASM`: `0.00 mm` from GTV

## Vertex validation

- `V1` pruned from `[-4.0, -4.0, -12.0]` mm because `candidate_center_outside_gtv`.
- `V2` kept at `[-26.0, -4.0, 2.0]` mm (edge margin `3.798` mm, critical OAR clearance `15.909` mm).
- `V3` kept at `[18.0, -6.0, 2.0]` mm (edge margin `8.967` mm, critical OAR clearance `10.125` mm).
- `V4` kept at `[-12.0, -22.0, -28.0]` mm (edge margin `4.392` mm, critical OAR clearance `15.633` mm).
- `V5` kept at `[2.0, 23.0, -23.0]` mm (edge margin `5.314` mm, critical OAR clearance `10.125` mm).
