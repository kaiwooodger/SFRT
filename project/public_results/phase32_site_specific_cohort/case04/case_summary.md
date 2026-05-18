# case04: Laryngo-hypopharyngeal elongated cylinder

- Site group: `laryngohypopharynx`
- Site note: Elongated laryngo-hypopharyngeal target along the cranio-caudal axis.
- Tumour model: `elongated cylinder`
- Estimated GTV: `76.0 cc`
- Instantiated GTV: `66.4 cc`
- Instantiated CTVboost: `120.3 cc`
- Proposed vertices: `2`
- Kept vertices: `1`
- Coordinate note: Coordinate system follows the synthetic benchmark convention: x = left (+) / right (-), y = anterior (+) / posterior (-), z = superior (+) / inferior (-).

## Nearby critical anatomy

- `ARTERIES`: `0.00 mm` from GTV
- `VEINS`: `0.00 mm` from GTV
- `SPINAL_CORD`: `0.00 mm` from GTV
- `BRAINSTEM`: `2.00 mm` from GTV
- `CHIASM`: `28.43 mm` from GTV

## Vertex validation

- `V1` pruned from `[0.0, -12.0, -14.0]` mm because `insufficient_oar_clearance`.
- `V2` kept at `[0.0, -3.0, -46.0]` mm (edge margin `3.0` mm, critical OAR clearance `11.0` mm).
