# case06: Buccal mucosa / cheek infiltrative crescent

- Site group: `cheek_superficial`
- Site note: Superficial buccal / cheek disease with skin and mandibular proximity.
- Tumour model: `superficial crescent`
- Estimated GTV: `100.0 cc`
- Instantiated GTV: `100.6 cc`
- Instantiated CTVboost: `144.6 cc`
- Proposed vertices: `2`
- Kept vertices: `1`
- Coordinate note: Coordinate system follows the synthetic benchmark convention: x = left (+) / right (-), y = anterior (+) / posterior (-), z = superior (+) / inferior (-).

## Nearby critical anatomy

- `ARTERIES`: `0.00 mm` from GTV
- `VEINS`: `0.00 mm` from GTV
- `OPTIC_NERVE_R`: `0.00 mm` from GTV
- `EYE_R`: `0.00 mm` from GTV
- `LENS_R`: `0.00 mm` from GTV

## Vertex validation

- `V1` pruned from `[-48.0, 20.0, 2.0]` mm because `insufficient_oar_clearance`.
- `V2` kept at `[-16.0, 38.0, -9.0]` mm (edge margin `0.472` mm, critical OAR clearance `10.0` mm).
