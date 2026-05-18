# case07: Oral tongue / floor-of-mouth horseshoe

- Site group: `oral_tongue_floor_mouth`
- Site note: Oral tongue / floor-of-mouth horseshoe geometry in the anterior oral cavity.
- Tumour model: `horseshoe / wraparound`
- Estimated GTV: `94.0 cc`
- Instantiated GTV: `101.6 cc`
- Instantiated CTVboost: `155.5 cc`
- Proposed vertices: `3`
- Kept vertices: `2`
- Coordinate note: Coordinate system follows the synthetic benchmark convention: x = left (+) / right (-), y = anterior (+) / posterior (-), z = superior (+) / inferior (-).

## Nearby critical anatomy

- `ARTERIES`: `0.00 mm` from GTV
- `VEINS`: `0.00 mm` from GTV
- `MANDIBLE`: `0.00 mm` from GTV
- `MAXILLA`: `0.00 mm` from GTV
- `SKULL`: `0.00 mm` from GTV

## Vertex validation

- `V1` kept at `[-17.0, 16.0, -8.0]` mm (edge margin `10.142` mm, critical OAR clearance `10.422` mm).
- `V2` kept at `[17.0, 18.0, -8.0]` mm (edge margin `10.142` mm, critical OAR clearance `10.142` mm).
- `V3` pruned from `[0.0, 7.0, -34.0]` mm because `insufficient_oar_clearance`.
