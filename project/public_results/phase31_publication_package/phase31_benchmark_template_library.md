# Phase 31 synthetic H&N benchmark template library

These templates are synthetic geometry benchmarks derived from the Yang-style lattice planning rules supplied in the study notes.

## Common assumptions

- Lattice boost subplan only: 15 Gy to vertices and 3.5 Gy to CTVboost = GTV + 5 mm in 1 fraction.
- Photon concept: 2 full VMAT arcs as the default Yang-like delivery concept.
- Vertex rule: keep each sphere wholly intratumoral; if a sphere would breach GTV, move or delete it.
- H&N pruning rule: if a vertex competes with carotid, jugular, spinal cord, brainstem, optic structures, cochlea, mandible, skull base, or skin, delete it rather than force symmetry.
- PET guidance is a secondary selector only and does not override OAR safety.
- These 10 geometries are synthetic benchmark templates, not published patient datasets.

## Template summary

### case01: Right sinonasal / maxillary bulky ellipsoid
- Tumour model: `ellipsoid`
- Bounding dimensions: `5.8 x 4.8 x 4.5 cm`
- Estimated GTV: `66.0 cc`
- Vertices: `2`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.225]`
- Coordinates: `[[-1.4, 0.0, 0.8], [1.4, 0.0, -0.8]]`
- Intent: Yang-like two-sphere layout for a compact midface tumour; keeps hot spots central and away from orbit/skull-base interfaces.

### case02: Left maxillary sinus crescent abutting orbit
- Tumour model: `crescent / banana`
- Bounding dimensions: `7.2 x 5.0 x 4.6 cm`
- Estimated GTV: `87.0 cc`
- Vertices: `2`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.247]`
- Coordinates: `[[-1.5, -0.6, 0.7], [1.2, 0.4, -0.8]]`
- Intent: Non-symmetric two-sphere layout for a crescent wrapped toward the orbit.
- Caution: Do not place a third vertex toward the superior-anterior orbital limb.

### case03: Base-of-tongue / oropharynx bilobed central mass
- Tumour model: `bilobed`
- Bounding dimensions: `7.0 x 5.8 x 5.2 cm`
- Estimated GTV: `111.0 cc`
- Vertices: `3`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.4, 2.97, 2.97]`
- Coordinates: `[[-1.7, 0.0, 0.9], [1.7, 0.0, 0.9], [0.0, -0.8, -1.4]]`
- Intent: Triangular central pattern avoiding lateral drift toward carotid spaces.

### case04: Laryngo-hypopharyngeal elongated cylinder
- Tumour model: `elongated cylinder`
- Bounding dimensions: `8.6 x 4.2 x 4.0 cm`
- Estimated GTV: `76.0 cc`
- Vertices: `2`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.276]`
- Coordinates: `[[0.0, -0.4, 1.6], [0.0, 0.3, -1.6]]`
- Intent: Superior-inferior stacking rather than lateral pairing in a narrow geometry.
- Caution: Do not force lateral paired vertices near carotid/cord interfaces.

### case05: Parapharyngeal / prestyloid deep-space bulky mass
- Tumour model: `asymmetric ellipsoid`
- Bounding dimensions: `7.5 x 5.6 x 5.2 cm`
- Estimated GTV: `114.0 cc`
- Vertices: `3`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.028, 3.192, 2.846]`
- Coordinates: `[[-1.5, -0.5, 1.2], [1.5, -0.4, 0.8], [0.0, 0.8, -1.3]]`
- Intent: Tripod in the medial solid core while staying away from skull-base and carotid-adjacent extremes.

### case06: Buccal mucosa / cheek infiltrative crescent
- Tumour model: `superficial crescent`
- Bounding dimensions: `8.8 x 6.0 x 3.6 cm`
- Estimated GTV: `100.0 cc`
- Vertices: `2`
- Recommended diameter: `0.8 cm`
- Spacing summary: `[3.002]`
- Coordinates: `[[-1.2, -0.4, 0.8], [1.2, 0.6, -0.7]]`
- Intent: Two small deep-core vertices for crowded superficial cheek anatomy.
- Caution: Stay off skin and mandibular cortex.

### case07: Oral tongue / floor-of-mouth horseshoe
- Tumour model: `horseshoe / wraparound`
- Bounding dimensions: `7.4 x 6.4 x 3.8 cm`
- Estimated GTV: `94.0 cc`
- Vertices: `3`
- Recommended diameter: `0.8 cm`
- Spacing summary: `[3.0, 3.134, 3.134]`
- Coordinates: `[[-1.5, 0.0, 0.8], [1.5, 0.0, 0.8], [0.0, -0.9, -1.8]]`
- Intent: Three-vertex deep tongue-body pattern with inferior-posterior core anchoring.
- Caution: Avoid anterior superficial floor-of-mouth and lingual cortex interfaces.

### case08: Bulky nodal conglomerate level II-IV with central necrosis
- Tumour model: `irregular nodal mass`
- Bounding dimensions: `9.0 x 6.6 x 5.4 cm`
- Estimated GTV: `168.0 cc`
- Vertices: `4`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[3.6, 2.786, 3.813, 3.677, 2.839, 2.627]`
- Coordinates: `[[-1.8, 0.0, 1.2], [1.8, 0.0, 1.2], [-0.8, -1.0, -1.2], [0.9, 1.0, -1.3]]`
- Intent: Four viable-rim vertices around an irregular necrotic core.
- Caution: Do not place a vertex into fully liquefactive central necrosis.

### case09: Deep parotid / infratemporal fossa bulky mass
- Tumour model: `irregular ellipsoid`
- Bounding dimensions: `8.0 x 6.0 x 5.8 cm`
- Estimated GTV: `146.0 cc`
- Vertices: `3`
- Recommended diameter: `1.0 cm`
- Spacing summary: `[2.818, 3.082, 2.871]`
- Coordinates: `[[-1.4, -0.6, 1.0], [1.4, -0.3, 0.9], [0.0, 0.9, -1.3]]`
- Intent: Three deep-biased vertices in preserved soft-tissue bulk.
- Caution: Stay off superficial parotid skin edge and skull-base foramina.

### case10: Composite oropharynx + upper-neck very bulky tumour
- Tumour model: `giant composite volume`
- Bounding dimensions: `10.8 x 8.0 x 7.2 cm`
- Estimated GTV: `326.0 cc`
- Vertices: `5`
- Recommended diameter: `1.2 cm`
- Spacing summary: `[2.608, 2.608, 2.538, 2.739, 4.4, 3.768, 4.844, 4.609, 3.829, 4.116]`
- Coordinates: `[[0.0, 0.0, 0.0], [-2.2, 0.0, 1.4], [2.2, 0.0, 1.4], [-0.8, -1.8, -1.6], [1.0, 1.9, -1.7]]`
- Intent: Sparse central five-point array borrowing from larger-volume lattice planning without automatically escalating to 6-8 vertices in H&N.
- Caution: Do not escalate to 6-8 vertices automatically in crowded H&N anatomy.

