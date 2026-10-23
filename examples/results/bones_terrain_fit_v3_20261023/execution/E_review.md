# E review

Source f49e5ab; actual run2026-10-24 on PRO6000.
120 fits,11 renders,zero processing/render failures; 119 one-level clips and one no-level clip.
All120 have zero elevated support boxes. No raised-edge support geometry was recovered.
Five clips exceed2cm in the sampled skin support metric; P95 of per-clip maxima0.0187m.
All240 variants passed the non-ground penetration gate.
111 walls were fitted; max plane shift0.7387354m in on_the_edge_004__A142.
The only fitted ceiling on_the_edge_002__A414 explains0% of13 sampled low-posture frames.

Three PNG observations (agent E inspected all referenced PNGs and matched TSV/viewer data):

1. on_the_edge_001__A236 shows upright skeletal poses with extended arms above one floor level, without walls, ceilings, or raised support boxes.
2. on_the_edge_001__A098 shows overlapping poses beside a purple wall; annotations report0.026m peak skin penetration and32 sampled frames over2cm.
3. on_the_edge_002__A414 shows a purple wall and a small teal ceiling; annotations report0% low-frame explanation and21 sampled frames over2cm penetration.

Agent:01a16a22-ae4f-7231-847b-83c88225d300. Main dispatched runtime through the sole SSH; agent performed read-only artifact review.
