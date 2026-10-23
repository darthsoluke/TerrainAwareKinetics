# S review

Agent S reviewed completed PRO6000 reports and compact per-fit audit evidence; source f49e5ab.
2907 fits and123renders succeeded,zero failures. Skin18056vertices,stride2; elapsed1446.4s.

- Variants:11029/11628 accepted;599rejected across152clips.
- Walls:2311clips fitted,59over2cm (flat43,ladder13,known dimensions1,obstacle1,stairs1).
- The wall of walk_backward_loop_004__A026 shifted10.7777m; its final position is far from the initial contact plane.
- Flat-control extra support boxes fell from31/900 in v1 to0/900 in v3.
-47clips have support_rejected;62clusters rejected. Unsupported evidence0 does not measure the support gap.
- stairs_climbing_down_loop_R_001__A300 has skin vertical depth0.524367m and support-gap P95 0.992220m; all4variants were accepted.
- jump_to_shoulder_roll_R_003__A304 has0retained support boxes and support-gap P95 1.709351m; all4variants were accepted.
- v1 sphere metrics deduct2cm tolerance; v3 main metrics use zero-tolerance sampled skin. Raw tables preserve that distinction.
- Support depth includesground and measures vertical distance belowtops. Scene acceptance uses nearest-face depth in non-ground solids; support gaps/explanation are not acceptance gates.
- Actor scale P5/P50/P95:1.04031/1.14434/1.23580;within-actorheight std median1.1243mm.

Main subsequently verified all fitted ceiling checks across all shards have max_m=0; geometry_gate_audit.json contains the complete result.
