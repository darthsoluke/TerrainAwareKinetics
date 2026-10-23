# L1 review

Agent L1 reviewed completed PRO6000 reports and per-fit audit evidence; source f49e5ab.
2131fits and58renders succeeded,zero failures;18056vertices,stride2;elapsed1394.1s.

| Group | Ceilings | Walls | Wall clips over2cm | Accepted variants |
| --- | ---: | ---: | ---: | --- |
| crouch |1009/1143|667|28|4460/4572|
| kneel |274/638|623|8|2433/2552|
| wall_contact |85/350|339|75|1100/1400|

All three groups have fitted-subset explanation P50/P10=1/1. Kneel has363rectangle failures and1tight-above-bound failure.
Maximum wall shifts:crouch7.125665m,kneel0.453500m,wall_contact1.065019m.
Contact scores describe the original plane; final contact is not remeasured after shifting.
kneeling_loop_003__A062 penetrates its wall0.048860m in146/146sampledframes;all4variants rejected.
crouch_ff_stop_270_R_003__A244 has tight explanation0.0625 and loose1.0.
Rejected variants:112/119/300 across28/32/75clips respectively. Acceptance tests non-ground solids, not explanation or support gaps.
All crouch and wall_contact fits have0supportboxes; their flagged support-skin depths measure the ground reference.
Posture ratios0.72/0.45 remain configured inputs; requested tight-height percentile calibration is in the main report.

Main's later complete geometry_gate_audit confirms every fitted ceiling max_m=0 in sampled frames. Walls and variant quality gates remain unmet.
