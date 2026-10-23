## Penetration comparison

| Group | Clips | v1 min clips | v3 min clips | v1 min P95 m | v3 min P95 m | v1 max clips | v3 max clips | v1 max P95 m | v3 max P95 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| box_support | 48 | 22 | 23 | 0.5966 | 0.1035 | 22 | 23 | 0.5966 | 0.109 |
| crouch | 1143 | 36 | 157 | 0.0091 | 0.0274 | 36 | 157 | 0.0091 | 0.0274 |
| flat_control | 900 | 0 | 80 | 0.002 | 0.0217 | 0 | 80 | 0.002 | 0.0217 |
| kneel | 638 | 619 | 601 | 0.0962 | 0.0924 | 619 | 602 | 0.0962 | 0.0924 |
| known_dimension_support | 1760 | 32 | 39 | 0.0004 | 0.0176 | 32 | 718 | 0.0004 | 0.158 |
| ladder | 64 | 16 | 9 | 0.7135 | 0.1695 | 16 | 15 | 0.7135 | 0.3928 |
| obstacle_clearance | 86 | 4 | 17 | 0.0156 | 0.0235 | 4 | 19 | 0.0156 | 0.0245 |
| prone | 919 | 904 | 919 | 0.1261 | 0.1259 | 904 | 919 | 0.1261 | 0.1259 |
| quadruped | 831 | 830 | 830 | 0.1298 | 0.1295 | 830 | 830 | 0.1298 | 0.1295 |
| stairs | 49 | 21 | 16 | 0.7385 | 0.1607 | 21 | 33 | 0.7385 | 0.3712 |
| wall_contact | 350 | 36 | 47 | 0.0694 | 0.0676 | 36 | 47 | 0.0694 | 0.0676 |

## Scene counts

| Group | Ceiling statuses | Explanation P50 / P10 | Walls fitted | Failing fitted checks | Variants accepted / total | Upright ceilings | Clutter mean | Ground P95 m |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| box_support | fitted: 1, no_conflict_free_rectangle: 19, no_sustained_low_posture: 9, unexplainable_tight_above_bound: 19 | 0.0 / 0.0 | 43 | 0 | 188 / 192 | 14 | 1.984 | 0.1 |
| crouch | fitted: 1009, no_conflict_free_rectangle: 12, no_sustained_low_posture: 122 | 1.0 / 1.0 | 667 | 28 | 4460 / 4572 | 223 | 2.062 | 0.0274 |
| flat_control | no_sustained_low_posture: 900 | None / None | 574 | 43 | 3428 / 3600 | 1828 | 2.0 | 0.0217 |
| kneel | fitted: 274, no_conflict_free_rectangle: 363, unexplainable_tight_above_bound: 1 | 1.0 / 1.0 | 623 | 8 | 2433 / 2552 | 0 | 1.989 | 0.0926 |
| known_dimension_support | fitted: 2, no_conflict_free_rectangle: 12, no_sustained_low_posture: 1736, unexplainable_tight_above_bound: 10 | 0.5625 / 0.2125 | 1562 | 1 | 6717 / 7040 | 3344 | 1.985 | 0.015 |
| ladder | fitted: 3, no_conflict_free_rectangle: 7, no_sustained_low_posture: 50, unexplainable_body_reaches_bound: 3, unexplainable_tight_above_bound: 1 | 0.6854 / 0.1371 | 51 | 13 | 192 / 256 | 42 | 2.023 | 0.1 |
| obstacle_clearance | no_conflict_free_rectangle: 9, no_sustained_low_posture: 77 | None / None | 59 | 1 | 340 / 344 | 149 | 1.977 | 0.0235 |
| prone | fitted: 831, no_conflict_free_rectangle: 88 | 1.0 / 1.0 | 514 | 420 | 1992 / 3676 | 0 | 1.967 | 0.1 |
| quadruped | fitted: 699, no_conflict_free_rectangle: 132 | 1.0 / 1.0 | 520 | 187 | 2576 / 3324 | 0 | 2.038 | 0.1 |
| stairs | no_conflict_free_rectangle: 7, no_sustained_low_posture: 41, unexplainable_tight_above_bound: 1 | None / None | 22 | 1 | 164 / 196 | 23 | 1.883 | 0.0811 |
| wall_contact | fitted: 85, no_conflict_free_rectangle: 19, no_sustained_low_posture: 246 | 1.0 / 1.0 | 339 | 75 | 1100 / 1400 | 479 | 1.979 | 0.0677 |

## Tight ceiling calibration

| Group | Fitted | P5 m | P50 m | P95 m |
| --- | ---: | ---: | ---: | ---: |
| box_support | 1 | 0.742 | 0.742 | 0.742 |
| crouch | 1009 | 1.0776 | 1.26 | 1.468 |
| flat_control | 0 | - | - | - |
| kneel | 274 | 1.239 | 1.29 | 1.698 |
| known_dimension_support | 2 | 1.6824 | 1.6855 | 1.6886 |
| ladder | 3 | 1.4938 | 1.546 | 1.6918 |
| obstacle_clearance | 0 | - | - | - |
| prone | 831 | 0.335 | 0.558 | 0.6465 |
| quadruped | 699 | 0.715 | 0.822 | 0.927 |
| stairs | 0 | - | - | - |
| wall_contact | 85 | 1.188 | 1.579 | 1.7078 |
