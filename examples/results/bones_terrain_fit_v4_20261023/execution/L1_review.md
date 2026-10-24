# L1 review

Agent L1 inspected all 14 selected PNGs (10 wall_contact, two crouch, two kneel) and five additional spots using view_image, matching the TSV.
The run completed 2,131 fits and 56 renders with zero failures, using source c9039da.

Wall coverage is 32.86% (115/350), comprising 101 lean and 14 brace walls. All 449 retained walls and 1,357 fitted ceilings have zero frames over 2 cm.
The kneel near-zero support target fails: 11/638 minimal and 35/638 maximal exceed 2 cm; P95 depths are 15.5/20.6 mm. Comparable v3 counts were 20/28.
The initial wall selection contains four retained walls and six no-wall results. Both crouch ceilings and the stationary kneel ceiling look reasonable.
h_b_w_crouch_to_stand_270_start_001__A344 has 40.3 mm maximal support penetration and is rejected; the headline's 9.8 mm is minimal.
kneeling_stop_003__A097 has 30.5 mm ground penetration with zero support penetration.
crouch_ff_stop_225_001__A146 has 24.3/40.3 mm vertical support depth while 4/4 sampled variants are accepted; record the different geometries and metrics separately.

Spot notes:
- idle_to_wall_leaning_idle_360_R_002__A289: reasonable wall following the lean; support 0, ground 9.0 mm, 4/4 accepted.
- drinking_wall_leaning_mug_090_R_003__A282: reasonable wall beside the extended arm/bracing; support 0, ground 9.9 mm, 4/4 accepted.
- drinking_wall_leaning_mug_360_R_003__A282: reasonable backward-lean wall; support 0, ground 6.2 mm, 4/4 accepted.
- kneeling_start_R_003__A408: ground is still questionable at 27.6 mm; standing/kneel overlap explains no ceiling; support 0, 0/4 accepted.
- kneeling_stop_003__A051: reasonable absence of a ceiling for an in-place upright transition; support 0, ground 0.8 mm, 0/4 accepted.
