# L2 review

Agent inspected all 10 selected PNGs (5 prone, 5 quadruped) and both named spot images against TSV and audit records. Runtime source c9039da; 1750 fits, 44 renders, zero execution failures.

Near-zero support penetration FAILS for both groups. Prone normalized v3 min/max clips over 2 cm change from 1/1 to 402/595; quadruped 1/1 become 118/311. Ground placement improves: prone ground peak P95 falls from 0.1259 m to 0.0145 m; quadruped from 0.1295 m to 0.0134 m.

ground_from_skin: prone 905/919 (98.48%), quadruped 801/831 (96.39%). Retained walls 1/37 and fitted ceilings 647/692 have no frames over 2 cm. Quadruped dropped 40 walls after the full-skin check.

Prone eligibility: environment_explained 238 clips/2171.26s; geometry_only_low_posture_unexplained 130/1109.68s; rejected 551/4976.76s.
Quadruped eligibility: environment_explained 205/1737.69s; geometry_only_low_posture_unexplained 74/433.56s; rejected 552/4255.02s.

Top prone rejection counts per variant: solid_penetrates_skin 1923, foot_evidence_not_supported 1637, low_posture_unexplained 1088.
Top quadruped counts: foot_evidence_not_supported 2184, low_posture_unexplained 556, solid_penetrates_skin 428.

All seven inverted-foot cases reviewed: eight dropped levels, no support penetration above 2 cm. support_rejected flags: 302 prone, 478 quadruped.

The two sampled continuous crawl loops have flat ground and clear overhead ceilings; all four variants are accepted. The other eight sampled clips are rejected; seven exceed 2 cm for maximal supports. PNG captions show minimal metrics, so maximal failures are read from the TSV.

Spot checks:
- crawl_ff_loop_315_005__A131: reasonable; sustained crawl under a 0.642-0.654 m ceiling, ground 4.2 mm, support 0, no walls, 4/4 accepted.
- idle_crawl_stop_001__A128: suspicious supports; four raised boxes penetrate 23.0/26.6 mm in 10/13 sampled frames; ground 7.5 mm. The standing transition explains omission of a ceiling, but supports are rejected; 0/4 accepted.
