# L2 review

Agent L2 reviewed completed PRO6000 reports and per-fit audit evidence; source f49e5ab.
1750fits and44renders succeeded,zero failures;18056vertices,stride2;elapsed1434.1s.

| Group | Ceilings | Rectangle failures | Walls | Wall clips over2cm | Accepted variants |
| --- | ---: | ---: | ---: | ---: | --- |
| quadruped |699/831|132|520|187|2576/3324|
| prone |831/919|88|514|420|1992/3676|

Fitted ceiling explanation P50/P10=1/1 in bothgroups. mohak_idle_stop_001__A125 has explanation0 but all4variants accepted.
607clips have wall penetration over2cm. quadruped maximum wallshift7.839142m is mohak_loop_003__A032, with remaining0.054048m penetration in6/227frames.
Prone maximum wallshift1.961044m. The fitting height band is0.2-2.0m; full-skin verification also checks points outside it.
2432 variants were rejected across 608 clips. L2's 607 failing walls account for 2428 rejected variants; one additional prone clip accounts for four more.
Ground solid-depth P95 is0.1m in bothgroups, capped by nearest-face depth of the0.2m thick floor. Skin vertical depth can be larger.
The complete per-fit audit rejects nonfinite JSON; allfiles parsed. Wall-scan RuntimeWarnings were preserved by main.

Main's later complete geometry_gate_audit confirms every fitted ceiling max_m=0 in sampled frames. Wall and variant gates remain unmet.
