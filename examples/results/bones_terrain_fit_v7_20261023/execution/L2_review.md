# BONES terrain v7: L2 visual review

**The prone and quadruped joint targets did not pass despite substantial reductions.** All ten selected clips changed from R to E, but eight still show fragmented low support boxes: all five prone clips and three quadruped clips. Two quadruped stopping clips have cleared their support boxes.

Reviewed all ten PNGs directly with the image tool, inspecting both the 3D overlay and side elevation. Within each group, review order is v7 minimal box count descending, then filename SHA-256 ascending. The selection contains five now-E, previously non-E clips per group and excludes the 71 targets. Run source: `c4b776a920d69fc3f997f8e38821f1300837f177`. Counts and status changes come from `L2_review_evidence.json`; links below use the publication layout.

E = `environment_explained`; G = `geometry_only_low_posture_unexplained`; R = `rejected`. Box counts below are equal for minimal and maximal fits. The patch column counts unique `body_rests_lower` drops / checked raised-foot patches; lower unchecked patches are outside that denominator. Orange shapes are fitted support boxes, cyan is the fitted ceiling. The clip judgments describe visible foot/knee-path fragments; exact cluster input kinds are recorded separately in the four saved-fit diagnostics below.

| Group / PNG | v6 status / boxes | v7 status / boxes | Raised-foot drops / checked | Judgment from both views |
| --- | --- | --- | ---: | --- |
| prone / [crawl_ff_loop_225_001__A133](../viz/prone/crawl_ff_loop_225_001__A133.png) | R / 11 | E / 13 | 15 / 15 | **Residual fragmentation increased.** A dense row of narrow orange pieces remains beneath the central crawl path, with a continuous ceiling above. All checked foot patches drop, yet the 6.26 cm layer still produces 13 pieces around the foot/knee path. |
| prone / [crawl_ff_loop_225_003__A132](../viz/prone/crawl_ff_loop_225_003__A132.png) | R / 14 | E / 9 | 22 / 24 | **Reduced, still fragmented.** Several separated groups of small boxes remain under successive foot/knee positions at 7.43 cm above ground. The blue poses stay low along the route, without a stepped ascent; two checked foot patches remain. |
| prone / [crawl_ff_start_225_R_002__A233](../viz/prone/crawl_ff_start_225_R_002__A233.png) | R / 12 | E / 9 | 24 / 28 | **Residual row remains.** Small boxes extend from the start through the middle and later crawl path, including narrow terminal pieces beneath the low body trajectory. The ceiling is continuous; the 8.05 cm fragmented support layer and four checked foot patches remain. |
| prone / [crawl_ff_start_315_R_002__A233](../viz/prone/crawl_ff_start_315_R_002__A233.png) | R / 10 | E / 8 | 9 / 16 | **Residual fragmentation remains.** The side view shows separated short blocks, a longer central/later block, and a thin end piece under the foot/knee route. The body remains low beneath the ceiling; the 8.24 cm layer and seven checked foot patches survive. |
| prone / [crawl_ff_loop_225_002__A131](../viz/prone/crawl_ff_loop_225_002__A131.png) | R / 7 | E / 7 | 9 / 12 | **No box-count improvement.** Seven scattered low pieces remain beneath the crawl trajectory, with narrow pieces near the middle and end. The ceiling covers the motion, while the 6.50 cm support layer and three checked foot patches remain. |
| quadruped / [on_all_fours_forward_start_R_003__A199](../viz/quadruped/on_all_fours_forward_start_R_003__A199.png) | R / 4 | E / 10 | 11 / 11 | **Fragmentation worsened.** Ten narrow posts follow the repeated foot/knee contact path over the long forward start, clearly separated in side elevation beneath a continuous ceiling. Every checked foot patch drops, while the 6.62 cm layer produces six additional boxes. |
| quadruped / [spider_crawl_R_004__A361](../viz/quadruped/spider_crawl_R_004__A361.png) | R / 6 | E / 4 | 5 / 5 | **Four residual pieces remain.** Small isolated orange blocks occupy the middle of the flattened crawl trajectory, beneath the legs and low body, with the motion extending beyond them. All checked foot patches drop; the 6.23 cm fragmented layer remains. |
| quadruped / [mohak_backward_stop_004__A121](../viz/quadruped/mohak_backward_stop_004__A121.png) | R / 4 | E / 3 | 5 / 5 | **Three residual posts remain.** The stopping sequence settles around three isolated blocks under the knee/foot region. Both views show a continuous overhead slab and separated ground-adjacent posts; all checked foot patches drop, but the 6.52 cm layer remains. |
| quadruped / [on_all_fours_backward_stop_R_003__A200](../viz/quadruped/on_all_fours_backward_stop_R_003__A200.png) | R / 1 | E / 0 | 9 / 9 | **Support fragments cleared.** Both views show the hand/knee stopping path over a continuous ground reference, with the ceiling overhead and no orange foot/knee boxes. The raised layer is removed and `levels_m` is `[0]`. |
| quadruped / [on_all_fours_forward_stop_R_004__A199](../viz/quadruped/on_all_fours_forward_stop_R_004__A199.png) | R / 4 | E / 0 | 7 / 7 | **Support fragments cleared.** The forward stopping poses share the ground reference beneath a continuous ceiling, with no isolated orange boxes in either view. All checked patches drop and only the ground level remains. |

All ten saved v6 examples had four `foot_evidence_not_supported` variant rejections; all ten v7 examples have zero. Their saved v7 minimal/maximal skin-support measurements have zero frames over 2 cm. Those results explain the status improvement alongside the eight visible residual cases.

## Full-group targets

The merged [comparison](../comparison.json) and [target results](../target_results.json) cover 919 prone and 831 quadruped clips. F counts scene-variant rejections for `foot_evidence_not_supported`; the four-plus statistic counts clips with at least four support boxes. The joint requirements are F below 30% of v6, four-plus clips below 10% of v6, and increased E+G share.

| Group | F: v6 -> v7; gate | Four-plus clips: v6 -> v7; gate | E/G/R: v6 -> v7 | E+G | Joint verdict |
| --- | --- | --- | --- | --- | --- |
| prone | 1815 -> 582; 32.07% retained, **fails** `<544.5` | 381 -> 150; 39.37% retained, **fails** `<38.1` | 306/163/450 -> 549/226/144 | 469 -> 775 | **Not passed** |
| quadruped | 2207 -> 436; 19.76% retained, **passes** `<662.1` | 82 -> 21; 25.61% retained, **fails** `<8.2` | 207/73/551 -> 636/86/109 | 280 -> 722 | **Not passed** |

Prone F falls 67.93% and quadruped F falls 80.24%; four-plus counts fall 60.63% and 74.39%. E+G increases in both groups. The remaining prone F and both four-plus counts leave the joint targets unmet; the saved overall `passed` value is also `false`.

## Verified residual mechanism

The [residual support diagnostic](../residual_support_diagnostic.json) reads four existing v6/v7 fits. These are additional diagnostic clips, separate from the ten PNGs above. Values below are saved-fit measurements; input kinds describe fitted clusters before any subset selection.

| Diagnostic clip | Foot patches v6 -> v7; other events | Clusters v6 -> v7 | Boxes v6 -> v7 | v7 fitted clusters by input kinds |
| --- | --- | --- | --- | --- |
| `crawl_ff_loop_225_004__A131` | 13 -> 2; 22 knees unchanged | 8 -> 9 | 8 -> 8 | 1 foot-only, 7 knee-only; 1 other cluster rejected for body conflict |
| `crawl_ff_start_225_002__A133` | 10 -> 4; 7 knees unchanged | 5 -> 6 | 5 -> 6 | 1 foot-only, 2 foot+knee, 3 knee-only |
| `crawl_ff_stop_225_006__A131` | 14 -> 7; 13 knees unchanged | 11 -> 13 | 9 -> 10 | 5 foot-only, 1 foot+knee, 4 knee-only; 3 other clusters rejected |
| `crawl_ff_start_360_002__A130` | 10 -> 5; 9 knees and 2 palms unchanged | 6 -> 7 | 6 -> 7 | 3 foot-only, 1 foot+knee, 2 knee-only, 1 palm-only |

For A131 loop 004, all 11 checked foot patches drop. Its two surviving foot patches are 5.9563 and 4.5964 cm above ground, below the 6 cm checking threshold; they are assigned upward by 0.7112 and 2.0712 cm to the 6.6675 cm layer. They keep that layer available to the unchanged knee events. Removing connecting foot polygons separates the evidence: v6's fitted clusters were 3 foot-only, 4 mixed foot+knee and 1 knee-only; v7 has 1 foot-only and 7 knee-only. For A130 start 360, the increase is in foot-only fitted clusters, from 2 to 3, with knee-only and palm-only counts unchanged.

The resulting residuals combine retained low foot evidence, a surviving raised layer, and fragmented limb-support clusters. The base ground slab remains continuous. This separates successful removal of contradicted raised-foot evidence from the unresolved small-box geometry visible in the review. No code, parameters or fits were changed for this review.
