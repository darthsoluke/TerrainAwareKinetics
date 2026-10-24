# BONES terrain fit v7: L1 visual review

Date: 2026-10-24. Source commit: `c4b776a920d69fc3f997f8e38821f1300837f177`.

Viewed all five selected v7 PNGs individually through the image tool, including each complete 3D and side-elevation panel. The selection covers five kneel clips previously classified G or R in v6. Paired numerical evidence comes from `examples` and `previous_v6_examples` in [L1_review_evidence.json](L1_review_evidence.json); the requested filenames and PNG links are in [L1_review_selection.json](L1_review_selection.json).

All five retain zero minimal and maximal support boxes and zero walls. The visible v7 scenes match the recorded v6 geometry: one ceiling-constrained kneeling loop and four floor-only transitions. Three clips gain more appropriate raised-foot evidence filtering; one changes qualification from rejected to environment-explained. The selected clips show no support-box geometry improvement. The remaining ground-depth error in A097 is unchanged.

## Five Visual Judgments

E = `environment_explained`; G = `geometry_only_low_posture_unexplained`; R = `rejected`. Box pairs are `(minimal, maximal)`; patch pairs are `(kept, dropped)`. Body-drop values are in metres, rounded to six decimals, and are the median of the per-patch body-drop medians. V6 did not record body-drop medians; `n/a` in v7 means no raised patch was checked.

| Clip / viewed PNG | One-line visual judgment and geometry change | Qualification v6 -> v7 | Support boxes v6 -> v7 | Raised patches v6 -> v7 | V7 body-drop median |
| --- | --- | --- | --- | --- | --- |
| [kneeling_loop_001__A409](../viz/kneel/kneeling_loop_001__A409.png) | The compact kneeling pose clears the cyan ceiling and has no raised support beneath the folded foot; geometry is unchanged, with improved foot-evidence classification. | R -> E | (0, 0) -> (0, 0) | (1, 0) -> (0, 1) | 0.085012 |
| [kneeling_start_003__A184](../viz/kneel/kneeling_start_003__A184.png) | Overlaid standing-to-kneeling poses occupy the same floor area without a ceiling or raised support; geometry and the unexplained transition classification are unchanged. | G -> G | (0, 0) -> (0, 0) | (0, 0) -> (0, 0) | n/a |
| [kneeling_stop_002__A023](../viz/kneel/kneeling_stop_002__A023.png) | The folded rear leg opens into a standing pose above a clear floor, with no support boxes or ceiling; patch filtering improves while fitted geometry stays unchanged. | G -> G | (0, 0) -> (0, 0) | (2, 0) -> (0, 2) | 0.099499 |
| [kneeling_stop_002__A037](../viz/kneel/kneeling_stop_002__A037.png) | The closely overlapping kneel-to-stand poses need no raised platform and have no ceiling; the right-foot patch is removed from evidence, with unchanged geometry. | G -> G | (0, 0) -> (0, 0) | (1, 0) -> (0, 1) | 0.063511 |
| [kneeling_stop_003__A097](../viz/kneel/kneeling_stop_003__A097.png) | The floor-only rise contains no added support or overhead geometry; geometry remains unchanged and the recorded ground penetration persists. | G -> G | (0, 0) -> (0, 0) | (0, 0) -> (0, 0) | n/a |

## Patch Evidence And Qualification

All four newly discarded patches have reason `body_rests_lower`. The body lies below the candidate sole patch while the foot-only drop is smaller:

| Clip / side | Patch height above ground (m) | Foot-drop median (m) | Body-drop median (m) | Body-below fraction |
| --- | ---: | ---: | ---: | ---: |
| A409 / left | 0.085863 | 0.025146 | 0.085012 | 1.000000 |
| A023 / left | 0.102660 | 0.022596 | 0.097906 | 1.000000 |
| A023 / right | 0.105914 | 0.022888 | 0.101092 | 0.888889 |
| A037 / right | 0.074216 | 0.008741 | 0.063511 | 0.750000 |

A409's retained support-evidence points fall from 16 to 0 in both minimal and maximal fits, and its `foot_evidence_not_supported` variant rejections fall from 4 to 0. Its existing ceiling bounds remain 1.3383615186810494 / 1.710514736175537 m and explain all 157 low-posture frames. The recorded qualification becomes E and `training_eligible=true`; the geometry itself stays the same.

A184 and A037 retain `unexplainable_in_place_transition`; A023 and A097 retain `no_conflict_free_rectangle`. All four remain G with `training_eligible=false`. A023 and A037 already had zero final support boxes in v6, so dropping their candidate patches does not remove fitted boxes. A184 and A097 have no raised patches in either version.

All five have zero measured skin-to-support penetration in both versions. A097 separately retains a maximum skin-to-ground depth of **0.030532932374626398 m**, with **7/41** sampled frames above 2 cm in both minimal and maximal fits, in both versions. This is the remaining geometry defect in the five-clip review.

## L1 Group Gates

The [comparison](../comparison.json) and [target results](../target_results.json) agree on the following results. F counts scene-variant `foot_evidence_not_supported` rejections.

| Gate | V6 | V7 | Judgment |
| --- | ---: | ---: | --- |
| Kneel F | 420 | 140 | Pass: 140 < 420; reduction of 280, or 66.67%. |
| Crouch fitted walls | 55 | 55 | Pass: 55 <= 58. |
| Wall-contact clips with fitted walls | 106/350 | 106/350 (30.29%) | Pass recorded in target results; coverage is unchanged. |

Across all 638 kneel clips, qualification counts change from E/G/R = **214/319/105** to **265/338/35**. This group-level improvement includes the reviewed A409 transition; the other four reviewed clips remain G. Flat-control review is handled by main.
