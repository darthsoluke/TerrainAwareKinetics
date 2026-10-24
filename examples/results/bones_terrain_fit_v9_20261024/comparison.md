# BONES terrain v7 / v9：全量逐组对照

相同6,788条、11组。保留v7全部比较字段，并添加`stage_seconds`。v7没有同口径阶段计时，以null表示。

F是四个场景变体的缺支撑拒绝次数；skin_support为支撑箱体蒙皮指标；尺寸单位为米，耗时单位为秒。

patches的dropped统计去重patch，dropped_by统计每个原因的命中数，一块patch可以命中多个原因。

[执行记录](../../../docs/records/bones_terrain_fit_v9_pro6000_20261024.md)包含主要变化、门槛和图片审查。

## box_support

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 48 | 48 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0192}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0192}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0178}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0175}}` |
| ground_skin_depth_m | `{"p50": 0.01, "p95": 0.0347, "p99": 0.0393}` | `{"p50": 0.01, "p95": 0.0347, "p99": 0.0393}` |
| ground_source | `{"lowest_foot_level": 28, "skin_lowest_contacts": 20}` | `{"lowest_foot_level": 28, "skin_lowest_contacts": 20}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 8, "ground_from_skin": 20, "inverted_foot_level_dropped": 3, "support_rejected": 9}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 5, "ground_from_skin": 20, "inverted_foot_level_dropped": 3, "support_rejected": 5}` |
| patches_resting_on_body | 15 | 7 |
| raised_sole_patches | `{"checked": 161, "dropped": 15, "dropped_by": {"body_rests_lower": 14, "body_under_sole": 1, "foot_lower_than_sole": 1}, "kept": 146}` | `{"checked": 115, "dropped": 7, "dropped_by": {"body_rests_lower": 6, "body_under_sole": 1, "foot_lower_than_sole": 1}, "kept": 108}` |
| median_foot_drop_median_m | 0.005514 | 0.005047 |
| median_body_drop_median_m | -0.076752 | -0.078267 |
| F | 32 | 24 |
| clips_with_4plus_support_boxes | 3 | 0 |
| max_support_boxes | 4 | 3 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 0 | 0 |
| wall_status | `{"no_contact_plane": 48}` | `{"no_contact_plane": 48}` |
| wall_kind | `{}` | `{}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 17, "no_trunk_or_bracing_hand_contact": 756}` | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 17, "no_trunk_or_bracing_hand_contact": 756}` |
| ceiling_status | `{"fitted": 1, "no_conflict_free_rectangle": 3, "no_sustained_low_posture": 9, "unexplainable_in_place_transition": 14, "unexplainable_tight_above_bound": 21}` | `{"fitted": 1, "no_conflict_free_rectangle": 3, "no_sustained_low_posture": 9, "unexplainable_in_place_transition": 15, "unexplainable_tight_above_bound": 20}` |
| explanation_p50 | 0.0 | 0.0 |
| explanation_p10 | 0.0 | 0.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 192 | 192 |
| variants_accepted | 36 | 36 |
| variant_acceptance_rate | 0.1875 | 0.1875 |
| rejection_reasons | `{"foot_evidence_not_supported": 32, "low_posture_partly_explained": 4, "low_posture_unexplained": 152}` | `{"foot_evidence_not_supported": 24, "low_posture_partly_explained": 4, "low_posture_unexplained": 152}` |
| eligibility | `{"environment_explained": {"clips": 9, "seconds": 41.97}, "geometry_only_low_posture_unexplained": {"clips": 31, "seconds": 138.97}, "rejected": {"clips": 8, "seconds": 38.27}}` | `{"environment_explained": {"clips": 9, "seconds": 41.97}, "geometry_only_low_posture_unexplained": {"clips": 33, "seconds": 147.6}, "rejected": {"clips": 6, "seconds": 29.64}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 42, "p50": 1.5926, "p95": 2.6202, "total": 68.7}, "motion_converted": {"clips": 6, "p50": 3.6479, "p95": 4.4098, "total": 20.7}, "scene": {"clips": 48, "p50": 1.1725, "p95": 3.1494, "total": 68.2}, "screen": {"clips": 48, "p50": 0.047, "p95": 0.1465, "total": 2.6}, "support_fit": {"clips": 48, "p50": 1.069, "p95": 2.757, "total": 58.0}}` |

## crouch

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 1143 | 1143 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0119, "p95": 0.025, "p99": 0.03}` | `{"p50": 0.0119, "p95": 0.025, "p99": 0.03}` |
| ground_source | `{"lowest_foot_level": 1116, "skin_lowest_contacts": 27}` | `{"lowest_foot_level": 1116, "skin_lowest_contacts": 27}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 3, "ground_from_skin": 27, "inverted_foot_level_dropped": 0, "support_rejected": 2}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 0, "ground_from_skin": 27, "inverted_foot_level_dropped": 0, "support_rejected": 0}` |
| patches_resting_on_body | 3 | 0 |
| raised_sole_patches | `{"checked": 53, "dropped": 3, "dropped_by": {"body_rests_lower": 3, "body_under_sole": 1}, "kept": 50}` | `{"checked": 0, "dropped": 0, "dropped_by": {}, "kept": 0}` |
| median_foot_drop_median_m | 0.005201 | null |
| median_body_drop_median_m | -0.073607 | null |
| F | 19 | 0 |
| clips_with_4plus_support_boxes | 4 | 0 |
| max_support_boxes | 8 | 0 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 55 | 55 |
| wall_status | `{"fitted": 55, "no_contact_plane": 679, "no_still_segment": 409}` | `{"fitted": 55, "no_contact_plane": 679, "no_still_segment": 409}` |
| wall_kind | `{"brace": 46, "lean": 9}` | `{"brace": 46, "lean": 9}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 18, "feet_at_the_plane": 147, "no_trunk_or_bracing_hand_contact": 15155, "plane_must_move_to_clear_body": 163}` | `{"contact_slides_along_plane": 18, "feet_at_the_plane": 147, "no_trunk_or_bracing_hand_contact": 15155, "plane_must_move_to_clear_body": 163}` |
| ceiling_status | `{"fitted": 1008, "no_conflict_free_rectangle": 9, "no_sustained_low_posture": 122, "unexplainable_in_place_transition": 4}` | `{"fitted": 1008, "no_conflict_free_rectangle": 9, "no_sustained_low_posture": 122, "unexplainable_in_place_transition": 4}` |
| explanation_p50 | 1.0 | 1.0 |
| explanation_p10 | 1.0 | 1.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 4572 | 4572 |
| variants_accepted | 4433 | 4444 |
| variant_acceptance_rate | 0.969598 | 0.972003 |
| rejection_reasons | `{"foot_evidence_not_supported": 19, "low_posture_partly_explained": 76, "low_posture_unexplained": 52}` | `{"low_posture_partly_explained": 76, "low_posture_unexplained": 52}` |
| eligibility | `{"environment_explained": {"clips": 1109, "seconds": 6632.54}, "geometry_only_low_posture_unexplained": {"clips": 30, "seconds": 274.09}, "rejected": {"clips": 4, "seconds": 36.97}}` | `{"environment_explained": {"clips": 1111, "seconds": 6644.44}, "geometry_only_low_posture_unexplained": {"clips": 32, "seconds": 299.16}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 49, "p50": 1.9728, "p95": 3.2908, "total": 99.7}, "motion_converted": {"clips": 1094, "p50": 3.5513, "p95": 6.1495, "total": 4208.0}, "scene": {"clips": 1143, "p50": 1.3999, "p95": 4.906, "total": 2091.5}, "screen": {"clips": 1143, "p50": 0.0447, "p95": 0.1091, "total": 61.8}, "support_fit": {"clips": 1143, "p50": 0.566, "p95": 1.6794, "total": 804.1}}` |

## flat_control

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 900 | 900 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0128, "p95": 0.0216, "p99": 0.0275}` | `{"p50": 0.0128, "p95": 0.0216, "p99": 0.0275}` |
| ground_source | `{"lowest_foot_level": 898, "skin_lowest_contacts": 2}` | `{"lowest_foot_level": 898, "skin_lowest_contacts": 2}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 0, "ground_from_skin": 2, "inverted_foot_level_dropped": 0, "support_rejected": 0}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 0, "ground_from_skin": 2, "inverted_foot_level_dropped": 0, "support_rejected": 0}` |
| patches_resting_on_body | 0 | 0 |
| raised_sole_patches | `{"checked": 0, "dropped": 0, "dropped_by": {"body_rests_lower": 0}, "kept": 0}` | `{"checked": 0, "dropped": 0, "dropped_by": {}, "kept": 0}` |
| median_foot_drop_median_m | null | null |
| median_body_drop_median_m | null | null |
| F | 0 | 0 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 0 | 0 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 3 | 3 |
| wall_status | `{"fitted": 3, "no_contact_plane": 618, "no_still_segment": 279}` | `{"fitted": 3, "no_contact_plane": 618, "no_still_segment": 279}` |
| wall_kind | `{"brace": 3}` | `{"brace": 3}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 7, "feet_at_the_plane": 374, "no_trunk_or_bracing_hand_contact": 11721, "plane_must_move_to_clear_body": 3}` | `{"contact_slides_along_plane": 7, "feet_at_the_plane": 374, "no_trunk_or_bracing_hand_contact": 11721, "plane_must_move_to_clear_body": 3}` |
| ceiling_status | `{"no_sustained_low_posture": 900}` | `{"no_sustained_low_posture": 900}` |
| explanation_p50 | null | null |
| explanation_p10 | null | null |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 3600 | 3600 |
| variants_accepted | 3600 | 3600 |
| variant_acceptance_rate | 1.0 | 1.0 |
| rejection_reasons | `{}` | `{}` |
| eligibility | `{"environment_explained": {"clips": 900, "seconds": 6277.69}}` | `{"environment_explained": {"clips": 900, "seconds": 6277.69}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 41, "p50": 2.5229, "p95": 4.8756, "total": 114.3}, "motion_converted": {"clips": 859, "p50": 4.2924, "p95": 10.2219, "total": 4515.7}, "scene": {"clips": 900, "p50": 1.4734, "p95": 6.5185, "total": 1948.7}, "screen": {"clips": 900, "p50": 0.05, "p95": 0.1618, "total": 66.6}, "support_fit": {"clips": 900, "p50": 0.7885, "p95": 2.5505, "total": 977.8}}` |

## kneel

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 638 | 638 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0013}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0006}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0006, "p95": 0.0284, "p99": 0.0312}` | `{"p50": 0.0006, "p95": 0.0284, "p99": 0.0312}` |
| ground_source | `{"lowest_foot_level": 143, "skin_lowest_contacts": 495}` | `{"lowest_foot_level": 143, "skin_lowest_contacts": 495}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 292, "ground_from_skin": 495, "inverted_foot_level_dropped": 92, "support_rejected": 31}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 95, "ground_from_skin": 495, "inverted_foot_level_dropped": 42, "support_rejected": 2}` |
| patches_resting_on_body | 470 | 137 |
| raised_sole_patches | `{"checked": 569, "dropped": 470, "dropped_by": {"body_rests_lower": 468, "body_under_sole": 61, "foot_lower_than_sole": 33}, "kept": 99}` | `{"checked": 143, "dropped": 137, "dropped_by": {"body_rests_lower": 137, "body_under_sole": 28, "foot_lower_than_sole": 17}, "kept": 6}` |
| median_foot_drop_median_m | 0.016264 | 0.020756 |
| median_body_drop_median_m | 0.074768 | 0.086361 |
| F | 140 | 4 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 2 | 0 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 2 | 2 |
| wall_status | `{"fitted": 2, "no_contact_plane": 636}` | `{"fitted": 2, "no_contact_plane": 636}` |
| wall_kind | `{"lean": 2}` | `{"lean": 2}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 2, "feet_at_the_plane": 60, "no_trunk_or_bracing_hand_contact": 12947, "plane_must_move_to_clear_body": 2}` | `{"contact_slides_along_plane": 2, "feet_at_the_plane": 60, "no_trunk_or_bracing_hand_contact": 12947, "plane_must_move_to_clear_body": 2}` |
| ceiling_status | `{"fitted": 272, "no_conflict_free_rectangle": 113, "unexplainable_in_place_transition": 245, "unexplainable_tight_above_bound": 8}` | `{"fitted": 273, "no_conflict_free_rectangle": 113, "unexplainable_in_place_transition": 252}` |
| explanation_p50 | 1.0 | 1.0 |
| explanation_p10 | 1.0 | 1.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 2552 | 2552 |
| variants_accepted | 1060 | 1088 |
| variant_acceptance_rate | 0.415361 | 0.426332 |
| rejection_reasons | `{"foot_evidence_not_supported": 140, "low_posture_unexplained": 1464}` | `{"foot_evidence_not_supported": 4, "low_posture_unexplained": 1460}` |
| eligibility | `{"environment_explained": {"clips": 265, "seconds": 2388.53}, "geometry_only_low_posture_unexplained": {"clips": 338, "seconds": 1533.43}, "rejected": {"clips": 35, "seconds": 206.81}}` | `{"environment_explained": {"clips": 272, "seconds": 2460.92}, "geometry_only_low_posture_unexplained": {"clips": 365, "seconds": 1663.48}, "rejected": {"clips": 1, "seconds": 4.37}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 42, "p50": 2.3527, "p95": 7.0565, "total": 131.9}, "motion_converted": {"clips": 596, "p50": 4.6801, "p95": 11.0051, "total": 3320.0}, "scene": {"clips": 638, "p50": 2.8533, "p95": 9.2881, "total": 2526.7}, "screen": {"clips": 638, "p50": 0.0358, "p95": 0.1003, "total": 29.2}, "support_fit": {"clips": 638, "p50": 1.0449, "p95": 2.9206, "total": 821.4}}` |

## known_dimension_support

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 1760 | 1760 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0185}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0152}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0185}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0152}}` |
| ground_skin_depth_m | `{"p50": 0.0093, "p95": 0.015, "p99": 0.0198}` | `{"p50": 0.0093, "p95": 0.015, "p99": 0.0198}` |
| ground_source | `{"lowest_foot_level": 1758, "skin_lowest_contacts": 2}` | `{"lowest_foot_level": 1758, "skin_lowest_contacts": 2}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 78, "ground_from_skin": 2, "inverted_foot_level_dropped": 0, "support_rejected": 12}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 2, "ground_from_skin": 2, "inverted_foot_level_dropped": 0, "support_rejected": 11}` |
| patches_resting_on_body | 84 | 2 |
| raised_sole_patches | `{"checked": 3615, "dropped": 84, "dropped_by": {"body_rests_lower": 82, "body_under_sole": 3}, "kept": 3531}` | `{"checked": 3613, "dropped": 2, "dropped_by": {"body_under_sole": 2}, "kept": 3611}` |
| median_foot_drop_median_m | 0.005335 | 0.005335 |
| median_body_drop_median_m | -0.074219 | -0.076575 |
| F | 172 | 196 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 3 | 3 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 50 | 50 |
| wall_status | `{"fitted": 50, "no_contact_plane": 1694, "no_still_segment": 16}` | `{"fitted": 50, "no_contact_plane": 1694, "no_still_segment": 16}` |
| wall_kind | `{"brace": 50}` | `{"brace": 50}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 206, "feet_at_the_plane": 868, "no_trunk_or_bracing_hand_contact": 26887, "plane_must_move_to_clear_body": 175}` | `{"contact_slides_along_plane": 206, "feet_at_the_plane": 868, "no_trunk_or_bracing_hand_contact": 26887, "plane_must_move_to_clear_body": 175}` |
| ceiling_status | `{"fitted": 2, "no_sustained_low_posture": 1736, "unexplainable_in_place_transition": 12, "unexplainable_tight_above_bound": 10}` | `{"fitted": 2, "no_sustained_low_posture": 1736, "unexplainable_in_place_transition": 13, "unexplainable_tight_above_bound": 9}` |
| explanation_p50 | 0.5625 | 0.5625 |
| explanation_p10 | 0.2125 | 0.2125 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 7040 | 7040 |
| variants_accepted | 6780 | 6768 |
| variant_acceptance_rate | 0.963068 | 0.961364 |
| rejection_reasons | `{"foot_evidence_not_supported": 172, "low_posture_partly_explained": 4, "low_posture_unexplained": 88}` | `{"foot_evidence_not_supported": 196, "low_posture_partly_explained": 4, "low_posture_unexplained": 88}` |
| eligibility | `{"environment_explained": {"clips": 1695, "seconds": 6477.36}, "geometry_only_low_posture_unexplained": {"clips": 22, "seconds": 121.71}, "rejected": {"clips": 43, "seconds": 187.21}}` | `{"environment_explained": {"clips": 1692, "seconds": 6451.53}, "geometry_only_low_posture_unexplained": {"clips": 19, "seconds": 90.51}, "rejected": {"clips": 49, "seconds": 244.24}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 116, "p50": 1.2592, "p95": 4.5402, "total": 230.8}, "motion_converted": {"clips": 1644, "p50": 2.165, "p95": 4.3091, "total": 4082.1}, "scene": {"clips": 1760, "p50": 0.7948, "p95": 2.9452, "total": 1866.3}, "screen": {"clips": 1760, "p50": 0.0338, "p95": 0.0717, "total": 69.1}, "support_fit": {"clips": 1760, "p50": 0.6429, "p95": 2.1911, "total": 1496.2}}` |

## ladder

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 64 | 64 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.016}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.016}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.016}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0146}}` |
| ground_skin_depth_m | `{"p50": 0.01, "p95": 0.0418, "p99": 0.0988}` | `{"p50": 0.01, "p95": 0.0418, "p99": 0.0988}` |
| ground_source | `{"lowest_foot_level": 47, "skin_lowest_contacts": 17}` | `{"lowest_foot_level": 47, "skin_lowest_contacts": 17}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 11, "ground_from_skin": 17, "inverted_foot_level_dropped": 10, "support_rejected": 8}` | `{"floating_slab": 1, "foot_patches_rest_on_body": 0, "ground_from_skin": 17, "inverted_foot_level_dropped": 0, "support_rejected": 9}` |
| patches_resting_on_body | 12 | 0 |
| raised_sole_patches | `{"checked": 36, "dropped": 12, "dropped_by": {"body_rests_lower": 12}, "kept": 24}` | `{"checked": 36, "dropped": 0, "dropped_by": {}, "kept": 36}` |
| median_foot_drop_median_m | 0.005647 | 0.005647 |
| median_body_drop_median_m | -0.067975 | -0.076286 |
| F | 32 | 40 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 1 | 2 |
| slab_min_thickness_m | null | 0.072916 |
| slab_min_bottom_above_ground_m | null | 0.290916 |
| walls_fitted | 3 | 3 |
| wall_status | `{"fitted": 3, "no_contact_plane": 60, "no_still_segment": 1}` | `{"fitted": 3, "no_contact_plane": 60, "no_still_segment": 1}` |
| wall_kind | `{"lean": 3}` | `{"lean": 3}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 35, "feet_at_the_plane": 97, "no_trunk_or_bracing_hand_contact": 964, "plane_must_move_to_clear_body": 1}` | `{"contact_slides_along_plane": 35, "feet_at_the_plane": 97, "no_trunk_or_bracing_hand_contact": 964, "plane_must_move_to_clear_body": 1}` |
| ceiling_status | `{"fitted": 3, "no_sustained_low_posture": 50, "unexplainable_body_reaches_bound": 3, "unexplainable_in_place_transition": 7, "unexplainable_tight_above_bound": 1}` | `{"fitted": 3, "no_sustained_low_posture": 50, "unexplainable_body_reaches_bound": 3, "unexplainable_in_place_transition": 7, "unexplainable_tight_above_bound": 1}` |
| explanation_p50 | 0.6854 | 0.6854 |
| explanation_p10 | 0.1371 | 0.1371 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 256 | 256 |
| variants_accepted | 188 | 188 |
| variant_acceptance_rate | 0.734375 | 0.734375 |
| rejection_reasons | `{"foot_evidence_not_supported": 32, "low_posture_partly_explained": 8, "low_posture_unexplained": 44}` | `{"foot_evidence_not_supported": 40, "low_posture_partly_explained": 8, "low_posture_unexplained": 44}` |
| eligibility | `{"excluded_unmodelled_geometry": {"clips": 64, "seconds": 332.79}}` | `{"excluded_unmodelled_geometry": {"clips": 64, "seconds": 332.79}}` |
| excluded_unmodelled_geometry | `{"clips": 64, "seconds": 332.769}` | `{"clips": 64, "seconds": 332.769}` |
| eligibility_before_exclusion | `{"environment_explained": {"clips": 47, "seconds": 246.502}, "geometry_only_low_posture_unexplained": {"clips": 9, "seconds": 50.568}, "rejected": {"clips": 8, "seconds": 35.699}}` | `{"environment_explained": {"clips": 47, "seconds": 246.502}, "geometry_only_low_posture_unexplained": {"clips": 7, "seconds": 42.001}, "rejected": {"clips": 10, "seconds": 44.266}}` |
| stage_seconds | null | `{"motion_cached": {"clips": 41, "p50": 2.5163, "p95": 5.2473, "total": 117.2}, "motion_converted": {"clips": 23, "p50": 3.6112, "p95": 6.7761, "total": 94.8}, "scene": {"clips": 64, "p50": 1.935, "p95": 5.3492, "total": 148.9}, "screen": {"clips": 64, "p50": 0.0558, "p95": 0.1455, "total": 4.0}, "support_fit": {"clips": 64, "p50": 0.8133, "p95": 3.7007, "total": 97.4}}` |

## obstacle_clearance

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 86 | 86 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0113}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0092}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0142}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.014}}` |
| ground_skin_depth_m | `{"p50": 0.0162, "p95": 0.0235, "p99": 0.0261}` | `{"p50": 0.0162, "p95": 0.0235, "p99": 0.0261}` |
| ground_source | `{"lowest_foot_level": 86}` | `{"lowest_foot_level": 86}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 3, "ground_from_skin": 0, "inverted_foot_level_dropped": 3, "support_rejected": 8}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 0, "ground_from_skin": 0, "inverted_foot_level_dropped": 0, "support_rejected": 7}` |
| patches_resting_on_body | 3 | 0 |
| raised_sole_patches | `{"checked": 25, "dropped": 3, "dropped_by": {"body_rests_lower": 3}, "kept": 22}` | `{"checked": 25, "dropped": 0, "dropped_by": {}, "kept": 25}` |
| median_foot_drop_median_m | 0.006556 | 0.006556 |
| median_body_drop_median_m | -0.076554 | -0.077374 |
| F | 36 | 28 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 3 | 3 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 0 | 0 |
| wall_status | `{"no_contact_plane": 79, "no_still_segment": 7}` | `{"no_contact_plane": 79, "no_still_segment": 7}` |
| wall_kind | `{}` | `{}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 13, "no_trunk_or_bracing_hand_contact": 853}` | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 13, "no_trunk_or_bracing_hand_contact": 853}` |
| ceiling_status | `{"fitted": 1, "no_sustained_low_posture": 77, "unexplainable_in_place_transition": 8}` | `{"fitted": 1, "no_sustained_low_posture": 77, "unexplainable_in_place_transition": 8}` |
| explanation_p50 | 0.0 | 0.0 |
| explanation_p10 | 0.0 | 0.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 344 | 344 |
| variants_accepted | 292 | 292 |
| variant_acceptance_rate | 0.848837 | 0.848837 |
| rejection_reasons | `{"foot_evidence_not_supported": 36, "low_posture_partly_explained": 4, "low_posture_unexplained": 32}` | `{"foot_evidence_not_supported": 28, "low_posture_partly_explained": 4, "low_posture_unexplained": 32}` |
| eligibility | `{"excluded_unmodelled_geometry": {"clips": 86, "seconds": 457.88}}` | `{"excluded_unmodelled_geometry": {"clips": 86, "seconds": 457.88}}` |
| excluded_unmodelled_geometry | `{"clips": 86, "seconds": 457.838}` | `{"clips": 86, "seconds": 457.838}` |
| eligibility_before_exclusion | `{"environment_explained": {"clips": 73, "seconds": 353.037}, "geometry_only_low_posture_unexplained": {"clips": 4, "seconds": 26.634}, "rejected": {"clips": 9, "seconds": 78.167}}` | `{"environment_explained": {"clips": 73, "seconds": 353.037}, "geometry_only_low_posture_unexplained": {"clips": 6, "seconds": 49.001}, "rejected": {"clips": 7, "seconds": 55.8}}` |
| stage_seconds | null | `{"motion_cached": {"clips": 40, "p50": 2.2471, "p95": 3.7923, "total": 95.2}, "motion_converted": {"clips": 46, "p50": 3.9667, "p95": 6.3955, "total": 194.9}, "scene": {"clips": 86, "p50": 1.2803, "p95": 2.8866, "total": 128.6}, "screen": {"clips": 86, "p50": 0.044, "p95": 0.1102, "total": 4.5}, "support_fit": {"clips": 86, "p50": 0.6525, "p95": 3.6256, "total": 99.9}}` |

## prone

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 919 | 919 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0199}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0196}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0024, "p95": 0.0145, "p99": 0.029}` | `{"p50": 0.0024, "p95": 0.0145, "p99": 0.029}` |
| ground_source | `{"lowest_foot_level": 14, "skin_lowest_contacts": 905}` | `{"lowest_foot_level": 14, "skin_lowest_contacts": 905}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 796, "ground_from_skin": 905, "inverted_foot_level_dropped": 373, "support_rejected": 159}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 329, "ground_from_skin": 905, "inverted_foot_level_dropped": 62, "support_rejected": 21}` |
| patches_resting_on_body | 6754 | 1393 |
| raised_sole_patches | `{"checked": 7197, "dropped": 6754, "dropped_by": {"body_rests_lower": 6750, "body_under_sole": 39, "foot_lower_than_sole": 56}, "kept": 443}` | `{"checked": 1507, "dropped": 1393, "dropped_by": {"body_rests_lower": 1392, "body_under_sole": 21, "foot_lower_than_sole": 31}, "kept": 114}` |
| median_foot_drop_median_m | 0.011196 | 0.012967 |
| median_body_drop_median_m | 0.060856 | 0.081024 |
| F | 582 | 98 |
| clips_with_4plus_support_boxes | 150 | 13 |
| max_support_boxes | 16 | 9 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 0 | 0 |
| wall_status | `{"no_contact_plane": 918, "no_still_segment": 1}` | `{"no_contact_plane": 918, "no_still_segment": 1}` |
| wall_kind | `{}` | `{}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 3, "feet_at_the_plane": 28, "no_trunk_or_bracing_hand_contact": 14552, "plane_must_move_to_clear_body": 1}` | `{"contact_slides_along_plane": 3, "feet_at_the_plane": 28, "no_trunk_or_bracing_hand_contact": 14552, "plane_must_move_to_clear_body": 1}` |
| ceiling_status | `{"fitted": 651, "no_conflict_free_rectangle": 102, "unexplainable_in_place_transition": 135, "unexplainable_tight_above_bound": 31}` | `{"fitted": 652, "no_conflict_free_rectangle": 103, "unexplainable_in_place_transition": 154, "unexplainable_tight_above_bound": 10}` |
| explanation_p50 | 1.0 | 1.0 |
| explanation_p10 | 1.0 | 1.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 3676 | 3676 |
| variants_accepted | 2191 | 2545 |
| variant_acceptance_rate | 0.596028 | 0.692329 |
| rejection_reasons | `{"foot_evidence_not_supported": 582, "low_posture_partly_explained": 59, "low_posture_unexplained": 1072}` | `{"foot_evidence_not_supported": 98, "low_posture_partly_explained": 59, "low_posture_unexplained": 1068}` |
| eligibility | `{"environment_explained": {"clips": 549, "seconds": 4939.92}, "geometry_only_low_posture_unexplained": {"clips": 226, "seconds": 1979.25}, "rejected": {"clips": 144, "seconds": 1338.53}}` | `{"environment_explained": {"clips": 637, "seconds": 5755.33}, "geometry_only_low_posture_unexplained": {"clips": 258, "seconds": 2282.91}, "rejected": {"clips": 24, "seconds": 219.46}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 81, "p50": 3.1489, "p95": 5.0101, "total": 263.1}, "motion_converted": {"clips": 838, "p50": 5.4325, "p95": 8.6602, "total": 4732.6}, "scene": {"clips": 919, "p50": 2.4551, "p95": 6.8796, "total": 2744.4}, "screen": {"clips": 919, "p50": 0.0371, "p95": 0.0765, "total": 38.0}, "support_fit": {"clips": 919, "p50": 0.9872, "p95": 3.1791, "total": 1197.3}}` |

## quadruped

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 831 | 831 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0188}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0169}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0013, "p95": 0.0134, "p99": 0.0296}` | `{"p50": 0.0013, "p95": 0.0134, "p99": 0.0296}` |
| ground_source | `{"lowest_foot_level": 30, "skin_lowest_contacts": 801}` | `{"lowest_foot_level": 30, "skin_lowest_contacts": 801}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 665, "ground_from_skin": 801, "inverted_foot_level_dropped": 465, "support_rejected": 107}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 362, "ground_from_skin": 801, "inverted_foot_level_dropped": 242, "support_rejected": 0}` |
| patches_resting_on_body | 4084 | 1876 |
| raised_sole_patches | `{"checked": 4306, "dropped": 4084, "dropped_by": {"body_rests_lower": 4075, "body_under_sole": 755, "foot_lower_than_sole": 270}, "kept": 222}` | `{"checked": 1882, "dropped": 1876, "dropped_by": {"body_rests_lower": 1875, "body_under_sole": 305, "foot_lower_than_sole": 104}, "kept": 6}` |
| median_foot_drop_median_m | 0.022584 | 0.022438 |
| median_body_drop_median_m | 0.082279 | 0.101232 |
| F | 436 | 0 |
| clips_with_4plus_support_boxes | 21 | 0 |
| max_support_boxes | 21 | 0 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 33 | 33 |
| wall_status | `{"dropped_penetrates_skin": 37, "fitted": 33, "no_contact_plane": 751, "no_still_segment": 10}` | `{"dropped_penetrates_skin": 37, "fitted": 33, "no_contact_plane": 751, "no_still_segment": 10}` |
| wall_kind | `{"lean": 33}` | `{"lean": 33}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 21, "feet_at_the_plane": 56, "no_trunk_or_bracing_hand_contact": 14114, "plane_must_move_to_clear_body": 79}` | `{"contact_slides_along_plane": 21, "feet_at_the_plane": 56, "no_trunk_or_bracing_hand_contact": 14114, "plane_must_move_to_clear_body": 79}` |
| ceiling_status | `{"fitted": 692, "no_conflict_free_rectangle": 9, "unexplainable_in_place_transition": 121, "unexplainable_tight_above_bound": 9}` | `{"fitted": 693, "no_conflict_free_rectangle": 9, "unexplainable_in_place_transition": 129}` |
| explanation_p50 | 1.0 | 1.0 |
| explanation_p10 | 1.0 | 1.0 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 3324 | 3324 |
| variants_accepted | 2544 | 2736 |
| variant_acceptance_rate | 0.765343 | 0.823105 |
| rejection_reasons | `{"foot_evidence_not_supported": 436, "low_posture_partly_explained": 32, "low_posture_unexplained": 556}` | `{"low_posture_partly_explained": 36, "low_posture_unexplained": 552}` |
| eligibility | `{"environment_explained": {"clips": 636, "seconds": 5162.99}, "geometry_only_low_posture_unexplained": {"clips": 86, "seconds": 546.12}, "rejected": {"clips": 109, "seconds": 717.16}}` | `{"environment_explained": {"clips": 684, "seconds": 5512.67}, "geometry_only_low_posture_unexplained": {"clips": 147, "seconds": 913.6}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 61, "p50": 3.7219, "p95": 8.6222, "total": 278.5}, "motion_converted": {"clips": 770, "p50": 5.7322, "p95": 11.3091, "total": 4824.5}, "scene": {"clips": 831, "p50": 3.3103, "p95": 8.1721, "total": 3160.1}, "screen": {"clips": 831, "p50": 0.0381, "p95": 0.103, "total": 38.4}, "support_fit": {"clips": 831, "p50": 1.4012, "p95": 3.3175, "total": 1316.8}}` |

## stairs

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 49 | 49 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0198}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0197}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0198}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0197}}` |
| ground_skin_depth_m | `{"p50": 0.0086, "p95": 0.0303, "p99": 0.0671}` | `{"p50": 0.0086, "p95": 0.0303, "p99": 0.0671}` |
| ground_source | `{"lowest_foot_level": 47, "skin_lowest_contacts": 2}` | `{"lowest_foot_level": 47, "skin_lowest_contacts": 2}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 3, "ground_from_skin": 2, "inverted_foot_level_dropped": 3, "support_rejected": 14}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 0, "ground_from_skin": 2, "inverted_foot_level_dropped": 0, "support_rejected": 15}` |
| patches_resting_on_body | 3 | 0 |
| raised_sole_patches | `{"checked": 242, "dropped": 3, "dropped_by": {"body_rests_lower": 3}, "kept": 239}` | `{"checked": 242, "dropped": 0, "dropped_by": {}, "kept": 242}` |
| median_foot_drop_median_m | 0.004753 | 0.004753 |
| median_body_drop_median_m | -0.083768 | -0.086237 |
| F | 60 | 60 |
| clips_with_4plus_support_boxes | 25 | 25 |
| max_support_boxes | 10 | 10 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 0 | 0 |
| wall_status | `{"no_contact_plane": 32, "no_still_segment": 17}` | `{"no_contact_plane": 32, "no_still_segment": 17}` |
| wall_kind | `{}` | `{}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 11, "no_trunk_or_bracing_hand_contact": 419}` | `{"contact_slides_along_plane": 0, "feet_at_the_plane": 11, "no_trunk_or_bracing_hand_contact": 419}` |
| ceiling_status | `{"no_sustained_low_posture": 41, "unexplainable_in_place_transition": 7, "unexplainable_tight_above_bound": 1}` | `{"no_sustained_low_posture": 41, "unexplainable_in_place_transition": 7, "unexplainable_tight_above_bound": 1}` |
| explanation_p50 | null | null |
| explanation_p10 | null | null |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 196 | 196 |
| variants_accepted | 108 | 108 |
| variant_acceptance_rate | 0.55102 | 0.55102 |
| rejection_reasons | `{"foot_evidence_not_supported": 60, "low_posture_unexplained": 32}` | `{"foot_evidence_not_supported": 60, "low_posture_unexplained": 32}` |
| eligibility | `{"environment_explained": {"clips": 27, "seconds": 111.59}, "geometry_only_low_posture_unexplained": {"clips": 7, "seconds": 39.0}, "rejected": {"clips": 15, "seconds": 85.73}}` | `{"environment_explained": {"clips": 27, "seconds": 111.59}, "geometry_only_low_posture_unexplained": {"clips": 7, "seconds": 39.0}, "rejected": {"clips": 15, "seconds": 85.73}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 40, "p50": 2.504, "p95": 3.3517, "total": 95.2}, "motion_converted": {"clips": 9, "p50": 5.8252, "p95": 6.8442, "total": 46.4}, "scene": {"clips": 49, "p50": 1.5012, "p95": 3.2698, "total": 80.8}, "screen": {"clips": 49, "p50": 0.0567, "p95": 0.1086, "total": 3.1}, "support_fit": {"clips": 49, "p50": 3.5494, "p95": 10.3181, "total": 215.0}}` |

## wall_contact

| 字段 | v7 | v9 |
| --- | --- | --- |
| clips | 350 | 350 |
| skin_support | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0099}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0089}}` | `{"maximal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}, "minimal": {"clips_over_2cm": 0, "peak_p95_m": 0.0}}` |
| ground_skin_depth_m | `{"p50": 0.0095, "p95": 0.0169, "p99": 0.0218}` | `{"p50": 0.0095, "p95": 0.0169, "p99": 0.0218}` |
| ground_source | `{"lowest_foot_level": 313, "skin_lowest_contacts": 37}` | `{"lowest_foot_level": 313, "skin_lowest_contacts": 37}` |
| flags | `{"floating_slab": 0, "foot_patches_rest_on_body": 24, "ground_from_skin": 37, "inverted_foot_level_dropped": 0, "support_rejected": 5}` | `{"floating_slab": 0, "foot_patches_rest_on_body": 7, "ground_from_skin": 37, "inverted_foot_level_dropped": 0, "support_rejected": 1}` |
| patches_resting_on_body | 41 | 9 |
| raised_sole_patches | `{"checked": 95, "dropped": 41, "dropped_by": {"body_rests_lower": 41, "body_under_sole": 1}, "kept": 54}` | `{"checked": 21, "dropped": 9, "dropped_by": {"body_rests_lower": 9}, "kept": 12}` |
| median_foot_drop_median_m | 0.005461 | 0.005822 |
| median_body_drop_median_m | -0.040473 | -0.075662 |
| F | 8 | 0 |
| clips_with_4plus_support_boxes | 0 | 0 |
| max_support_boxes | 3 | 2 |
| slab_min_thickness_m | null | null |
| slab_min_bottom_above_ground_m | null | null |
| walls_fitted | 106 | 106 |
| wall_status | `{"fitted": 106, "no_contact_plane": 244}` | `{"fitted": 106, "no_contact_plane": 244}` |
| wall_kind | `{"brace": 8, "lean": 98}` | `{"brace": 8, "lean": 98}` |
| wall_rejection_reasons | `{"contact_slides_along_plane": 20, "feet_at_the_plane": 231, "no_trunk_or_bracing_hand_contact": 7607, "plane_must_move_to_clear_body": 26}` | `{"contact_slides_along_plane": 20, "feet_at_the_plane": 231, "no_trunk_or_bracing_hand_contact": 7607, "plane_must_move_to_clear_body": 26}` |
| ceiling_status | `{"fitted": 77, "no_conflict_free_rectangle": 7, "no_sustained_low_posture": 246, "unexplainable_in_place_transition": 11, "unexplainable_tight_above_bound": 9}` | `{"fitted": 79, "no_conflict_free_rectangle": 8, "no_sustained_low_posture": 246, "unexplainable_in_place_transition": 17}` |
| explanation_p50 | 1.0 | 1.0 |
| explanation_p10 | 0.935 | 0.9333 |
| fitted_ceiling_checks_over_2cm | 0 | 0 |
| variants | 1400 | 1400 |
| variants_accepted | 1264 | 1272 |
| variant_acceptance_rate | 0.902857 | 0.908571 |
| rejection_reasons | `{"foot_evidence_not_supported": 8, "low_posture_partly_explained": 28, "low_posture_unexplained": 108}` | `{"low_posture_partly_explained": 28, "low_posture_unexplained": 100}` |
| eligibility | `{"environment_explained": {"clips": 316, "seconds": 2445.48}, "geometry_only_low_posture_unexplained": {"clips": 32, "seconds": 122.58}, "rejected": {"clips": 2, "seconds": 6.4}}` | `{"environment_explained": {"clips": 318, "seconds": 2454.95}, "geometry_only_low_posture_unexplained": {"clips": 32, "seconds": 119.51}}` |
| excluded_unmodelled_geometry | `{"clips": 0, "seconds": 0.0}` | `{"clips": 0, "seconds": 0.0}` |
| eligibility_before_exclusion | `{}` | `{}` |
| stage_seconds | null | `{"motion_cached": {"clips": 40, "p50": 2.751, "p95": 5.6299, "total": 118.2}, "motion_converted": {"clips": 310, "p50": 4.3059, "p95": 8.2643, "total": 1490.8}, "scene": {"clips": 350, "p50": 2.7775, "p95": 6.4823, "total": 1099.0}, "screen": {"clips": 350, "p50": 0.0585, "p95": 0.1637, "total": 24.9}, "support_fit": {"clips": 350, "p50": 0.8386, "p95": 2.2312, "total": 356.6}}` |
