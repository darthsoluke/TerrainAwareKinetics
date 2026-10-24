# 脚本：按任务选择入口

这里保留实验脚本的既有名称，使Python模块导入保持一致。下表是阅读/运行的主要入口；历史脚本应结合对应记录使用。

| 任务 | 入口 | 状态 |
| --- | --- | --- |
| 查看三臂结果表 | [compare_arms.py](compare_arms.py) | 可使用仓库内精简结果直接运行 |
| 最新统一源支撑评估 | [review_pfnn_ablation_fixed_masks.py](review_pfnn_ablation_fixed_masks.py) | 需要相应checkpoint与数据 |
| PFNN全量索引审计 | [audit_pfnn_full_coverage.py](audit_pfnn_full_coverage.py) | 需要既有数据/cache |
| BONES地形拟合候选摸底 | [survey_bones_terrain_candidates.py](survey_bones_terrain_candidates.py) | 只读标注；[PRO6000真实数据结果](../docs/records/bones_terrain_survey_pro6000_20261023.md)已完成 |
| BONES地形几何筛查 | [screen_bones_terrain_geometry.py](screen_bones_terrain_geometry.py) | 读BVH、不改动作；组定义见[配置](../configs/bones_terrain_screen_v1.json) |
| BONES支撑、顶棚、侧墙与场景变体 | [fit_bones_terrain_support.py](fit_bones_terrain_support.py) | 需要筛查结果目录；方法见[方法v2](../docs/records/bones_terrain_method_v2_20261023.md)；输出拟合JSON、图片和[浏览器查看页](../web/terrain_fit_review/index.html)数据；`--motion-cache` 复用转换结果（精确），`--clips` 只跑开发集，`--sample-per-group`/`--render-list` 指定审查图，`report.json` 的 `stage_seconds` 给出各阶段耗时 |
| BONES动作—场景训练集 | [build_bones_env_training_set.py](build_bones_env_training_set.py)、[audit_bones_env_training_set.py](audit_bones_env_training_set.py) | 从v9拟合和动作缓存构建分片；审计在线观测并试跑未训练模型；[说明](../docs/records/bones_env_training_set_20261024.md) |
| 地形感知动作Transformer训练 | [train_terrain_controller.py](train_terrain_controller.py) | 4卡torchrun，分阶段（一步监督→可微闭环），时间上限、原子保存、曝光账本；需按[计划](../docs/records/terrain_controller_pilot_plan_20261024.md)获批后运行 |
| 场景观测计时 | [benchmark_scene_observation.py](benchmark_scene_observation.py) | 部署显卡上测骨骼射线与体积观测的单帧延迟；[性能约定](../docs/records/skeleton_ray_observation_20261023.md) |
| BONES原始BVH迁移 | [transfer_bones_originals.py](transfer_bones_originals.py) | 逐文件清单、按日期打包、私有Drive与目标全量哈希校验；[执行记录](../docs/records/bones_terrain_survey_v2_pro6000_20261023.md) |
| P1训练 | [train_pfnn_frame_actor.py](train_pfnn_frame_actor.py) | 旧P1已结束，不应重复执行 |
| A/B/C小试 | [train_pfnn_frame_ablation.py](train_pfnn_frame_ablation.py) | v3三臂已完成 |
| 逐帧模型评估 | [evaluate_pfnn_frame_actor.py](evaluate_pfnn_frame_actor.py) | 已用于P1零训练对照 |
| 保存轨迹的足部审阅 | [review_pfnn_frame_gait.py](review_pfnn_frame_gait.py) | 复用轨迹，不进行模型推理 |
| 单帧完整计算链性能 | [benchmark_pfnn_frame_actor.py](benchmark_pfnn_frame_actor.py) | 测计算性能，结果与质量分别评估 |
| 旧MB混合训练 | [train_unified_full_corpus.py](train_unified_full_corpus.py) | 正式38.4M运行已完成 |
| PFNN地图界面 | [soma_pfnn_maps_server.py](soma_pfnn_maps_server.py) | 依赖本机模型/场景制品 |
| 人体运动学实时界面 | [soma_reactive_server.py](soma_reactive_server.py) | 历史可操控入口，质量见记录 |
| G1实验界面 | [mb_sonic_lab_server.py](mb_sonic_lab_server.py) | 依赖独立机器人资源 |

数据准备、probe、review、export等历史脚本按前缀可检索。其输出属于本地研究制品，不应提交动捕或模型到Git。
