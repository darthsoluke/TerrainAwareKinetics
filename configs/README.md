# 实验配置

配置记录模型结构、输入格式、训练预算和相应资源身份。请结合 [实验账本](../docs/history/README.md) 与详细执行记录阅读。

| 文件 | 用途 |
| --- | --- |
| [pfnn_frame_actor_pro6000_gpu3_v1.json](pfnn_frame_actor_pro6000_gpu3_v1.json) | 已完成P1的单卡配置 |
| [pfnn_frame_actor_p2_pro6000_gpu3_draft.json](pfnn_frame_actor_p2_pro6000_gpu3_draft.json) | P2课程与预算草案，尚未代表执行 |
| [pfnn_frame_actor_v1.json](pfnn_frame_actor_v1.json) | P1迁移前的双卡配置历史 |
| [pfnn_codec_repair_pilot_v1.json](pfnn_codec_repair_pilot_v1.json) | 历史Codec修复方案 |
| [bones_terrain_screen_v1.json](bones_terrain_screen_v1.json) | BONES地形几何筛查的分组与每组词干；只做选择，不是拟合结果 |
| [bones_terrain_screen_edge_v1.json](bones_terrain_screen_edge_v1.json) | 单独筛查 `on_the_edge`，确认是否为窄支撑数据 |

`draft`、`preflight`、`forward`与正式训练的含义不同。配置存在不代表模型已训练，正式结果应有实际checkpoint、更新数与评估记录。

公开副本使用模板机器路径；源数据/权重身份和运行资源按私有制品配置绑定。
