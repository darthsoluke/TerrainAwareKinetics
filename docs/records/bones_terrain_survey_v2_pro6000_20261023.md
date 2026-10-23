# BONES-SEED survey v2：词干清单与原始 BVH 迁移

2026-10-23，PRO6000，调查源码 `d313b65c4f42a939b9d354ce7e9a2ac93780babb`。

**已完成：survey v2与范围A的全部原始BVH迁移。** 71,132个目标文件均通过独立SHA256验证，差异0、额外BVH 0；完成后根分区可用414.888GiB。完整验证产物见 [迁移结果](../../examples/results/bones_terrain_survey_v2_20261023/transfer/README.md)。

## Survey v2 已完成

测试命令与实际最后一行：

```text
python -m pytest tests/test_bones_terrain_survey.py -q
9 passed, 1 skipped in 2.04s
```

调查使用与 v1 相同的 timeline，SHA256 为 `379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8`。没有使用 metadata 或 BVH 观测。

```bash
python scripts/survey_bones_terrain_candidates.py \
  --timelines "${BONES_INPUTS}/timelines.jsonl" \
  --output runs/bones_terrain_survey_v2_timeline
```

原始输出：

```text
stems: {'unique': 2863, 'with_any_hit': 812, 'with_dimension_tokens': 17}
bones-terrain-survey-v2: wrote runs/bones_terrain_survey_v2_timeline
clips: {'total': 142220, 'timeline_only': 142220, 'metadata_only': 0, 'both': 0}
bucket                         clips    orig  mirror      orig_s   event_s  families  actors
support_height_change           5430    2715    2715     20664.8   14237.1       310     346
hand_or_obstacle_support        2730    1365    1365     10561.8    4600.8       135     348
low_posture                    12509    6256    6253     50095.7   37445.1       586     380
any_candidate                  19935    9969    9966     77145.0         -       927     496
flat_remainder                122285   61163   61122    435575.1         -      3411     518
```

完整文件位于 [v2结果目录](../../examples/results/bones_terrain_survey_v2_20261023/README.md)。其中 [stems.tsv](../../examples/results/bones_terrain_survey_v2_20261023/stems.tsv) 包含全部2,863个词干，2,051个没有关键词命中。词干表原始/镜像条数求和分别为71,132/71,088，覆盖检查通过。

17个词干包含尺寸词，包括 `come_up_50cm_box`、`jump_on_50cm`、`jump_over_obstacle_0.75m` 等，可供后续几何核查。关键词规则沿用v1，进入拟合的依据仍是下一阶段的几何证据。完整 `candidates.jsonl` 保留在PRO6000。

## BVH 迁移：A，全部原始动作

用户已选择迁移全部71,132条原始BVH，跳过stem以`_M`结尾的镜像，保留日期子目录。

11:07（Asia/Singapore）实测空间：

| 存储位置 | 可用 bytes | 可用 GiB |
| --- | ---: | ---: |
| PRO6000根分区 |594,284,519,424 |553.471 |
| PRO6000数据分区 |239,587,082,240 |223.133 |

目标选择根分区下的用户存储目录。复制前与解包时均检查剩余原始文件体积和压缩包暂存空间，完成后至少保留200GiB。

源文件SHA256清单已完整生成：71,132个文件、148,382,694,286 bytes、124个日期目录。清单SHA256为 `aae7f5c9fe585c264867488488315857020ebb11dd58c6981856b0315ae9dfc1`，摘要见 [transfer_source_summary.json](../../examples/results/bones_terrain_survey_v2_20261023/transfer_source_summary.json)。

源端全部124个日期包已上传到私有Drive，压缩后合计27,821,576,214 bytes，约25.91GiB。[完整上传收据](../../examples/results/bones_terrain_survey_v2_20261023/transfer_staging_complete.json)包含各包SHA256，文件数、原始字节数、压缩字节数和日期数汇总均已核对。

全部日期包已在PRO6000下载、解包并核验，接收程序退出码为0。上传器记录每个成功日期，恢复时复用完成记录；大文件分块设为32MiB。g3直连Drive实测约26～29MiB/s，接收端使用M78。

源清单与timeline原始名称已完成交叉核对：71,132个源文件名与71,132条原始标注全部匹配，没有单边缺失。最终BVH验证与空间测量如下：

| 项目 | 实测结果 |
| --- | ---: |
| 预期文件 |71,132 |
| 独立全量SHA256通过 |71,132 |
| 哈希差异 / 额外BVH |0 / 0 |
| 总字节 |148,382,694,286 |
| 接收准备时根分区可用量 |594,274,402,304 bytes / 553.461 GiB |
| 最终根分区可用量 |445,482,201,088 bytes / 414.888 GiB |
| 保留空间门槛 |214,748,364,800 bytes / 200 GiB |

测量原件：[最终验证](../../examples/results/bones_terrain_survey_v2_20261023/transfer/verification.json)、[接收准备时df](../../examples/results/bones_terrain_survey_v2_20261023/transfer/space_before.txt)、[完成后df](../../examples/results/bones_terrain_survey_v2_20261023/transfer/space_after.txt)。`verification.json.free_bytes_before`是缓存恢复时的空间快照，迁移前空间采用单独保存的df输出。

迁移工具：[transfer_bones_originals.py](../../scripts/transfer_bones_originals.py)，最新[测试](../../tests/test_bones_transfer.py)有13项通过，覆盖原始/镜像筛选、日期目录、错误哈希、已有文件保护、空间门槛、完成日期复用、预下载完成条件，以及缓存路径的范围/数量和哈希验证。工具分为：

1. `index`：只读g3源BVH，写出每个相对路径、大小与SHA256；清单完整后才允许打包。
2. `upload`：每个日期一个tar.gz，经私有Drive上传；成功后清理本任务自己的临时压缩包。
3. `receive --prefetch`：预下载时预留全部原始BVH、最坏情况压缩包与额外10GiB空间。源端完成标志到达后，PRO6000验证压缩包、保留日期目录解包、核对每个BVH，最后再独立读取所有目标文件进行全量哈希验证。

源文件、NAS与其他用户目录保持不变；全过程CPU，无训练。机器路径和传输凭证采用任务私有配置。

## 接收阶段修复

处理完63个日期目录后，旧接收器对本地已存在的`230320.tar.gz`仍执行Drive查询，返回 `Source doesn't exist or is a directory and destination is a file`，接收进程退出。

现场核对剩余61个本地压缩包，大小和SHA256全部匹配源端完成记录，源清单SHA256也匹配。接收器改为复用完整本地缓存，保留包哈希、逐文件哈希和最后独立全量核对；已有目标文件先核验，再继续处理剩余目录。恢复后完成全部71,132个文件，差异为0。新增测试同时覆盖无需网络的缓存接收、损坏缓存必须报错，以及缓存数量必须匹配请求。

原始错误、缓存核验及实际运行命令分别见 [中断日志](../../examples/results/bones_terrain_survey_v2_20261023/transfer/receive_interruption.txt)、[缓存检查](../../examples/results/bones_terrain_survey_v2_20261023/transfer/cache_resume_check.json)、[恢复执行记录](../../examples/results/bones_terrain_survey_v2_20261023/transfer/resume_execution.json)。实际完成接收的工具SHA256保留在执行记录中，后续发布的数量检查不会改写此次运行身份。
