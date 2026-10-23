# 全部原始 BVH 迁移完成

2026-10-23，PRO6000。范围A：全部原始动作，保留124个日期子目录，跳过镜像。

| 项目 | 结果 |
| --- | ---: |
| 预期文件 |71,132 |
| 最终逐文件SHA256通过 |71,132 |
| 哈希差异 |0 |
| 额外BVH |0 |
| 原始文件总字节 |148,382,694,286 |
| 压缩传输总字节 |27,821,576,214 |
| 接收准备时可用空间 |553.461 GiB |
| 完成时可用空间 |414.888 GiB |
| 最低保留空间 |200 GiB |

源清单SHA256：`aae7f5c9fe585c264867488488315857020ebb11dd58c6981856b0315ae9dfc1`。

| 文件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 范围、总量、完成时间与空间测量 |
| [verification.json](verification.json) | 最终独立全量文件哈希验证 |
| [annotation_coverage.json](annotation_coverage.json) | 71,132个源文件名与原始timeline一一对应 |
| [archive_format_check.json](archive_format_check.json) | 首个日期包的实际tar结构检查 |
| [cache_resume_check.json](cache_resume_check.json) | 中断后剩余61个缓存包全部通过SHA256 |
| [resume_execution.json](resume_execution.json) | 最终接收进程的命令与实际工具SHA256 |
| [receive_interruption.txt](receive_interruption.txt) | 旧接收器重复查询Drive时的完整错误 |
| [space_before.txt](space_before.txt) | 接收准备时的df原始输出 |
| [space_after.txt](space_after.txt) | 完成后的df原始输出 |
| [SHA256SUMS](SHA256SUMS) | 本目录九个结果文件的公开SHA256 |
| [publication_manifest.json](publication_manifest.json) | 原始文件与公开副本的哈希对应 |

`summary.json`中的预检空间为11:07的553.471GiB，`space_before.txt`是在接收准备时再次测得的553.461GiB。`verification.json.free_bytes_before`记录缓存恢复时的459.478GiB；最终可用量为414.888GiB。

公开副本替换了私人机器路径，测量结果保持原样。目标目录以`${BONES_BVH_ROOT}`表示，运行时实际路径保存在私有交付记录。完整BVH、逐文件源清单和`candidates.jsonl`存放在私有数据环境。

调查与执行背景见 [完整v2记录](../../../../docs/records/bones_terrain_survey_v2_pro6000_20261023.md)。
