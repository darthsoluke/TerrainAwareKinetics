# BONES-SEED 几何筛查：PRO6000 执行记录

2026-10-23，源码版本 `0f97409173586919f35afa8c74a9d601d5788ac2`。

**试运行与全量筛查均已完成。** 试运行 196 条全部成功，耗时 20.4 秒；全量处理配置选中的 6,795 条，成功 6,788 条，7 条因骨架几何偏差被拒绝，耗时 422.8 秒（约 7 分 3 秒）。

完整结果见 [结果目录](../../examples/results/bones_terrain_screen_v1_20261023/README.md)。本次运行只使用 PRO6000 的 CPU，16 个 worker，每个 worker 的计算线程数设为 1。没有训练，也没有改动源 BVH。

## 1. 环境修复与测试

独立项目环境中已有 Kimodo 源码和人体资源，但 Kimodo 目录没有加入 Python 的导入路径。首次导入错误完整保存在 [imports.txt](../../examples/results/bones_terrain_screen_v1_20261023/execution/imports.txt)：

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'kimodo'
```

复用项目已有 Kimodo 快照，通过 `PYTHONPATH` 加入导入路径。本次任务的环境设置如下；`${KIMODO_ROOT}` 指向已有 Kimodo 包目录的上一级：

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

未安装或升级包，未修改筛查源码、Kimodo 快照或 cloth 环境。后续复现需保留上述导入路径设置。

修复后执行：

```bash
$PY -m pytest tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py -q
$PY -c "import kimodo, scipy; print(kimodo.__file__, scipy.__version__)"
```

输出：

```text
...................                                                      [100%]
19 passed in 4.46s

${KIMODO_ROOT}/kimodo/__init__.py 1.17.1
```

原始日志：[测试](../../examples/results/bones_terrain_screen_v1_20261023/execution/tests_fixed.txt)、[导入](../../examples/results/bones_terrain_screen_v1_20261023/execution/imports_fixed.txt)。环境修复前的测试为 `18 passed, 1 skipped in 2.62s`，也保留在结果目录中。

Python 3.11.16，NumPy 1.26.4，SciPy 1.17.1，PyTorch 2.7.1+cu128，pandas 2.3.3，pytest 9.1.1。worker 初始化记录确认骨架位于 `cpu`，站立髋高参考为 `1.0092086791992188 m`。包版本、Kimodo 源文件集合哈希和官方蒙皮资源哈希见 [environment.json](../../examples/results/bones_terrain_screen_v1_20261023/execution/environment.json)。

## 2. 选样和资源

BVH 根目录包含迁移完成的 71,132 条原始动作。本次按 [v1 配置](../../configs/bones_terrain_screen_v1.json) 选择 98 个词干、11 个组；平地对照按配置每词干最多取 60 条。

```bash
$PY scripts/screen_bones_terrain_geometry.py \
  --bvh-root "${BONES_BVH_ROOT}" \
  --output runs/bones_screen_v1_dry --dry-run
```

完整打印行：

```json
{"selected": 6795, "per_group": {"box_support": 48, "crouch": 1145, "flat_control": 900, "kneel": 638, "known_dimension_support": 1760, "ladder": 64, "obstacle_clearance": 86, "prone": 919, "quadruped": 833, "stairs": 49, "wall_contact": 353}, "configured_stems_without_files": []}
```

硬件查询的输出如下：

```text
72
               total        used        free      shared  buff/cache   available
Mem:             498         110          11           3         384         388
Swap:              7           0           7
affinity_cpus 72
```

`nproc` 第一次继承了 `OMP_NUM_THREADS=1`，返回 1。随后仅在硬件查询时清除 `OMP_NUM_THREADS` 和 `OMP_THREAD_LIMIT`，得到 72；`os.sched_getaffinity(0)` 同样确认 72。运行时仍保留每个 worker 单线程设置。worker 数按 `min(16, 72/2)` 取 16。两次资源查询都保存在 [execution/](../../examples/results/bones_terrain_screen_v1_20261023/execution/resources_host.txt)。

## 3. 试运行和全量执行

```bash
$PY -u scripts/screen_bones_terrain_geometry.py \
  --bvh-root "${BONES_BVH_ROOT}" \
  --output runs/bones_screen_v1_pilot --limit-per-stem 2 --workers 16

nohup "$PY" -u scripts/screen_bones_terrain_geometry.py \
  --bvh-root "${BONES_BVH_ROOT}" \
  --output runs/bones_screen_v1_full --workers 16 \
  > runs/bones_screen_v1_execution/full.stdout.txt 2>&1 < /dev/null &
```

实际全量命令外层使用 `bash -c` 记录退出状态，日志末行为 `SCREEN_EXIT=0`。后台任务已结束。

| 运行 | 处理条数 | 成功 | 拒绝 | 失败比例 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每词干 2 条试运行 | 196 | 196 | 0 | 0% | 20.4 秒 |
| 配置范围内全量 | 6,795 | 6,788 | 7 | 0.1030% | 422.8 秒 |

试运行的 `failure_reasons` 为 `{}`，低于 10% 停止线，随后才启动全量任务。`report.json` 的 `screened` 字段包含处理过的失败项；`by_group.json` 的 `screened` 则是该组成功条数，表中分别列出以便阅读。

| 组 | 试运行条数 / 失败 | 全量条数 | 全量成功 | 全量失败 |
| --- | ---: | ---: | ---: | ---: |
| 已知尺寸支撑 `known_dimension_support` | 20 / 0 | 1,760 | 1,760 | 0 |
| 箱体支撑 `box_support` | 8 / 0 | 48 | 48 | 0 |
| 楼梯 `stairs` | 12 / 0 | 49 | 49 | 0 |
| 越障 `obstacle_clearance` | 18 / 0 | 86 | 86 | 0 |
| 梯子 `ladder` | 20 / 0 | 64 | 64 | 0 |
| 蹲姿 `crouch` | 32 / 0 | 1,145 | 1,143 | 2 |
| 手膝着地 `quadruped` | 24 / 0 | 833 | 831 | 2 |
| 匍匐 `prone` | 14 / 0 | 919 | 919 | 0 |
| 跪姿 `kneel` | 2 / 0 | 638 | 638 | 0 |
| 靠墙 `wall_contact` | 16 / 0 | 353 | 350 | 3 |
| 平地对照 `flat_control` | 30 / 0 | 900 | 900 | 0 |

完整打印的分组 JSON：[试运行](../../examples/results/bones_terrain_screen_v1_20261023/pilot/by_group.json)、[全量](../../examples/results/bones_terrain_screen_v1_20261023/full/by_group.json)。完整逐片段表：[196 行试运行 TSV](../../examples/results/bones_terrain_screen_v1_20261023/pilot/clips_compact.tsv)、[6,795 行全量 TSV](../../examples/results/bones_terrain_screen_v1_20261023/full/clips_compact.tsv)，均另含一行表头。

## 4. 七条拒绝记录

全部触发同一项检查：BVH 中立骨架与官方 SOMA77 的最大关节位置偏差超过 `0.002 m`。没有放宽阈值或改写动作。

| 文件名（省略 `.bvh`） | 报错中的偏差 |
| --- | ---: |
| `crouch_ff_loop_315_004__A145` | 0.0038 m |
| `crouch_ff_start_225_103__A128` | 0.0027 m |
| `h_b_w_stand_to_crouch_270_stop_001__A342` | 0.0030 m |
| `h_b_w_stand_to_crouch_270_stop_002__A342` | 0.0035 m |
| `h_b_w_stand_to_crouch_270_stop_003__A342` | 0.0035 m |
| `mohak_forward_loop_001__A035` | 0.0021 m |
| `mohak_forward_loop_002__A035` | 0.0021 m |

全量 `failure_reasons`：

```json
{
  "ValueError: BVH reference geometry differs from official SOMA77 (0.0038 m)": 1,
  "ValueError: BVH reference geometry differs from official SOMA77 (0.0027 m)": 1,
  "ValueError: BVH reference geometry differs from official SOMA77 (0.0030 m)": 1,
  "ValueError: BVH reference geometry differs from official SOMA77 (0.0035 m)": 2,
  "ValueError: BVH reference geometry differs from official SOMA77 (0.0021 m)": 2
}
```

## 5. 几何结果的重点

下列比例的分母均为各组成功转换的片段数。`multi_level` 要求支撑高度跨度至少 10 cm；`sustained_*` 要求某种姿态连续至少 0.5 秒。

- 平地对照 900 条，`multi_level` 命中 0 条，站立帧比例的逐片段均值为 100%。
- 已知尺寸支撑组的多级支撑比例为 99.38%；箱体支撑组为 97.92%；楼梯组为 83.67%。
- 蹲姿组持续蹲姿比例为 78.39%；手膝组持续手膝姿态比例为 99.88%；匍匐组持续匍匐比例为 99.56%。
- 已知尺寸组的“支撑层跨度减文件名尺寸”：P5 为 +2.09 cm，中位数 +7.57 cm，P95 为 +11.81 cm。试运行中位数为 +9.72 cm。这个尺寸偏差需要在拟合前复核。
- 越障组的“最低身体最大净空减文件名尺寸”：P5 为 -5.25 cm，中位数 +3.24 cm，P95 为 +27.58 cm。越障组主要查看净空指标；起落点在同一平面时，支撑高度跨度可以为零。

当前输出提供支撑层、肢体静止事件、姿态和净空证据。下一步按这些证据选择拟合输入，并检查尺寸偏差；本轮没有生成地形资产。

## 6. 输入核验与发布

筛查读取每条源 BVH 时记录了 SHA256。将这些哈希与迁移时的逐文件清单比较，试运行 196 条、全量 6,795 条全部匹配，差异为 0，包含被拒绝的七条。

源迁移清单 SHA256：`aae7f5c9fe585c264867488488315857020ebb11dd58c6981856b0315ae9dfc1`。核验摘要见 [input_lineage.json](../../examples/results/bones_terrain_screen_v1_20261023/execution/input_lineage.json)。

发布目录：`examples/results/bones_terrain_screen_v1_20261023/`。包含两组完整报告、分组与词干统计、完整紧凑 TSV、测试和导入输出、stdout 及末尾日志、环境指纹和文件哈希。`selection.jsonl`、`clips.jsonl` 保留在 PRO6000 的各运行目录。

发布文件经私有 Drive 取回并核验 SHA256；公开副本只把私人路径替换为角色占位符，数值和 TSV 行保持原样。原始与公开文件的哈希对应见 [publication_manifest.json](../../examples/results/bones_terrain_screen_v1_20261023/publication_manifest.json)。
