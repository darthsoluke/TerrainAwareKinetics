# 第7步A100 pilot：迁移、单卡冒烟与resume检查

执行日期：**2026-10-25（Asia/Singapore）**，文件名日期沿用任务约定。用户已批准[训练计划](terrain_controller_pilot_plan_20261024.md)，执行源码固定为`9785c012cfd51b113fd8dc245034a6e46f98d36f`。

**依赖已修复，82项测试、全量迁移校验和单卡冒烟均通过；resume未通过，正式pilot尚未启动。** resume应从10次更新继续到12次，实际仍为10次，脚本累计exposures也保持400而非528。恢复前后的模型与critic权重逐tensor一致，测量器记录新增更新为0。

按预检失败即停止的规则，双卡冒烟、H=16显存预检和正式pilot均未启动。没有修改训练实现、调整超参数或追加预算。

## 1. 环境修复与测试

最初补装pandas使用`--no-deps`，遗漏pytz，造成81通过、1失败；这份[初次依赖诊断](../../examples/results/terrain_controller_pilot_a100_20261024/preflight_report.json)保留原始traceback。用户确认依赖小问题自行修复后，在同一独立依赖目录补齐pytz和tzdata：

```bash
$PY -m pip install --target "$STEP7_ROOT/runtime_extra" pytz==2025.2 tzdata==2025.2
export PYTHONPATH="$STEP7_ROOT/runtime_extra:$KIMODO_ROOT"
```

实际环境：Python3.11.17、Torch2.5.1+cu124、NumPy1.26.4、pandas2.3.3、pytz2025.2、tzdata2025.2、pytest9.1.1。基础环境没有修改，Kimodo沿用项目已有资源。CPU准备工作使用`CUDA_VISIBLE_DEVICES=''`，OMP/OpenBLAS/MKL为单线程，`TMPDIR`使用项目临时目录。

```bash
$PY -m pytest tests/test_bones_env_dataset.py tests/test_terrain_controller.py \
  tests/test_skeleton_ray_sampler.py tests/test_bones_terrain_fit.py \
  tests/test_bones_terrain_scene.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py -q --full-trace --color=no \
  --junitxml=runs/tmt_pilot_coord/tests_ready.xml
```

结果：**`82 passed, 11 warnings in 7.96s`，0失败、0跳过**。原始日志和JUnit留在A100的`runs/tmt_pilot_coord/`。

## 2. 从PRO6000迁移默认E集

源为PRO6000的`runs/bones_env_v1`，包含5,650个shard、`manifest.jsonl`和`report.json`。源端逐文件计算SHA-256，在新传输目录打包，经既有M78链路上传私有Google Drive；A100从Drive下载并逐文件复核。A100 SSH仅用于控制和状态，没有承载文件payload。

| 项目 | 结果 |
| --- | ---: |
| 源文件数 / 已核对文件数 | 5,652 / 5,652 |
| 源字节 / 目的字节 | 1,711,809,448 / 1,711,809,448 |
| 不一致文件 | 0 |
| 归档字节 | 1,399,488,336 |
| 源端打包秒数 | 45.502 |
| 源端打包加上传秒数 | 297.914 |
| A100下载、解包及复核秒数 | 50.221 |

| 对象 | SHA-256 |
| --- | --- |
| 归档 | `2998ba8131aba43976083cb02c6d01a2186e9d26e955f57abefa7b8923c4872c` |
| 逐文件SHA清单 | `5aaa9e1ada746ad02d61a5c499ba896213f52fba0d75200e7104ec90d4497e5a` |
| 源与目的dataset manifest | `67808b6813a42b676b58607b47bedc0bda9566819bee258f6ac4416c5a33c74d` |

数据位于A100任务仓库的`runs/bones_env_v1/`；逐文件清单和校验回执为`runs/tmt_pilot_coord/transfer/files.json`、`transfer_verified.json`。后续接手可复用这份已核验的数据。

## 3. 单卡冒烟

每次GPU启动前均扫描所有GPU的index、UUID、显存、利用率和计算PID。最初候选为GPU1/3；迁移期间空闲状态改变，启动时重新选择并固定**物理GPU0、GPU1**。单卡阶段只使用该对卡中的GPU0，通过UUID设置`CUDA_VISIBLE_DEVICES`。

```bash
$PY scripts/train_terrain_controller.py --dataset runs/bones_env_v1 \
  --output runs/tmt_smoke_1gpu --phases teacher:5 loop:4:5 \
  --batch 16 --max-hours 0.2 --eval-every 5 --log-every 1 --eval-batch 32
```

实际由协调目录的测量包装器执行上述原始脚本和参数，外层`timeout`为750秒，记录全部命令、退出码和每次更新的耗时。训练函数和AdamW仍调用原实现；计时从前向/损失开始到模型optimizer.step结束，两端同步CUDA，不含数据加载和评估。

**单卡冒烟通过，退出0，完成10次更新，损失与评估有限。** 外层壁钟13.109秒，原脚本最终elapsed为7.3秒。写出`phase0_teacher.pt`、`phase1_loop4.pt`及`latest.pt`，均留在A100。

| 阶段 | 更新 | 平均秒/更新 | 中位秒/更新 | 最大秒/更新 |
| --- | --- | --- | --- | --- |
| teacher:0 | 5 | 0.112330 | 0.072061 | 0.279179 |
| loop:4 | 5 | 0.320269 | 0.317822 | 0.339143 |

`torch.cuda.max_memory_allocated`为**136,466,432 bytes，约130.14MiB**；`torch.cuda.get_device_properties().total_memory`为**85,100,068,864 bytes**。这是batch16、H=4冒烟的峰值；计划中的batch512、H=16显存预检未执行，因此尚未决定pilot批量。

## 4. Resume失败证据

```bash
$PY scripts/train_terrain_controller.py --dataset runs/bones_env_v1 \
  --output runs/tmt_smoke_resume --phases teacher:5 loop:4:7 \
  --resume runs/tmt_smoke_1gpu/latest.pt \
  --batch 16 --max-hours 0.2 --eval-every 5 --log-every 1 --eval-batch 32
```

进程退出0、输出`stopped=complete`，外层壁钟8.262秒，但计数检查失败：

| 字段 | 恢复起点 | 预期结束 | 实际结束 |
| --- | ---: | ---: | ---: |
| update | 10 | 12 | 10 |
| 脚本exposures | 400 | 528 | 400 |
| 本次新增更新 | 0 | 2 | 0 |
| 本次新增exposures | 0 | 128 | 0 |

checkpoint的恢复状态为`phase_index=2, phase_update=0`。训练入口[第128行](../../scripts/train_terrain_controller.py#L128)直接恢复这份state，[第171行](../../scripts/train_terrain_controller.py#L171)从该phase_index迭代；新命令仍只有两个阶段，因此`range(2, 2)`为空。原运行在[第223行](../../scripts/train_terrain_controller.py#L223)将完成游标推进到2，最终`latest.pt`保存了这个游标。把第二阶段配额从5改为7，没有重新定位到第二阶段已完成的第5次更新。

测量器记录resume的optimizer调用次数为0；恢复前后model、critic各自所有state tensor逐一相等。这里没有异常traceback，失败证据是计数和权重不变。结构化证据在[report.json](../../examples/results/terrain_controller_pilot_a100_20261024/report.json)。训练源码保持批准版本，本次未修补恢复逻辑或换用其他checkpoint继续。

## 5. 已有评估与曲线

固定验证批量为32窗。以下保留原脚本产生的全部5行评估：单卡阶段更新0/5/10，以及单卡结束时的重复更新10、resume结束时的更新10。重复评估没有增加优化器更新。

| 运行/序号 | 更新 | 一步cm | 保持基线cm | 根漂移P50cm | 根漂移P90cm | 穿透超2cm帧比例 | 骨盆高度误差cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a/0 | 0 | 0.928659 | 0.928659 | 13.4661 | 52.7991 | 0 | 2.69591 |
| a/1 | 5 | 0.959057 | 0.928659 | 13.4137 | 52.7458 | 0 | 2.69694 |
| a/2 | 10 | 0.975893 | 0.928659 | 13.3902 | 52.7117 | 0 | 2.70074 |
| a/3 | 10 | 0.975893 | 0.928659 | 13.3902 | 52.7117 | 0 | 2.70074 |
| b/0 | 10 | 0.975893 | 0.928659 | 13.3902 | 52.7117 | 0 | 2.70074 |

| 运行/序号 | command_velocity | command_heading | penetration | support | skating | smoothness |
| --- | --- | --- | --- | --- | --- | --- |
| a/0 | 0.417704 | 0.0258499 | 1.71906e-08 | 0.00532174 | 7.02294e-12 | 0.418298 |
| a/1 | 0.4173 | 0.025811 | 2.32541e-07 | 0.00520067 | 0.000159428 | 0.415534 |
| a/2 | 0.417004 | 0.0258274 | 4.01226e-07 | 0.0048651 | 0.000221213 | 0.414351 |
| a/3 | 0.417004 | 0.0258274 | 4.01226e-07 | 0.0048651 | 0.000221213 | 0.414351 |
| b/0 | 0.417004 | 0.0258274 | 4.01226e-07 | 0.0048651 | 0.000221213 | 0.414351 |

全部精度与闭环代价的原始数值见[评估TSV](../../examples/results/terrain_controller_pilot_a100_20261024/evaluation_table.tsv)。

![A100预检曲线](../../examples/results/terrain_controller_pilot_a100_20261024/preflight_curves.png)

曲线来自已保存的`progress.jsonl`与`eval.jsonl`，未新增推理或训练。正式pilot未启动，六个验证窗口的source/闭环接触图未生成。

## 6. 计划第4节的阈值诊断

下表使用10次冒烟更新的末值。正式pilot验收状态为**未评估**。

| 判据 | 阈值 | 冒烟末值 | 冒烟诊断 |
| --- | --- | --- | --- |
| 数值 | 全程有限；完成或到时保存 | 两次命令退出0、数值有限且有checkpoint | 通过 |
| 一步全身误差 | 比保持基线至少低30% | 0.975893cm，基线0.928659cm | 失败 |
| 根漂移P50 | 低于初始且<10cm | 13.466064 → 13.390215cm | 失败 |
| 关节穿透超2cm帧比例 | ≤1% | 0.000% | 通过 |
| 骨盆高度误差 | <5cm | 2.700739cm | 通过 |
| 滑步和支撑代价 | 均低于初始 | skating 7.02294e-12 → 0.000221213；support 0.00532174 → 0.0048651 | 失败 |

## 7. 最终账本与执行状态

| 范围 | 控制器更新 | 窗口曝光 | 训练闭环生成帧 |
| --- | ---: | ---: | ---: |
| 单卡teacher预检 | 5 | 80 | 0 |
| 单卡H=4预检 | 5 | 80 | 320 |
| resume新增 | 0 | 0 | 0 |
| 预检合计 | **10** | **160** | **320** |
| 正式pilot | **0** | **0** | **0** |

脚本的`exposures=400`把teacher窗口80与loop生成帧320相加，因此上表将窗口与生成帧分开。另有5次固定批次验证，共160窗次、2,560个闭环诊断帧和160个单步预测，不计入训练曝光。

critic另执行了5次优化器更新。读取checkpoint确认，控制器optimizer的step值为10，critic为5；resume后的critic仍为5。正式pilot的控制器与critic更新均为0。

双卡冒烟、H=16显存预检、正式pilot均未执行；没有触发3小时停止、没有恢复或扩展pilot预算，也没有部署模型。后续应先修复并验证“已完成阶段延长后恢复”的计数行为，再继续剩余预检。

## 8. 发布与文件位置

checkpoint和原始日志留在A100的`runs/tmt_smoke_1gpu/`、`runs/tmt_smoke_resume/`、`runs/tmt_pilot_coord/`。公共[结果目录](../../examples/results/terrain_controller_pilot_a100_20261024/README.md)仅发布报告、评估表和预检曲线；checkpoint只记录路径、大小与SHA-256。数据传输和结果取回均使用私有Drive。

发布前检查源码哈希未改变、迁移总数一致、resume权重相等证据、账本、PNG、TSV列宽，以及仓库链接和所有变更记录的UTF-8/U+FFFD。

## 9. ee4db6f 续跑

以下为**2026-10-25**的追加记录，上文旧失败及其结果文件原样保留。新执行源码固定为`ee4db6f40bcd5dbe50ff56292967b5a40d8c0d2e`。

**84项测试和resume2通过；双卡冒烟完成运行，但检查未通过，显存预检与正式pilot未启动。** 双卡输出了2个checkpoint文件，而要求为1个；另外，两个rank从不同的模型参数开始训练，结束时仍不一致。原始checkpoint、日志和rank记录留在A100。

### 9.1 更新、测试与复用

A100 checkout经`git pull --ff-only origin main`更新到上述版本，独立依赖目录、Torch2.5.1+cu124与Kimodo沿用已验证环境。运行原七个测试文件加新增的`tests/test_train_terrain_controller.py`：

```bash
$PY -m pytest tests/test_bones_env_dataset.py tests/test_terrain_controller.py \
  tests/test_skeleton_ray_sampler.py tests/test_bones_terrain_fit.py \
  tests/test_bones_terrain_scene.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py tests/test_train_terrain_controller.py \
  -q --full-trace --color=no --junitxml=runs/tmt_pilot_coord_v2/tests.xml
```

结果为**`84 passed, 11 warnings in 8.13s`**，0失败、0跳过。

直接复用已通过的5,652文件迁移及`runs/tmt_smoke_1gpu/latest.pt`，没有重传数据或重跑单卡冒烟。再次读取旧checkpoint的SHA-256，与先前记录一致。每个新GPU任务前重新扫描全部GPU的UUID、显存、利用率与计算PID，锁定空闲的物理**GPU0和GPU1**；resume2仅使用GPU0。

### 9.2 Resume2通过

```bash
$PY scripts/train_terrain_controller.py --dataset runs/bones_env_v1 \
  --output runs/tmt_smoke_resume2 --phases teacher:5 loop:4:7 \
  --resume runs/tmt_smoke_1gpu/latest.pt \
  --batch 16 --max-hours 0.2 --eval-every 5 --log-every 1 --eval-batch 32
```

日志记录的恢复位置完全符合要求：

```json
{"resumed_at_update": 10, "phase_index": 1, "phase_update": 5}
```

最终**update=12、exposures=528**，较旧单卡冒烟新增**2次控制器更新、2次critic更新、32个窗口、128个H=4生成帧**。退出码0，外层耗时9.532秒，损失与评估均有限。

### 9.3 双卡运行与检查

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 2 \
  scripts/train_terrain_controller.py --dataset runs/bones_env_v1 \
  --output runs/tmt_smoke_2gpu --phases teacher:5 \
  --batch 64 --max-hours 0.2 --log-every 1 --eval-batch 64
```

实际通过只记录计时、参数摘要和保存事件的包装器执行同一训练入口与参数。没有改变损失、随机种子、梯度平均或保存逻辑。外层耗时**13.458秒**，两个rank均退出0、各完成5次控制器更新；全局batch128，累计640窗口，exposures=640。

| Rank | 退出码 | 更新 | 保存事件 |
| --- | --- | --- | --- |
| 0 | 0 | 5 | phase0_teacher.pt, latest.pt |
| 1 | 0 | 5 | 无 |

只有rank0写文件，实际文件为`phase0_teacher.pt`与`latest.pt`。原入口在阶段结束时保存前者，最终再次保存后者，分别见冻结源码的[阶段保存](https://github.com/darthsoluke/TerrainAwareKinetics/blob/ee4db6f40bcd5dbe50ff56292967b5a40d8c0d2e/scripts/train_terrain_controller.py#L240)与[最终保存](https://github.com/darthsoluke/TerrainAwareKinetics/blob/ee4db6f40bcd5dbe50ff56292967b5a40d8c0d2e/scripts/train_terrain_controller.py#L243)。因此“两个rank完成”通过，“恰好一个checkpoint文件”未通过。

另记录了每个rank在首个训练步骤之前、训练结束之后的逐参数SHA-256。摘要计算位于更新计时区间之外，没有新增训练步骤。

| 时点 | 控制器不同参数张量 | Critic不同参数张量 |
| --- | --- | --- |
| 训练前 | 72/134 | 6/6 |
| 训练后 | 72/134 | 6/6 |

控制器初始化摘要为：

- rank0：`bf930a301bd4e6db089189680dadbb0520be223901f16c3b86a97d55441e1e18`
- rank1：`fc1a16fd712ef5c7146d32a97182149f3af6afcef9e8796a17d26728a660eb8a`

不同的参数包括`age`、`command_in.0.weight`、`command_in.0.bias`、`command_in.2.weight`、`command_in.2.bias`和`contact_head.bias`；完整名称与摘要见[续跑报告](../../examples/results/terrain_controller_pilot_a100_20261024/continuation_ee4db6f/report.json)。这是额外的多卡一致性诊断。

源码[第97行](https://github.com/darthsoluke/TerrainAwareKinetics/blob/ee4db6f40bcd5dbe50ff56292967b5a40d8c0d2e/scripts/train_terrain_controller.py#L97)使用`torch.manual_seed(args.seed + rank)`初始化各rank；随后只在[梯度平均](https://github.com/darthsoluke/TerrainAwareKinetics/blob/ee4db6f40bcd5dbe50ff56292967b5a40d8c0d2e/scripts/train_terrain_controller.py#L148)中all-reduce，没有同步初始模型或critic参数。实测两rank的模型从不同参数开始，5次平均梯度更新之后仍不同。

本次不修改训练实现或保存规则。按预检失败即停止的条件，未进入H=16/batch512显存检查，未选择pilot的512/256批量，也未启动`runs/tmt_pilot_v1`。

### 9.4 每阶段耗时与显存

| 运行 | Rank | 阶段 | 更新 | 平均秒/更新 | 中位秒/更新 | 峰值MiB |
| --- | --- | --- | --- | --- | --- | --- |
| b2 | 0 | loop:4 | 2 | 0.635094 | 0.635094 | 137.267 |
| c | 0 | teacher:0 | 5 | 0.181422 | 0.090268 | 159.279 |
| c | 1 | teacher:0 | 5 | 0.181444 | 0.089624 | 104.179 |

峰值来自`torch.cuda.max_memory_allocated`；每卡`total_memory`为85,100,068,864 bytes。表中仅为已经执行的batch16/H=4及batch64/teacher数据，计划要求的batch512/H=16峰值尚无结果。计时从前向和损失开始到模型optimizer.step结束，两端同步CUDA，不含数据加载、评估或参数摘要。

### 9.5 新增评估与曲线

resume2使用固定32窗验证批量，双卡运行使用64窗；下面分别保留各自的全部评估。

| 运行 | 验证batch | 更新 | 一步cm | 保持基线cm | 根漂移P50cm | P90cm | 穿透超2cm帧比例 | 骨盆高度误差cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resume2 | 32 | 12 | 0.976279 | 0.928659 | 13.3894 | 52.7109 | 0 | 2.70091 |
| two_gpu | 64 | 0 | 0.967278 | 0.967277 | 14.9659 | 58.1541 | 0 | 2.81307 |
| two_gpu | 64 | 5 | 0.975396 | 0.967277 | 14.9754 | 58.1535 | 0 | 2.81281 |

| 运行/更新 | command_velocity | command_heading | penetration | support | skating | smoothness |
| --- | --- | --- | --- | --- | --- | --- |
| resume2/12 | 0.416996 | 0.0258266 | 3.98981e-07 | 0.00485196 | 0.000223669 | 0.414344 |
| two_gpu/0 | 0.430068 | 0.0216219 | 8.91541e-09 | 0.00455835 | 7.00315e-12 | 0.448475 |
| two_gpu/5 | 0.429861 | 0.0215963 | 3.56801e-08 | 0.00450555 | 5.50788e-05 | 0.446976 |

完整精度数值见[评估TSV](../../examples/results/terrain_controller_pilot_a100_20261024/continuation_ee4db6f/evaluation_table.tsv)。

![ee4db6f预检续跑曲线](../../examples/results/terrain_controller_pilot_a100_20261024/continuation_ee4db6f/preflight_curves.png)

图表均从已经保存的progress/eval日志生成。没有执行额外闭环推理；六个验证窗口的source/闭环接触图仍未生成。计划第4节正式pilot的数值、一步误差、漂移、穿透、骨盆高度以及滑步/支撑验收均为**未评估**，本次停止位置在预检。

### 9.6 新增及累计账本

| 范围 | 控制器全局迭代 | Critic更新 | 窗口曝光 | 训练闭环生成帧 |
| --- | ---: | ---: | ---: | ---: |
| resume2新增 | 2 | 2 | 32 | 128 |
| 双卡teacher冒烟 | 5 | 0 | 640 | 0 |
| 本次续跑合计 | **7** | **2** | **672** | **128** |
| 含旧单卡冒烟的全部预检 | **17** | **7** | **832** | **448** |
| 正式pilot | **0** | **0** | **0** | **0** |

双卡每一轮按一个全局迭代计，两个rank各执行5次optimizer.step。本次控制器optimizer调用合计12次，全部预检合计22次。`resume2`checkpoint累计update=12；双卡任务是另一个新初始化运行，checkpoint update=5，不把它们描述成单个模型的17次连续训练。

正式3,000更新/3小时预算未启动，没有恢复或增加pilot预算。后续需要处理checkpoint保存数量，并使两rank的控制器和critic初始化一致，再继续剩余预检。

### 9.7 产物与发布

旧记录和旧结果文件保留。新公共结果位于[continuation_ee4db6f](../../examples/results/terrain_controller_pilot_a100_20261024/continuation_ee4db6f/README.md)，包括结构化报告、评估表与预检曲线；原始日志、rank记录和checkpoint保留在A100的`runs/tmt_pilot_coord_v2/`、`runs/tmt_smoke_resume2/`、`runs/tmt_smoke_2gpu/`。结果经私有Drive取回，没有导出checkpoint。

发布前核对新增源码哈希、84项测试、resume计数与日志、双rank保存记录和参数摘要、账本、图表与表格，并确认本节之前的Markdown逐字节不变，再运行仓库及UTF-8/U+FFFD检查。
