# 开发、复现与源码整理

## 从无需模型的结果表开始

在仓库根目录运行：

```bash
python scripts/compare_arms.py --root examples/results/ablation_v3 --fixed-review examples/results/fixed_masks_summary.json
python tools/check_repository.py
```

第一个命令只读取精简结果。第二个检查文档链接、Python语法、JSON和公开源码内容。

## 安装Python包

使用Python3.10或3.11，先按硬件安装适合的PyTorch，再在独立环境中安装本项目：

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

三维人体代码依赖 [Kimodo](https://github.com/nv-tlabs/kimodo)。本项目记录的参考版本为 `58e781898b3d7e328a676a75d3e338c45dce3ad9`。历史运行代码约定把它放在 `out/vendor/kimodo`，并通过Python安装或PYTHONPATH读取。

源代码准备示例：

```bash
git clone https://github.com/nv-tlabs/kimodo.git out/vendor/kimodo
git -C out/vendor/kimodo checkout 58e781898b3d7e328a676a75d3e338c45dce3ad9
python -m pip install -e out/vendor/kimodo
```

Kimodo的人体骨架、蒙皮资源，模型权重和相应实验数据需要按来源说明另外准备。运行地图查看器可参照 `requirements-map-viewer.txt`；历史G1参考求解使用过的依赖见 `requirements-reference-ik.txt`。

SONIC、ProtoMotions与MuJoCo的机器人分支有独立模型和环境合同，按其历史接入记录准备外部依赖。当前主线是人体运动学生成。

## 训练和评估入口

- [scripts索引](../../scripts/README.md) 按任务列出常用命令入口。
- [configs说明](../../configs/README.md) 解释配置版本和draft状态。
- [tests说明](../../tests/README.md) 解释可运行的轻量检查与依赖实际资源的测试。

任何新训练先锁定数据/源码、实际输入时间语义、预算和预检。当前公开配置中的私人路径已改成模板位置，运行前使用自己的本机配置。

## 本次源码同步

本仓库是后续代码与文档的统一入口。初次导入从旧MotionBricks工作区读取源码，保留模块路径，并对公开副本进行机器信息清理。

对应工具：

```bash
python tools/sync_to_terrain_repo.py --source /path/to/legacy/motionbricks --target /path/to/TerrainAwareKinetics
```

该导入工具只处理源码、配置、测试、网页和Markdown白名单；记录每个源文件与公开文件的哈希。遇到目标仓库里独立修改的同名文件会停止，留给维护者合并。后续直接在本仓库开发时，无需重复导入旧副本。

工具会清理机器地址、用户路径和私有制品链接。额外的私人域名、账户名和存储别名放在仓库外的JSON替换表，通过 `--redactions /path/to/private-replacements.json` 传入；发布前运行仓库检查并人工审阅差异。

导入清单：[source_snapshot.json](source_snapshot.json)。历史文档引用、但未放进Git的资源位置：[private_artifact_references.json](private_artifact_references.json)。

本次CPU测试、结果表与发布检查记录见 [2026-10-23检查](publication_checks_20261023.md)。
