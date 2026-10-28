# 交接目录

规划端（写代码、写计划的会话）把给执行端的指令写在这里；执行端直接读仓库，不再需要用户在两边之间复制粘贴。

## 文件

| 文件 | 用途 | 谁写 |
| --- | --- | --- |
| [CURRENT.md](CURRENT.md) | 当前要执行的任务，只指向一个任务文件 | 规划端 |
| `tasks/YYYY-MM-DD_<短名>.md` | 一次完整的指令：背景、批准情况、步骤、停止条件、要写的记录 | 规划端写指令；执行端只改文件末尾的“状态”一节 |

旧任务文件不删除，保留为历史；`CURRENT.md` 换指向即可。实验结果仍写在 `docs/records/` 和 `examples/results/`，任务文件只链接过去。

## Protocol for the execution agent

1. After `git pull --ff-only origin main`, read `handoff/CURRENT.md` and the task file it points to. That file is the instruction; nothing else needs to be pasted.
2. Do only what the task file says, within its budget and approvals. If the task file and the plan document it cites disagree, stop and record the disagreement instead of choosing.
3. Before starting, set the task's Status to `running` (date, source commit) and push. When you finish or stop, set it to `done` or `stopped`, with a one-line outcome, the results commit, and links to the records you wrote. Do not edit anything above the Status section.
4. If `CURRENT.md` says there is no active task, do nothing.

## Protocol for the planning side

- One task file per instruction, self-contained (no "as in the previous message"). State which user approval covers it.
- Point `CURRENT.md` at it in the same commit. A newer task supersedes the old one only when `CURRENT.md` is repointed; mark the old one `superseded` in its Status if it never ran.
