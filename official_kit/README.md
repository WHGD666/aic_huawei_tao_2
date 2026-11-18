# 官方任务包摘录（只读参考 + 交付物模板）

来源：`F:\huawei_aic_tao\task2\..\..\mnt\2\赛题2\contest-source.tar.gz`（用户下载的官方材料），2026-11-19 摘录。

| 目录 | 用途 |
|---|---|
| `local_tasks/` | **官方静态校验器**（`tp_gnn_local_task_definitions.py::validate_program`）+ 两份 `prompt.md`（官方接口说明）+ 固定配置 |
| `local_bench/program_safety.py` | 校验器依赖（禁外部副作用、归一化） |
| `contest2/` | 外层 harness（预算/调度/评分，只读参考） |
| `agent_templates/` | **我们的交付物**：两份 Agent + `program.py` 模板；每个目录要整体复制到官方 `AgenticBench/agent/<baseline>/` |

## 本地怎么用

```bash
cd official_kit
python3 -c "
from local_tasks.tp_gnn_local_task_definitions import BINARY_TASK, validate_program
from pathlib import Path
print(len(validate_program(Path('agent_templates/binary_loss/program.py').read_text(), BINARY_TASK)))
"
```

## 交付物结构（每份基线）

```
agent_templates/binary_loss/      →  复制到 AgenticBench/agent/binary_loss/
├── agent.py                      Agent.run(context) 入口（我们写的）
├── program.py                    当前父代程序（会被 Agent 覆写）
└── common/agent_common.py        共享：离线知识、本地校验、记忆、反馈摘要、切分变化预检
```

`common/` 不放 `__init__.py`（官方要求 namespace package 风格，且 Agent 目录里允许普通辅助文件）。
