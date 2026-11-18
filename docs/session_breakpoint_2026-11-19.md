# 会话断点（2026-11-19 凌晨，用户关机）

> 下次开机**从这里接着做**，不要重新审题，不要重新解压。

## 一、当前状态

### 1.1 已解压到 `F:\wsl\task2`（沙箱路径 `/mnt/task2`）

| 内容 | 状态 | 大小 |
|---|---|---|
| `contest-data/` | ✅ **完整**（261 个文件，`task2/{binary_loss,loss_decoder}/{train,validation}/` 都在） | 985 MB |
| `contest-source/` | ⏳ **解压到 29,692 / 33,960 个文件（87%）**，目录结构已齐（AGENTS.md / AgenticBench / OpenROAD-3D / Place-3D / open3dbench / third_party / tests / docker / docs / release） | 2.4 GB |
| `open3dbench-contest-runtime.tar.gz` | ❌ **还没复制/加载**（7.0 GB，需要 `docker load`，只能用户在 Windows 侧做） | — |

**下次第一件事**（约 2–3 分钟）：
```bash
cd /sessions/serene-sleepy-edison/mnt/task2 && \
  timeout 280 tar -xzf "/sessions/serene-sleepy-edison/mnt/2/赛题2/contest-source.tar.gz" --skip-old-files -C . ; \
  find contest-source -type f | wc -l
```
反复执行，直到 `find contest-source -type f | wc -l` **稳定在 33,960**（官方 `bundle.json: physical_files=33960`）。
解压完成后用 `python3 -c "import json;d=json.load(open('contest-source/bundle.json'));print(d['status'])"` 核对，或直接跑 `contest2 check`（它会逐文件校验 sha256）。

> 注意：沙箱每次 bash 调用上限约 290 秒，`tar` 必须在**前台**分批跑；`nohup &` 会在调用结束时被回收。

### 1.2 我们自己的代码（`F:\huawei_aic_tao\task2`，git 已提交到 `00c73cb` 之后若干 commit）

| 文件 | 说明 |
|---|---|
| `official_kit/agent_templates/binary_loss/` | **交付物 1**：`agent.py`（按官方接口重写）+ `program.py`（官方原始基线，928 字节的 EVOLVE-BLOCK）+ `common/agent_common.py` |
| `official_kit/agent_templates/loss_decoder/` | **交付物 2**：同上，双函数（846 字节） |
| `official_kit/local_tasks/` | 官方**静态校验器**（`validate_program`），本地已验证两份 program 都通过 |
| `docs/search_policy_10.md` | 10 次评估的分配（**已按饱和度实测重排**：平衡 > 调度 > 割项） |
| `docs/local_findings_saturation.md` | 本地实测：割项在饱和后 0% 影响，平衡项 48% |
| `docs/offline_knowledge_*.md` + `official_kit/.../agent_common.py` | 离线知识 **170 / 182 计数单位**（上限 200），计数脚本 `workspace/tools/count_offline_knowledge.py` |
| `docs/windows_setup_checklist.md` | 用户侧步骤（**已按 `F:\wsl\task2` 路径更新**） |
| `workspace/baseline_stub/common/official_adapter.py` | 官方签名 → 替身物理的适配层（无 GPU 本地验候选） |
| `workspace/tools/mock_cycle.py` | 10 次预算闭环演练（已在沙箱副本中跑通，不改交付物） |

### 1.3 已确认的关键事实（不要重新推导）

- 每基线 **10 次评估 + 15 元**（`contest2/budget.py:108`）
- 分母 = 该算例两条基线的**共同参考值**（α=1.854，`manifests/common-timing-reference.json`）
- 物理失败单组 −100 且留在矩阵；接口/范围违规不产生分数
- 交付物 = `AgenticBench/agent/<baseline>/`，官方替你跑评估
- 离线知识 ≤200 计数单位；**运行时自学的经验允许**（写 `MEMORY.md`）
- 官方固定配方下概率饱和 99.7%，割项梯度≈0

## 二、下次的剩余步骤（按顺序）

1. **补完解压**（上面那条命令，跑到 33,960）
2. **用户侧**：
   a. 确认 Docker 数据盘在 F 盘（Docker Desktop → Settings → Resources → Advanced → Disk image location）
   b. `cd F:\wsl\task2 && docker load -i <path>\open3dbench-contest-runtime.tar.gz`（7 GB，20–40 min）
   c. `python -m pip install PyYAML`
   d. `cd F:\wsl\task2\contest-source\AgenticBench && python -m contest2 check --baseline binary_loss`
3. **装我们的 Agent**（复制 `official_kit/agent_templates/<baseline>/*` 覆盖 `AgenticBench/agent/<baseline>/`），再跑 `python -m contest2 check` 确认入口可导入
4. **`python -m contest2 smoke --baseline binary_loss`**（真评估 1 组，验链路）
5. **正式 `run`**（10 次/基线），之后 `validate`

## 三、关机前要提醒用户的

- 解压还没完（87%），**不要删 `F:\wsl\task2`**，下次我接着补
- 7 GB 镜像**还没加载**；加载前务必确认 Docker 数据盘在 F 盘
- 我这边沙箱没有 docker，**`docker load` 和所有 GPU 物理流程只能你执行**
- 两个仓库都有未 push 的 commit：`task2`（`F:\huawei_aic_tao\task2`）和 `task1`；推送用各自的 `git_sync.bat`
