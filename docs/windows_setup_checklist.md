# 本机环境准备清单（Windows 侧，你执行；我无法代劳）

> 目标：把官方三件套变成"能跑 `python3 -m contest2 check/smoke/run`"的状态。
> 全部放在 `F:\` 下；已确认 F 盘可用 276 GB（7 GB 镜像解压后约 20–25 GB，数据 1.1 GB，加上工作目录余量足够）。
> 每一步都写了**预期输出**和**常见失败**，出错时把最后 20 行贴给我。

---

## 步骤 0 · 先确认三件事（30 秒）

在 **PowerShell** 里跑：

```powershell
nvidia-smi                          # 预期：表格里有 GPU 名称、Driver Version、CUDA Version
docker info --format '{{.DockerRootDir}}'   # 预期：一个路径；若是 C:\ 下的路径，先做步骤 1
wsl --list --verbose                # 预期：有 docker-desktop 与 docker-desktop-data（或默认发行版）
```

**预期**：
- `nvidia-smi` 出表格（RTX 4060 Laptop 或类似；显存 ≥ 8 GB 更稳）
- `docker info` 正常返回（Docker Desktop 已启动）

**常见失败**：
- `nvidia-smi` 不是内部或外部命令 → NVIDIA 驱动没装好 / PATH 没配
- `docker info` 报 `error during connect` → Docker Desktop 没启动
- `wsl` 报错 → 没启用 WSL2（需要管理员 PowerShell 跑 `wsl --install`）

---

## 步骤 1 · 把 Docker 数据盘放到 F 盘（若步骤 0 显示在 C 盘）

Docker Desktop → 右上齿轮 **Settings** → **Resources** → **Advanced** → **Disk image location**
改成 `F:\docker\DockerDesktopWSL` → **Apply & Restart**。

**预期**：重启后 `docker info --format '{{.DockerRootDir}}'` 仍在容器内报 `/var/lib/docker`（正常），但宿主机实际占用的目录变成 F 盘。可以在 F 盘看到 `DockerDesktopWSL\` 目录增长。

**为什么必须做**：7 GB 压缩包 `docker load` 后是多层镜像，连同 WSL 的 ext4.vhdx，实际占用通常 20–30 GB；C 盘不够会直接失败。

---

## 步骤 2 · 解压源码与数据（Windows 自带 tar 即可，或 7-Zip）

```powershell
cd F:\huawei_aic_tao\task2
mkdir official
copy ..\..\mnt\2\赛题2\*.tar.gz official\      # 或直接用资源管理器从下载目录复制
cd official
tar -xzf contest-source.tar.gz
tar -xzf contest-data.tar.gz
```

> **注意**：我这边看到的路径是 `F:\huawei_aic_tao\task2\2\赛题2\`（沙箱里显示为 `/mnt/2/赛题2`）。
> 以你实际下载位置为准；关键是把三个 `.tar.gz` 放在**同一个目录**再解压，因为 `contest-data.tar.gz` 解出来是 `contest-data/task2/...`，`start`/`run` 依赖这个相对结构。

**预期**：
```
official\
├── contest-source\        (AgenticBench\ Place-3D\ open3dbench\ OpenROAD-3D\ README.md)
├── contest-data\task2\    (runtime.json + binary_loss\ + loss_decoder\)
└── open3dbench-contest-runtime.tar.gz
```

**常见失败**：
- `tar` 报 "无法识别的归档" → 用 7-Zip 解压（`.tar.gz` 是两层）
- 解出中文乱码 → Windows `tar` 的 codepage 问题；用 7-Zip 更稳
- 磁盘不足 → 先做步骤 1

---

## 步骤 3 · 加载镜像（最慢的一步，20–40 分钟）

```powershell
cd F:\huawei_aic_tao\task2\official
docker load -i open3dbench-contest-runtime.tar.gz
```

**预期输出**：最后一行形如
```
Loaded image: open3dbench-contest-runtime:xxxx
```
（具体 tag 以 `runtime-images.json` 里的 `config_id` 为准；`contest2 check` 会核对 `sha256:f2401e97...`）

**失败排查**：
- `no space left on device` → 步骤 1 没做，或 F 盘不足
- `docker load` 卡住 → Docker Desktop 的 WSL 后端内存不足；在 Settings → Resources 把内存调到 ≥ 12 GB
- 校验失败（`check` 阶段报 config_id 不符）→ 镜像文件下载不完整，重新下载那个 7 GB 包

---

## 步骤 4 · Python 与依赖

```powershell
python --version          # 需要 3.10+
python -m pip install PyYAML
```

**预期**：`PyYAML` 安装成功（官方 README 只要求这一个）。

> 如果 `python` 不是 3.10+：用 `py -3.11 -m pip install PyYAML`，后面命令都换成 `py -3.11`。
> **不要**在 WSL 里装 Python 来跑这个——`contest2` 需要直接调 Docker，在 Windows 侧跑最省事。

---

## 步骤 5 · DeepSeek API key

官方要求：**key 不能放在源码/数据/工作目录里**。

```powershell
mkdir F:\huawei_aic_tao\secrets
notepad F:\huawei_aic_tao\secrets\deepseek.key     # 粘贴 sk-... 后保存
```

后续命令加 `--api-key-file F:\huawei_aic_tao\secrets\deepseek.key`，或设环境变量 `DEEPSEEK_API_KEY`。

**花费提醒**：每基线 15 元；先在步骤 6 的 `check` 与步骤 7 的 `smoke` 跑通，再跑正式。

---

## 步骤 6 · 官方自检（不耗评估次数、不花模型费）

```powershell
cd F:\huawei_aic_tao\task2\official\contest-source\AgenticBench
python -m contest2 check --baseline binary_loss
python -m contest2 check --baseline loss_decoder
```

**预期**：输出里包含数据/参考/镜像核对通过，并打印 GPU 型号、计算能力、运行库版本。

**失败排查**：
| 报错 | 处理 |
|---|---|
| 找不到 `contest-data` | 步骤 2 的解压位置不对，`contest-data` 必须与 `contest-source` 同级 |
| 镜像 config_id 不符 | 镜像不完整 |
| CUDA 算子检查失败 | 驱动/CUDA 版本与镜像不匹配；把报错贴我 |
| `ModuleNotFoundError: yaml` | 步骤 4 没做 |

---

## 步骤 7 · 闭环自检（官方 smoke，会**真的评估 1 组**，但不计正式额度）

```powershell
python -m contest2 smoke --baseline binary_loss
```

**预期**：`contest-work/smoke/<新目录>/session-result.json` 存在，退出码 0。日志会显示两轮 Codex 调用 + 1 组物理评估（bp_fe-ng45 / seed 1003）。

**这一步很关键**：它验证"镜像 + 物理流程 + 模型网关 + Agent 容器"整条链路。**先跑它，再跑正式**。

**耗时**：单组物理流程按官方说明不设时限；参考数值显示切分布局任务明显短于全流程，但首次仍可能要几十分钟。

---

## 步骤 8 · 装上我们的 Agent

```powershell
cd F:\huawei_aic_tao\task2\official\contest-source\AgenticBench
# backup官方模板
ren agent\binary_loss agent\binary_loss.official
ren agent\loss_decoder agent\loss_decoder.official
# 复制我们的
xcopy /E /I /Y F:\huawei_aic_tao\task2\official_kit\agent_templates\binary_loss agent\binary_loss
xcopy /E /I /Y F:\huawei_aic_tao\task2\official_kit\agent_templates\loss_decoder agent\loss_decoder
```

然后**不花额度**地验证入口可导入 + 候选能过官方校验：

```powershell
cd F:\huawei_aic_tao\task2\official\contest-source\AgenticBench
python -c "from agent.binary_loss.agent import Agent; print('binary OK', callable(Agent().run))"
python -c "from agent.loss_decoder.agent import Agent; print('decoder OK', callable(Agent().run))"
python -c "from local_tasks.tp_gnn_local_task_definitions import BINARY_TASK, validate_program; from pathlib import Path; print(len(validate_program(Path('agent/binary_loss/program.py').read_text(), BINARY_TASK)))"
```

**预期**：两行 OK + 一个数字（≈928）。

---

## 步骤 9 · 正式跑（10 次评估 / 基线）

```powershell
python -m contest2 run --baseline binary_loss --api-key-file F:\huawei_aic_tao\secrets\deepseek.key
```

跑完后：

```powershell
python -m contest2 validate --baseline binary_loss --candidate-id <返回的候选ID>
```

**监控要点**：`contest-work\task2\binary_loss\` 下会有评估记录；`agent\binary_loss\MEMORY.md` 是 Agent 自写的运行记忆（可随时查看它学到了什么）。

---

## 我这边能做什么 / 不能做什么

| 能做 | 不能做 |
|---|---|
| 改 `agent.py` / `program.py` / 离线知识 | 执行 `docker load`、跑 GPU 流程 |
| 本地用官方静态校验器验候选 | 读你的 `deepseek.key`（也不需要） |
| 用替身做本地 A/B（无 GPU） | 访问你 C 盘 / 装驱动 |
| 分析你贴回来的日志 | 直接操作 Docker Desktop GUI |

**下一步建议**：先做步骤 0–2（10 分钟），把 `nvidia-smi`、`docker info`、`wsl --list --verbose` 三条输出贴给我，我确认没问题你再花 30 分钟做 `docker load`。
