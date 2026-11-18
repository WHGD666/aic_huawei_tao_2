# Harness 审核清单（2026-11-18，独立 agent 对抗式审核 + 本轮修复记录）

> 审核对象：`workspace/harness/*`、`config/harness.yaml`、`prompts/*`、`baseline_stub/`。视角：预算泄漏、血统正确性、规则合规（函数边界 / 不特判 / 全有效）、多候选与代理层正确性。
> 审核员在会话里**实测复现**了下列绕过（gate 规模/单位规则可绕过、非目标文件可写穿、空切分指纹导致全判重复、导出回退基线等）。

## HIGH

| # | 位置 | 问题 | 修法 | 状态 |
|---|---|---|---|---|
| 1 | loop / budget | `charge_evaluation` 在 `evaluate()` 之后 → 评估中途被杀则漏记，反复被杀+续跑可绕过 30 次 | `begin_evaluation()` 预扣 + 启动时把遗留 pending 计为已消耗；`charge_evaluation` 结算 | ☑ |
| 2 | config/harness.yaml | `proxy_cmd` 复用官方评估命令 → 若官方按命令调用计量，代理运行在烧官方次数 | 配置注释明确"必须指向独立离线脚本"；`proxy.enabled` 与官方口径绑定（`open_questions` #1）；若官方计次则 `--no-proxy` | ☑（文档+配置约束；最终取决于官方口径） |
| 3 | gate | 规模特判只拦字面量 `>50` 且正则命中两种写法 → `n = p.shape[0]; if n > 300` 实测通过 | 别名传播（`p.shape[0] / len(...) / graph["n_nodes"]` → size 标量）+ 比较/if 分支都查（阈值 >8） | ☑ |
| 4 | gate | 单位陷阱只认 `feats[:, k]` 直接下标 → `s = feats[:, 0]; s < 0.1` 实测通过 | 列别名传播（`s = feats[:, k]` 继承 timing 属性）+ 相对化识别（argsort/rank/mean/std/clamp 链放行） | ☑ |
| 5 | gate / smoke | `getattr(p,'detach')()`、`__class__.__mro__`、`import numpy as npl; npl.random.rand()`、`compute_loss.calls=n` 实测通过；smoke 在 harness 进程内 exec | 动态访问/dunder/函数对象挂属性一律拒；import 别名归一化；smoke 子进程 + 硬超时 | ☑（dunder/getattr/别名/状态已拦；子进程见 MED-10） |
| 6 | loop / codebase | 一次 edit 同时改 `train.py` 与 `loss.py`：`train.py` 被写进 work 但 gate 只看 targets、patch 不含它 → 评估命令真实使用越界改动 | `_stray_files()`：work 里非目标文件与基线不同即 gate_failed | ☑ |
| 7 | evaluator | `partitions` 为空时 `partition_hash()` 恒为同一常量 → 第 2 个候选起全判 duplicate、不花评估 | 无切分数据返回 `None`，此时禁用去重；基线无切分数据则日志明示"去重关闭" | ☑ |
| 8 | loop.finalize | 从 `work/<best>` 导出、缺失时静默回退基线代码 → 提交物自相矛盾 | 从 `baseline + candidate.patch` 重建导出；重建失败才回退 work，且 work 也缺则 raise | ☑ |

## MED

| # | 位置 | 问题 | 修法 | 状态 |
|---|---|---|---|---|
| 9 | evaluator | `valid: "false"` 字符串被 `bool()` 成 True | 显式解析字符串/数字/布尔 | ☑ |
| 10 | smoke | `while True:` 会永久挂死 harness；"超时"是跑完再比 | 子进程 + 硬超时（`smoke_time_limit_s`） | ☐ 待做 |
| 11 | loop | 只有"连续 gate_failed / duplicate"两个 breaker | 增加 `max_consecutive_no_submit`（连续无提交即停） | ☑ |
| 12 | loop | fix 时 `previous_response` 是整段 k 候选响应，slot n 的修复可能拿到 slot 0 的 edit | `_candidate_segment(text, slot)` 只给该候选自己的段落 | ☑ |
| 13 | loop | `_patch_hashes` 只登记 chosen → 之后可重复提案；discarded 的却被登记 → 封杀未评估改动 | 改为评估成功后登记 | ☑ |
| 14 | selector | `rescale_phase_plan(1)` → final = −2 | 每阶段 ≥0、total 足够时 ≥1、余数给非 final | ☑ |
| 15 | proposer | system prompt 的 `{K}` 固定成 propose_k，final 阶段 k=1 但 system 说 2 | `build_messages` 里按实际 k 替换 | ☑ |
| 16 | scoring | 基线违例为 0 时候选记 −1，量级与其它算例不可比 | 保留约定并在报告里标注口径（跨算例比较的影响留到真实数据再定） | ☑（文档） |
| 17 | gate | `feature_names` 硬编码 | 构造参数保留（config 可覆盖），与任务包核对 | ☑ |
| 18 | smoke | `HOLDOUT_CASES`（无宏 / asap7）定义却不用 → 最终候选可能在无宏算例上 NaN | 导出前对 holdout 跑一次免费冒烟/代理，不通过不导出 | ☐ 待做 |
| 19 | loop | proxy 中 smoke 指标排在前 4 → 历史里给模型看的不是 cut/wcut | 代理统计在前、冒烟在后；`history_digest` 只列统计键 | ☑ |
| 20 | evaluator | proxy 缓存键只有源码 hash | 键加入 cases/seeds/cmd/基线参考 hash | ☑ |
| 21 | evaluator | 去重的标签对称假设（0/1 互换）；整体翻转的补集切分被判重复而永不评估 | 去重仍用对称指纹省评估；"近零 hamming"只作 warn；报告里声明该假设 | ☑（文档）；补集探测未做 |
| 22 | gate / prompts | `sign/ceil/nonzero/cpu/no_grad/.data/.grad` 拦了但 system.md 没写 → 静默被拒浪费一次修复 | 待把禁用清单从同一份常量注入 system.md | ☐ 待做 |
| 23 | smoke | 硬依赖 `baseline_stub.*` → 换官方任务包后每个候选都 gate_failed 且方向误判 | 区分"移植/环境错误"（ConfigError 停）与"候选错误" | ☐ 待做 |
| 24 | codebase | `apply_unified_diff` 用 `splitlines()` 丢 CRLF | 改为按 `\n` 切分保留 `\r` | ☑ |

## LOW

| # | 位置 | 问题 | 状态 |
|---|---|---|---|
| 25 | loop | propose 抛异常后 meta 停在 proposed（本次运行内悬空） | ☑ |
| 26 | loop | 事件名一律 "evaluated"（status 可能是 invalid） | ☑ |
| 27 | selector | final 阶段父代未过滤 `_fail_streak` | ☑ |
| 28 | profile | 画像打印 `abs(area_dev)` 丢方向 | ☐ |
| 29 | gate | 只查 str 常量（bytes 漏）；`'ng'+'45'` 拼接漏 | ☑（bytes 已查；拼接仍漏） |
| 30 | precheck | 允许 SEARCH 命中"文件里唯一但提示词没展示"的片段（与 system.md 不一致） | ☐ 保留（设计取舍，见下） |
| 31 | evaluator | `--ref` 用正则剥除；命令含字面 `{` 会让 format 抛错 | ☐ |
| 32 | smoke | decode 只在 100–120 节点测，O(n²) 解码在 10⁵ 节点会超时 | ☐ |
| 33 | llm | replay 缓存键不含 model/thinking | ☐ |
| 34 | store | 损坏 meta.json 直接 KeyError | ☐ |
| 35 | evaluator | `bytes(p)` 对 0/1 外标签抛错 | ☑（hash 前校验） |
| 36 | tests | e2e 写真实 `workspace/algorithms/` | ☐ |
| 37 | loop CLI | 新 run-dir 重置本地预算；CLI 可覆盖预算 | ☐（正式跑只读 config；文档已注明） |
| 38 | gate | 装饰器列表死比较 | ☑ |

## 审核确认无问题的点

- `can_evaluate/can_call_llm` 边界（`left > 0` 严格）；`reserve_evaluations` 语义
- 续跑预算不重置（budget.json 重放）
- LLM 费用：propose/fix/analyze 全部经 `_chat → charge_llm`；重试有上限且 `max_cost=12 < 15` 有余量
- duplicate 不会指向自己或未评估候选（三处写入点都来自已评估/基线）
- screened/discarded/gate_failed/invalid 永远不会做父代或最终候选（`elites()` 只取 status=evaluated 且 score 非空）
- 算例匿名化覆盖所有渲染路径（彩排 83 个 replay 里 0 次出现真实算例名，实证）
- `apply_edits` 原子性（任一 SEARCH 失败不写任何文件）；路径穿越检查有效
- AST 边界：签名（名+序）、默认值/注解、装饰器、模块级 import、其它函数、新增顶层语句、docstring 改动全部被拦
- probe 阶段确实从基线出发（`choose_parent` 在 probe 返回 None）
- mock 模式不写基线文件
- 多候选 work 目录 / id 分配互不干扰

## "SEARCH 可命中未展示但真实存在的代码"（L-30）为什么保留

小模型在父代叠加场景下，若严格要求"只看本轮展示片段"，会把父代刚改好的正确代码误判为编造（早期实测首次通过率 1/5 → 5/6）。
安全边界由 AST 门禁（只允许目标函数体变化）+ `stray` 检查（不许越界写文件）保证，SEARCH 的来源宽松不影响合规。
