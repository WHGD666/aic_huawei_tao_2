# Harness 设计 —— 模块与数据流

> 代码在 `workspace/harness/`；怎么跑见 `workspace/README.md`；策略动机见 `PLAN.md` §3、§5。

## 1. 设计原则

1. **演化对象是函数，不是文件。** 白名单是 `(文件, 函数名, 种类)` 三元组；门禁用 AST 比较保证"除目标函数体外逐字不变"。
2. **免费检查做到极致，再花昂贵评估。** 候选在提交前经过：语法 → AST 边界 → 禁用符号 → 单位陷阱 → 合成图可微冒烟 → 真实图代理运行 → 指纹去重 → 微小改动弃权。每一层都免费。
3. **代理指标是线索不是裁判。** 本地切分统计只用于过滤与给 LLM 提供方向（Spearman 秩相关），排名永远用真实评估。
4. **稳健 > 峰值。** 7 个计分算例里 4 个看不见；选择指标是 mean − σ(算例) − κ·σ(种子)，最终候选优先"三算例全非负"。
5. **可追溯。** 每个候选在拿到 id 的瞬间落 `meta.json` + 血统事件；评估与计费严格配对；崩溃也有记录。
6. **不特判。** 画像与历史里算例匿名为 C1..C3；gate 拒绝算例名 / 工艺名 / 规模分支字面量。

## 2. 模块

| 模块 | 职责 |
|---|---|
| `loop.py` | 主循环与 CLI；每轮：k 候选 → gate → 代理 → 去重 → 提交 1 个 → 分析 → 选择；`finalize` 导出 algorithms/ 与 summary.json |
| `proposer.py` | 提示词组织（system + knowledge + phase 指引 + 画像 + 代理相关性 + 历史 + 父代 patch + 当前代码）、多候选协议解析、SEARCH 预检 |
| `gate.py` | AST 边界（签名 / 装饰器 / 模块级语句 / 其它函数不变）、禁用调用、梯度中断操作、算例/工艺字面量、规模分支、时序特征绝对阈值 |
| `smoke.py` | 合成小图（含宏 / 无宏 + ps 单位）上真跑候选：损失 → 标量/有限/requires_grad/backward/梯度非零；解码 → 0/1、确定性、平衡偏差 |
| `evaluator.py` | `CommandEvaluator`（外部命令 + JSON 协议）、`ProxyRunner`（同协议、只切分、按源码 hash 缓存）、`MockEvaluator`（秒级单测） |
| `scoring.py` | 题面计分；mean / min / mean−σ；种子间 σ；TNS 均值 |
| `selector.py` | 精英集、softmax 父代采样、血统淘汰、阶段推进、`final_pick`、历史摘要 |
| `analyzer.py` | 规则摘要（按算例按种子）、Spearman 代理↔真实、LLM 解读提示 |
| `profile.py` | 算例匿名化、基线画像、候选切分统计差值、代理向量 |
| `store.py` | `candidates/<id>/{meta.json, candidate.patch, eval.json, partitions.json, llm/*.json}` + `lineage.jsonl` |
| `budget.py` / `llm.py` | 影子计量（评估次数 / token / 费用）；DeepSeek 客户端（thinking 关闭、缓存命中记账、record/replay） |
| `codebase.py` | 目标文件索引与展示、SEARCH/REPLACE 应用（原子）、diff 生成、patch 指纹 |

## 3. 评估接口协议（`CommandEvaluator`）

命令模板占位：`{workdir}`（含目标文件的目录）`{out}`（结果 JSON 路径）`{ref}`（基线结果 JSON，用于 hamming_ref；无则省略）`{cases}` `{seeds}`。

返回 JSON：
```json
{"runs": {"<case>/<seed>": {"wns": -0.6, "tns": -23.4, "valid": true, "fail_reason": "",
                            "stats": {"cut": 154, "wcut": 460.1, "crit_cross": 0.016, "area_dev": 0.029,
                                      "macro_cnt_dev": 0.0, "macro_area_dev": 0.01, "repair_flips": 112, "hamming_ref": 0.14},
                            "train_s": 0.3, "log": "..."}},
 "partitions": {"<case>/<seed>": [0, 1, ...]}}
```
`stats` 与 `partitions` 可选；缺 `stats` 则代理层只做去重与失败过滤。官方接口到手后，要么直接满足此协议，要么写一个薄适配脚本。

## 4. 候选状态机

`proposed → gate_failed | screened | duplicate | invalid | evaluated | discarded`

- `gate_failed`：本地检查未过（含一次修复后仍失败）；不花评估；不影响父代计数
- `screened`：过了 gate 与代理，但本轮另一个候选被提交，或与基线差异 < 0.5%
- `duplicate`：切分指纹与已评估候选相同；继承其分数；不花评估
- `invalid`：代理运行失败，或官方评估任一 run 无效；永不做父代
- `evaluated`：有真实分数；进入精英集候选

## 5. 消融开关（`config/harness.yaml: ablation`）

`knowledge_block / problem_profile / precheck / fix_round / boost_parent_lines / analysis_in_history / proxy_feedback / llm_analysis`，全 true = 完整 harness。技术报告"实验与创新点"用它们做部件消融。
