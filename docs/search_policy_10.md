# 搜索策略 —— 10 次评估 / 基线的硬预算下怎么花（赛题 2）

> 依据：官方 `README.md`（每基线 **10 次训练评估提交** + 15 元）、`contest2/budget.py`（`evaluation_limit=10`）、`task.yaml`（`search_rounds_template: 30` 明示"模板不是执行授权"）以及 `manifests/common-timing-reference.json` 的参考值结构。
> 两份基线**额度独立**（`binary_loss` 和 `loss_decoder` 各有 10 次 + 15 元），所以本文给两套分配。

---

## 0. 先算清楚"一次评估值多少钱"

- 1 次评估 = **9 组**（bp_fe-ng45 / bp-ng45 / ariane133-ng45 × 种子 1001/1002/1003），一次提交、一次计费。
- 单组的代价：`100 × (0.4·ΔWNS + 0.6·ΔTNS)`，下限 −100。
- **一次被拒的提交（接口/范围/语法错）也占 1 次额度且不产生分数** → 本地静态预检是**唯一**能保住额度的免费手段。
- 官方 `validate_program` 在我们手上（`task2/official_kit/local_tasks/`），本地运行不耗额度、不调模型。

**结论 1：任何候选在提交前必须过官方校验器。** 这不是优化，是硬要求。

## 1. 参考值的结构决定"往哪打"

官方共同参考值（`common-timing-reference.json`，α = 1.854；两基线共用分母）：

| 算例 | 划分 | W̄ (ns) | T̄ (ns) |
|---|---|---|---|
| bp_fe-ng45 | train | 0.900 | 716.9 |
| bp-ng45 | train | 8.663 | 6622.8 |
| ariane133-ng45 | train | 0.529 | 96.4 |
| swerv_wrapper-ng45 | validation | 2.482 | 1978.9 |
| bp_fe-asap7 | validation | 0.450 | 11.6 |

评分是"**差值 ÷ 本算例均值**"，所以**每个算例的权重相同，与绝对量级无关**。这意味着：

- `bp-ng45` 的 TNS 有 6622 ns，要改善 1% 需要减 66 ns；`bp_fe-asap7` 的 TNS 只有 11.6 ns，减 **0.12 ns** 就是 1%。→ **小分母算例（ariane133、bp_fe-asap7、bp_fe-ng45）的 1 个百分点"更便宜"**，但它们的参考值本身很小、噪声占比大，改一个机制可能把 0.5 ns 变成 −0.5 ns（±10%）。
- 因此：**不是"抓大算例"，而是"在所有算例上都不退化"**。任何一个算例掉到 −100（物理失败）就要用另外几个算例的 100 分去填。
- 9 组里有 1 组物理失败 → 该算例只能拿 (−100)/3 + 其他两组… → 单个算例最多丢 −33.3 分（按算例等权）；若该算例 3 组全失败 → 该算例 −100 → 总分被拉低 20 分（5 算例等权）。

**结论 2：稳健性 = 分数。** 验证集（2 个算例）在最终候选后运行，共 6 组；隐藏 2 算例。所以**训练集 3 个算例的 +x 必须能迁移**。

## 2. 两套基线的机制菜单（来自官方 prompt.md 的"允许"清单，不是我们编的）

### 2.1 binary_loss（只有 12 个参数，只能改损失）

官方 prompt.md 明确允许：`改变损失表达式、加有界正则、强调关键网、对每网惩罚做非线性、用 epoch 做调度`；明确**不能**碰 assignment / embedding / logits / temperature / optimizer。

输入里**现成可以用**但基线没用到的：
- `net_cut_probabilities` [H] + `net_weights` [H]：**逐网割概率与冻结的时序权重**——基线只用了聚合的 `expected_cut`。这是最直接的可乘项（`weighted_cut = Σ w_e · cut_e / Σ w_e`）。
- `node_features` [N,10]：**10 列时序/结构特征**（worst slack、加权平均 slack、max slew、max arrival、min required、cell delay、coverage、hops、1-hop 度、2-hop 度）——基线完全没用。
- `epoch / max_epochs`：可做调度。
- `probabilities`：可用**面积加权分布**做更细的平衡项（不只是均值）。

### 2.2 loss_decoder（损失 + 解码）

- 损失侧：`positive_scores / negative_scores / embeddings / graph_scores / graph_weights / net_weights / node_features / node_areas`。基线损失 = `−(F.logsigmoid(pos).sum() + F.logsigmoid(−neg).sum())/batch`，**完全没用 graph_scores / net_weights / node_features / node_areas**——这些是官方为"关键性"和"结构"准备的。
- 解码侧：**不能对数组元素赋值**、**没有 np.zeros**、**只能一个 `for ... in range(max_iterations)`**、**不能调 np.linalg**。可用：`argsort / cumsum / where / take / concatenate / stack / lexsort / clip / maximum / minimum / reshape / sum / mean / sqrt / exp / log / tanh / square / abs / all / any / argmax / argmin`。参考助手 `reference_global_two_way` 可调一次（5 个参数原样）。

## 2.5 **实测修正**：割项在训练后期失效（见 `local_findings_saturation.md`）

在官方固定配方下（割权重 1 / 平衡权重 200 / 二值权重 0→10 / 训练 400 步 / 温度 1→0.1），本地替身实测：

- 概率饱和到 **99.7%**（p<0.01 或 p>0.99）；
- 饱和后 `cut_e` 对 p 的梯度 ≈ 0 → **只改割项，切分 0.0% 变化**；
- 只改平衡项（加 ±1% 死区 hinge），切分 **48%** 变化。

所以下面的原始列表要按这个优先级重排：**先打平衡/二值，再打 epoch 调度，最后才是割项**。

## 3. binary_loss：10 次怎么花

| # | 目的 | 候选内容 | 为什么先/后 |
|---|---|---|---|
| 1 | 基线确认 + 建画像 | **原样提交** `initial_program.py`（不加任何改动） | 官方参考已给，但**本机实测基线**能校准"我们的容器与官方一致"；万一不一致，前十次全白做。**必须第 1 次花** |
| 2 | 探针 B1 | **平衡 hinge** 死区 ±1%（实测唯一能显著改变切分的改动） | 本地已证实 48% 切分变化 |
| 3 | 探针 B2 | 平衡项换**面积加权方差**（基线只有均值偏差） | 与 B1 不同轴；对"面积均匀性"更敏感 |
| 4 | 探针 C | **epoch 调度**：割项权重从 2 线性降到 0.2、平衡项恒定 | 唯一能让割项在饱和前起作用的写法 |
| 5 | 结构 1 | 把 2 有效机制合并（A + B） | 基于 2/3 的结论 |
| 6 | 结构 2 | 用 node_features 的 coverage 做"未知时序"门控 + 割项在**早期**加权（饱和前） | coverage=0 = 未知是官方点名的坑；割项要趁早 |
| 7 | 结构 3 | 在 5 的基础上加二值间隔的非线性（如 `margin^4`） | 二值化质量影响阈值化结果 |
| 8 | 组合 | 取历史最优 + 1 个新机制 | 交叉验证 |
| 9 | 微调 | 最优候选的一个系数 ×0.5 / ×2 | 低成本确认方向 |
| 10 | 收尾 | 最优候选的保守变体（或原样再提交确认） | **保底**：必须让最终返回的候选在 10 次里有被验证过 |

**关键纪律**：
- **每个候选先在本地跑替身适配层**（`baseline_stub/common/official_adapter.py`），要求 `Hamming(candidate, baseline) > 1%`；0% 变化的候选**不要提交**（那是一次纯浪费）。
- 1 次评估若返回 `rejected`（接口/范围），**立刻用官方 `validate_program` 定位**，不要重试同一候选。
- 物理失败（−100 组）说明候选"能跑但不合法"，**不要把这种候选当父代**——把失败原因写进 MEMORY，下一轮避开。
- **不要用第 1 次评估跑新候选**。

## 4. loss_decoder：10 次怎么花

| # | 目的 | 候选内容 |
|---|---|---|
| 1 | 基线确认 | 原样 `initial_program.py`（两函数不变） |
| 2 | 探针 D1 | **只改解码器**：`argsort(投影)` + `cumsum` 平衡切分（不用参考助手），向量化、O(N log N) |
| 3 | 探针 D2 | **只改解码器**：两中心距离（`embeddings` 的两列做投影） |
| 4 | 探针 L1 | **只改损失**：把 `graph_scores × graph_weights` 作为额外正样本项（官方准备但没用） |
| 5 | 探针 L2 | **只改损失**：用 `net_weights` 给采样对加权（关键网上的正样本更重） |
| 6 | 组合 D1+L1 | 若 D1、L1 都正 |
| 7 | 组合 D2+L2 | 若 D2、L2 都正 |
| 8 | 结构 | 在最优组合上加 `node_areas` 的体量归一化（防大组主导） |
| 9 | 微调 | 系数 ×0.5 / ×2 |
| 10 | 收尾 | 保守变体 / 原样确认 |

**关键纪律**：
- 解码器最容易**物理失败**（空层、不平衡）→ 每次提交前在本地用合成 embeddings/areas 跑一遍 `decode_tp_gnn_embeddings`，检查 `labels` 是 0/1、两层非空。
- 官方"共同参考值"使两基线均分为 0，所以**任何正分就是净收益**，不需要追极大值。

## 5. 父代选择（10 次预算下的极简规则）

官方历史只给 `score`（单个数字，9 组平均后的总分）。没有按算例的信息（除非反馈里带 `cells`）。所以：

```
best = argmax(history 中 status==succeeded 且 score 最大者)
若本轮候选 score > best.score: 下一轮父代 = 本轮候选
否则: 下一轮父代 = best（回到最优）
```

**不用精英集**（10 次样本太少，方差比可解释性更重要）。但要在 MEMORY 里记录**每个候选的机制标签**，避免重复。

## 6. 失败恢复

| 情况 | 动作 |
|---|---|
| `status == "rejected"` | 本地 `validate_program` 跑一遍，把错误文本写进 MEMORY，下一轮避开同一写法 |
| `status == "failed"`（物理失败） | 读 `feedback.json` / `artifacts_dir` 里的失败阶段，写进 MEMORY；**不把该候选当父代** |
| `status == "uncertain" / "reconciling"` | `wait_evaluation` 会继续等；**绝不重复提交** |
| 模型调用失败（`retryable: false`） | 立即停止，返回当前最优（`RuntimeError`），不要再烧钱 |
| 预算耗尽 | 返回 `best`（必须 ≥1 次 succeeded） |

## 7. "离线知识 ≤200 单位"下的策略

- 初始提示词只放一段 **≤200 计数单位**的通用经验（纯自然语言），内容限定在"方向选择"（比如：先看加权割、注意 coverage 为零、平衡修复会被翻转、小分母算例对绝对变化敏感）。
- **真正的机制细节写进 `MEMORY.md`**——由 Agent 在每次评估后**自己**总结（"运行中自主形成的经验"被明确允许）。
- MEMORY 的模板：
  ```
  [eval #k] 候选：<一句话机制>
    score: <n>, per-case: <若反馈有>
    结果: <成立/不成立/失败>
    结论: <下一轮该做什么>
  ```

## 8. 与旧 harness 的关系

旧 harness（`workspace/harness/`）**不能提交**（官方不接受自建评估器）。但其中三样东西要**移植进 Agent**：
1. `gate.py` 的 AST 边界检查 → 本地 `precheck(program)`（等价于官方 `validate_program`，但更快、更早）；
2. `selector.py` 的 `best()` 逻辑 → Agent 里的 `pick_best()`；
3. `analyzer.py` 的"规则摘要" → 把 `feedback.json` 压成一句话写进 MEMORY。

`loop.py` / `budget.py` / `evaluator.py` **删除**（`contest2/` 已经提供了账本、调度、评分）。
