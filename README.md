# aic_huawei_tao_2 —— "韬"智能杯 EDA 创新大赛 · 赛题 2

面向 3D 芯片网表切分的优化目标演化：实现一个 **预算感知的 Agent Harness**，让 DeepSeek-V4.1-Flash 在 30 次候选评估 / 15 元 API 预算内，迭代改进 GNN 网表切分的**损失函数**（概率预测基线）或**损失 + 聚类解码函数**（表示学习基线），以固定物理流程后的 WNS/TNS 改善比例计分（7 个算例等权，仅初赛，无答辩）。

| 入口 | 说明 |
|---|---|
| [`PLAN.md`](PLAN.md) | **总控文档**：题目拆解、本题结构特点、得分策略、harness 设计、里程碑、风险、进度 |
| [`docs/hypothesis_library.md`](docs/hypothesis_library.md) | 两种基线的演化假设库（公式 / 伪代码 / 代理指标） |
| [`docs/open_questions.md`](docs/open_questions.md) | 待组委会确认事项 + 基线 A/B 决策清单 |
| [`docs/harness_design.md`](docs/harness_design.md) | harness 模块设计、评估接口协议、候选状态机、消融开关 |
| [`docs/submission_checklist.md`](docs/submission_checklist.md) | 提交清单 |
| `2/` | 赛题原文（md / pdf / zip） |
| [`workspace/`](workspace/) | harness 代码 + 基线替身 + 测试（README 讲怎么跑） |

## 协作方式

- 本地工作在 `F:\huawei_aic_tao\task2`，推送用根目录 `git_sync.bat "说明"`（会先检查 `.env` 没被跟踪）
- `workspace/.env` 保存 API key，永不入库
- 环境：Python ≥ 3.10 + `workspace/requirements-harness.txt`；有 torch 用 torch，没有时自动退化到 numpy 自动微分替身；所有环境与缓存落 F 盘，C 盘零写入
