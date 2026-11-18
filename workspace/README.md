# workspace —— 面向 3D 芯片网表切分的优化目标演化 harness

> 总控规划见上一级 `../PLAN.md`；模块设计见 `../docs/harness_design.md`；本题的机制假设库见 `../docs/hypothesis_library.md`。本文件只讲"怎么跑、产物是什么"。

## 目录

```
workspace/
├── run.sh / run.bat            统一启动入口（提交要求）
├── .env                        DEEPSEEK_API_KEY / BASE_URL / MODEL（gitignore，不入库）
├── requirements-harness.txt    harness 依赖（requests / pyyaml / numpy / pytest；torch 可选）
├── config/
│   ├── harness.yaml            基线选择、演化目标（文件+函数）、预算、搜索、评估命令、门禁、消融开关
│   └── pricing.yaml            token 单价（元/百万）→ 本地影子计费
├── harness/                    演化循环（模块说明见 ../docs/harness_design.md）
│   ├── loop.py                 主循环：提出→生成→检查→代理筛选→评估→分析→选择
│   ├── proposer.py             提示词组织、多候选协议解析、SEARCH 预检
│   ├── gate.py                 函数级门禁：AST 边界 / 禁用符号 / 单位陷阱 / 冒烟
│   ├── smoke.py                合成小图上的可微性 / 可行性冒烟
│   ├── evaluator.py            CommandEvaluator（外部命令 + JSON 协议）/ ProxyRunner（只切分）/ Mock
│   ├── scoring.py              题面计分 + 稳健指标 + 种子方差
│   ├── selector.py             精英集 / 父代采样 / 血统淘汰 / 阶段 / 最终候选
│   ├── analyzer.py             规则摘要 + 代理↔真实秩相关 + LLM 解读提示
│   ├── profile.py              算例匿名化（C1..Cn）/ 基线画像 / 候选差值 / 代理向量
│   ├── store.py                候选历史：meta.json / candidate.patch / eval.json / partitions.json / llm/
│   ├── codebase.py             目标文件展示、SEARCH/REPLACE 应用、纯 Python unified-diff
│   ├── budget.py               评估次数 / token / 费用的影子计量（budget.json 可续跑）
│   ├── llm.py                  DeepSeek 客户端（thinking 关闭、缓存命中记账、record/replay）
│   └── prompts/                system / knowledge / phase_{probe,structural,combine,final} / propose / fix
├── baseline_stub/              本地替身基线（官方任务包到手后替换 baseline/ 与 config 的 targets）
│   ├── common/                 合成超图 + 2D 时序特征、numpy 自动微分、固定 GNN/Adam、修复程序与平衡检查
│   ├── prob/                   概率预测基线：loss.py（可演化）+ train.py（固定训练器）
│   ├── embed/                  表示学习基线：loss.py + decode.py（可演化）+ train.py
│   └── evaluate.py             模拟官方评估接口：跑 3 算例 × 3 种子切分 +（可选）伪物理流程 → JSON
├── baseline/                   官方初始算法（任务包到手后放这里，config 里改 root）
├── algorithms/                 最终导出的函数文件 + candidate.patch + SELECTED.json（循环结束自动写）
├── logs/                       每次运行：candidates/、lineage.jsonl、lineage_tree.txt、budget.json、summary.json
│   └── rehearsal/R30_prob/     30 次评估彩排样例（数字见 ../docs/rehearsal_report_R30_prob.md）
├── tools/                      report.py（运行目录→技术报告素材）/ probe_models / calibrate_pricing
└── tests/                      单元 + 端到端（FakeLLM + 替身，不联网、零费用）
```

## 首次使用

```bash
cd workspace
python -m pip install -r requirements-harness.txt     # 或用已有环境；torch 不必装
cp .env.example .env 2>/dev/null || true              # 填 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
python tests/_runner_nopytest.py tests.test_units     # 17 项单元测试（约 10 秒）
python tests/_runner_nopytest.py tests.test_e2e       # 7 项端到端（约 3 分钟）
```

无 torch 环境下 `baseline_stub/common/backend.py` 会自动把 `import torch` 解析到内部 numpy 自动微分替身（`tiny_torch.py`），候选函数源码在两种环境下逐字一致。

## 跑演化循环

```bash
# 官方预算口径（题面：读取任务与预算）
./run.sh prob  --max-evaluations 30 --max-cost 15

# 本地替身 + 真实模型，几毛钱一次完整彩排
./run.sh prob  --llm-mode record --max-evaluations 30 --run-dir logs/my_run
./run.sh embed --llm-mode record --max-evaluations 30 --run-dir logs/my_embed

# 零费用回放已录制的响应（调试循环）
./run.sh prob --llm-mode replay --max-evaluations 10 --run-dir logs/my_run

# 只跑 rounds 上限，然后续跑（同一 --run-dir 自动接着来）
./run.sh prob --max-evaluations 30 --max-rounds 5 --run-dir logs/my_run

# 关闭本地代理层（若官方不允许本地跑切分）
./run.sh prob --no-proxy

# 运行目录 → 技术报告素材
python tools/report.py logs/my_run -o logs/my_run/report.md
```

Windows：`run.bat prob --max-evaluations 30`。

## 接入官方评估接口

只改 `config/harness.yaml`：

```yaml
baselines:
  prob:
    root: baseline/prob                      # 官方初始算法目录
    targets:
      - {file: loss.py, func: compute_loss, kind: loss}
    eval_cmd:  "<官方命令> --candidate {workdir} --out {out}"    # {workdir} {out} {ref} {cases} {seeds}
    proxy_cmd: "<官方训练脚本，只跑切分>"                          # 可为空 → 关闭代理筛选
evaluator:
  mode: command
  cases: [<三个训练算例名>]
  seeds: [1001, 1002, 1003]
  baseline_costs_evaluation: false        # 官方若不提供基线参考值且跑基线计次，置 true
```

返回 JSON 协议见 `../docs/harness_design.md` §3。若官方接口只留下每个 `<case>/<seed>` 的目录，写一个薄适配脚本转成该 JSON 即可。

## 产物与提交对应关系

| 提交要求 | 来源 |
|---|---|
| `harness/` + `run.sh` + 提示模板 + 配置 | 本目录 `harness/`、`run.sh`、`config/` |
| 算法代码（损失 / 解码函数） | 循环结束写入 `algorithms/<baseline>/*.py`（签名与任务包一致）+ `candidate.patch` + `SELECTED.json` |
| `logs/`（父子关系 / 代码版本 / 评估结果 / 失败原因） | `logs/run_*/candidates/*/meta.json`、`lineage.jsonl`、`lineage_tree.txt`、`eval.json`、`llm/*.json`、`summary.json` |
| 切分方案 / 物理结果 | `logs/run_*/candidates/<id>/partitions.json`（切分）+ 官方评估接口产出的流程结果目录；按"算例/种子"组织后放 `results/` |
