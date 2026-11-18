"""端到端：FakeLLM + 基线替身 + 命令评估器（伪物理）把循环跑若干轮，验证 logs/ 产物、预算刹车、代理筛选与去重。"""
import json
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from harness import loop as L  # noqa: E402
from tests.fake_llm import ABS_THRESH, BAD_SEARCH, FIXED_SEARCH, FakeLLM, GOOD, GOOD2, NO_CHANGE, SIGNATURE, SYNTAX_ERR, TWO, hinge, varied  # noqa: E402


def make_harness(tmp_path, script, max_evals=3, baseline="prob", evaluator="command", proxy=True, k=1, fail_rate=0.0):
    cfg = L.load_config(WS / "config" / "harness.yaml")
    cfg["baseline"] = baseline
    cfg["budget"]["max_evaluations"] = max_evals
    cfg["budget"]["max_cost_cny"] = 5.0
    probe = max(1, max_evals // 2)
    cfg["search"]["phase_plan"] = {"probe": probe, "structural": max(0, max_evals - probe)}
    cfg["search"]["propose_k"] = k
    cfg["evaluator"]["mode"] = evaluator
    cfg["evaluator"]["mock_fail_rate"] = fail_rate
    cfg["proxy"]["enabled"] = proxy
    # 测试用更短的算例（替身生成器按名字查规格；这里用小图加速）
    h = L.Harness(cfg, llm_mode="replay", run_dir=tmp_path / "run")
    h.llm = FakeLLM(script)
    return h


def test_e2e_happy_path_command_evaluator(tmp_path):
    h = make_harness(tmp_path, varied(3), max_evals=2)
    best = h.run()
    assert best is not None and best.status == "evaluated"
    assert h.budget.state.evaluations == 2
    run = tmp_path / "run"
    assert (run / "candidates" / "lineage.jsonl").exists() and (run / "lineage_tree.txt").exists()
    meta = json.loads((run / "candidates" / best.id / "meta.json").read_text(encoding="utf-8"))
    assert meta["hypothesis"].startswith("Balance hinge") and meta["analysis"]
    assert meta["proxy"].get("cut") is not None and meta["seed_std"] is not None
    ev = json.loads((run / "candidates" / best.id / "eval.json").read_text(encoding="utf-8"))
    assert ev["partition_hash"] and ev.get("proxy")
    assert (run / "candidates" / best.id / "partitions.json").exists()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["code_version"]["baseline_targets_sha256"] and summary["case_aliases"]
    out = WS / "algorithms" / "prob"
    assert (out / "candidate.patch").exists() and (out / "loss.py").exists()
    assert "def compute_loss(p, graph, feats)" in (out / "loss.py").read_text(encoding="utf-8")
    assert h.budget.state.llm_calls >= 2 and h.budget.state.cost_cny > 0
    # 画像与历史里不出现算例名
    prompt = [c for c in h.llm.calls if c[0] == "propose"][-1][1][1]["content"]
    assert "case_small" not in prompt and "C1" in prompt


def test_e2e_fix_path_and_gate(tmp_path):
    # r1: SEARCH 不匹配 -> precheck -> fix 成功 -> 评估
    # r2: 签名改动 -> gate -> fix 仍改签名 -> gate_failed（不耗评估）
    # r3: 语法错 -> fix 给绝对阈值 -> gate_failed
    # r4: 正常
    script = [BAD_SEARCH, FIXED_SEARCH, SIGNATURE, SIGNATURE, SYNTAX_ERR, ABS_THRESH, GOOD2]
    h = make_harness(tmp_path, script, max_evals=2)
    h.run()
    metas = h.store.all()
    st = [m.status for m in metas]
    assert st[0] == "evaluated" and "5.0 * torch.clamp" in (h.store.dir(metas[0].id) / "candidate.patch").read_text()
    assert st[1] == "gate_failed" and "signature" in metas[1].gate_report
    assert st[2] == "gate_failed"
    assert st[3] == "evaluated"
    assert h.budget.state.evaluations == 2
    assert [c[0] for c in h.llm.calls].count("fix") == 3


def test_e2e_multi_candidate_screening_and_duplicate(tmp_path):
    # k=2：一轮两个候选，只提交一个，另一个 screened；下一轮重复同一 patch -> 指纹去重（gate 层）；再给一个无变化 -> gate_failed
    script = [TWO, TWO, NO_CHANGE + NO_CHANGE.replace("CANDIDATE 1", "CANDIDATE 2")]
    h = make_harness(tmp_path, script, max_evals=2, k=2)
    h.selector.phase_plan = {"probe": 2}          # 两轮都从基线出发，第二轮的 patch 才会与第一轮完全相同
    h.run()
    metas = h.store.all()
    st = [m.status for m in metas]
    assert st[:2].count("evaluated") == 1 and st[:2].count("screened") == 1
    # 第二轮：两个 patch 都与第一轮相同 -> 已评估过的那个被判 identical patch（修复后仍无变化 -> gate_failed），
    # 第一轮 screened 过的那个这次可提交
    assert st[2] == "gate_failed" and st[3] == "evaluated"
    log = (tmp_path / "run" / "harness.log").read_text(encoding="utf-8")
    assert "identical patch already evaluated" in log
    assert h.budget.state.evaluations == 2
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_screened"] >= 1


def test_e2e_invalid_results_not_selected(tmp_path):
    h = make_harness(tmp_path, varied(3), max_evals=2, evaluator="mock", proxy=False, fail_rate=1.0)
    best = h.run()
    assert best is None
    assert all(m.status == "invalid" for m in h.store.all() if m.status != "gate_failed")
    assert not (tmp_path / "run" / "work" / "c001" / "proxy_result.json").exists()


def test_e2e_embed_baseline(tmp_path):
    dec = ("CANDIDATE 1\nHYPOTHESIS: Decoder balance target tightened to 49.5-50.5 so repair moves fewer nodes.\n"
           "EXPECTED: repair flips down.\nKEYWORDS: decode\n<<<EDIT file=decode.py\n<<<SEARCH\n"
           "        if 0.49 <= r <= 0.51:\n===\n        if 0.495 <= r <= 0.505:\n>>>\n>>>\n")
    dec2 = dec.replace("        if 0.49 <= r <= 0.51:\n===\n        if 0.495 <= r <= 0.505:", "        if 0.495 <= r <= 0.505:\n===\n        if 0.498 <= r <= 0.502:")
    h = make_harness(tmp_path, [dec, dec2], max_evals=2, baseline="embed")
    best = h.run()
    assert best is not None and best.files_changed == ["decode.py"]
    out = WS / "algorithms" / "embed"
    assert (out / "decode.py").exists() and (out / "loss.py").exists()


def test_budget_stop_on_cost(tmp_path):
    h = make_harness(tmp_path, varied(5), max_evals=5)
    h.budget.max_cost_cny = 0.012   # 两三次调用就触顶
    h.run()
    assert h.budget.state.evaluations < 5
    assert (tmp_path / "run" / "summary.json").exists()


def test_resume_run_dir(tmp_path):
    h = make_harness(tmp_path, varied(2), max_evals=1)
    h.run()
    n1 = len(h.store.all())
    h2 = make_harness(tmp_path, [hinge(33.0)], max_evals=2)
    h2.run()
    assert len(h2.store.all()) > n1 and h2.budget.state.evaluations == 2
    assert h2.store.all()[0].id == "c001"
