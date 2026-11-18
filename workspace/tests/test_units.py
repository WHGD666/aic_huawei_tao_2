"""单元测试：计分、门禁（AST 边界 / 禁用符号 / 单位 / 冒烟）、提案解析、选择器、代理去重。"""
import math
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from harness.gate import Gate, Target  # noqa: E402
from harness.proposer import parse_proposals  # noqa: E402
from harness.scoring import RunMetrics, candidate_score, improvement  # noqa: E402
from harness.evaluator import EvalResult, MockEvaluator, parse_result_json  # noqa: E402
from harness.analyzer import proxy_correlation, spearman  # noqa: E402
from tests.fake_llm import ABS_THRESH, ARGMAX, GOOD, GOOD2, IMPORT, SIGNATURE, SYNTAX_ERR, TWO, hinge  # noqa: E402

PROB = WS / "baseline_stub" / "prob"
EMBED = WS / "baseline_stub" / "embed"


# ---------------- scoring ----------------
def _runs(vals):
    return {f"c{i}/{s}": RunMetrics(f"c{i}", s, w, t) for i, per in enumerate(vals) for s, (w, t) in zip((1001, 1002, 1003), per)}


def test_improvement_and_zero_baseline():
    assert improvement(1.0, 0.5) == pytest.approx(0.5)
    assert improvement(1.0, 1.5) == pytest.approx(-0.5)
    assert improvement(0.0, 0.0) == 0.0 and improvement(0.0, 0.1) == -1.0


def test_candidate_score_shape():
    base = _runs([[(-1, -10)] * 3, [(-2, -20)] * 3])
    cand = _runs([[(-0.5, -5)] * 3, [(-2, -20), (-2.2, -22), (-1.8, -18)]])
    s = candidate_score(base, cand)
    assert s.all_valid and s.per_case["c0"] == pytest.approx(0.5) and s.per_case["c1"] == pytest.approx(0.0)
    assert s.total == pytest.approx(0.5) and s.robust_min == pytest.approx(0.0)
    assert s.seed_std_mean > 0 and s.tns_mean == pytest.approx(0.25)
    assert s.robust("mean_minus_std", kappa_seed=1.0) < s.robust_mean_minus_std


def test_any_invalid_run_sinks_candidate():
    base = _runs([[(-1, -10)] * 3])
    cand = _runs([[(-0.5, -5)] * 3])
    cand["c0/1002"].valid = False
    s = candidate_score(base, cand)
    assert not s.all_valid and s.total == -math.inf


def test_parse_result_json_nonfinite_is_invalid():
    d = {"runs": {"a/1": {"wns": float("nan"), "tns": -1.0}, "a/2": {"wns": -1, "tns": -2, "stats": {"cut": 3}}}}
    r = parse_result_json("x", d, ["a"], [1, 2, 3])
    assert not r.runs["a/1"].valid and r.runs["a/2"].valid and r.runs["a/2"].stats["cut"] == 3
    assert not r.runs["a/3"].valid and "missing" in r.runs["a/3"].fail_reason


def test_partition_hash_symmetric():
    a = EvalResult("a", {}, {"c/1": [0, 1, 1, 0]})
    b = EvalResult("b", {}, {"c/1": [1, 0, 0, 1]})
    c = EvalResult("c", {}, {"c/1": [0, 0, 1, 1]})
    assert a.partition_hash() == b.partition_hash() != c.partition_hash()


# ---------------- proposer ----------------
def test_parse_multi_candidates():
    props = parse_proposals(TWO, 3)
    assert len(props) == 2 and props[0].hypothesis.startswith("Balance hinge") and props[1].edits[0].file == "loss.py"
    assert not props[0].parse_errors and not props[1].parse_errors


def test_parse_single_without_marker():
    txt = GOOD.replace("CANDIDATE 1\n", "")
    props = parse_proposals(txt, 2)
    assert len(props) == 1 and props[0].edits


# ---------------- gate ----------------
def _gate(kind="prob", smoke=True):
    if kind == "prob":
        return Gate([Target("loss.py", "compute_loss", "loss")], PROB, case_names=["case_small", "case_mid", "case_large"], smoke_enabled=smoke)
    return Gate([Target("loss.py", "compute_loss", "loss"), Target("decode.py", "decode_partition", "decode")], EMBED,
                case_names=["case_small"], smoke_enabled=smoke)


def _apply(script: str, root: Path):
    from harness.codebase import apply_edits
    from harness.proposer import parse_proposals
    import tempfile
    p = parse_proposals(script, 1)[0]
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        changed, errs = apply_edits(root, work, p.edits, fallback_root=root)
        assert not errs, errs
        return {f: (work / f).read_text(encoding="utf-8") for f in changed}


def _base(root: Path, files):
    return {f: (root / f).read_text(encoding="utf-8") for f in files}


def test_gate_accepts_good_candidates():
    g = _gate()
    for s in (GOOD, GOOD2):
        files = _apply(s, PROB)
        rep = g.check(files, _base(PROB, files))
        assert rep.ok, rep.text()
        assert rep.smoke and rep.smoke["loss_tiny_gnorm"] > 0


def test_gate_rejects_signature_change():
    g = _gate(smoke=False)
    files = _apply(SIGNATURE, PROB)
    rep = g.check(files, _base(PROB, files))
    assert not rep.ok and "signature" in rep.text()


def test_gate_rejects_import_inside_function():
    g = _gate(smoke=False)
    files = _apply(IMPORT, PROB)
    rep = g.check(files, _base(PROB, files))
    assert not rep.ok and "import" in rep.text()


def test_gate_rejects_module_level_change():
    g = _gate(smoke=False)
    src = (PROB / "loss.py").read_text(encoding="utf-8")
    files = {"loss.py": src.replace("import torch\n", "import torch\nALPHA = 2.0\n")}
    rep = g.check(files, _base(PROB, files))
    assert not rep.ok and "outside the body" in rep.text()


def test_gate_rejects_grad_breakers_and_abs_threshold():
    g = _gate(smoke=False)
    files = _apply(ARGMAX, PROB)
    rep = g.check(files, _base(PROB, files))
    assert not rep.ok and "autograd" in rep.text()
    files = _apply(ABS_THRESH, PROB)
    rep = g.check(files, _base(PROB, files))
    assert not rep.ok and "absolute threshold" in rep.text()


def test_gate_rejects_syntax_error_and_circuit_name():
    g = _gate(smoke=False)
    files = _apply(SYNTAX_ERR, PROB)
    assert not g.check(files, _base(PROB, files)).ok
    src = (PROB / "loss.py").read_text(encoding="utf-8").replace("    l_bin = (p * (1.0 - p)).mean()", "    tag = 'case_mid'\n    l_bin = (p * (1.0 - p)).mean()")
    rep = g.check({"loss.py": src}, _base(PROB, ["loss.py"]))
    assert not rep.ok and "circuit name" in rep.text()


def test_gate_smoke_catches_constant_loss():
    g = _gate()
    src = (PROB / "loss.py").read_text(encoding="utf-8").replace("    return l_cut + 10.0 * l_bal + 1.0 * l_bin", "    return (area * 0.0).sum() + 1.0")
    rep = g.check({"loss.py": src}, _base(PROB, ["loss.py"]))
    assert not rep.ok and ("does not require grad" in rep.text() or "identically zero" in rep.text())


def test_gate_embed_decode_rules():
    g = _gate("embed")
    files = _base(EMBED, ["loss.py", "decode.py"])
    rep = g.check({"decode.py": files["decode.py"]}, files)
    # 基线解码本身 == 基线 → "no files changed"不在这里判，gate 只看边界/符号/冒烟
    assert rep.ok, rep.text()
    bad = files["decode.py"].replace("    rng = np.random.default_rng(int(seed))", "    rng = np.random.default_rng()")
    rep = g.check({"decode.py": bad}, files)
    assert not rep.ok and "seeded" in rep.text()
    bad2 = files["decode.py"].replace("    return part", "    return np.ones(n, dtype=np.int64)")
    rep = g.check({"decode.py": bad2}, files)
    assert not rep.ok and "same tier" in rep.text()


# ---------------- analyzer ----------------
def test_spearman_and_proxy_correlation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    c = proxy_correlation([({"cut": 10, "x": 1}, 0.1), ({"cut": 5, "x": 2}, 0.3), ({"cut": 1}, 0.5)])
    assert c["cut"][0] == pytest.approx(-1.0) and "x" not in c


# ---------------- mock evaluator determinism ----------------
def test_mock_evaluator_deterministic(tmp_path):
    (tmp_path / "loss.py").write_text("x = 1\n", encoding="utf-8")
    ev = MockEvaluator(["case_small"], [1001, 1002], fail_rate=0.0)
    a, b = ev.evaluate("c1", tmp_path), ev.evaluate("c1", tmp_path)
    assert {k: (r.wns, r.tns) for k, r in a.runs.items()} == {k: (r.wns, r.tns) for k, r in b.runs.items()}
