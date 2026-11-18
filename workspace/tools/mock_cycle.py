"""
mock_cycle.py -- 在无 GPU、无额度的情况下把 agent 的决策循环跑通。

模拟对象（全部本地）：
  * FakeContext  —— 与 contest2.agent_api.AgentContext 同名的接口，返回伪造 history/额度
  * FakeGateway  —— 用一个确定性的"假模型"响应 context.model.responses_create
                     （首轮返回父代原程序，之后返回一个预先写好的候选，用来测预检/提交/记忆）
  * FakeMatrix   —— 用 task2/baseline_stub 的合成图 + 伪物理给候选打分，模拟 9 组结果

用法:
    python tools/mock_cycle.py --baseline binary_loss --rounds 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
ROOT = WS.parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(ROOT / "official_kit" / "agent_templates"))
sys.path.insert(0, str(ROOT / "official_kit"))

TEMPLATES = ROOT / "official_kit" / "agent_templates"


def _load_agent(baseline: str):
    spec = importlib.util.spec_from_file_location(f"agent_{baseline}", TEMPLATES / baseline / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    # 让 `from common.agent_common import ...` 解析到模板目录
    sys.modules.setdefault("common", types.ModuleType("common"))
    sys.modules["common"].__path__ = [str(TEMPLATES / baseline / "common")]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class FakeGateway:
    """Deterministic stand-in for context.model.responses_create.

    Round 0 returns the parent program unchanged (so the submit is the baseline);
    later rounds return a small textual mutation supplied by the caller.
    """

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = 0

    def responses_create(self, payload):
        self.calls += 1
        if not self.candidates:
            return {"output_text": "no more candidates"}
        text = self.candidates.pop(0)
        return {"output_text": text}


class FakeContext:
    def __init__(self, task, baseline_program: str, gateway: FakeGateway, limit: int = 3):
        self.task = task
        self.baseline_candidate = "baseline"
        self.evaluation_mode = "formal"
        self.files = {}
        self.source_policy = {}
        self.model = gateway
        self._history = []
        self._limit = limit
        self._pending = {}
        self._results = {}
        self._matrix = None

    # ---- organizer-like API ----
    def status(self):
        return {
            "task": self.task,
            "evaluations_remaining": max(0, self._limit - len([h for h in self._history if h["status"] != "rejected"])),
            "model_cost_available_rmb": 15.0,
            "model_dispatch_allowed": True,
            "evaluation_dispatch_allowed": True,
            "model_stop_reasons": [],
            "unresolved_model_calls": 0,
            "dispatch_allowed": True,
            "stop_reasons": [],
            "history": list(self._history),
        }

    def submit_evaluation(self, source, request_id=None):
        req = request_id or f"req{len(self._history)}"
        self._pending[req] = source
        return {"request_id": req, "status": "accepted"}

    def wait_evaluation(self, request_id, poll_seconds=5):
        if request_id not in self._results:
            source = self._pending.get(request_id, "")
            result = self._evaluate(request_id, source)
            self._results[request_id] = result
            self._history.append({"request_id": request_id, "candidate_id": result["candidate_id"],
                                  "status": result["status"], "reason": result.get("reason"),
                                  "score": (result.get("feedback") or {}).get("score")})
        return self._results[request_id]

    # ---- fake evaluation ----
    def set_matrix(self, fn):
        self._matrix = fn

    def _evaluate(self, request_id, source):
        cid = "cand-" + str(abs(hash(source)) % 100000)
        if self._matrix is None:
            score, cells = 0.0, []
        else:
            score, cells = self._matrix(source)
        return {"request_id": request_id, "candidate_id": cid, "status": "succeeded",
                "feedback": {"score": score, "cells": cells}}


_BASELINE_SNAPSHOT = {}


def _make_matrix(baseline: str):
    """用官方签名适配层 + 伪物理给候选打分（无 GPU）。"""
    import tempfile
    from baseline_stub.common.synth import DEFAULT_CASES, make_case
    from baseline_stub.common.official_adapter import run_binary_loss, run_loss_decoder
    from baseline_stub.common.partition_utils import mini_physics

    runner = run_binary_loss if baseline == "binary_loss" else run_loss_decoder

    def matrix(source: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "program.py"
            p.write_text(source, encoding="utf-8")
            # 先用同一 runner 跑基线程序，得到"参考负裕量"，再按官方公式算分
            cells = []
            ref = {}
            # program.py 会被 agent 覆盖，基线必须用"运行开始时的快照"
            if baseline not in _BASELINE_SNAPSHOT:
                _BASELINE_SNAPSHOT[baseline] = (Path(td) / "baseline_snapshot.py")
                _BASELINE_SNAPSHOT[baseline].write_text(
                    (TEMPLATES / baseline / "program.py").read_text(encoding="utf-8"), encoding="utf-8")
            base_p = _BASELINE_SNAPSHOT[baseline]
            for name, spec in DEFAULT_CASES.items():
                g = make_case(name, **spec)
                seed = {"case_small": 1001, "case_mid": 1002, "case_large": 1003}[name]
                try:
                    rb = runner(g, seed, base_p)
                    ref[name] = mini_physics(g, rb["part"], seed)
                except Exception:  # noqa: BLE001
                    ref[name] = (0.0, 0.0)
            total = 0.0
            ncase = 0
            for name, spec in DEFAULT_CASES.items():
                g = make_case(name, **spec)
                seed = {"case_small": 1001, "case_mid": 1002, "case_large": 1003}[name]
                try:
                    r = runner(g, seed, p)
                    w, t = mini_physics(g, r["part"], seed)
                    rbw, rbt = ref[name]
                    dw = (max(0, -rbw) - max(0, -w)) / max(max(0, -rbw), 1e-9)
                    dt = (max(0, -rbt) - max(0, -t)) / max(max(0, -rbt), 1e-9)
                    sc = 100 * (0.4 * dw + 0.6 * dt)
                    total += max(sc, -100); ncase += 1
                    cells.append({"case": name, "seed": seed, "valid": True, "score": max(sc, -100),
                                  "wns_ns": w, "tns_ns": t})
                except Exception as error:  # noqa: BLE001
                    total += -100; ncase += 1
                    cells.append({"case": name, "seed": seed, "valid": False, "reason": str(error)[:160]})
            return (total / max(ncase, 1)), cells

    return matrix


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="binary_loss", choices=["binary_loss", "loss_decoder"])
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)

    mod = _load_agent(a.baseline)
    folder = TEMPLATES / a.baseline
    original = (folder / "program.py").read_text(encoding="utf-8")
    # 演练在临时副本里做，绝不改动交付物
    import shutil, tempfile
    sandbox = Path(tempfile.mkdtemp(prefix="mock_agent_"))
    shutil.copytree(folder, sandbox / a.baseline)
    shutil.copytree(TEMPLATES / "common", sandbox / "common", dirs_exist_ok=True)
    folder = sandbox / a.baseline

    # Build a fake reply stream: unchanged parent first, then a marked variant that
    # still passes the organizer's static validator.
    body = "# EVOLVE-BLOCK-START" + original.split("# EVOLVE-BLOCK-START", 1)[1]
    # 一个**真的会改变切分**的改动：面积平衡加 ±1% 死区 hinge（本地实测 48% 变化）
    variants = {
        1: body.replace("area_loss = (top_area_ratio - balance_target).square()",
                        "dev = (top_area_ratio - balance_target).abs()\n"
                        "    area_loss = (top_area_ratio - balance_target).square() + 20.0 * F.relu(dev - 0.01).square()"),
        2: body.replace("area_loss = (top_area_ratio - balance_target).square()",
                        "dev = (top_area_ratio - balance_target).abs()\n"
                        "    area_loss = (top_area_ratio - balance_target).square() + 60.0 * F.relu(dev - 0.005).square()"),
    }
    replies = []
    for i in range(a.rounds):
        replies.append(body if i == 0 else variants.get(i, body))

    ctx = FakeContext(("tp_gnn_binary_loss_timing" if a.baseline == "binary_loss"
                       else "tp_gnn_loss_decoder_timing"),
                      original, FakeGateway(replies), limit=a.rounds)
    ctx.set_matrix(_make_matrix(a.baseline))

    # 让 agent 使用沙箱目录
    mod.__file__ = str(sandbox / a.baseline / "agent.py")
    agent = mod.Agent()
    best = agent.run(ctx)
    print(json.dumps({"best": best, "status": ctx.status()}, ensure_ascii=False, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
