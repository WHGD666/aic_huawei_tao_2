"""
evaluate.py -- 模拟"官方评估接口"的命令行：对一个候选（损失 / 解码文件）在 3 个训练算例 × 3 种子上跑切分 +
伪物理流程，输出指标 JSON。真实接口到手后由 harness/evaluator.py 的 OfficialEvaluator 直接换命令。

    python -m baseline_stub.evaluate --baseline prob  --candidate <dir含loss.py>            --out eval.json
    python -m baseline_stub.evaluate --baseline embed --candidate <dir含loss.py,decode.py> --out eval.json
    [--cases case_small,case_mid,case_large] [--seeds 1001,1002,1003] [--no-physics]（只出切分统计，用于代理层）

输出
{
  "runs": {"case/seed": {"wns": -0.6, "tns": -23.4, "valid": true, "fail_reason": "",
                          "stats": {cut, wcut, crit_cross, area_dev, macro_cnt_dev, macro_area_dev, repair_flips, hamming_ref},
                          "train_s": 0.3}},
  "partitions": {"case/seed": [0,1,...]}        # 切分方案（提交材料要求"切分方案"）
}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_stub.common.partition_utils import mini_physics, stats  # noqa: E402
from baseline_stub.common.synth import DEFAULT_CASES, HOLDOUT_CASES, make_case  # noqa: E402


def evaluate(baseline: str, cand_dir: Path, cases, seeds, physics: bool = True, ref_parts: dict | None = None,
             timeout_s: float = 120.0) -> dict:
    runs, parts = {}, {}
    spec = {**DEFAULT_CASES, **HOLDOUT_CASES}
    if baseline == "prob":
        from baseline_stub.prob.train import load_loss, run_partition
        loss_fn = load_loss(cand_dir / "loss.py")
        runner = lambda g, s: run_partition(g, s, loss_fn)  # noqa: E731
    elif baseline == "embed":
        from baseline_stub.embed.train import load_decode, load_loss, run_partition
        loss_fn = load_loss(cand_dir / "loss.py")
        dec_fn = load_decode(cand_dir / "decode.py")
        runner = lambda g, s: run_partition(g, s, loss_fn, dec_fn)  # noqa: E731
    else:
        raise ValueError(baseline)
    for c in cases:
        g = make_case(c, **spec[c])
        for s in seeds:
            key = f"{c}/{s}"
            t0 = time.time()
            try:
                r = runner(g, int(s))
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"partition exceeded {timeout_s}s")
                ref = None
                if ref_parts and key in ref_parts:
                    ref = np.asarray(ref_parts[key])
                st = stats(g, r["part"], r["flips"], ref=ref).to_dict()
                wns, tns = mini_physics(g, r["part"], int(s)) if physics else (0.0, 0.0)
                runs[key] = {"wns": wns, "tns": tns, "valid": True, "fail_reason": "", "stats": st,
                             "train_s": round(time.time() - t0, 3)}
                parts[key] = r["part"].astype(int).tolist()
            except Exception as ex:  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()[-1]
                runs[key] = {"wns": 0.0, "tns": 0.0, "valid": False, "fail_reason": f"{type(ex).__name__}: {ex}"[:200],
                             "stats": {}, "train_s": round(time.time() - t0, 3), "log": tb}
    return {"runs": runs, "partitions": parts}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["prob", "embed"], required=True)
    ap.add_argument("--candidate", required=True, help="目录，含 loss.py（embed 还需 decode.py）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--seeds", default="1001,1002,1003")
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--ref", default=None, help="参考切分 json（通常是基线 eval.json），用于 hamming_ref")
    a = ap.parse_args(argv)
    ref = None
    if a.ref and Path(a.ref).exists():
        ref = json.loads(Path(a.ref).read_text(encoding="utf-8")).get("partitions")
    res = evaluate(a.baseline, Path(a.candidate), a.cases.split(","), [int(x) for x in a.seeds.split(",")],
                   physics=not a.no_physics, ref_parts=ref)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    ok = sum(1 for r in res["runs"].values() if r["valid"])
    print(f"evaluated {len(res['runs'])} runs, {ok} valid -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
