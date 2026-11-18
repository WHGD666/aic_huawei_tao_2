"""
smoke.py -- 候选函数的可微性 / 可行性冒烟（合成小图，秒级，免费）。

loss  : 标量、有限、requires_grad、backward 通过、梯度有限且非全零、耗时；再跑两个退化输入（无宏 / 全同 slack）
decode: 返回 0/1 向量、长度正确、确定性（同 seed 两次一致）、面积与宏偏差（修复前）、耗时；退化输入不崩

张量后端：有 torch 用 torch，否则用 numpy 自动微分替身（baseline_stub.common.backend）。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

WS = Path(__file__).resolve().parent.parent
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))


def _load_fn(src: str, fn_name: str):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(src)
        tmp = tf.name
    spec = importlib.util.spec_from_file_location(f"smoke_{fn_name}_{abs(hash(src))}", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    Path(tmp).unlink(missing_ok=True)
    return getattr(mod, fn_name)


def _cases():
    from baseline_stub.common.synth import make_case
    return [
        ("tiny", make_case("smoke_tiny", n_nodes=120, n_macros=4, seed=5, pdk="ng45", clock_period=1.0, n_crit_paths=3, crit_len=6)),
        ("nomacro", make_case("smoke_nomacro", n_nodes=100, n_macros=0, seed=6, pdk="asap7", clock_period=0.5, n_crit_paths=2, crit_len=5)),
    ]


def smoke_test(kind: str, src: str, fn_name: str, time_limit_s: float = 20.0) -> Tuple[List[str], List[str], Dict[str, float]]:
    """在**子进程**里跑真实冒烟：模型写的 `while True:` 或死循环只会被硬超时杀掉，不会挂死 harness。"""
    import json as _json
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tf

    with _tf.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(src)
        src_path = tf.name
    try:
        cmd = [_sys.executable, "-m", "harness.smoke", "--kind", kind, "--fn", fn_name, "--src", src_path,
               "--time-limit", str(time_limit_s)]
        try:
            r = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                        timeout=time_limit_s + 5.0, cwd=str(WS))
        except _sp.TimeoutExpired:
            return [f"smoke exceeded {time_limit_s:.0f}s on a 120-node synthetic graph (infinite loop or runaway complexity; the real circuits are 100x larger)"], [], {}
        if r.returncode != 0 and not r.stdout.strip():
            return [f"smoke subprocess failed (rc={r.returncode}): {(r.stderr or '')[-400:]}"], [], {}
        try:
            d = _json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return [f"smoke subprocess produced no result: {(r.stdout or r.stderr)[-400:]}"], [], {}
        return d.get("errors", []), d.get("warnings", []), d.get("metrics", {})
    finally:
        Path(src_path).unlink(missing_ok=True)


def smoke_test_inproc(kind: str, src: str, fn_name: str, time_limit_s: float = 20.0) -> Tuple[List[str], List[str], Dict[str, float]]:
    errs, warns, metrics = [], [], {}
    try:
        from baseline_stub.common.backend import torch
        from baseline_stub.common.gnn import to_numpy
        from baseline_stub.prob.train import graph_tensors
        fn = _load_fn(src, fn_name)
    except Exception as ex:  # noqa: BLE001
        return [f"cannot load candidate function: {type(ex).__name__}: {ex}"], warns, metrics

    for tag, g in _cases():
        t0 = time.time()
        try:
            graph, feats = graph_tensors(g)
            if kind == "loss":
                _smoke_loss(fn, g, graph, feats, torch, to_numpy, tag, errs, warns, metrics)
            elif kind == "decode":
                _smoke_decode(fn, g, graph, feats, to_numpy, tag, errs, warns, metrics)
        except Exception as ex:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            where = next((ln.strip() for ln in reversed(tb) if "smoke_" in ln and "line" in ln), "")
            errs.append(f"[{tag}] raised {type(ex).__name__}: {str(ex)[:160]} {('@ ' + where) if where else ''}")
        dt = time.time() - t0
        metrics[f"{tag}_s"] = round(dt, 3)
        if dt > time_limit_s:
            errs.append(f"[{tag}] took {dt:.1f}s > {time_limit_s}s on a {g.n_nodes}-node graph; too slow for real circuits")
        if errs:
            break
    return errs, warns, metrics


def _smoke_loss(fn, g, graph, feats, torch, to_numpy, tag, errs, warns, metrics):
    n = g.n_nodes
    gen = torch.Generator(); gen.manual_seed(0)
    # 两种输入：概率（prob 基线）或表示（embed 基线）—— 按签名参数个数判断
    import inspect
    nparams = len(inspect.signature(fn).parameters)
    if nparams == 3:
        logits = torch.randn(n, generator=gen)
        logits = logits.detach() if hasattr(logits, "detach") else logits
        logits.requires_grad_(True)
        inp = torch.sigmoid(logits)
        loss = fn(inp, graph, feats)
        leaf = logits
    else:
        h = torch.randn(n, 16, generator=gen)
        h = h.detach() if hasattr(h, "detach") else h
        h.requires_grad_(True)
        z = torch.nn.functional.normalize(h, p=2, dim=1)
        E = int(graph["edge_index"].shape[1])
        neg_idx = torch.randint(0, n, (E, 5), generator=gen)
        loss = fn(z, graph, feats, neg_idx)
        leaf = h
    if not hasattr(loss, "backward"):
        errs.append(f"[{tag}] compute_loss returned {type(loss).__name__}, not a tensor"); return
    shape = tuple(loss.shape) if hasattr(loss, "shape") else ()
    if shape not in ((), (1,)):
        errs.append(f"[{tag}] loss must be a scalar, got shape {shape}"); return
    lv = float(to_numpy(loss).reshape(-1)[0])
    if not np.isfinite(lv):
        errs.append(f"[{tag}] loss is {lv} (not finite)"); return
    if not getattr(loss, "requires_grad", False):
        errs.append(f"[{tag}] loss does not require grad — the computation graph is broken (detach/item/numpy/argmax?)"); return
    loss.backward()
    gr = leaf.grad
    if gr is None:
        errs.append(f"[{tag}] no gradient reached the model output"); return
    ga = to_numpy(gr)
    if not np.all(np.isfinite(ga)):
        errs.append(f"[{tag}] gradient contains NaN/Inf"); return
    gnorm = float(np.linalg.norm(ga))
    metrics[f"{tag}_loss"] = round(lv, 6)
    metrics[f"{tag}_gnorm"] = round(gnorm, 6)
    if gnorm == 0.0:
        errs.append(f"[{tag}] gradient is identically zero — the loss does not depend on the prediction"); return
    frac_zero = float((np.abs(ga) < 1e-12).mean())
    if frac_zero > 0.9:
        warns.append(f"[{tag}] {frac_zero:.0%} of nodes receive zero gradient (saturation / hard thresholds?)")
    if gnorm > 1e4:
        warns.append(f"[{tag}] gradient norm {gnorm:.2e} is very large; consider normalising terms")


def _smoke_decode(fn, g, graph, feats, to_numpy, tag, errs, warns, metrics):
    from baseline_stub.common.partition_utils import area_ratio, macro_ratios
    from baseline_stub.embed.train import _graph_numpy
    n = g.n_nodes
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 16)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    gnp = _graph_numpy(graph)
    fx = g.features.astype(np.float32)
    out1 = fn(z.copy(), dict(gnp), fx.copy(), 1001)
    out2 = fn(z.copy(), dict(gnp), fx.copy(), 1001)
    out1 = np.asarray(out1)
    if out1.shape != (n,):
        errs.append(f"[{tag}] decode must return shape ({n},), got {out1.shape}"); return
    if not np.all((out1 == 0) | (out1 == 1)):
        errs.append(f"[{tag}] decode must return 0/1 values"); return
    if not np.array_equal(out1, np.asarray(out2)):
        errs.append(f"[{tag}] decode is not deterministic for the same seed"); return
    r = area_ratio(g, out1.astype(np.int64))
    metrics[f"{tag}_area_dev"] = round(abs(r - 0.5), 4)
    if abs(r - 0.5) > 0.03:
        warns.append(f"[{tag}] pre-repair upper-tier area ratio {r:.3f}; the repair step will flip many nodes and destroy structure")
    mc, ma = macro_ratios(g, out1.astype(np.int64))
    if mc is not None:
        metrics[f"{tag}_macro_dev"] = round(max(abs(mc - 0.5), abs(ma - 0.5)), 4)
        if abs(mc - 0.5) > 0.05 or abs(ma - 0.5) > 0.05:
            warns.append(f"[{tag}] pre-repair macro ratios count={mc:.2f} area={ma:.2f} violate 45–55%; repair will move macros")
    if out1.min() == out1.max():
        errs.append(f"[{tag}] decode put every node on the same tier")


def _main(argv=None):
    """子进程入口：跑冒烟并把 (errors, warnings, metrics) 以 JSON 打到 stdout。"""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--fn", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--time-limit", type=float, default=20.0)
    a = ap.parse_args(argv)
    src = Path(a.src).read_text(encoding="utf-8")
    try:
        e, w, m = smoke_test_inproc(a.kind, src, a.fn, time_limit_s=a.time_limit)
    except Exception as ex:  # noqa: BLE001
        import traceback
        e, w, m = [f"smoke crashed: {type(ex).__name__}: {ex}\n{traceback.format_exc()[-600:]}"], [], {}
    print(json.dumps({"errors": e, "warnings": w, "metrics": m}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
