"""
train.py -- 表示学习基线的固定训练器（不可演化）：图构建、特征归一化、GNN、负采样、优化器、训练预算、
L2 归一化、调用解码、修复。

    python -m baseline_stub.embed.train --case case_small --seed 1001 [--loss loss.py --decode decode.py]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_stub.common.backend import torch, HAS_TORCH  # noqa: E402
from baseline_stub.common.gnn import Adam, TinyGNN, normalize_features, to_numpy  # noqa: E402
from baseline_stub.common.partition_utils import check_balance, repair_balance  # noqa: E402
from baseline_stub.common.synth import CaseGraph, DEFAULT_CASES, HOLDOUT_CASES, make_case  # noqa: E402
from baseline_stub.prob.train import graph_tensors  # noqa: E402

EPOCHS = 60
LR = 0.01
HIDDEN = 32
EMB = 16
NEG_K = 5


def _load(path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location(f"candidate_{fn_name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, fn_name):
        raise RuntimeError(f"{path.name} must define {fn_name}")
    return getattr(mod, fn_name)


def load_loss(path: Path):
    return _load(path, "compute_loss")


def load_decode(path: Path):
    return _load(path, "decode_partition")


def _graph_numpy(graph: dict) -> dict:
    out = {}
    for k, v in graph.items():
        out[k] = to_numpy(v) if hasattr(v, "shape") and not isinstance(v, (int, list, str)) else v
    # 整型保持整型
    for k in ("edge_index", "net_ptr", "net_nodes", "net_of_pin"):
        out[k] = np.asarray(out[k]).astype(np.int64)
    out["is_macro"] = np.asarray(out["is_macro"]).astype(bool)
    return out


def run_partition(g: CaseGraph, seed: int, loss_fn, decode_fn, epochs: int = EPOCHS, return_emb: bool = False) -> Dict:
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    graph, feats = graph_tensors(g)
    x = torch.as_tensor(normalize_features(g.features), dtype=torch.float32)
    extra = np.stack([np.log(g.area + 1e-6), g.is_macro.astype(np.float32)], axis=1).astype(np.float32)
    x = torch.cat([x, torch.as_tensor(normalize_features(extra))], dim=1)
    model = TinyGNN(x.shape[1], HIDDEN, EMB, seed)
    opt = Adam(model.params, lr=LR)
    E = int(graph["edge_index"].shape[1])
    curve = []
    best_z, best_loss = None, float("inf")
    for ep in range(epochs):
        model.zero_grad()
        h = model.forward(x, graph["edge_index"], g.n_nodes)
        z = torch.nn.functional.normalize(h, p=2, dim=1)          # 固定的表示归一化
        neg_idx = torch.randint(0, g.n_nodes, (E, NEG_K), generator=gen)   # 固定采样
        loss = loss_fn(z, graph, feats, neg_idx)
        if not hasattr(loss, "backward"):
            raise RuntimeError("compute_loss must return a tensor")
        lv = float(loss.item())
        if not np.isfinite(lv):
            raise RuntimeError(f"loss is not finite at epoch {ep}: {lv}")
        loss.backward()
        opt.step()
        curve.append(lv)
        if lv < best_loss:
            best_loss, best_z = lv, to_numpy(z).copy()
    z_np = best_z if best_z is not None else to_numpy(z)
    t1 = time.time()
    raw = decode_fn(z_np, _graph_numpy(graph), g.features.astype(np.float32), int(seed))
    raw = np.asarray(raw)
    if raw.shape != (g.n_nodes,) or not np.all((raw == 0) | (raw == 1)):
        raise RuntimeError("decode_partition must return a 0/1 vector of length n_nodes")
    raw = raw.astype(np.int64)
    # 修复（官方程序角色）：score 用"到自己簇的归属强度"的替代——这里没有概率，用随机扰动 + 原标签
    part, flips = repair_balance(g, raw, score=raw.astype(np.float64) + np.random.default_rng(seed).random(g.n_nodes) * 0.01)
    ok, why = check_balance(g, part)
    if not ok:
        raise RuntimeError(f"repair failed: {why}")
    out = {"part": part, "raw_part": raw, "flips": int(flips), "loss_curve": curve,
           "train_s": t1 - t0, "decode_s": time.time() - t1}
    if return_emb:
        out["emb"] = z_np
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--loss", default=str(Path(__file__).with_name("loss.py")))
    ap.add_argument("--decode", default=str(Path(__file__).with_name("decode.py")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args(argv)
    spec = {**DEFAULT_CASES, **HOLDOUT_CASES}
    if a.case not in spec:
        print(f"unknown case {a.case}", file=sys.stderr)
        return 2
    g = make_case(a.case, **spec[a.case])
    r = run_partition(g, a.seed, load_loss(Path(a.loss)), load_decode(Path(a.decode)), epochs=a.epochs)
    if a.out:
        np.save(a.out, r["part"])
    print(json.dumps({"case": a.case, "seed": a.seed, "flips": r["flips"], "final_loss": r["loss_curve"][-1],
                      "train_s": round(r["train_s"], 2), "decode_s": round(r["decode_s"], 2),
                      "backend": "torch" if HAS_TORCH else "tiny"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
