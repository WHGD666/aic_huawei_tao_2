"""
train.py -- 概率预测基线的固定训练器（不可演化）：图构建、特征归一化、GNN、优化器、训练预算、阈值化、修复都在这里。

    python -m baseline_stub.prob.train --case case_small --seed 1001 --loss path/to/loss.py --out part.npy

被 harness 的代理层与冒烟调用；也可直接 import run_partition()。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_stub.common.backend import torch, HAS_TORCH  # noqa: E402
from baseline_stub.common.gnn import Adam, TinyGNN, normalize_features, to_numpy  # noqa: E402
from baseline_stub.common.partition_utils import check_balance, repair_balance  # noqa: E402
from baseline_stub.common.synth import CaseGraph, DEFAULT_CASES, HOLDOUT_CASES, make_case  # noqa: E402

EPOCHS = 60
LR = 0.01
HIDDEN = 32
THRESHOLD = 0.5


def load_loss(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_loss", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "compute_loss"):
        raise RuntimeError("loss module must define compute_loss(p, graph, feats)")
    return mod.compute_loss


def graph_tensors(g: CaseGraph) -> Tuple[dict, object]:
    ei = g.edge_index()
    net_of_pin = np.repeat(np.arange(g.n_nets), np.diff(g.net_ptr))
    graph = {
        "edge_index": torch.as_tensor(ei, dtype=torch.int64),
        "net_ptr": torch.as_tensor(g.net_ptr, dtype=torch.int64),
        "net_nodes": torch.as_tensor(g.net_nodes, dtype=torch.int64),
        "net_of_pin": torch.as_tensor(net_of_pin, dtype=torch.int64),
        "area": torch.as_tensor(g.area, dtype=torch.float32),
        "is_macro": torch.as_tensor(g.is_macro, dtype=torch.bool),
        "n_nodes": int(g.n_nodes),
        "n_nets": int(g.n_nets),
        "feature_names": list(g.feature_names),
    }
    feats = torch.as_tensor(g.features, dtype=torch.float32)
    return graph, feats


def run_partition(g: CaseGraph, seed: int, loss_fn, epochs: int = EPOCHS, return_prob: bool = False) -> Dict:
    """返回 {'part': np[N] 0/1 (修复后), 'raw_part': 阈值化前的, 'flips': int, 'loss_curve': [...], 'train_s': float}"""
    t0 = time.time()
    torch.manual_seed(seed)
    graph, feats = graph_tensors(g)
    x = torch.as_tensor(normalize_features(g.features), dtype=torch.float32)
    # 固定：面积 log 与宏标记也进输入
    extra = np.stack([np.log(g.area + 1e-6), g.is_macro.astype(np.float32)], axis=1).astype(np.float32)
    x = torch.cat([x, torch.as_tensor(normalize_features(extra))], dim=1)
    model = TinyGNN(x.shape[1], HIDDEN, 1, seed)
    opt = Adam(model.params, lr=LR)
    curve = []
    best_p, best_loss = None, float("inf")
    for ep in range(epochs):
        model.zero_grad()
        out = model.forward(x, graph["edge_index"], g.n_nodes).reshape(-1)
        logit = 4.0 * torch.tanh(out / 4.0)          # 固定：有界 logit，避免 sigmoid 饱和后梯度消失
        p = torch.sigmoid(logit)
        loss = loss_fn(p, graph, feats)
        if not hasattr(loss, "backward"):
            raise RuntimeError("compute_loss must return a tensor")
        lv = float(to_numpy(loss).reshape(-1)[0]) if hasattr(loss, "shape") and tuple(loss.shape) != () else float(loss.item())
        if not np.isfinite(lv):
            raise RuntimeError(f"loss is not finite at epoch {ep}: {lv}")
        loss.backward()
        opt.step()
        curve.append(lv)
        if lv < best_loss:
            best_loss, best_p = lv, to_numpy(p).copy()
    p_np = best_p if best_p is not None else to_numpy(p)
    raw = (p_np > THRESHOLD).astype(np.int64)
    part, flips = repair_balance(g, raw, score=p_np)
    ok, why = check_balance(g, part)
    if not ok:
        raise RuntimeError(f"repair failed: {why}")
    out = {"part": part, "raw_part": raw, "flips": int(flips), "loss_curve": curve, "train_s": time.time() - t0}
    if return_prob:
        out["prob"] = p_np
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--loss", default=str(Path(__file__).with_name("loss.py")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args(argv)
    spec = {**DEFAULT_CASES, **HOLDOUT_CASES}
    if a.case not in spec:
        print(f"unknown case {a.case}; known: {sorted(spec)}", file=sys.stderr)
        return 2
    g = make_case(a.case, **spec[a.case])
    r = run_partition(g, a.seed, load_loss(Path(a.loss)), epochs=a.epochs)
    if a.out:
        np.save(a.out, r["part"])
    print(json.dumps({"case": a.case, "seed": a.seed, "flips": r["flips"], "final_loss": r["loss_curve"][-1],
                      "train_s": round(r["train_s"], 2), "backend": "torch" if HAS_TORCH else "tiny"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
