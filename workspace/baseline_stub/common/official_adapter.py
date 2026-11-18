"""
official_adapter.py -- 把**官方签名**的候选函数接到本地替身物理上。

官方接口（binary_loss）：
    compute_binary_partition_loss(
        probabilities, expected_cut, net_cut_probabilities, net_weights,
        node_features, node_areas, balance_target,
        cut_weight, area_balance_weight, binarization_weight, epoch, max_epochs)

官方接口（loss_decoder）：
    compute_tp_gnn_loss(positive_scores, negative_scores, embeddings, ..., node_areas)
    decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target)

本模块从替身的 CaseGraph 构造出与官方**语义一致**的这些张量：
  * 逐网割概率 cut_e = 1 − Π p − Π(1−p)（与官方 prompt.md 的公式逐字对应）
  * net_weights = 冻结时序权重（替身里用"网上最关键引脚的临界度"近似官方 paper_delay_slack 权重）
  * node_features 取替身 8 列里的前 7 列 + 补 3 列（hops / 1-hop 度 / 2-hop 度）凑官方 10 列顺序
  * balance_target = 0.5，cut_weight = 1，area_balance_weight = 200（与官方固定值一致）
  * binarization_weight 按官方 0→10 的调度（25%–75% 线性升）
  * 温度 1→0.1 + logit 扰动 0.02→0 的调度也按官方复现（否则本地测不出"训练稳定性")

这不是官方评估，只用于**无 GPU 时验证候选能否跑、方向对不对**。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_stub.common.backend import torch  # noqa: E402
from baseline_stub.common.gnn import Adam, TinyGNN, normalize_features, to_numpy  # noqa: E402
from baseline_stub.common.partition_utils import check_balance, criticality, repair_balance  # noqa: E402
from baseline_stub.common.synth import CaseGraph  # noqa: E402

# 官方的固定超参（task.yaml fixed_training / limits）
BINARY_LR = 0.003
BINARY_UPDATES = 400
TEMP_INIT, TEMP_FINAL = 1.0, 0.1
WARMUP, RAMP_END = 0.25, 0.75
CUT_WEIGHT, AREA_WEIGHT, FINAL_BIN = 1.0, 200.0, 10.0
NOISE_INIT, NOISE_FINAL = 0.02, 0.0
FEATURE_SCALE = 0.05

DECODER_MAX_ITER = 50
BALANCE_TARGET = 0.5


def _reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target):
    """官方冻结助手 area_balanced_weighted_kmeans 的**行为等价替身**：
    种子化的两中心初始化 → 面积加权中心 → 按面积平衡切点分配。"""
    rng = np.random.default_rng(int(seed))
    z = np.asarray(embeddings, dtype=np.float64)
    a = np.asarray(node_areas, dtype=np.float64)
    n = z.shape[0]
    # 两中心初始化：投影方向取最大方差维，按面积中位切
    order = np.argsort(z[:, 0])
    csum = np.cumsum(a[order])
    k0 = int(np.searchsorted(csum, a.sum() * balance_target))
    k0 = max(1, min(n - 1, k0))
    c0 = z[order[:k0]].mean(axis=0)
    c1 = z[order[k0:]].mean(axis=0)
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(max_iterations))):
        d0 = ((z - c0) ** 2).sum(axis=1)
        d1 = ((z - c1) ** 2).sum(axis=1)
        new = (d1 < d0).astype(np.int64)
        if new.min() == new.max():
            break
        if np.array_equal(new, lab) and _ > 0:
            lab = new
            break
        lab = new
        # 面积加权中心
        w0, w1 = a * (1 - lab), a * lab
        if w0.sum() > 0:
            c0 = (z * w0[:, None]).sum(axis=0) / w0.sum()
        if w1.sum() > 0:
            c1 = (z * w1[:, None]).sum(axis=0) / w1.sum()
    return _area_balance_labels(z, a, lab, balance_target)


def _area_balance_labels(z, a, lab, balance_target):
    """按"到对面中心距离"排序翻转，直到上层面积占比进入 ±2%（官方容差 3%）。"""
    n = z.shape[0]
    total = a.sum()
    for _ in range(n):
        r = a[lab == 1].sum() / total
        if abs(r - balance_target) <= 0.02:
            break
        if r > balance_target:
            cand = np.where(lab == 1)[0]
            order = cand[np.argsort(a[cand])]
            lab[order[0]] = 0
        else:
            cand = np.where(lab == 0)[0]
            order = cand[np.argsort(-a[cand])]
            lab[order[0]] = 1
    return lab


def load_module(path: Path, name: str = "candidate", baseline: str = "binary_loss"):
    """加载候选程序，并注入官方宿主提供的可信助手（torch / F / np / reference_global_two_way）。

    官方容器里这些是**调用前注入的全局名**，候选程序不 import；本适配层复现这一点，
    否则带 F.logsigmoid / reference_global_two_way 的候选在本机跑不起来。
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.torch = torch
    mod.F = torch.nn.functional
    mod.np = np
    mod.reference_global_two_way = _reference_global_two_way
    return mod


# ---------------------------------------------------------------------------
def _edge_list(g: CaseGraph) -> np.ndarray:
    return g.edge_index()


def _two_hop_degree(g: CaseGraph) -> np.ndarray:
    ei = _edge_list(g)
    n = g.n_nodes
    adj = np.zeros((n, n), dtype=bool)
    adj[ei[0], ei[1]] = True
    a2 = adj.astype(np.int32) @ adj.astype(np.int32)
    return (a2 > 0).sum(axis=1).astype(np.float32)


def official_features(g: CaseGraph) -> np.ndarray:
    """替身 8 列 → 官方 10 列顺序：
       0 worst slack           ← slack
       1 加权平均 slack        ← slack（替身没有引脚加权，用同列）
       2 max slew              ← slew
       3 max arrival           ← arrival
       4 min required          ← required
       5 summed cell delay     ← cell_delay
       6 timing-coverage       ← timing_coverage
       7 distance-to-source    ← 用 degree 的倒数排名近似
       8 one-hop degree        ← degree
       9 two-hop degree        ← 二分邻域度数
    """
    c = {n: i for i, n in enumerate(g.feature_names)}
    deg = g.features[:, c["degree"]]
    hops = np.argsort(np.argsort(deg)).astype(np.float32) / max(g.n_nodes - 1, 1)
    cols = [g.features[:, c["slack"]],       # worst slack
            g.features[:, c["slack"]],       # weighted mean slack
            g.features[:, c["slew"]],
            g.features[:, c["arrival"]],
            g.features[:, c["required"]],
            g.features[:, c["cell_delay"]],
            g.features[:, c["timing_coverage"]],
            hops,
            deg,
            _two_hop_degree(g)]
    return np.stack(cols, axis=1).astype(np.float32)


def net_timing_weights(g: CaseGraph) -> np.ndarray:
    """官方权重 = 1 + alpha·minmax(minmax(delay) − minmax(slack))；替身用关键度近似。"""
    crit = criticality(g, beta=1.0)
    w = np.zeros(g.n_nets, dtype=np.float32)
    for k in range(g.n_nets):
        nd = g.net_nodes[g.net_ptr[k]:g.net_ptr[k + 1]]
        if len(nd) >= 2:
            w[k] = 1.0 + 3.0 * float(crit[nd].max())
        else:
            w[k] = 1.0
    return w


def cut_probabilities(p: "object", g: CaseGraph) -> Tuple["object", "object"]:
    """逐网软割概率 cut_e = 1 − Π p − Π(1−p)，以及按权重归一化的期望割。"""
    net_of_pin = np.repeat(np.arange(g.n_nets), np.diff(g.net_ptr))
    nodes = torch.as_tensor(g.net_nodes, dtype=torch.int64)
    nop = torch.as_tensor(net_of_pin, dtype=torch.int64)
    pp = p[nodes]
    lp = torch.log(torch.clamp(pp, 1e-6, 1.0))
    lq = torch.log(torch.clamp(1.0 - pp, 1e-6, 1.0))
    slp = torch.zeros(g.n_nets).index_add(0, nop, lp)
    slq = torch.zeros(g.n_nets).index_add(0, nop, lq)
    cut = 1.0 - torch.exp(slp) - torch.exp(slq)
    w = torch.as_tensor(net_timing_weights(g), dtype=torch.float32)
    expected = (w * cut).sum() / w.sum()
    return cut, expected, w


def schedule(step: int, total: int) -> Tuple[float, float, float]:
    """温度的倒数（官方是对 logit 除以温度）与二值权重、扰动尺度。"""
    frac = step / max(total - 1, 1)
    if frac < WARMUP:
        temp, wbin, noise = TEMP_INIT, 0.0, NOISE_INIT
    elif frac < RAMP_END:
        t = (frac - WARMUP) / (RAMP_END - WARMUP)
        temp = TEMP_INIT + (TEMP_FINAL - TEMP_INIT) * t
        wbin = FINAL_BIN * t
        noise = NOISE_INIT + (NOISE_FINAL - NOISE_INIT) * t
    else:
        temp, wbin, noise = TEMP_FINAL, FINAL_BIN, NOISE_FINAL
    return temp, wbin, noise


# ---------------------------------------------------------------------------
def run_binary_loss(g: CaseGraph, seed: int, program_path: Path) -> Dict:
    """用候选的 compute_binary_partition_loss 训练替身 GNN（官方 400 步调度），返回切分。"""
    mod = load_module(program_path, "cand_bin", baseline="binary_loss")
    fn = getattr(mod, "compute_binary_partition_loss", None)
    if fn is None:
        raise RuntimeError("program must define compute_binary_partition_loss")

    torch.manual_seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    x_np = official_features(g)
    x = torch.as_tensor(normalize_features(x_np), dtype=torch.float32)
    extra = np.stack([np.log(g.area + 1e-6), g.is_macro.astype(np.float32)], axis=1).astype(np.float32)
    x = torch.cat([x, torch.as_tensor(normalize_features(extra))], dim=1)
    ei = torch.as_tensor(_edge_list(g), dtype=torch.int64)
    model = TinyGNN(x.shape[1], 32, 1, seed)
    opt = Adam(model.params, lr=BINARY_LR)
    areas = torch.as_tensor(g.area, dtype=torch.float32)
    feats = torch.as_tensor(x_np, dtype=torch.float32)
    n = g.n_nodes

    best_p, best_loss = None, float("inf")
    curve = []
    for step in range(BINARY_UPDATES):
        temp, wbin, noise = schedule(step, BINARY_UPDATES)
        model.zero_grad()
        logit = model.forward(x, ei, n).reshape(-1)
        scaled = logit / temp
        if noise > 0:
            jitter = torch.randn(n, generator=gen) * noise
            jitter = jitter - (areas * jitter).sum() / areas.sum()   # 面积居中
            scaled = scaled + jitter
        p = torch.sigmoid(scaled)
        cut, expected, w = cut_probabilities(p, g)
        loss = fn(p, expected, cut, w, feats, areas, BALANCE_TARGET,
                  CUT_WEIGHT, AREA_WEIGHT, wbin, step + 1, BINARY_UPDATES)
        lv = float(loss.item())
        if not np.isfinite(lv):
            raise RuntimeError(f"loss not finite at step {step}: {lv}")
        loss.backward()
        opt.step()
        curve.append(lv)
        if step >= int(RAMP_END * BINARY_UPDATES):     # 最终稳定阶段才参与选点
            if lv < best_loss:
                best_loss, best_p = lv, to_numpy(p).copy()
    p_np = best_p if best_p is not None else to_numpy(p)
    raw = (p_np >= 0.5).astype(np.int64)
    part, flips = repair_balance(g, raw, score=p_np)
    ok, why = check_balance(g, part)
    if not ok:
        raise RuntimeError(f"closure failed: {why}")
    return {"part": part, "raw": raw, "prob": p_np, "flips": flips, "curve": curve,
            "uncertainty": float(np.abs(p_np - 0.5).mean())}


# ---------------------------------------------------------------------------
def run_loss_decoder(g: CaseGraph, seed: int, program_path: Path) -> Dict:
    """用候选的 compute_tp_gnn_loss + decode_tp_gnn_embeddings 跑替身训练与解码。"""
    mod = load_module(program_path, "cand_dec", baseline="loss_decoder")
    loss_fn = getattr(mod, "compute_tp_gnn_loss", None)
    dec_fn = getattr(mod, "decode_tp_gnn_embeddings", None)
    if loss_fn is None or dec_fn is None:
        raise RuntimeError("program must define compute_tp_gnn_loss and decode_tp_gnn_embeddings")

    EPOCHS = 20
    torch.manual_seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    x_np = official_features(g)
    x = torch.as_tensor(normalize_features(x_np), dtype=torch.float32)
    extra = np.stack([np.log(g.area + 1e-6), g.is_macro.astype(np.float32)], axis=1).astype(np.float32)
    x = torch.cat([x, torch.as_tensor(normalize_features(extra))], dim=1)
    ei = torch.as_tensor(_edge_list(g), dtype=torch.int64)
    n = g.n_nodes
    model = TinyGNN(x.shape[1], 32, 16, seed)
    opt = Adam(model.params, lr=0.01)
    areas = torch.as_tensor(g.area, dtype=torch.float32)
    feats = torch.as_tensor(x_np, dtype=torch.float32)
    E = int(ei.shape[1])
    w_net = torch.as_tensor(net_timing_weights(g), dtype=torch.float32)
    src, dst = ei[0], ei[1]
    net_of_pin = np.repeat(np.arange(g.n_nets), np.diff(g.net_ptr))
    graph_net_ids = torch.as_tensor(net_of_pin[:E] if len(net_of_pin) >= E else
                                    np.resize(net_of_pin, E), dtype=torch.int64)
    curve = []
    best_z, best_loss = None, float("inf")
    for ep in range(EPOCHS):
        model.zero_grad()
        h = model.forward(x, ei, n)
        z = torch.nn.functional.normalize(h, p=2, dim=1)
        pos = (z[src] * z[dst]).sum(dim=1)
        neg_idx = torch.randint(0, n, (E, 5), generator=gen)
        neg = (z[src].unsqueeze(1) * z[neg_idx]).sum(dim=2)
        loss = loss_fn(pos, neg, z, src, dst, src, dst, float(E), ep, EPOCHS,
                       pos, src, dst, graph_net_ids, torch.ones(E), w_net, feats, areas)
        lv = float(loss.item())
        if not np.isfinite(lv):
            raise RuntimeError(f"loss not finite at epoch {ep}: {lv}")
        loss.backward()
        opt.step()
        curve.append(lv)
        if ep >= EPOCHS // 2 and lv < best_loss:
            best_loss, best_z = lv, to_numpy(z).copy()
    z_np = best_z if best_z is not None else to_numpy(z)
    gnp = {"edge_index": _edge_list(g), "net_ptr": g.net_ptr, "net_nodes": g.net_nodes,
           "net_of_pin": np.asarray(net_of_pin), "area": g.area, "is_macro": g.is_macro,
           "n_nodes": n, "n_nets": g.n_nets}
    labels = np.asarray(dec_fn(z_np.copy(), g.area.astype(np.float32).copy(), int(seed),
                               DECODER_MAX_ITER, BALANCE_TARGET))
    if labels.shape != (n,):
        raise RuntimeError(f"decoder returned shape {labels.shape}, expected ({n},)")
    if not np.all((labels == 0) | (labels == 1)):
        raise RuntimeError("decoder must return 0/1 labels")
    if labels.min() == labels.max():
        raise RuntimeError("decoder put every group on one tier")
    part, flips = repair_balance(g, labels.astype(np.int64), score=labels.astype(np.float64))
    ok, why = check_balance(g, part)
    if not ok:
        raise RuntimeError(f"closure failed: {why}")
    return {"part": part, "raw": labels.astype(np.int64), "flips": flips, "curve": curve}
