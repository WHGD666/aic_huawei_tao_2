"""
partition_utils.py -- 切分结果的通用处理（纯 numpy，不可演化、模拟"官方程序"的角色）：

  * repair_balance : 约束修复 —— 上层实例面积占比 47–53%；有宏时宏数量占比与宏面积占比 45–55%
  * check_balance  : 合法性判定
  * stats          : 割网数 / 关键加权割 / 关键相邻对跨层比例 / 平衡偏差 / 宏偏差 / 与参考切分的 Hamming
  * mini_physics   : 一个"伪物理流程"——从切分 + 2D 时序特征估一个 WNS/TNS，只用于离线演练，
                     让 selector / analyzer 的行为有意义。它故意只与割网数、关键路径跨层数、
                     层间关键面积不均衡相关，并带种子噪声。真实评估接口到手后不再使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .synth import CaseGraph

AREA_LO, AREA_HI = 0.47, 0.53
MACRO_LO, MACRO_HI = 0.45, 0.55


# ---------------------------------------------------------------------------
def area_ratio(g: CaseGraph, part: np.ndarray) -> float:
    return float(g.area[part == 1].sum() / g.area.sum())


def macro_ratios(g: CaseGraph, part: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    m = g.is_macro
    if m.sum() == 0:
        return None, None
    cnt = float((part[m] == 1).mean())
    ar = float(g.area[m & (part == 1)].sum() / g.area[m].sum())
    return cnt, ar


def check_balance(g: CaseGraph, part: np.ndarray) -> Tuple[bool, str]:
    if part.shape != (g.n_nodes,) or not np.all((part == 0) | (part == 1)):
        return False, "partition must be a 0/1 vector of length n_nodes"
    r = area_ratio(g, part)
    if not (AREA_LO - 1e-9 <= r <= AREA_HI + 1e-9):
        return False, f"upper-tier area ratio {r:.4f} outside [{AREA_LO},{AREA_HI}]"
    mc, ma = macro_ratios(g, part)
    if mc is not None:
        if not (MACRO_LO - 1e-9 <= mc <= MACRO_HI + 1e-9):
            return False, f"upper-tier macro count ratio {mc:.3f} outside [{MACRO_LO},{MACRO_HI}]"
        if not (MACRO_LO - 1e-9 <= ma <= MACRO_HI + 1e-9):
            return False, f"upper-tier macro area ratio {ma:.3f} outside [{MACRO_LO},{MACRO_HI}]"
    return True, ""


def repair_balance(g: CaseGraph, part: np.ndarray, score: Optional[np.ndarray] = None,
                   max_iter: int = 100000) -> Tuple[np.ndarray, int]:
    """模拟官方修复程序：贪心翻转直到满足约束。score[i] 越大表示节点 i 越"想在上层"
    （概率基线传 p；没有就按面积小的先翻）。返回 (修复后的 part, 翻动节点数)。

    顺序：先修宏（数量 + 面积），再修总面积（只翻标准单元，避免又把宏约束弄坏）。
    """
    part = part.astype(np.int64).copy()
    n = g.n_nodes
    if score is None:
        score = np.zeros(n, dtype=np.float64)
    flips = 0
    macro = g.is_macro
    # ---- 宏：数量占比与面积占比都要落在 [45%,55%]。宏很少（≤16）时穷举子集，取"翻动代价最小"的可行解；
    #      多时用贪心 + 交换。代价 = 翻动节点的置信度 (0.5 + |score-0.5|)，即尽量只翻模棱两可的宏。
    if macro.sum() > 0:
        midx = np.where(macro)[0]
        init = part[midx].copy()
        if len(midx) <= 16:
            best, best_cost = None, np.inf
            sc = score[midx]
            conf = 0.5 + np.abs(sc - (sc.min() + sc.max()) / 2.0) / max(sc.max() - sc.min(), 1e-9) * 0.5
            areas = g.area[midx]
            tot = areas.sum()
            k = len(midx)
            for mask in range(1 << k):
                sel = np.array([(mask >> i) & 1 for i in range(k)], dtype=np.int64)
                cr = sel.mean()
                ar = (areas * sel).sum() / tot
                if MACRO_LO - 1e-9 <= cr <= MACRO_HI + 1e-9 and MACRO_LO - 1e-9 <= ar <= MACRO_HI + 1e-9:
                    cost = float((conf * (sel != init)).sum())
                    if cost < best_cost:
                        best, best_cost = sel, cost
            if best is None:
                # 无可行解（宏数/面积分布本身不允许）：取违反量最小的，交给 check_balance 报错
                best_v = np.inf
                for mask in range(1 << k):
                    sel = np.array([(mask >> i) & 1 for i in range(k)], dtype=np.int64)
                    v = max(0, abs(sel.mean() - 0.5) - 0.05) + max(0, abs((areas * sel).sum() / tot - 0.5) - 0.05)
                    if v < best_v:
                        best, best_v = sel, v
            part[midx] = best
            flips += int((best != init).sum())
        else:
            for _ in range(len(midx) * 2):
                mc, ma = macro_ratios(g, part)
                if MACRO_LO <= mc <= MACRO_HI and MACRO_LO <= ma <= MACRO_HI:
                    break
                if mc > MACRO_HI or (MACRO_LO <= mc <= MACRO_HI and ma > MACRO_HI):
                    cand = midx[part[midx] == 1]
                    if len(cand) == 0:
                        break
                    i = cand[np.argmin(score[cand])]
                    part[i] = 0
                else:
                    cand = midx[part[midx] == 0]
                    if len(cand) == 0:
                        break
                    i = cand[np.argmax(score[cand])]
                    part[i] = 1
                flips += 1
    # ---- 总面积（只动标准单元） ----
    cells = np.where(~macro)[0]
    order_down = cells[np.argsort(score[cells])]             # 最不想在上层的先下去
    order_up = cells[np.argsort(-score[cells])]
    ptr_d = ptr_u = 0
    for _ in range(max_iter):
        r = area_ratio(g, part)
        if AREA_LO <= r <= AREA_HI:
            break
        if r > AREA_HI:
            while ptr_d < len(order_down) and part[order_down[ptr_d]] == 0:
                ptr_d += 1
            if ptr_d >= len(order_down):
                break
            part[order_down[ptr_d]] = 0
        else:
            while ptr_u < len(order_up) and part[order_up[ptr_u]] == 1:
                ptr_u += 1
            if ptr_u >= len(order_up):
                break
            part[order_up[ptr_u]] = 1
        flips += 1
    return part, flips


# ---------------------------------------------------------------------------
def criticality(g: CaseGraph, beta: float = 2.0) -> np.ndarray:
    """算例内相对关键度 c_i ∈ [0,1]：slack 的算例内排名分位数（slack 越小越关键）。单位无关。"""
    s = g.column("slack")
    rank = np.argsort(np.argsort(s))
    q = rank / max(g.n_nodes - 1, 1)
    return (1.0 - q) ** beta


@dataclass
class PartStats:
    cut: int
    cut_ratio: float
    wcut: float                 # 关键加权割（权 1 + 4c_n）
    crit_cross: float           # 关键节点（c>0.8）相邻对里跨层的比例
    area_dev: float             # 上层面积占比 − 0.5
    macro_cnt_dev: Optional[float]
    macro_area_dev: Optional[float]
    repair_flips: int
    hamming_ref: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: (None if v is None else (float(v) if not isinstance(v, int) else v)) for k, v in vars(self).items()}


def stats(g: CaseGraph, part: np.ndarray, repair_flips: int = 0, ref: Optional[np.ndarray] = None) -> PartStats:
    c = criticality(g)
    cut = 0
    wcut = 0.0
    for k in range(g.n_nets):
        nd = g.net_nodes[g.net_ptr[k]:g.net_ptr[k + 1]]
        if len(nd) < 2:
            continue
        p = part[nd]
        if p.min() != p.max():
            cut += 1
            wcut += 1.0 + 4.0 * c[nd].max()
    ei = g.edge_index()
    crit = c > 0.8
    both = crit[ei[0]] & crit[ei[1]]
    crit_cross = float((part[ei[0]] != part[ei[1]])[both].mean()) if both.any() else 0.0
    mc, ma = macro_ratios(g, part)
    ham = None
    if ref is not None:
        h = float((part != ref).mean())
        ham = min(h, 1.0 - h)      # 标签对称
    return PartStats(cut, cut / max(g.n_nets, 1), wcut, crit_cross, area_ratio(g, part) - 0.5,
                     None if mc is None else mc - 0.5, None if ma is None else ma - 0.5, repair_flips, ham)


# ---------------------------------------------------------------------------
def mini_physics(g: CaseGraph, part: np.ndarray, seed: int) -> Tuple[float, float]:
    """伪物理流程 → (WNS, TNS)，单位 ns，负值。只用于离线演练。
    机制：每个节点的 3D slack = 2D slack − HBT 延迟 × (该节点关键邻边跨层数) − 拥塞项(关键面积不均衡) + 噪声。
    """
    rng = np.random.default_rng(seed * 7919 + g.n_nodes)
    scale = g.meta.get("pdk_scale", 1.0)
    slack = g.column("slack") / scale                          # → ns
    period = g.meta["clock_period"] / scale
    c = criticality(g)
    ei = g.edge_index()
    cross = (part[ei[0]] != part[ei[1]]).astype(np.float64)
    n_cross = np.bincount(ei[0], weights=cross, minlength=g.n_nodes)
    hbt_delay = 0.045 * period
    # 关键面积不均衡 → 拥塞 → 线延迟
    crit_area_up = (g.area * c * (part == 1)).sum() / max((g.area * c).sum(), 1e-9)
    congest = 0.08 * period * abs(crit_area_up - 0.5) * 2
    s3 = slack - hbt_delay * n_cross * (0.3 + 0.7 * c) - congest * c + rng.normal(0, 0.01 * period, g.n_nodes)
    neg = np.minimum(s3, 0.0)
    return float(neg.min()), float(neg.sum())
