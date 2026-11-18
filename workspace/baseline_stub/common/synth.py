"""
synth.py -- 合成"网表超图 + 2D 时序特征"的生成器，模拟任务包给每个算例的输入。

设计目标：让替身基线、门禁冒烟和代理评估在没有真实数据时也有**结构上像电路**的输入：
  * 节点 = 标准单元 + 少量宏；面积长尾分布；宏面积远大于单元
  * 网 = 超边；扇出分布长尾（少量高扇出的时钟/复位网），带局部性（相邻 id 更可能连）
  * 若干条"关键路径"：沿路径的节点 slack 为负、彼此相邻
  * 2D 时序特征：slack / slew / arrival / required / cell_delay / timing_coverage，单位按 pdk 不同（ns 或 ps）

输出 CaseGraph（numpy）。真实任务包的图结构到手后，只需在 loader 里把真实数据装成同样的 CaseGraph。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CaseGraph:
    name: str
    n_nodes: int
    area: np.ndarray            # [N] float
    is_macro: np.ndarray        # [N] bool
    net_ptr: np.ndarray         # [M+1] CSR 指针：net k 的引脚节点 = net_nodes[net_ptr[k]:net_ptr[k+1]]
    net_nodes: np.ndarray       # [sum fanout] int
    features: np.ndarray        # [N, F] float，列名见 feature_names
    feature_names: List[str]
    time_unit: str = "ns"
    meta: Dict[str, float] = field(default_factory=dict)

    # ---------- 派生 ----------
    @property
    def n_nets(self) -> int:
        return len(self.net_ptr) - 1

    def net_sizes(self) -> np.ndarray:
        return np.diff(self.net_ptr)

    def edge_index(self, max_clique: int = 8) -> np.ndarray:
        """超图 → 边列表 [2, E]（无向，两个方向都给）。大网用星型展开（以第一个引脚为中心）避免 O(k²)。"""
        src, dst = [], []
        for k in range(self.n_nets):
            nodes = self.net_nodes[self.net_ptr[k]:self.net_ptr[k + 1]]
            if len(nodes) < 2:
                continue
            if len(nodes) <= max_clique:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        src.append(nodes[i]); dst.append(nodes[j])
            else:
                c = nodes[0]
                for j in nodes[1:]:
                    src.append(c); dst.append(j)
        s, d = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
        return np.stack([np.concatenate([s, d]), np.concatenate([d, s])])

    def column(self, name: str) -> np.ndarray:
        return self.features[:, self.feature_names.index(name)]


FEATURES = ["slack", "slew", "arrival", "required", "cell_delay", "timing_coverage", "degree", "fanout_max"]


def _macro_feasible(areas: np.ndarray, lo: float = 0.45, hi: float = 0.55) -> bool:
    """是否存在一个子集同时满足数量占比与面积占比都在 [lo, hi]（宏 ≤ 16 时穷举）。"""
    k = len(areas)
    if k == 0:
        return True
    if k > 16:
        return True
    tot = areas.sum()
    for m in range(1 << k):
        sel = np.array([(m >> i) & 1 for i in range(k)])
        if lo <= sel.mean() <= hi and lo <= (areas * sel).sum() / tot <= hi:
            return True
    return False


def make_case(name: str, n_nodes: int = 400, n_macros: int = 6, seed: int = 0, pdk: str = "ng45",
              clock_period: float = 1.0, n_crit_paths: int = 8, crit_len: int = 10) -> CaseGraph:
    rng = np.random.default_rng(seed)
    unit_scale = 1000.0 if pdk == "asap7" else 1.0           # asap7 特征用 ps，ng45 用 ns
    period = clock_period * unit_scale

    # ---- 面积 ----
    area = np.exp(rng.normal(0.0, 0.5, n_nodes)).astype(np.float32)      # 单元 ~ 1
    is_macro = np.zeros(n_nodes, dtype=bool)
    if n_macros > 0:
        macro_idx = rng.choice(n_nodes, n_macros, replace=False)
        is_macro[macro_idx] = True
        # 宏面积要保证"存在满足 45–55% 数量 + 面积双约束的子集"（真实数据未必保证，见 open_questions）：
        # 成对生成相近面积的宏；奇数个时最后一个取中位数
        for _ in range(200):
            base = rng.uniform(40, 120, (n_macros + 1) // 2)
            pair = np.repeat(base, 2)[:n_macros] * rng.uniform(0.9, 1.1, n_macros)
            if _macro_feasible(pair):
                break
        area[macro_idx] = pair.astype(np.float32)

    # ---- 网：局部性 + 长尾扇出 ----
    nets: List[np.ndarray] = []
    n_nets = int(n_nodes * 1.1)
    for _ in range(n_nets):
        k = int(np.clip(rng.geometric(0.45) + 1, 2, 12))
        center = rng.integers(0, n_nodes)
        spread = rng.integers(3, 40)
        cand = np.clip(center + rng.integers(-spread, spread + 1, k * 2), 0, n_nodes - 1)
        nodes = np.unique(cand)[:k]
        if len(nodes) >= 2:
            nets.append(nodes.astype(np.int64))
    # 2 条高扇出网（时钟/复位）
    for _ in range(2):
        k = int(n_nodes * rng.uniform(0.15, 0.3))
        nets.append(np.sort(rng.choice(n_nodes, k, replace=False)).astype(np.int64))

    # ---- 关键路径：沿路径加 2 引脚网，并给负 slack ----
    slack = rng.normal(0.35, 0.15, n_nodes) * period                     # 大多数为正
    arrival = rng.uniform(0.1, 0.7, n_nodes) * period
    crit_nodes = set()
    for p in range(n_crit_paths):
        start = rng.integers(0, n_nodes - crit_len)
        path = np.arange(start, start + crit_len)
        depth = rng.uniform(0.05, 0.4)
        for i in range(crit_len - 1):
            nets.append(np.array([path[i], path[i + 1]], dtype=np.int64))
        slack[path] = -depth * period * rng.uniform(0.6, 1.0, crit_len)
        arrival[path] = np.linspace(0.3, 1.0 + depth, crit_len) * period
        crit_nodes.update(path.tolist())

    net_ptr = np.zeros(len(nets) + 1, dtype=np.int64)
    net_ptr[1:] = np.cumsum([len(n) for n in nets])
    net_nodes = np.concatenate(nets).astype(np.int64)

    # ---- 其它时序特征 ----
    required = arrival + slack
    slew = np.abs(rng.normal(0.05, 0.02, n_nodes)) * period
    cell_delay = np.abs(rng.normal(0.04, 0.015, n_nodes)) * period
    degree = np.bincount(net_nodes, minlength=n_nodes).astype(np.float32)
    fan_max = np.zeros(n_nodes, dtype=np.float32)
    sizes = np.diff(net_ptr)
    for k in range(len(nets)):
        nd = net_nodes[net_ptr[k]:net_ptr[k + 1]]
        fan_max[nd] = np.maximum(fan_max[nd], sizes[k])
    coverage = np.clip(degree / (degree.max() + 1e-9) + rng.uniform(0, 0.3, n_nodes), 0, 1)
    for i in crit_nodes:
        coverage[i] = min(1.0, coverage[i] + 0.5)

    feats = np.stack([slack, slew, arrival, required, cell_delay, coverage, degree, fan_max], axis=1).astype(np.float32)
    return CaseGraph(name, n_nodes, area, is_macro, net_ptr, net_nodes, feats, list(FEATURES),
                     time_unit="ps" if pdk == "asap7" else "ns",
                     meta={"clock_period": period, "n_crit_paths": float(n_crit_paths), "pdk_scale": unit_scale})


# 三个"训练算例"的合成配置：规模、宏数、周期各不同，避免候选函数偷懒
DEFAULT_CASES = {
    "case_small": dict(n_nodes=300, n_macros=4, seed=11, pdk="ng45", clock_period=1.0),
    "case_mid": dict(n_nodes=600, n_macros=8, seed=22, pdk="ng45", clock_period=2.0),
    "case_large": dict(n_nodes=1000, n_macros=12, seed=33, pdk="ng45", clock_period=4.0),
}
# 用于本地泛化自检（不进入搜索决策）：无宏 / 跨工艺
HOLDOUT_CASES = {
    "holdout_nomacro": dict(n_nodes=500, n_macros=0, seed=44, pdk="ng45", clock_period=2.0),
    "holdout_asap7": dict(n_nodes=300, n_macros=4, seed=11, pdk="asap7", clock_period=1.0),
}


def build_cases(spec: Optional[Dict[str, dict]] = None) -> Dict[str, CaseGraph]:
    spec = spec or DEFAULT_CASES
    return {k: make_case(k, **v) for k, v in spec.items()}
