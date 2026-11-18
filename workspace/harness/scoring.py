"""
scoring.py -- 题面第六节的计分公式，作为本地 selector 的唯一打分口径。

算例得分 = 0.4 × WNS 改善比例 + 0.6 × TNS 改善比例
改善比例 = (基线违例量 − 候选违例量) / 基线违例量        （违例量 = |负 slack|，可为负）
先每算例每种子打分 → 种子平均 → 算例等权加和。

本地约定（原文未定义，需向组委会确认）：基线违例量为 0 时，候选也为 0 记 0，否则记 −1。
任一 run 无效 → 整个候选 −∞（题面：最终候选必测算例与种子必须全部有效，不得剔除失败种子）。
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

W_WNS, W_TNS = 0.4, 0.6


@dataclass
class RunMetrics:
    """一个 (算例, 种子) 的结果。wns/tns 为负 slack（ns）。"""
    case: str
    seed: int
    wns: float
    tns: float
    valid: bool = True
    fail_reason: str = ""
    stats: Dict[str, float] = field(default_factory=dict)   # 评估接口附带的切分统计（可为空）

    @property
    def wns_viol(self) -> float:
        return abs(min(self.wns, 0.0))

    @property
    def tns_viol(self) -> float:
        return abs(min(self.tns, 0.0))


def improvement(base_viol: float, cand_viol: float) -> float:
    if base_viol <= 0:
        return 0.0 if cand_viol <= 0 else -1.0
    return (base_viol - cand_viol) / base_viol


def run_score(base: RunMetrics, cand: RunMetrics) -> Optional[float]:
    if not cand.valid:
        return None
    return W_WNS * improvement(base.wns_viol, cand.wns_viol) + W_TNS * improvement(base.tns_viol, cand.tns_viol)


def run_tns_improvement(base: RunMetrics, cand: RunMetrics) -> Optional[float]:
    if not cand.valid:
        return None
    return improvement(base.tns_viol, cand.tns_viol)


@dataclass
class CandidateScore:
    total: float                          # 算例等权加和（官方口径，训练集 3 个）
    per_case: Dict[str, float]            # 每算例（种子平均）
    per_run: Dict[str, Optional[float]]   # "case/seed" -> score 或 None
    all_valid: bool
    robust_mean: float                    # 算例均值
    robust_min: float                     # 最差算例
    robust_mean_minus_std: float          # 算例均值 − 算例间标准差
    seed_std_mean: float                  # 各算例"种子间标准差"的均值（同一机制在不同种子下的稳定性）
    tns_mean: float                       # TNS 改善比例的算例均值（同分比 TNS）

    def robust(self, metric: str, kappa_seed: float = 0.0) -> float:
        """selector 用的稳健分：metric ∈ mean | min | mean_minus_std，再减 κ × 种子间方差。"""
        base = {"mean": self.robust_mean, "min": self.robust_min, "mean_minus_std": self.robust_mean_minus_std}.get(metric, self.robust_mean_minus_std)
        if not math.isfinite(base):
            return base
        return base - kappa_seed * self.seed_std_mean


def candidate_score(baseline: Dict[str, RunMetrics], candidate: Dict[str, RunMetrics]) -> CandidateScore:
    """baseline / candidate 都是 {"case/seed": RunMetrics}。"""
    per_run: Dict[str, Optional[float]] = {}
    by_case: Dict[str, List[float]] = {}
    tns_by_case: Dict[str, List[float]] = {}
    all_valid = True
    for key, b in baseline.items():
        c = candidate.get(key)
        if c is None:
            per_run[key] = None
            all_valid = False
            continue
        s = run_score(b, c)
        per_run[key] = s
        if s is None:
            all_valid = False
            continue
        by_case.setdefault(b.case, []).append(s)
        tns_by_case.setdefault(b.case, []).append(run_tns_improvement(b, c) or 0.0)
    # 评估接口返回了基线没有的 run 也不算有效（口径以基线 key 为准，多出来的忽略）
    per_case = {c: statistics.fmean(v) for c, v in by_case.items()}
    vals = list(per_case.values())
    if not all_valid or not vals:
        return CandidateScore(-math.inf, per_case, per_run, False, -math.inf, -math.inf, -math.inf, 0.0, -math.inf)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    seed_std = statistics.fmean([statistics.pstdev(v) if len(v) > 1 else 0.0 for v in by_case.values()])
    tns_mean = statistics.fmean([statistics.fmean(v) for v in tns_by_case.values()])
    return CandidateScore(sum(vals), per_case, per_run, True, mean, min(vals), mean - std, seed_std, tns_mean)
