"""
profile.py -- 把基线 / 候选的评估结果压成"问题画像"给 Proposer（≤ ~40 行）。

画像里**不出现算例名**：三个训练算例按规模排序别名为 C1/C2/C3（历史摘要同理），防止模型按名字特判。
内容：每算例基线 WNS/TNS、切分统计（cut / 关键加权割 / 关键相邻对跨层比 / 修复翻动 / 平衡偏差）、种子间差异。
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from .evaluator import EvalResult
from .scoring import RunMetrics

STAT_KEYS = ["cut", "cut_ratio", "wcut", "crit_cross", "area_dev", "macro_cnt_dev", "macro_area_dev", "repair_flips", "hamming_ref"]


def case_aliases(cases: List[str], base: Optional[EvalResult] = None) -> Dict[str, str]:
    """按基线 |TNS| 升序（≈规模）给别名 C1..Cn；没有基线就按名字排序。"""
    if base:
        size = {}
        for c in cases:
            vals = [abs(r.tns) for k, r in base.runs.items() if r.case == c]
            size[c] = statistics.fmean(vals) if vals else 0.0
        order = sorted(cases, key=lambda c: size[c])
    else:
        order = sorted(cases)
    return {c: f"C{i + 1}" for i, c in enumerate(order)}


def _by_case(res: EvalResult) -> Dict[str, List[RunMetrics]]:
    out: Dict[str, List[RunMetrics]] = {}
    for r in res.runs.values():
        out.setdefault(r.case, []).append(r)
    return out


def _stat_mean(rs: List[RunMetrics], key: str) -> Optional[float]:
    v = [r.stats[key] for r in rs if r.valid and key in r.stats and r.stats[key] is not None]
    return statistics.fmean(v) if v else None


def render_baseline_profile(base: EvalResult, aliases: Dict[str, str]) -> str:
    lines = ["Baseline (official initial algorithm) on the training circuits, averaged over 3 seeds. Circuits are anonymised C1..Cn by size."]
    lines.append("circuit | WNS(ns) | TNS(ns) | seed-spread(TNS) | nets cut | crit-weighted cut | crit-pair cross-tier | repair flips | area dev (+ = upper-heavy)")
    for c, rs in sorted(_by_case(base).items(), key=lambda kv: aliases.get(kv[0], kv[0])):
        w = statistics.fmean(r.wns for r in rs)
        t = statistics.fmean(r.tns for r in rs)
        ts = statistics.pstdev([r.tns for r in rs]) if len(rs) > 1 else 0.0
        cut, wcut, cc, fl, ad = (_stat_mean(rs, k) for k in ("cut", "wcut", "crit_cross", "repair_flips", "area_dev"))
        f = lambda x, nd=1: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
        lines.append(f"{aliases.get(c, c)} | {w:.3f} | {t:.1f} | ±{ts:.1f} | {f(cut, 0)} | {f(wcut, 0)} | {f(cc, 3)} | {f(fl, 0)} | {f(ad, 3)}")
    lines.append("Notes: 'crit-pair cross-tier' = fraction of adjacent pairs of timing-critical nodes (top-20% by 2D slack rank) that end up on different tiers; "
                 "'repair flips' = nodes moved by the official balance-repair step after thresholding (large = the loss did not learn balance itself).")
    return "\n".join(lines)


def render_candidate_delta(base: EvalResult, cand: EvalResult, aliases: Dict[str, str]) -> str:
    """候选相对基线的切分统计变化（给 analyzer / 历史）。"""
    out = []
    bc, cc = _by_case(base), _by_case(cand)
    for c in sorted(bc, key=lambda x: aliases.get(x, x)):
        if c not in cc:
            continue
        parts = []
        for k, nd in (("cut", 0), ("wcut", 0), ("crit_cross", 3), ("repair_flips", 0)):
            b, v = _stat_mean(bc[c], k), _stat_mean(cc[c], k)
            if b is not None and v is not None:
                parts.append(f"{k} {b:.{nd}f}->{v:.{nd}f}")
        h = _stat_mean(cc[c], "hamming_ref")
        if h is not None:
            parts.append(f"changed-vs-baseline {h:.1%}")
        out.append(f"  {aliases.get(c, c)}: " + ", ".join(parts))
    return "\n".join(out)


def proxy_vector(res: EvalResult) -> Dict[str, float]:
    """候选的代理指标向量（跨算例平均）。"""
    v: Dict[str, float] = {}
    rs = [r for r in res.runs.values() if r.valid]
    for k in STAT_KEYS:
        m = _stat_mean(rs, k)
        if m is not None:
            v[k] = round(m, 4)
    return v
