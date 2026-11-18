"""
analyzer.py -- 评估结果 → 结构化摘要。规则提取（免费）+ 可选 LLM ≤120 词解读 + 代理指标↔真实得分的秩相关。
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

from .evaluator import EvalResult
from .profile import render_candidate_delta
from .scoring import CandidateScore


def rule_digest(base: EvalResult, cand: EvalResult, score: CandidateScore, aliases: Dict[str, str]) -> str:
    lines = [f"total={score.total:+.3f} robust(mean-std)={score.robust_mean_minus_std:+.3f} min_case={score.robust_min:+.3f} "
             f"seed_std={score.seed_std_mean:.3f} tns_mean={score.tns_mean:+.3f} all_valid={score.all_valid}"]
    for case, s in sorted(score.per_case.items(), key=lambda kv: aliases.get(kv[0], kv[0])):
        bw = [base.runs[k].wns_viol for k in base.runs if base.runs[k].case == case]
        cw = [cand.runs[k].wns_viol for k in cand.runs if cand.runs[k].case == case and cand.runs[k].valid]
        bt = [base.runs[k].tns_viol for k in base.runs if base.runs[k].case == case]
        ct = [cand.runs[k].tns_viol for k in cand.runs if cand.runs[k].case == case and cand.runs[k].valid]
        if cw and ct:
            dw = (statistics.fmean(bw) - statistics.fmean(cw)) / max(statistics.fmean(bw), 1e-9)
            dt = (statistics.fmean(bt) - statistics.fmean(ct)) / max(statistics.fmean(bt), 1e-9)
            per_seed = [f"{v:+.2f}" for k, v in score.per_run.items() if k.startswith(case + "/") and v is not None]
            lines.append(f"  {aliases.get(case, case)}: score={s:+.3f} dWNS={dw:+.1%} dTNS={dt:+.1%} per-seed=[{' '.join(per_seed)}]")
        else:
            lines.append(f"  {aliases.get(case, case)}: score=n/a (all seeds invalid)")
    fails = [(aliases.get(r.case, r.case) + "/" + str(r.seed), r.fail_reason) for r in cand.runs.values() if not r.valid]
    if fails:
        lines.append("  failures: " + "; ".join(f"{k}:{r}" for k, r in fails[:6]))
    delta = render_candidate_delta(base, cand, aliases)
    if delta:
        lines.append("  partition stats vs baseline:\n" + delta)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def proxy_correlation(samples: List[Tuple[Dict[str, float], float]]) -> Dict[str, Tuple[Optional[float], int]]:
    """samples: [(proxy_vector, true_score)] → {metric: (spearman, n)}。方向：指标值越小越好的用负号，
    这里直接报告 ρ，正 ρ = 指标越大分越高。"""
    out = {}
    # 冒烟指标（smoke_*）与真实得分无关，不参与相关性面板
    keys = sorted({k for v, _ in samples for k in v if not k.startswith("smoke_")})
    for k in keys:
        pairs = [(v[k], s) for v, s in samples if k in v and math.isfinite(s)]
        if len(pairs) >= 3:
            out[k] = (spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs))
    return out


def render_proxy_correlation(corr: Dict[str, Tuple[Optional[float], int]]) -> str:
    if not corr:
        return "(not enough evaluated candidates yet to correlate partition statistics with the real score)"
    lines = ["Spearman rank correlation between a candidate's partition statistics (averaged over circuits) and its real score, over evaluated candidates "
             "(positive = larger statistic goes with higher score; use this to judge which direction is actually paying off):"]
    for k, (rho, n) in sorted(corr.items(), key=lambda kv: -abs(kv[1][0] or 0)):
        lines.append(f"  {k}: rho={rho:+.2f} (n={n})" if rho is not None else f"  {k}: n/a (n={n})")
    return "\n".join(lines)


ANALYZE_PROMPT = """You are analyzing the evaluation of one candidate objective function for a GNN-based netlist tier-partitioner (2-tier 3D IC).
Hypothesis of the change: {hyp}
Rule-based digest (scores are improvement ratios vs. the official baseline; C1..C3 are the training circuits by size):
{digest}
Write <=120 words. If some runs are INVALID, first classify the failure (crash / non-finite loss / balance repair failure / timeout) and whether it is systematic or sporadic - an invalid candidate says nothing about the hypothesis itself; suggest how to make the same idea legal. Otherwise: (1) did the hypothesis hold? (2) which circuit/metric moved and the plausible mechanism (cut nets, critical-pair cross-tier, repair flips)? (3) is the seed spread acceptable? (4) one concrete next step. No preamble."""


def build_analyze_messages(hypothesis: str, digest: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": ANALYZE_PROMPT.format(hyp=hypothesis, digest=digest)}]
