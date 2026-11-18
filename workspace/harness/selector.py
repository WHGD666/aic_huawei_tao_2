"""
selector.py -- 选择下一轮父代、维护精英集、决定阶段与停止、挑最终候选。

  * 稳健分：mean − std（跨训练算例）− κ × 种子间标准差；惩罚"只在某个算例/某个种子上好"
  * 精英集 top-k；父代从精英集 softmax 采样，温度随阶段降低
  * 血统淘汰：同一父代连续 N 个子代（真实评估）不改善 → 不再被选；gate/screen 失败不计
  * 阶段：probe（从基线出发单机制）→ structural → combine → final
  * 最终候选：精英里"训练集三算例全 ≥ 0"的稳健分最高者；没有则稳健分最高者
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from .store import CandidateMeta, CandidateStore


class Selector:
    def __init__(self, store: CandidateStore, elite_size: int = 4, robustness: str = "mean_minus_std",
                 max_consecutive_failures: int = 2, phase_plan: Optional[Dict[str, int]] = None, seed: int = 0,
                 kappa_seed: float = 0.5, aliases: Optional[Dict[str, str]] = None):
        self.store = store
        self.elite_size = elite_size
        self.robustness = robustness
        self.max_fail = max_consecutive_failures
        self.phase_plan = phase_plan or {"probe": 5, "structural": 13, "combine": 8, "final": 4}
        self.rng = random.Random(seed)
        self.kappa_seed = kappa_seed
        self.aliases = aliases or {}
        self.force_phase: Optional[str] = None
        self._fail_streak: Dict[str, int] = {}
        for m in sorted(self.store.all(), key=lambda x: x.created_at):
            if m.status in ("evaluated", "invalid"):
                self.record_outcome(m)

    def rescale_phase_plan(self, total: int) -> None:
        keys = list(self.phase_plan)
        old = sum(self.phase_plan.values()) or 1
        new = {k: max(1, int(round(self.phase_plan[k] * total / old))) for k in keys}
        new[keys[-1]] += total - sum(new.values())
        self.phase_plan = new

    # ---------- 阶段 ----------
    def phase_for(self, evaluations_done: int) -> str:
        if self.force_phase:
            return self.force_phase
        acc = 0
        for name, n in self.phase_plan.items():
            acc += n
            if evaluations_done < acc:
                return name
        return "final"

    # ---------- 精英 ----------
    def robust_value(self, m: CandidateMeta) -> float:
        return m.score_robust if m.score_robust is not None else -math.inf

    def elites(self) -> List[CandidateMeta]:
        ev = [m for m in self.store.evaluated() if m.all_valid]
        ev.sort(key=self.robust_value, reverse=True)
        return ev[: self.elite_size]

    def best(self) -> Optional[CandidateMeta]:
        e = self.elites()
        return e[0] if e else None

    def final_pick(self) -> Optional[CandidateMeta]:
        """最终候选：精英中三算例全部非负者优先（宁要 +3/+3/+3 不要 +15/+2/−5），再按稳健分。"""
        e = self.elites()
        if not e:
            return None
        allpos = [m for m in e if m.per_case and all(v >= 0 for v in m.per_case.values())]
        pool = allpos or e
        return max(pool, key=lambda m: (self.robust_value(m), m.score_total or -math.inf))

    # ---------- 父代选择 ----------
    def choose_parent(self, phase: str) -> Optional[str]:
        elites = [m for m in self.elites() if self._fail_streak.get(m.id, 0) < self.max_fail]
        if not elites or phase == "probe":
            return None
        if phase == "final":
            return (self.final_pick() or elites[0]).id
        temp = {"structural": 1.0, "combine": 0.4}.get(phase, 1.0)
        vals = [self.robust_value(m) for m in elites]
        mx = max(vals)
        ws = [math.exp((v - mx) / max(temp, 1e-6)) for v in vals]
        r = self.rng.random() * sum(ws)
        acc = 0.0
        for m, w in zip(elites, ws):
            acc += w
            if r <= acc:
                return m.id
        return elites[0].id

    # ---------- 反馈 ----------
    def record_outcome(self, child: CandidateMeta) -> None:
        if child.parent is None or child.status in ("gate_failed", "screened", "duplicate", "discarded", "proposed"):
            return
        if child.parent not in self.store._metas:
            return
        parent = self.store.get(child.parent)
        improved = (child.status == "evaluated" and child.all_valid and child.score_robust is not None
                    and (parent.score_robust is None or child.score_robust > parent.score_robust))
        self._fail_streak[child.parent] = 0 if improved else self._fail_streak.get(child.parent, 0) + 1

    # ---------- 历史摘要 ----------
    def history_digest(self, max_items: int = 10, with_analysis: bool = True) -> str:
        """最近的已评估 / 无效 / 门禁失败候选 + 当前精英。screened（只过了代理层没提交）的只列一行代理指标。"""
        recent = sorted(self.store.all(), key=lambda m: m.created_at)[-max_items:]
        ids = {m.id for m in recent}
        items = list(recent)
        for e in self.elites()[:2]:
            if e.id not in ids:
                items.insert(0, e)
        lines = []
        for m in items:
            s = f"{m.score_total:+.3f}" if m.score_total is not None else "n/a"
            robust = f"{m.score_robust:+.3f}" if m.score_robust is not None else "n/a"
            if m.status == "gate_failed":
                note = " (not evaluated: local check failed; the idea itself is untested)"
            elif m.status == "screened":
                px = ", ".join(f"{k}={v:.3g}" for k, v in list(m.proxy.items())[:4])
                note = f" (not submitted this round; local partition stats: {px})"
            elif m.status == "duplicate":
                note = (f" (DUPLICATE: the partition is byte-identical to {m.duplicate_of}; score inherited, no evaluation spent — "
                        f"this coefficient granularity does not change the partition, do not repeat it)")
            else:
                note = f" FAIL:{'; '.join(m.fail_reasons)[:80]}" if m.fail_reasons else ""
            pc = ""
            if m.per_case:
                pc = " [" + " ".join(f"{self.aliases.get(k, k)}:{v:+.2f}" for k, v in sorted(m.per_case.items(), key=lambda kv: self.aliases.get(kv[0], kv[0]))) + "]"
            sd = f" seed_std={m.seed_std:.2f}" if m.seed_std is not None else ""
            lines.append(f"- {m.id} (parent={m.parent or 'base'}, {m.phase}) [{m.status}] score={s} robust={robust}{sd}{pc}: {m.hypothesis[:110]}{note}")
            if with_analysis and m.analysis and m.status in ("evaluated", "invalid"):
                lines.append(f"    analysis: {m.analysis[:240]}")
        return "\n".join(lines) if lines else "(no history yet)"
