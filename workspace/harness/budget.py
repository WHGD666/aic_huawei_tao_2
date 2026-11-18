"""
budget.py -- 本地"影子计量"。官方外层循环才是权威计量，这里只做提前刹车。

记录三类消耗：
  * evaluations : 已提交的候选评估次数
  * tokens      : prompt / completion / cache_hit
  * cost_cny    : 按 config/pricing.yaml 折算的人民币
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    reasoning_tokens: int = 0          # deepseek-flash thinking 模式下计入 completion 的推理 token（已含在 completion_tokens 内）

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cache_hit_tokens + other.cache_hit_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


class Pricing:
    """元 / 百万 token。"""

    def __init__(self, pricing_file: Path):
        with open(pricing_file, "r", encoding="utf-8") as f:
            self.table: Dict[str, Dict[str, float]] = yaml.safe_load(f)["models"]

    def cost(self, model: str, u: Usage) -> float:
        key = model.lower()
        if key not in self.table:
            # 未知模型按最贵的档估，宁可高估
            p = max(self.table.values(), key=lambda d: d["input_cache_miss"] + d["output"])
        else:
            p = self.table[key]
        miss = max(u.prompt_tokens - u.cache_hit_tokens, 0)
        return (
            u.cache_hit_tokens * p["input_cache_hit"]
            + miss * p["input_cache_miss"]
            + u.completion_tokens * p["output"]
        ) / 1_000_000


@dataclass
class BudgetState:
    evaluations: int = 0
    cost_cny: float = 0.0
    usage: Usage = field(default_factory=Usage)
    llm_calls: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class Budget:
    def __init__(
        self,
        max_evaluations: int,
        max_cost_cny: float,
        pricing: Pricing,
        state_file: Optional[Path] = None,
        reserve_evaluations: int = 0,
    ):
        self.max_evaluations = max_evaluations
        self.max_cost_cny = max_cost_cny
        self.reserve_evaluations = reserve_evaluations
        self.pricing = pricing
        self.state_file = state_file
        self.state = BudgetState()
        if state_file and state_file.exists():
            self._load()

    # ---------- 查询 ----------
    @property
    def evaluations_left(self) -> int:
        return self.max_evaluations - self.state.evaluations

    @property
    def cost_left(self) -> float:
        return self.max_cost_cny - self.state.cost_cny

    def can_evaluate(self, keep_reserve: bool = True) -> bool:
        left = self.evaluations_left - (self.reserve_evaluations if keep_reserve else 0)
        return left > 0 and self.cost_left > 0

    def can_call_llm(self, expected_cost: float = 0.05) -> bool:
        return self.cost_left - expected_cost > 0

    # ---------- 记账 ----------
    def charge_llm(self, model: str, usage: Usage) -> float:
        c = self.pricing.cost(model, usage)
        self.state.cost_cny += c
        self.state.usage = self.state.usage + usage
        self.state.llm_calls += 1
        self._save()
        return c

    def charge_evaluation(self) -> None:
        self.state.evaluations += 1
        self._save()

    # ---------- 持久化 ----------
    def _save(self) -> None:
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        with open(self.state_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        u = d.pop("usage", {})
        self.state = BudgetState(**d, usage=Usage(**u))

    def summary(self) -> str:
        s = self.state
        return (
            f"eval {s.evaluations}/{self.max_evaluations} | "
            f"cost {s.cost_cny:.3f}/{self.max_cost_cny} CNY | "
            f"llm_calls {s.llm_calls} | "
            f"tok in {s.usage.prompt_tokens} (hit {s.usage.cache_hit_tokens}) out {s.usage.completion_tokens}"
        )
