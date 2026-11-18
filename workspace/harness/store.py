"""
store.py -- 候选历史库。直接产出提交要求里 logs/ 需要的"父子关系 / 代码版本 / 评估结果 / 失败原因"。

目录结构：
  candidates/
    <id>/
      meta.json         parent, hypothesis, files_changed, created_at, status, score...
      candidate.patch   相对 baseline 的 git diff
      eval.json         评估结果（如果有）
      llm/              本候选相关的 prompt/response 存档（省钱复盘用）
  lineage.jsonl         每行一个事件，追加写，方便画演化树
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CandidateMeta:
    id: str
    parent: Optional[str]
    hypothesis: str                       # LLM 提出的假设（一句话）
    expected_effect: str                  # 预期影响
    files_changed: List[str]
    phase: str                            # probe | structural | combine | final
    created_at: float = field(default_factory=time.time)
    status: str = "proposed"              # proposed | gate_failed | screened | duplicate | evaluated | invalid | discarded
    gate_report: str = ""
    score_total: Optional[float] = None
    score_robust: Optional[float] = None
    per_case: Dict[str, float] = field(default_factory=dict)
    all_valid: Optional[bool] = None
    fail_reasons: List[str] = field(default_factory=list)
    analysis: str = ""                    # Analyzer 的 ≤300 token 摘要
    llm_cost_cny: float = 0.0
    eval_wall_s: float = 0.0
    tags: List[str] = field(default_factory=list)
    prompt_chars: int = 0                 # 本轮 propose prompt 字符数（费用自适应用）
    proxy: Dict[str, float] = field(default_factory=dict)      # 本地代理指标（切分统计，不花评估）
    seed_std: Optional[float] = None       # 三算例"种子间得分标准差"的均值（稳健性）
    duplicate_of: Optional[str] = None     # 切分结果与已评估候选完全相同 -> 继承其分数，不花评估
    slot: int = 0                          # 同一轮内的第几个候选（多候选每轮）


class CandidateStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lineage = root / "lineage.jsonl"
        self._metas: Dict[str, CandidateMeta] = {}
        self._load()

    # ---------- CRUD ----------
    def new_id(self) -> str:
        """取 max(已有 id)+1，不按计数——删过目录也不会撞 id。"""
        mx = 0
        for cid in list(self._metas) + [p.name for p in self.root.glob("c[0-9]*")]:
            try:
                mx = max(mx, int(cid[1:]))
            except ValueError:
                pass
        return f"c{mx + 1:03d}"

    def write_patch(self, cid: str, patch_text: str) -> None:
        (self.root / cid / "candidate.patch").write_text(patch_text, encoding="utf-8", newline="")

    def create(self, meta: CandidateMeta, patch_text: str) -> Path:
        d = self.root / meta.id
        (d / "llm").mkdir(parents=True, exist_ok=True)
        (d / "candidate.patch").write_text(patch_text, encoding="utf-8", newline="")
        self._metas[meta.id] = meta
        self._flush(meta)
        self._event("create", meta.id, parent=meta.parent, phase=meta.phase, hypothesis=meta.hypothesis)
        return d

    def update(self, meta: CandidateMeta, event: str = "update", **extra) -> None:
        self._metas[meta.id] = meta
        self._flush(meta)
        self._event(event, meta.id, status=meta.status, score=meta.score_total, **extra)

    def save_eval(self, cid: str, eval_dict: dict) -> None:
        (self.root / cid / "eval.json").write_text(json.dumps(eval_dict, indent=1, ensure_ascii=False), encoding="utf-8")

    def save_llm(self, cid: str, name: str, prompt: list, response: str, usage: dict) -> None:
        p = self.root / cid / "llm" / f"{name}.json"
        p.write_text(json.dumps({"prompt": prompt, "response": response, "usage": usage}, indent=1, ensure_ascii=False), encoding="utf-8")

    def get(self, cid: str) -> CandidateMeta:
        return self._metas[cid]

    def dir(self, cid: str) -> Path:
        return self.root / cid

    def all(self) -> List[CandidateMeta]:
        return sorted(self._metas.values(), key=lambda m: m.created_at)

    def evaluated(self) -> List[CandidateMeta]:
        return [m for m in self.all() if m.status == "evaluated" and m.score_total is not None]

    def children(self, cid: str) -> List[CandidateMeta]:
        return [m for m in self.all() if m.parent == cid]

    def lineage_of(self, cid: str) -> List[str]:
        chain, cur = [], cid
        while cur:
            chain.append(cur)
            cur = self._metas[cur].parent if cur in self._metas else None
        return list(reversed(chain))

    # ---------- 导出 ----------
    def export_tree_text(self) -> str:
        roots = [m for m in self.all() if m.parent is None]
        lines: List[str] = []

        def walk(m: CandidateMeta, depth: int):
            s = f"{m.score_total:+.3f}" if m.score_total is not None else "  --  "
            lines.append(f"{'  ' * depth}{m.id} [{m.status:>11}] {s}  {m.hypothesis[:60]}")
            for ch in self.children(m.id):
                walk(ch, depth + 1)

        for r in roots:
            walk(r, 0)
        return "\n".join(lines)

    # ---------- internals ----------
    def _flush(self, meta: CandidateMeta) -> None:
        (self.root / meta.id).mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in vars(meta).items() if not k.startswith("_")}   # 运行期临时字段（如 _smoke_metrics）不落盘
        (self.root / meta.id / "meta.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")

    def _event(self, kind: str, cid: str, **kw) -> None:
        with open(self.lineage, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": kind, "id": cid, **kw}, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        import dataclasses
        known = {f.name for f in dataclasses.fields(CandidateMeta)}
        for mp in self.root.glob("*/meta.json"):
            try:
                d = json.loads(mp.read_text(encoding="utf-8"))
                if "id" not in d:
                    raise KeyError("id")
                self._metas[d["id"]] = CandidateMeta(**{k: v for k, v in d.items() if k in known})
            except (json.JSONDecodeError, KeyError, TypeError) as ex:
                print(f"[store] WARNING skipping unreadable {mp}: {type(ex).__name__}: {ex}", file=__import__("sys").stderr)
