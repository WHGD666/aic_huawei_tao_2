"""Shared agent machinery for both baselines: model calls, local validation, memory, budget.

This module lives in each baseline's agent directory (the organizer mounts only
that directory). It never touches the network directly: every model call goes
through `context.model.responses_create`, which the organizer gateway meters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Offline expert knowledge. MUST stay within 200 counting units
# (1 per CJK character, 2 per English word, 1 per run of digits; punctuation,
# spaces and newlines do not count; title and body count together).
# Verified by workspace/tools/count_offline_knowledge.py.
OFFLINE_KNOWLEDGE_BINARY = """离线经验：切分把实例分到上下两层，跨层网经混合键合端子并增加延迟，相邻且时序紧的实例宜同层。该配方训练后期概率高度饱和，割项梯度趋近零，只改割项往往不改变最终切分；优先调面积平衡的形状与死区，其次用训练进度前期加大割项权重，最后才动二值项形态。特征中覆盖率为零表示未知而非零裕量，用前按算例内排名归一化。面积偏离一半时官方修复会大量翻转，损失宜自行把面积压到一半附近。"""

OFFLINE_KNOWLEDGE_DECODER = """离线经验：切分把网表分到上下两层，跨层网经混合键合端子并增加延迟，关键路径上跨层越少越好，相邻且时序紧的单元宜同层。嵌入损失只用随机游走正负样本，图分数、逐边权重与节点时序特征都未利用，可先试按冻结时序权重强化关键边的正样本。解码只拿到嵌入、面积与标量控制；可用主方向投影后按累计面积取平衡点做二类划分，两层都须非空，不要构造全距离矩阵。两基线五公开算例初始均分为零，任何正分都是净收益，稳健优先。"""


def knowledge_for(baseline: str) -> str:
    return OFFLINE_KNOWLEDGE_BINARY if baseline == "binary_loss" else OFFLINE_KNOWLEDGE_DECODER


# ---------------------------------------------------------------------------
def program_path(folder: Path) -> Path:
    return folder / "program.py"


def read_program(folder: Path) -> str:
    return program_path(folder).read_text(encoding="utf-8")


def write_program(folder: Path, source: str) -> None:
    program_path(folder).write_text(source, encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
_MEMORY_LIMIT = 6000


def read_memory(folder: Path) -> str:
    p = folder / "MEMORY.md"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")[-_MEMORY_LIMIT:]
    except OSError:
        return ""


def append_memory(folder: Path, text: str) -> None:
    """Runtime-acquired experience. Allowed by the knowledge-injection rules
    (knowledge the agent forms itself during the run is not offline injection)."""
    p = folder / "MEMORY.md"
    try:
        old = p.read_text(encoding="utf-8") if p.exists() else "# Runtime memory (agent-written)\n\n"
        if len(old) > _MEMORY_LIMIT:
            old = old[-_MEMORY_LIMIT:]
        p.write_text(old.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BLOCK = re.compile(r"#\s*EVOLVE-BLOCK-START(.*?)#\s*EVOLVE-BLOCK-END", re.DOTALL)


def extract_program(text: str) -> str:
    """Pull the program source out of a model reply.

    Accepted shapes, in order: a complete program (def ... at top level), an
    EVOLVE-BLOCK pair, or a fenced python block. Returns "" when nothing of the
    right shape is found.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    # 1. EVOLVE-BLOCK markers: take the block and re-add the markers
    m = _BLOCK.search(t)
    if m:
        return "# EVOLVE-BLOCK-START" + m.group(1).rstrip() + "\n# EVOLVE-BLOCK-END\n"
    # 2. fenced code blocks
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    for body in fences:
        if _looks_like_program(body):
            return body.strip() + "\n"
    # 3. the whole reply, if it looks like a program
    if _looks_like_program(t):
        return t.strip() + "\n"
    return ""


def _looks_like_program(text: str) -> bool:
    return bool(re.search(r"^def\s+compute_(binary_partition|tp_gnn)_loss\s*\(", text, re.MULTILINE))


# ---------------------------------------------------------------------------
class LocalCheck:
    """Runs the organizer's own static validator when it is importable.

    The validator is mounted read-only at /opt/contest/AgenticBench; locally we
    may also have a copy. Failure to import is reported, never fatal.
    """

    def __init__(self):
        self.kind = None
        self._fn = None
        for path in ("/opt/contest/AgenticBench", "AgenticBench", "."):
            if not Path(path).is_dir():
                continue
            try:
                import sys
                if path not in sys.path:
                    sys.path.insert(0, path)
                from local_tasks.tp_gnn_local_task_definitions import (  # noqa: F401
                    BINARY_TASK, DECODER_TASK, validate_program,
                )
                self._validate = validate_program
                self._BINARY_TASK, self._DECODER_TASK = BINARY_TASK, DECODER_TASK
                self.kind = path
                break
            except Exception:  # noqa: BLE001
                continue

    @property
    def available(self) -> bool:
        return self.kind is not None

    def check(self, source: str, baseline: str):
        if not self.available:
            return True, "validator unavailable in this container; relying on the organizer"
        task = self._BINARY_TASK if baseline == "binary_loss" else self._DECODER_TASK
        try:
            self._validate(source, task)
            return True, "ok"
        except Exception as error:  # noqa: BLE001
            return False, f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
def feedback_digest(result: dict, limit: int = 2400) -> str:
    """Compress an evaluation result into the short text we keep in memory."""
    try:
        status = result.get("status")
        fb = result.get("feedback") or {}
        parts = [f"status={status}"]
        if fb.get("score") is not None:
            parts.append(f"score={fb['score']:.3f}")
        cells = fb.get("cells") or []
        if cells:
            bad = [c for c in cells if not c.get("valid", True)]
            parts.append(f"cells={len(cells)} invalid={len(bad)}")
            if bad:
                for c in bad[:3]:
                    parts.append(f"  invalid {c.get('case')}/{c.get('seed')}: {str(c.get('reason'))[:120]}")
            ok = [c for c in cells if c.get("valid", True) and c.get("score") is not None]
            if ok:
                by_case = {}
                for c in ok:
                    by_case.setdefault(c.get("case"), []).append(c["score"])
                parts.append("  per-case: " + ", ".join(
                    f"{k}={sum(v) / len(v):+.1f}" for k, v in sorted(by_case.items())))
        if fb.get("reason"):
            parts.append(f"reason={str(fb['reason'])[:200]}")
        if result.get("reason"):
            parts.append(f"result_reason={str(result['reason'])[:200]}")
        text = "\n".join(parts)
        return text[:limit]
    except Exception:  # noqa: BLE001
        return json.dumps(result, ensure_ascii=False)[:limit]


# ---------------------------------------------------------------------------
def local_partition_change(source: str, baseline: str, threshold: float = 0.01):
    """Free local smoke: does this candidate actually change the partition?

    Uses the bunded stub only when it is importable (it lives next to this
    repository, not in the organizer image). Inside the official container the
    module is absent, so we return (None, "unavailable") and rely on the
    organizer. Returning None never blocks a submission.
    """
    try:
        import sys
        from pathlib import Path as _P
        for cand in (str(_P(__file__).resolve().parents[3] / "workspace"),
                     "/workspace/workspace", "."):
            if _P(cand).is_dir() and cand not in sys.path:
                sys.path.insert(0, cand)
        from baseline_stub.common.synth import DEFAULT_CASES, make_case
        from baseline_stub.common.official_adapter import run_binary_loss, run_loss_decoder
    except Exception:  # noqa: BLE001
        return None, "stub unavailable in this container"
    import tempfile
    folder = _P(__file__).resolve().parent.parent
    base_program = folder / "program.py"
    if not base_program.exists():
        return None, "no baseline program to compare"
    runner = run_binary_loss if baseline == "binary_loss" else run_loss_decoder
    try:
        with tempfile.TemporaryDirectory() as td:
            p = _P(td) / "candidate.py"
            p.write_text(source, encoding="utf-8")
            changed_frac, total = 0.0, 0
            best_frac = 0.0
            for name, spec in DEFAULT_CASES.items():
                g = make_case(name, **spec)
                rb = runner(g, 1001, base_program)
                rc = runner(g, 1001, p)
                frac = float((rb["part"] != rc["part"]).mean())
                best_frac = max(best_frac, frac)
                changed_frac += frac
                total += 1
            return (changed_frac / max(total, 1)), f"max single-case change {best_frac:.1%}"
    except Exception as error:  # noqa: BLE001
        return None, f"stub raised {type(error).__name__}: {str(error)[:100]}"
