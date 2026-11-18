"""
proposer.py -- 组织提示词、解析模型输出成 Edit 列表、免费预检。

输出协议（对小模型最稳的格式；一次响应可含多个 CANDIDATE 段落）：
  CANDIDATE 1
  HYPOTHESIS: <一句话>
  EXPECTED: <预期对 WNS/TNS 的方向与幅度、对切分统计的预期>
  KEYWORDS: a, b, c
  <<<EDIT file=relative/path.py
  <<<SEARCH
  ...exact lines...
  ===
  ...replacement...
  >>>
  >>>
  CANDIDATE 2
  ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .codebase import Codebase, Edit

PROMPTS_DIR = Path(__file__).parent / "prompts"

EDIT_RE = re.compile(
    r"<<<EDIT\s+file=(?P<file>[^\n]+?)\s*\n<<<SEARCH[ \t]*\n(?P<search>.*?)\n===[ \t]*\n(?P<replace>.*?)\n>>>[ \t]*(?:\n>>>[ \t]*)?(?=\n|$)",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)\n```", re.DOTALL)
_CAND_SPLIT = re.compile(r"^\**\s*CANDIDATE\s*(\d+)\s*\**\s*:?\s*$", re.MULTILINE | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    if "<<<EDIT" not in text:
        return text
    out = text
    for m in _FENCE_RE.finditer(text):
        inner = m.group(1)
        if "<<<EDIT" in inner or "<<<SEARCH" in inner:
            out = out.replace(m.group(0), inner)
    return out


@dataclass
class Proposal:
    hypothesis: str
    expected: str
    keywords: List[str]
    edits: List[Edit]
    raw: str
    parse_errors: List[str] = field(default_factory=list)
    slot: int = 0


def _parse_one(text: str, slot: int) -> Proposal:
    def grab(tag: str) -> str:
        m = re.search(rf"^\**{tag}\**\s*[:：]\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip().strip("*") if m else ""

    hyp, exp = grab("HYPOTHESIS"), grab("EXPECTED")
    kws = [k.strip().strip("`") for k in grab("KEYWORDS").split(",") if k.strip()]
    edits, errs = [], []
    for m in EDIT_RE.finditer(text):
        edits.append(Edit(m.group("file").strip().strip("`'\""), m.group("search"), m.group("replace")))
    if not edits:
        hint = " (EDIT markers present but malformed: check 'file=' line, '<<<SEARCH', '===' separator, closing '>>>')" if "<<<" in text else ""
        errs.append("no EDIT blocks parsed" + hint)
    if not hyp:
        errs.append("missing HYPOTHESIS")
    return Proposal(hyp, exp, kws, edits, text, errs, slot)


def parse_proposals(text: str, max_candidates: int = 3) -> List[Proposal]:
    """一次响应 → 1..k 个 Proposal。没有 CANDIDATE 标记就当一个。"""
    text = _strip_fences(text)
    marks = list(_CAND_SPLIT.finditer(text))
    if not marks:
        return [_parse_one(text, 0)]
    out = []
    for i, m in enumerate(marks[:max_candidates]):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[start:end].strip()
        if seg:
            out.append(_parse_one(seg, i))
    return out or [_parse_one(text, 0)]


def parse_proposal(text: str) -> Proposal:
    return parse_proposals(text, 1)[0]


def _load(name: str) -> str:
    p = PROMPTS_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


class Proposer:
    def __init__(self, codebase: Codebase, baseline: str, targets_desc: str, propose_k: int = 2):
        self.cb = codebase
        self.baseline = baseline
        self.targets_desc = targets_desc
        self.propose_k = propose_k
        self.last_chunks = []
        self.use_knowledge = True
        self.rebuild_system()

    def rebuild_system(self) -> None:
        know = _load("knowledge.md") if self.use_knowledge else ""
        self.system = (_load("system.md")
                       .replace("{BASELINE}", self.baseline)
                       .replace("{TARGETS}", self.targets_desc)
                       .replace("{KNOWLEDGE}", know)
                       .replace("{K}", str(self.propose_k))
                       .replace("{OUTLINE}", self.cb.outline()))
        self._system_template = self.system

    def _phase_guide(self, phase: str) -> str:
        for name in (f"phase_{phase}_{self.baseline}.md", f"phase_{phase}.md"):
            if (PROMPTS_DIR / name).exists():
                return _load(name)
        return ""

    def precheck(self, prop: Proposal, chunks=None) -> List[str]:
        errs = []
        chunks = self.last_chunks if chunks is None else chunks
        by_file: Dict[str, str] = {}
        for c in chunks:
            by_file[c.file] = by_file.get(c.file, "") + "\n" + c.source
        for e in prop.edits:
            if e.file.startswith(("/", "\\")) or ".." in e.file.split("/") or ":" in e.file[:3]:
                errs.append(f"{e.file}: illegal path"); continue
            if e.file in self.cb.files:
                try:
                    real = self.cb.read(e.file)
                except OSError:
                    real = ""
                if e.search and real.count(e.search) == 1:
                    continue
            if e.file not in by_file:
                errs.append(f"{e.file}: not an evolvable file or not shown; editable files: {sorted(by_file) or self.cb.files}")
            elif e.search not in by_file[e.file]:
                first = e.search.strip().splitlines()[0].strip() if e.search.strip() else ""
                hint = ""
                for ln in by_file[e.file].splitlines():
                    if first and first[:30] in ln:
                        hint = f" Closest shown line: {ln.strip()[:100]!r}"; break
                errs.append(f"{e.file}: SEARCH block does not appear verbatim in the shown code (copy exactly, same indentation).{hint}")
        return errs

    def build_messages(self, phase: str, profile: str, history: str, parent_patch: Optional[str], proxy_info: str,
                       feedback_for_fix: Optional[str] = None, previous_response: Optional[str] = None,
                       code_chars: int = 8000, boost_lines: Optional[Dict[str, List[int]]] = None, k: Optional[int] = None):
        k = int(k or self.propose_k)
        chunks = self.cb.search(["compute_loss", "decode_partition", "def "], k=4, max_chars=code_chars, max_chunk_chars=code_chars, boost_lines=boost_lines)
        self.last_chunks = chunks
        code = self.cb.render(chunks) if chunks else "(no code shown)"
        if parent_patch and len(parent_patch) > 5000:
            parent_patch = parent_patch[:5000] + "\n... (patch truncated; full patch applied in working copy)"
        user = (_load("propose.md")
                .replace("{PHASE}", phase)
                .replace("{PHASE_GUIDE}", self._phase_guide(phase))
                .replace("{PROFILE}", profile)
                .replace("{HISTORY}", history)
                .replace("{PROXY}", proxy_info)
                .replace("{PARENT_PATCH}", parent_patch or "(starting from the official baseline)")
                .replace("{CODE}", code)
                .replace("{K}", str(k)))
        sys_text = self.system
        if k != self.propose_k:
            import re as _re
            sys_text = _re.sub(r"produce exactly \d+ candidate", f"produce exactly {k} candidate", sys_text)
        msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
        if feedback_for_fix and previous_response:
            msgs.append({"role": "assistant", "content": previous_response})
            msgs.append({"role": "user", "content": _load("fix.md").replace("{FEEDBACK}", feedback_for_fix)})
        return msgs, chunks
