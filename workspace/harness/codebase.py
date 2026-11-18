"""
codebase.py -- 基线代码的索引与片段检索。目的是控制 prompt 长度：永远不把整个文件塞给模型。

* 只索引白名单内文件
* 函数级分块（用 ast），每块记录 (file, name, lineno range, source)
* 关键字检索：按名称 / docstring / 源码命中次数打分，返回前 k 块
* apply_edits：把 SEARCH/REPLACE 块应用到工作副本，并生成 git diff
"""
from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def glob_match(path: str, pattern: str) -> bool:
    """gitignore 风格匹配：`**/` 表示零个或多个目录层级，`*` 不跨越 `/`。"""
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.fullmatch(out, path) is not None


def match_any(path: str, patterns) -> bool:
    return any(glob_match(path, p) for p in patterns)


@dataclass
class Chunk:
    file: str          # 相对路径
    name: str          # 函数/类名 或 "<module>"
    start: int
    end: int
    source: str

    def header(self) -> str:
        return f"### {self.file}:{self.start}-{self.end}  {self.name}"


class Codebase:
    def __init__(self, root: Path, allowed_globs: List[str], denied_globs: List[str] = (), overlay_root: Optional[Path] = None):
        """overlay_root：若给定，同名文件优先从这里读（父代工作副本），其余从 root（官方基线）。"""
        self.root = root
        self.overlay_root = overlay_root
        self.allowed = allowed_globs
        self.denied = list(denied_globs)
        self.files: List[str] = []
        self.chunks: List[Chunk] = []
        self._scan()

    @classmethod
    def overlay(cls, base: "Codebase", overlay_root: Path) -> "Codebase":
        return cls(base.root, base.allowed, base.denied, overlay_root=overlay_root)

    def _read_rel(self, rel: str) -> str:
        if self.overlay_root is not None and (self.overlay_root / rel).exists():
            return (self.overlay_root / rel).read_text(encoding="utf-8", errors="ignore")
        return (self.root / rel).read_text(encoding="utf-8", errors="ignore")

    # ---------- 索引 ----------
    def _is_allowed(self, rel: str) -> bool:
        return match_any(rel, self.allowed) and not match_any(rel, self.denied)

    def _scan(self) -> None:
        if not self.root.exists():
            return
        rels = set()
        for base in (self.root, self.overlay_root):
            if base is None or not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(base)).replace("\\", "/")
                    if rel != "candidate.patch" and self._is_allowed(rel):
                        rels.add(rel)
        for rel in sorted(rels):
            self.files.append(rel)
            src = self._read_rel(rel)
            if rel.endswith(".py"):
                self.chunks.extend(self._chunk_py(rel, src))
            else:
                self.chunks.append(Chunk(rel, "<file>", 1, src.count("\n") + 1, src))

    @staticmethod
    def _chunk_py(rel: str, src: str) -> List[Chunk]:
        """函数级分块；类按方法再切（大类整体塞 prompt 会挤掉其它片段）。"""
        out: List[Chunk] = []
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return [Chunk(rel, "<module>", 1, len(lines), src)]
        covered = set()

        def add(node, name):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            # 把装饰器行也带上
            for d in getattr(node, "decorator_list", []):
                s = min(s, d.lineno)
            out.append(Chunk(rel, name, s, e, "\n".join(lines[s - 1:e])))
            covered.update(range(s, e + 1))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node, node.name)
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if not methods or (getattr(node, "end_lineno", node.lineno) - node.lineno) < 80:
                    add(node, node.name)
                    continue
                # 类头（签名 + 类级属性，到第一个方法之前）
                first = min(min([m.lineno] + [d.lineno for d in m.decorator_list]) for m in methods)
                s = node.lineno
                for d in node.decorator_list:
                    s = min(s, d.lineno)
                out.append(Chunk(rel, node.name, s, first - 1, "\n".join(lines[s - 1:first - 1])))
                covered.update(range(s, first))
                for m in methods:
                    add(m, f"{node.name}.{m.name}")
        # 模块级剩余代码（import、常量、配置）作为一块
        rest = [ln for i, ln in enumerate(lines, 1) if i not in covered]
        if any(l.strip() for l in rest):
            out.insert(0, Chunk(rel, "<module>", 1, len(lines), "\n".join(rest)))
        return out

    # ---------- 检索 ----------
    def search(self, keywords: List[str], k: int = 6, max_chars: int = 6000, max_chunk_chars: int = 2500,
               max_per_file: int = 3, boost_lines: Optional[Dict[str, List[int]]] = None) -> List[Chunk]:
        """关键字打分（名称命中 ×3），按分数取前 k 块；单块超过 max_chunk_chars 的截断到命中行附近。
        每个文件最多 max_per_file 块。boost_lines={file: [行号...]}：覆盖这些行的块（通常是父代 patch 改过的区域）
        强制排到最前并且窗口以这些行为中心 —— 子代要改父代改过的地方，必须能看到它。"""
        boost_lines = boost_lines or {}
        scored: List[Tuple[float, Chunk, bool]] = []
        for c in self.chunks:
            low = c.source.lower()
            s = sum(low.count(kw.lower()) * (3 if kw.lower() in c.name.lower() else 1) for kw in keywords)
            hot = [ln for ln in boost_lines.get(c.file, []) if c.start <= ln <= c.end]
            if hot:
                s = s / (1 + len(c.source) / 4000) + 1000 + len(hot)
                scored.append((s, c, True))
            elif s:
                scored.append((s / (1 + len(c.source) / 4000), c, False))
        scored.sort(key=lambda t: -t[0])
        picked, used = [], 0
        per_file: Dict[str, int] = {}
        for _, c, is_hot in scored:
            if per_file.get(c.file, 0) >= max_per_file and not is_hot:
                continue
            if len(c.source) <= max_chunk_chars:
                cc = c
            elif is_hot:
                hot = [ln for ln in boost_lines[c.file] if c.start <= ln <= c.end]
                cc = self._window(c, keywords, max_chunk_chars, center_line=hot[len(hot) // 2] - c.start)
            else:
                cc = self._window(c, keywords, max_chunk_chars)
            if used + len(cc.source) > max_chars:
                continue
            picked.append(cc)
            per_file[c.file] = per_file.get(c.file, 0) + 1
            used += len(cc.source)
            if len(picked) >= k:
                break
        return picked

    @staticmethod
    def _window(c: Chunk, keywords: List[str], max_chars: int, center_line: Optional[int] = None) -> Chunk:
        """从大块里截取"关键字命中最密集"的连续行窗口（或以 center_line 为中心），并修正行号。"""
        lines = c.source.splitlines()
        if center_line is None:
            kws = [k.lower() for k in keywords]
            hits = [i for i, ln in enumerate(lines) if any(k in ln.lower() for k in kws)]
            if not hits:
                hits = [0]
            center = hits[len(hits) // 2]
        else:
            center = max(0, min(center_line, len(lines) - 1))
        lo = hi = center
        size = len(lines[center]) + 1
        while True:
            grown = False
            if lo > 0 and size + len(lines[lo - 1]) + 1 <= max_chars:
                lo -= 1; size += len(lines[lo]) + 1; grown = True
            if hi < len(lines) - 1 and size + len(lines[hi + 1]) + 1 <= max_chars:
                hi += 1; size += len(lines[hi]) + 1; grown = True
            if not grown:
                break
        return Chunk(c.file, c.name + " (excerpt)", c.start + lo, c.start + hi, "\n".join(lines[lo:hi + 1]))

    def render(self, chunks: List[Chunk]) -> str:
        return "\n\n".join(f"{c.header()}\n```python\n{c.source}\n```" for c in chunks)

    def outline(self) -> str:
        """文件 + 顶层符号清单，用于系统提示（很短）。"""
        by_file: Dict[str, List[str]] = {}
        for c in self.chunks:
            if c.name not in ("<module>", "<file>"):
                by_file.setdefault(c.file, []).append(c.name)
        lines = []
        for f in self.files:
            syms = by_file.get(f, [])
            lines.append(f"- {f}: {', '.join(syms[:12])}{' ...' if len(syms) > 12 else ''}")
        return "\n".join(lines)

    # ---------- 编辑 ----------
    def read(self, rel: str) -> str:
        return self._read_rel(rel)


@dataclass
class Edit:
    file: str
    search: str
    replace: str


def apply_edits(base_root: Path, work_root: Path, edits: List[Edit], fallback_root: Optional[Path] = None) -> Tuple[Dict[str, str], List[str]]:
    """
    依次应用 SEARCH/REPLACE。文件来源优先级：work_root 已有 > base_root > fallback_root。
    原子性：任一 SEARCH 失败则**不写任何文件**（避免 gate_failed 候选的 patch 记录半应用状态）。
    返回 ({relpath: new_content}, errors)。SEARCH 必须唯一精确匹配。
    """
    changed: Dict[str, str] = {}
    errors: List[str] = []
    for e in edits:
        if e.file.startswith(("/", "\\")) or ".." in e.file.replace("\\", "/").split("/") or ":" in e.file[:3]:
            errors.append(f"{e.file}: illegal path")
            continue
        if e.file not in changed:
            src = None
            for root in (work_root, base_root, fallback_root):
                if root is not None and (root / e.file).exists():
                    src = root / e.file
                    break
            if src is None:
                errors.append(f"{e.file}: file does not exist in baseline")
                continue
            changed[e.file] = src.read_text(encoding="utf-8", errors="ignore")
        content = changed[e.file]
        n = content.count(e.search)
        if n == 0:
            errors.append(f"{e.file}: SEARCH block not found (must match exactly, incl. indentation)")
            continue
        if n > 1:
            errors.append(f"{e.file}: SEARCH block matches {n} places; make it more specific")
            continue
        changed[e.file] = content.replace(e.search, e.replace, 1)
    if errors:
        return changed, errors
    for f, c in changed.items():
        (work_root / f).parent.mkdir(parents=True, exist_ok=True)
        (work_root / f).write_text(c, encoding="utf-8", newline="")
    return changed, errors


def patch_changed_lines(patch_text: str) -> Dict[str, List[int]]:
    """从 unified diff 里抽出每个文件在**新版本**中被改动/新增的行号（用于让子代检索时优先看到父代改过的区域）。"""
    out: Dict[str, List[int]] = {}
    cur: Optional[str] = None
    new_ln = 0
    for ln in patch_text.splitlines():
        if ln.startswith("+++ "):
            cur = ln[4:].strip()
            cur = cur[2:] if cur.startswith("b/") else cur
            out.setdefault(cur, [])
        elif ln.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            new_ln = int(m.group(1)) if m else 0
        elif cur is None or ln.startswith("---"):
            continue
        elif ln.startswith("+"):
            out[cur].append(new_ln)
            new_ln += 1
        elif ln.startswith("-"):
            out[cur].append(new_ln)
        elif ln.startswith("\\"):
            continue
        else:
            new_ln += 1
    return {k: sorted(set(v)) for k, v in out.items() if v}


def _difflib_diff(f: str, a: Path, b: Path) -> str:
    """无 git 时的兜底：difflib 生成 unified diff（git apply 可读；无 index 行）。"""
    import difflib
    at = a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if a.exists() else []
    bt = b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if b.exists() else []
    out = list(difflib.unified_diff(at, bt, fromfile=f"a/{f}" if a.exists() else "/dev/null",
                                    tofile=f"b/{f}" if b.exists() else "/dev/null", n=3))
    if not out:
        return ""
    # difflib 的 --- / +++ 行自带 \n，其余 hunk 行保留原换行
    body = "".join(out)
    if not body.endswith("\n"):
        body += "\n"
    return f"diff --git a/{f} b/{f}\n" + body


def patch_fingerprint(patch_text: str) -> str:
    """去重用：只看 hunk 内容（+/- 行与上下文），忽略 index/mode/行号偏移。"""
    import hashlib as _h
    keep = [ln for ln in patch_text.splitlines()
            if not ln.startswith(("diff --git", "index ", "old mode", "new mode", "--- ", "+++ ", "@@"))]
    return _h.sha256("\n".join(keep).encode("utf-8")).hexdigest()


def git_diff(base_root: Path, work_root: Path, files: List[str]) -> str:
    """用 git diff --no-index 生成标准 unified diff，路径规整为 a/<rel> b/<rel>（可直接 git apply 到基线）。
    无 git 可执行文件时退化为 difflib。"""
    import shutil as _sh
    have_git = _sh.which("git") is not None
    parts = []
    for f in files:
        a, b = base_root / f, work_root / f
        if not have_git:
            parts.append(_difflib_diff(f, a, b))
            continue
        r = subprocess.run(
            ["git", "diff", "--no-index", "--", str(a), str(b)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        lines = []
        for ln in r.stdout.splitlines():
            if ln.startswith("diff --git "):
                lines.append(f"diff --git a/{f} b/{f}")
            elif ln.startswith(("old mode ", "new mode ", "index ")):
                continue   # 文件模式/blob 哈希与内容无关（挂载盘上 chmod 不稳定），去掉让 patch 只反映内容
            elif ln.startswith("--- ") and not ln.startswith("--- /dev/null"):
                lines.append(f"--- a/{f}")
            elif ln.startswith("+++ ") and not ln.startswith("+++ /dev/null"):
                lines.append(f"+++ b/{f}")
            else:
                lines.append(ln)
        parts.append("\n".join(lines) + "\n")
    return "".join(parts)


def apply_unified_diff(patch_text: str, root: Path) -> Tuple[bool, str]:
    """纯 Python 应用 unified diff（git diff --no-index 或 difflib 风格；路径 a/<rel> b/<rel>）。
    逐 hunk 校验上下文与删除行必须逐字匹配；任一不匹配则不写任何文件。"""
    files: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    hunks: Dict[str, List[Tuple[int, List[str]]]] = {}
    # 用 "\n" 切分而不是 splitlines()：行尾 \r 保留在行内容里，CRLF 文件不会被静默改写
    lines = patch_text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("+++ "):
            cur = ln[4:].strip()
            cur = cur[2:] if cur.startswith("b/") else cur
            hunks.setdefault(cur, [])
            i += 1
            continue
        if ln.startswith("@@") and cur is not None:
            m = re.search(r"-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?", ln)
            if not m:
                return False, f"bad hunk header: {ln}"
            old_start = int(m.group(1))
            body: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git", "--- ", "+++ ")):
                if lines[i].startswith("\\"):
                    i += 1
                    continue
                body.append(lines[i])
                i += 1
            hunks[cur].append((old_start, body))
            continue
        i += 1
    new_contents: Dict[str, str] = {}
    for rel, hs in hunks.items():
        path = root / rel
        if not path.exists():
            return False, f"{rel}: file missing in work dir"
        raw = path.read_text(encoding="utf-8", errors="replace")
        src = raw.split("\n")
        out: List[str] = []
        pos = 0  # 0-based index into src
        for old_start, body in hs:
            start = old_start - 1
            if start < pos:
                return False, f"{rel}: overlapping hunks"
            out.extend(src[pos:start])
            pos = start
            for b in body:
                tag, text = (b[0], b[1:]) if b else (" ", "")
                if tag == " ":
                    if pos >= len(src) or src[pos] != text:
                        return False, f"{rel}: context mismatch at line {pos + 1}"
                    out.append(text); pos += 1
                elif tag == "-":
                    if pos >= len(src) or src[pos] != text:
                        return False, f"{rel}: removed line mismatch at line {pos + 1}"
                    pos += 1
                elif tag == "+":
                    out.append(text)
                else:
                    return False, f"{rel}: bad hunk line {b!r}"
        out.extend(src[pos:])
        new_contents[rel] = "\n".join(out)
    for rel, content in new_contents.items():
        (root / rel).write_text(content, encoding="utf-8", newline="")
    return True, ""
