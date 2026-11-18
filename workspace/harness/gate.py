"""
gate.py -- 提交评估前的免费本地门禁。演化对象是**函数**而不是文件，所以检查是 AST 级的：

  1. 边界     : 候选与基线逐文件比较 AST，除"目标函数体"外（含签名、装饰器、模块级语句、其它函数）必须完全一致
  2. 语法     : py_compile
  3. 禁用符号 : 在目标函数体内禁止 —— 文件/网络/子进程/import、非确定性随机、中断梯度的操作
                （.item()/.numpy()/.detach()/.tolist()/argmax/round/floor 作用于概率或表示）、
                算例名 / 规模分支字面量、工艺名分支
  4. 单位陷阱 : 对时序特征列做绝对阈值比较（feats[:,k] < 0.1 之类）→ 错误；要求用算例内相对量
  5. 冒烟     : 合成小图上真跑候选函数 —— 损失：标量 / 有限 / backward 通过 / 梯度非零有限 / 耗时；
                解码：0/1 向量、确定性（同 seed 两次一致）、无宏与极端输入不崩
  6. 平衡预检 : 解码输出修复前的面积/宏偏差超过阈值 → 警告（修复会把结构冲掉）
每项失败都返回可读报告，供 LLM 做一次自动修复。
"""
from __future__ import annotations

import ast
import py_compile
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Target:
    file: str          # 相对 baseline 根
    func: str          # 函数名
    kind: str          # loss | decode


@dataclass
class GateReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    smoke: Dict[str, float] = field(default_factory=dict)

    def text(self) -> str:
        out = [f"ERROR: {e}" for e in self.errors] + [f"WARN : {w}" for w in self.warnings]
        return "\n".join(out) if out else "OK"


# ---------------------------------------------------------------------------
# AST 工具
def _strip(node: ast.AST) -> ast.AST:
    """去掉位置信息与 docstring 差异，便于比较。"""
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, 0)
    return node


def _find_func(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _without_body(tree: ast.Module, name: str) -> str:
    """把目标函数体替换为 `pass`，dump 其余全部。"""
    t = ast.parse(ast.unparse(tree))  # 深拷贝
    f = _find_func(t, name)
    if f is not None:
        f.body = [ast.Pass()]
    return ast.dump(_strip(t), include_attributes=False)


def _signature(f: ast.FunctionDef) -> str:
    a = f.args
    parts = [x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{f.name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
class Gate:
    # 函数体内禁止调用的名字（全名或属性尾名）
    FORBIDDEN_CALLS = {
        "open", "exec", "eval", "compile", "__import__", "input", "print",
        "os.system", "os.popen", "subprocess", "requests", "urllib", "socket",
        "np.random.seed", "numpy.random.seed", "random.seed", "torch.manual_seed", "torch.seed",
        "np.random.rand", "np.random.randn", "np.random.random", "np.random.randint", "np.random.choice", "np.random.permutation",
        "random.random", "random.randint", "random.choice", "random.shuffle",
        "torch.rand", "torch.randn", "torch.randint", "torch.randperm", "torch.bernoulli", "torch.multinomial",
        "time.sleep", "sys.exit", "exit", "quit", "globals", "locals", "setattr", "delattr", "vars",
        "np.save", "np.load", "np.savez", "np.savetxt", "np.memmap", "np.fromfile", "np.tofile", "np.genfromtxt", "np.loadtxt",
        "torch.save", "torch.load", "torch.load_state_dict", "torch.from_file",
    }
    # 损失函数里中断梯度链的属性 / 调用
    GRAD_BREAKERS = {"item", "numpy", "detach", "tolist", "cpu", "no_grad", "argmax", "argmin", "round", "floor", "ceil",
                     "sign", "nonzero", "searchsorted", "bincount", "unique", "scatter_", "histc"}
    # 允许的属性链根（其它一律视为可疑：动态属性 / dunder 爬升 / 模块别名）
    ALLOWED_ROOTS = {"torch", "np", "numpy", "math", "p", "graph", "feats", "z", "neg_idx", "n", "i", "j", "k", "out",
                     "self", "loss", "ratio", "dev", "crit", "cw", "w", "w_edge", "diff2", "slack", "sl", "mu", "sd"}
    BANNED_ATTRS = {"__class__", "__mro__", "__globals__", "__dict__", "__subclasses__", "__bases__", "__code__",
                    "__self__", "__func__", "__getattribute__", "__init_subclass__", "f_back", "f_globals", "f_locals"}
    PDK_RE = re.compile(r"(?<![A-Za-z0-9_])(ng45|nangate45|asap7|sky130|gf180|tsmc|smic)(?![A-Za-z0-9_])", re.I)

    def __init__(self, targets: List[Target], baseline_root: Path, case_names: Optional[List[str]] = None,
                 smoke_enabled: bool = True, smoke_time_limit_s: float = 20.0, feature_names: Optional[List[str]] = None,
                 allow_rng_from_seed: bool = True):
        # feature_names 必须与任务包一致（时序列下标由它决定）；默认值与替身相同，config/gate.feature_names 可覆盖
        self.targets = targets
        self.baseline_root = baseline_root
        self.smoke_enabled = smoke_enabled
        self.smoke_time_limit_s = smoke_time_limit_s
        self.feature_names = feature_names or ["slack", "slew", "arrival", "required", "cell_delay", "timing_coverage", "degree", "fanout_max"]
        self.allow_rng_from_seed = allow_rng_from_seed
        toks = set()
        for c in case_names or []:
            toks.add(c)
            circuit = c.rsplit("-", 1)[0] if "-" in c else c
            if len(circuit) >= 4:
                toks.add(circuit)
            for part in re.split(r"[-_]", c):
                if len(part) >= 4 and part.lower() not in {"case", "ng45", "asap7", "top", "core", "wrapper"}:
                    toks.add(part)
        self.case_tokens = sorted(toks, key=len, reverse=True)
        self._case_re = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, self.case_tokens)) + r")(?![A-Za-z0-9_])", re.I) if self.case_tokens else None

    # ---------- public ----------
    def check(self, changed_files: Dict[str, str], baseline_files: Dict[str, str]) -> GateReport:
        """changed_files: {relpath: new_content}；baseline_files: {relpath: baseline_content}。"""
        rep = GateReport(ok=True)
        by_file = {t.file: t for t in self.targets}
        if not changed_files:
            rep.errors.append("no files changed")
            rep.ok = False
            return rep
        for f, content in changed_files.items():
            if f not in by_file:
                rep.errors.append(f"file not evolvable: {f} (only {sorted(by_file)})")
                continue
            t = by_file[f]
            base = baseline_files.get(f)
            if base is None:
                rep.errors.append(f"{f}: no baseline content to compare against")
                continue
            if not self._check_syntax(f, content, rep):
                continue
            tree_c, tree_b = ast.parse(content), ast.parse(base)
            if not self._check_boundary(f, t, tree_c, tree_b, rep):
                continue
            fn = _find_func(tree_c, t.func)
            self._check_forbidden(f, t, fn, rep)
            self._check_structure(f, t, fn, rep)
            self._check_units(f, t, fn, rep)
        if self.smoke_enabled and not rep.errors:
            self._smoke(changed_files, rep)
        rep.ok = not rep.errors
        return rep

    # ---------- 1/2 ----------
    def _check_syntax(self, f: str, content: str, rep: GateReport) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(content)
            tmp = tf.name
        try:
            py_compile.compile(tmp, doraise=True)
            return True
        except py_compile.PyCompileError as e:
            rep.errors.append(f"syntax error in {f}: {e.msg.splitlines()[-1] if e.msg else e}")
            return False
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _check_boundary(self, f: str, t: Target, tree_c: ast.Module, tree_b: ast.Module, rep: GateReport) -> bool:
        fc, fb = _find_func(tree_c, t.func), _find_func(tree_b, t.func)
        if fb is None:
            rep.errors.append(f"{f}: baseline has no function {t.func} (config error)")
            return False
        if fc is None:
            rep.errors.append(f"{f}: function {t.func} was removed or renamed; it must keep its name")
            return False
        if _signature(fc) != _signature(fb):
            rep.errors.append(f"{f}: signature changed to {_signature(fc)}; must stay {_signature(fb)} (names and order)")
            return False
        if ast.dump(_strip(ast.parse(ast.unparse(fc.args)))) != ast.dump(_strip(ast.parse(ast.unparse(fb.args)))):
            rep.errors.append(f"{f}: defaults/annotations of {t.func} changed; keep the signature identical")
            return False
        if ast.dump(_strip(ast.Module(fc.decorator_list, []))) != ast.dump(_strip(ast.Module(fb.decorator_list, []))):
            rep.errors.append(f"{f}: decorators of {t.func} changed")
            return False
        if _without_body(tree_c, t.func) != _without_body(tree_b, t.func):
            # 找出是哪类改动，给模型可操作的提示
            imp_c = {ast.unparse(n) for n in tree_c.body if isinstance(n, (ast.Import, ast.ImportFrom))}
            imp_b = {ast.unparse(n) for n in tree_b.body if isinstance(n, (ast.Import, ast.ImportFrom))}
            if imp_c != imp_b:
                rep.errors.append(f"{f}: module-level imports changed ({sorted(imp_c - imp_b)} added, {sorted(imp_b - imp_c)} removed); "
                                  f"only the body of {t.func} may change — put helper logic inside the function")
            else:
                rep.errors.append(f"{f}: code outside the body of {t.func} changed (other functions / constants / module statements); "
                                  f"only the body of {t.func} may change — define helpers as nested functions inside it")
            return False
        return True

    # ---------- 3 ----------
    @staticmethod
    def _call_name(node: ast.Call) -> str:
        f = node.func
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        return ".".join(reversed(parts))

    @staticmethod
    def _import_aliases(fn: ast.FunctionDef) -> Dict[str, str]:
        """{别名: 真模块名}：`import numpy as npl` → {npl: numpy}。"""
        out: Dict[str, str] = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Import):
                for a in n.names:
                    out[(a.asname or a.name.split(".")[0])] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    out[(a.asname or a.name)] = f"{n.module}.{a.name}"
        return out

    def _canon_call(self, node: ast.Call, aliases: Dict[str, str]) -> str:
        name = self._call_name(node)
        root, _, rest = name.partition(".")
        if root in aliases:
            name = aliases[root] + (("." + rest) if rest else "")
        if name.startswith("numpy."):
            name = "np." + name[6:]
        return name

    def _check_forbidden(self, f: str, t: Target, fn: ast.FunctionDef, rep: GateReport) -> None:
        aliases = self._import_aliases(fn)
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
                ok = all(m.split(".")[0] in {"torch", "numpy", "math"} for m in mods)
                if not ok:
                    rep.errors.append(f"{f}:{n.lineno}: import of {mods} inside {t.func} is not allowed (only torch / numpy / math)")
            elif isinstance(n, ast.Call):
                name = self._canon_call(n, aliases)
                tail = name.split(".")[-1]
                if name in self.FORBIDDEN_CALLS or tail in {"open", "exec", "eval", "__import__", "exit", "quit", "save", "load", "savetxt", "fromfile", "tofile"}:
                    if self.allow_rng_from_seed and t.kind == "decode" and name.startswith(("np.random.default_rng", "numpy.random.default_rng")):
                        continue
                    rep.errors.append(f"{f}:{n.lineno}: call to {name}() is not allowed inside {t.func} "
                                      f"(no I/O, no global RNG; decode may only use np.random.default_rng(seed))")
                elif name in ("np.random.default_rng", "numpy.random.default_rng") and t.kind == "decode":
                    if not n.args or not self._derives_from_seed(n.args[0], fn):
                        rep.errors.append(f"{f}:{n.lineno}: np.random.default_rng must be seeded from the `seed` argument")
                elif name in ("np.random.default_rng", "numpy.random.default_rng") and t.kind == "loss":
                    rep.errors.append(f"{f}:{n.lineno}: randomness inside the loss is not allowed (sampling is fixed by the trainer)")
                elif t.kind == "loss" and tail in self.GRAD_BREAKERS:
                    rep.errors.append(f"{f}:{n.lineno}: .{tail}() inside the loss breaks the autograd chain or is non-differentiable; "
                                      f"use smooth operations (sigmoid / softplus / clamp / soft ranks)")
            elif isinstance(n, ast.Attribute) and t.kind == "loss" and n.attr in {"data", "grad"}:
                rep.errors.append(f"{f}:{n.lineno}: access to .{n.attr} inside the loss is not allowed")
            elif isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
                if self._case_re is not None and self._case_re.search(n.value):
                    rep.errors.append(f"{f}:{n.lineno}: string literal references a circuit name ({n.value!r}); per-circuit special-casing is forbidden")
                elif self.PDK_RE.search(n.value):
                    rep.errors.append(f"{f}:{n.lineno}: string literal references a PDK/platform ({n.value!r}); candidates must be platform-agnostic")
            elif isinstance(n, ast.Compare):
                # n_nodes == 12345 / n_nets > 50000 之类的规模分支
                left = n.left
                names = [ast.unparse(left)] + [ast.unparse(c) for c in n.comparators]
                if any(re.search(r"n_nodes|n_nets|shape\[0\]|len\(", s) for s in names) and \
                   any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and c.value > 50 for c in [left] + n.comparators):
                    rep.errors.append(f"{f}:{n.lineno}: branching on problem size ({ast.unparse(n)}) looks like per-circuit special-casing")
            elif isinstance(n, ast.Global):
                rep.errors.append(f"{f}:{n.lineno}: `global` inside {t.func} is not allowed")
            elif isinstance(n, ast.While):
                has_break = any(isinstance(x, ast.Break) for x in ast.walk(n))
                cond_true = (isinstance(n.test, ast.Constant) and n.test.value is True) or \
                            (isinstance(n.test, ast.Name) and n.test.id == "True")
                if cond_true and not has_break:
                    rep.errors.append(f"{f}:{n.lineno}: `while True:` without break inside {t.func} is an infinite loop")
                elif t.kind == "loss":
                    rep.warnings.append(f"{f}:{n.lineno}: while-loop inside the loss; make sure it terminates and stays differentiable")

    @staticmethod
    def _derives_from_seed(expr: ast.AST, fn: ast.FunctionDef) -> bool:
        return any(isinstance(x, ast.Name) and x.id == "seed" for x in ast.walk(expr))

    # ---------- 4 ----------
    def _col_aliases(self, fn: ast.FunctionDef) -> Dict[str, str]:
        """函数体内的别名传播：`n = p.shape[0]` / `sl = feats[:, 0]` → {n: "size", sl: "timing"}。
        只做单层赋值（足够拦住最常见的写法），多跳再迭代两轮。"""
        kind: Dict[str, str] = {}
        for _ in range(3):
            changed = False
            for n in ast.walk(fn):
                if not isinstance(n, ast.Assign):
                    continue
                k = self._dtype_value_kind(n.value, kind)
                if k is None:
                    continue
                for t in n.targets:
                    if isinstance(t, ast.Name) and kind.get(t.id) != k:
                        kind[t.id] = k; changed = True
            if not changed:
                break
        return kind

    @staticmethod
    def _dtype_value_kind(v: ast.AST, kind: Dict[str, str]) -> Optional[str]:
        """一个表达式是否"从问题规模派生"（size）或"是一列原始时序特征"（timing）。"""
        # size：p.shape[...] / .shape[0] / len(feats) / graph["n_nodes"] / graph["n_nets"] / .numel()
        if isinstance(v, ast.Attribute) and v.attr in {"shape", "numel", "size"}:
            return "size"
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute) and v.value.attr == "shape":
            return "size"
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Name) and f.id == "len":
                return "size"
            if isinstance(f, ast.Attribute) and f.attr in {"numel", "item", "size"} and isinstance(f.value, (ast.Attribute,)):
                return "size"
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id == "graph" \
                and isinstance(v.slice, ast.Constant) and v.slice.value in {"n_nodes", "n_nets"}:
            return "size"
        if isinstance(v, ast.BinOp) and any(True for _ in [0]):
            lk, rk = Gate._dtype_value_kind(v.left, kind), Gate._dtype_value_kind(v.right, kind)
            if "size" in (lk, rk):
                return "size"
            if "timing" in (lk, rk):
                return "timing"
        if isinstance(v, ast.Name) and v.id in kind:
            return kind[v.id]
        # timing：feats[:, k]（k 在时序列里）/ feats[:, k] 的算术组合
        if isinstance(v, ast.Subscript):
            base = v.value
            if isinstance(base, ast.Name) and base.id in {"feats", "features", "node_feats", "x_feat"}:
                sl = v.slice
                idx = None
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    c = sl.elts[1]
                    idx = c.value if isinstance(c, ast.Constant) else None
                elif isinstance(sl, ast.Constant):
                    idx = sl.value
                if idx is not None:
                    return "timing:" + str(idx)
            return None
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and isinstance(v.func.value, ast.Name) \
                and v.func.value.id in kind and kind[v.func.value.id].startswith("timing"):
            return kind[v.func.value.id]           # sl.mean() / sl.std() 等仍视为时序量
        return None

    def _check_units(self, f: str, t: Target, fn: ast.FunctionDef, rep: GateReport) -> None:
        """时序特征的单位陷阱（含列别名传播）+ 问题规模派生标量的比较/分支（含别名传播）。
        - feats[:, k]（或经赋值别名）与非零常数比较/相加 → 错误（单位跨 PDK 不同，必须先用算例内相对量）
        - p.shape[0] / len(...) / graph["n_nodes"]（或别名）参与比较 → 错误（按规模特判）
        允许：与 0 比较（符号）、乘除常数、先经过 argsort/rank/z-score/归一化（识别为相对量）。
        """
        timing_cols = {i for i, n in enumerate(self.feature_names) if n in {"slack", "slew", "arrival", "required", "cell_delay"}}
        alias = self._col_aliases(fn)

        def is_timing(e: ast.AST) -> bool:
            if isinstance(e, ast.Name) and alias.get(e.id, "").startswith("timing"):
                return True
            k = self._dtype_value_kind(e, alias)
            return k is not None and k.startswith("timing")

        def is_size(e: ast.AST) -> bool:
            if isinstance(e, ast.Name) and alias.get(e.id) == "size":
                return True
            return self._dtype_value_kind(e, alias) == "size"

        def is_relativised(e: ast.AST) -> bool:
            """是否经过相对化调用（argsort / rank / mean / std / min / max / clamp 归一化链）。"""
            if isinstance(e, ast.Call):
                f = e.func
                if isinstance(f, ast.Attribute) and f.attr in {"argsort", "sort", "mean", "std", "median", "norm", "softmax", "log_softmax", "rank"}:
                    return True
                if isinstance(f, ast.Name) and f.id in {"argsort", "rank"}:
                    return True
            if isinstance(e, ast.BinOp):
                for side in (e.left, e.right):
                    if is_size(side):
                        return True
            return False

        for n in ast.walk(fn):
            if isinstance(n, ast.Compare):
                ops = [n.left] + list(n.comparators)
                for side in ops:
                    if is_timing(side) and not is_relativised(side):
                        consts = [c for c in ops if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and c.value != 0]
                        if consts:
                            rep.errors.append(f"{f}:{n.lineno}: absolute threshold on a timing feature ({ast.unparse(n)[:90]}); "
                                              f"units differ across PDKs — convert to case-relative quantities first "
                                              f"(rank / quantile / z-score within the case) and never compare to a non-zero constant")
                            break
                    if is_size(side) and not is_relativised(side):
                        if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and c.value > 8 for c in ops):
                            rep.errors.append(f"{f}:{n.lineno}: branching on problem size ({ast.unparse(n)[:90]}) looks like "
                                              f"per-circuit special-casing; the objective must be circuit-agnostic")
                            break
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
                if is_timing(n.left) ^ is_timing(n.right):
                    other = n.right if is_timing(n.left) else n.left
                    if isinstance(other, ast.Constant) and isinstance(other.value, (int, float)) and other.value != 0:
                        rep.errors.append(f"{f}:{n.lineno}: adding an absolute constant to a timing feature "
                                          f"({ast.unparse(n)[:90]}); units differ across PDKs")
            if isinstance(n, ast.If) and is_size(n.test):
                if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and c.value > 8 for c in ast.walk(n.test)):
                    rep.errors.append(f"{f}:{n.lineno}: if-statement branches on problem size ({ast.unparse(n.test)[:90]}); not allowed")

    # ---------- 3b: 结构化禁用（H5 的允许列表部分） ----------
    def _check_structure(self, f: str, t: Target, fn: ast.FunctionDef, rep: GateReport) -> None:
        """动态属性、dunder 爬升、模块别名、跨调用隐状态。"""
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"getattr", "setattr", "delattr", "vars", "globals", "locals", "hasattr"}:
                rep.errors.append(f"{f}:{n.lineno}: {n.func.id}() is not allowed (dynamic attribute access)")
            elif isinstance(n, ast.Attribute):
                if n.attr in self.BANNED_ATTRS:
                    rep.errors.append(f"{f}:{n.lineno}: attribute .{n.attr} is not allowed")
                # 根名检查：a.b.c 链条的根
                root = n
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id not in self.ALLOWED_ROOTS and root.id not in {"torch", "np", "numpy", "math"}:
                    if n.attr in {"detach", "numpy", "item", "tolist", "cpu", "data", "grad", "requires_grad", "train", "eval"}:
                        rep.errors.append(f"{f}:{n.lineno}: .{n.attr} accessed on `{root.id}`, which is not a tensor/math object "
                                          f"(dynamic access to tensors must go through the named parameters)")
            elif isinstance(n, ast.Assign):
                # 给函数对象挂属性：compute_loss.calls = 0（跨 epoch 隐状态）
                for tg in n.targets:
                    if isinstance(tg, ast.Attribute) and isinstance(tg.value, ast.Name) and tg.value.id in {"compute_loss", "decode_partition", "self"}:
                        rep.errors.append(f"{f}:{n.lineno}: assigning attributes to the function/module object ({ast.unparse(tg)}) is not allowed; "
                                          f"the objective must be stateless")

    # ---------- 5/6 ----------
    def _smoke(self, changed_files: Dict[str, str], rep: GateReport) -> None:
        try:
            from .smoke import smoke_test
        except Exception as ex:  # noqa: BLE001
            rep.warnings.append(f"smoke test unavailable: {ex!r}")
            return
        for t in self.targets:
            if t.file not in changed_files:
                continue
            t0 = time.time()
            errs, warns, metrics = smoke_test(t.kind, changed_files[t.file], t.func, time_limit_s=self.smoke_time_limit_s)
            rep.errors.extend(f"smoke[{t.kind}]: {e}" for e in errs)
            rep.warnings.extend(f"smoke[{t.kind}]: {w}" for w in warns)
            rep.smoke.update({f"{t.kind}_{k}": v for k, v in metrics.items()})
            rep.smoke[f"{t.kind}_wall_s"] = round(time.time() - t0, 3)
