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
    }
    # 损失函数里中断梯度链的属性 / 调用
    GRAD_BREAKERS = {"item", "numpy", "detach", "tolist", "cpu", "no_grad", "argmax", "argmin", "round", "floor", "ceil", "sign", "nonzero", "where_"}
    PDK_RE = re.compile(r"(?<![A-Za-z0-9_])(ng45|nangate45|asap7|sky130|gf180|tsmc|smic)(?![A-Za-z0-9_])", re.I)

    def __init__(self, targets: List[Target], baseline_root: Path, case_names: Optional[List[str]] = None,
                 smoke_enabled: bool = True, smoke_time_limit_s: float = 20.0, feature_names: Optional[List[str]] = None,
                 allow_rng_from_seed: bool = True):
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
        if fc.decorator_list != fb.decorator_list and ast.dump(_strip(ast.Module(fc.decorator_list, []))) != ast.dump(_strip(ast.Module(fb.decorator_list, []))):
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

    def _check_forbidden(self, f: str, t: Target, fn: ast.FunctionDef, rep: GateReport) -> None:
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
                ok = all(m.split(".")[0] in {"torch", "numpy", "math"} for m in mods)
                if not ok:
                    rep.errors.append(f"{f}:{n.lineno}: import of {mods} inside {t.func} is not allowed (only torch / numpy / math)")
            elif isinstance(n, ast.Call):
                name = self._call_name(n)
                tail = name.split(".")[-1]
                if name in self.FORBIDDEN_CALLS or tail in {"open", "exec", "eval", "__import__", "exit", "quit"}:
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
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
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
            elif isinstance(n, (ast.While,)) and t.kind == "loss":
                rep.warnings.append(f"{f}:{n.lineno}: while-loop inside the loss; make sure it terminates and stays differentiable")

    @staticmethod
    def _derives_from_seed(expr: ast.AST, fn: ast.FunctionDef) -> bool:
        return any(isinstance(x, ast.Name) and x.id == "seed" for x in ast.walk(expr))

    # ---------- 4 ----------
    def _check_units(self, f: str, t: Target, fn: ast.FunctionDef, rep: GateReport) -> None:
        """feats[:, k] 与非零常数直接比较 / 相加 → 单位相关的绝对阈值 → 错误。
        允许：与 0 比较（符号）、乘除常数（缩放）、先归一化后再比较（无法完美判断，只看直接比较）。"""
        timing_cols = {i for i, n in enumerate(self.feature_names) if n in {"slack", "slew", "arrival", "required", "cell_delay"}}
        feat_names = {a.arg for a in fn.args.args} & {"feats", "features", "x_feat", "node_feats"}

        def is_timing_slice(e: ast.AST) -> bool:
            if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in feat_names:
                sl = e.slice
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant) and sl.elts[1].value in timing_cols:
                    return True
                if isinstance(sl, ast.Constant) and sl.value in timing_cols:
                    return True
            return False

        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and (is_timing_slice(n.left) or any(is_timing_slice(c) for c in n.comparators)):
                consts = [c for c in [n.left] + n.comparators if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and c.value != 0]
                if consts:
                    rep.errors.append(f"{f}:{n.lineno}: absolute threshold on a timing feature ({ast.unparse(n)}); units differ across PDKs — "
                                      f"use case-relative quantities (rank / quantile / z-score within the case)")
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and (is_timing_slice(n.left) or is_timing_slice(n.right)):
                other = n.right if is_timing_slice(n.left) else n.left
                if isinstance(other, ast.Constant) and isinstance(other.value, (int, float)) and other.value != 0:
                    rep.errors.append(f"{f}:{n.lineno}: adding an absolute constant to a timing feature ({ast.unparse(n)}); units differ across PDKs")

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
