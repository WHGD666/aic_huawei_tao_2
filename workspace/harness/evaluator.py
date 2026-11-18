"""
evaluator.py -- 候选评估接口的抽象 + 实现。

* CommandEvaluator : 调一个外部命令（官方评估接口 / 本地替身），占位 {workdir} {out} {ref}；
                     解析返回 JSON {"runs": {"case/seed": {wns, tns, valid, fail_reason, stats?, log?}}, "partitions": {...}}
* ProxyRunner      : 同样的命令形式，但只跑切分不跑物理流程（免费、不计次），返回切分统计 + 切分 hash（去重用）；
                     按候选源码 hash 缓存
* MockEvaluator    : 确定性伪评估器（哈希 → 伪 WNS/TNS），只用于秒级单元测试
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .scoring import RunMetrics


@dataclass
class EvalResult:
    candidate_id: str
    runs: Dict[str, RunMetrics]                      # "case/seed" -> metrics
    partitions: Dict[str, List[int]] = field(default_factory=dict)   # "case/seed" -> 0/1 list（切分方案）
    stage_logs: Dict[str, str] = field(default_factory=dict)
    wall_s: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def all_valid(self) -> bool:
        return bool(self.runs) and all(r.valid for r in self.runs.values())

    def partition_hash(self) -> Optional[str]:
        """所有 (case/seed) 切分的联合指纹（标签对称：0/1 互换视为相同）。
        没有任何切分数据时返回 None —— 此时禁止去重（否则所有候选指纹相同，第一个之后全被判重复）。"""
        if not self.partitions:
            return None
        for k, p in self.partitions.items():
            if p and not set(p) <= {0, 1}:
                return None
        h = hashlib.sha256()
        for k in sorted(self.partitions):
            p = self.partitions[k]
            if p and p[0] == 1:
                p = [1 - x for x in p]
            h.update(k.encode()); h.update(bytes(p))
        return h.hexdigest()[:16]

    def to_dict(self, with_partitions: bool = True) -> dict:
        d = {"candidate_id": self.candidate_id, "wall_s": self.wall_s,
             "runs": {k: vars(v) for k, v in self.runs.items()}, "stage_logs": self.stage_logs}
        if with_partitions:
            d["partitions"] = self.partitions
        return d


def parse_result_json(cid: str, d: dict, cases: List[str], seeds: List[int]) -> EvalResult:
    runs, logs = {}, {}
    for c in cases:
        for s in seeds:
            key = f"{c}/{s}"
            r = d.get("runs", {}).get(key)
            if r is None:
                runs[key] = RunMetrics(c, s, 0.0, 0.0, valid=False, fail_reason="missing in result")
                continue
            try:
                wns, tns = float(r.get("wns")), float(r.get("tns"))
                finite = wns == wns and tns == tns and abs(wns) != float("inf") and abs(tns) != float("inf")
            except (TypeError, ValueError):
                wns, tns, finite = 0.0, 0.0, False
            v = r.get("valid", True)
            valid_flag = (v is True) or (str(v).strip().lower() not in {"false", "0", "no", "none", ""}) if not isinstance(v, bool) else v
            valid = bool(valid_flag) and finite
            reason = str(r.get("fail_reason", "")) or ("" if finite else "non-finite metric")
            runs[key] = RunMetrics(c, s, wns, tns, valid=valid, fail_reason=reason,
                                   stats={k: v for k, v in (r.get("stats") or {}).items() if isinstance(v, (int, float))})
            if r.get("log"):
                logs[key] = str(r["log"])[-1500:]
    parts = {k: [int(x) for x in v] for k, v in (d.get("partitions") or {}).items()}
    return EvalResult(cid, runs, parts, logs, raw={k: v for k, v in d.items() if k not in ("runs", "partitions")})


class Evaluator(ABC):
    def __init__(self, cases: List[str], seeds: List[int]):
        self.cases = cases
        self.seeds = seeds

    @abstractmethod
    def evaluate(self, candidate_id: str, workdir: Path) -> EvalResult: ...

    @abstractmethod
    def baseline(self) -> EvalResult: ...


# ---------------------------------------------------------------------------
class CommandEvaluator(Evaluator):
    """外部命令评估器。cmd 里可用 {workdir} {out} {ref} {cases} {seeds} 占位。
    基线：若 baseline_json 存在直接读；否则用 baseline_root 跑一次命令（会消耗一次官方评估，由上层决定是否允许）。"""

    def __init__(self, cases, seeds, cmd: str, baseline_json: Path, baseline_root: Optional[Path] = None,
                 timeout_s: float = 6 * 3600, workdir_out: str = "eval_result.json", cwd: Optional[Path] = None):
        super().__init__(cases, seeds)
        self.cmd = cmd
        self.cwd = cwd
        self.baseline_json = baseline_json
        self.baseline_root = baseline_root
        self.timeout_s = timeout_s
        self.workdir_out = workdir_out
        self.last_stderr = ""

    def _run(self, cid: str, workdir: Path, out: Path, ref: Optional[Path]) -> EvalResult:
        t0 = time.time()
        import shlex
        # 参数化：先按模板 format，再 shlex 切分；空 {ref} 及其 "--ref" 标志整体去掉（不做字符串手术，命令含 `{` 也不炸）
        filled = self.cmd.replace("{ref}", "__REF__")
        for k2, v2 in (("workdir", str(workdir)), ("out", str(out)),
                       ("cases", ",".join(self.cases)), ("seeds", ",".join(map(str, self.seeds)))):
            filled = filled.replace("{" + k2 + "}", v2)
        argv = shlex.split(filled, posix=True)
        if ref is not None:
            argv = [str(ref) if a == "__REF__" else a for a in argv]
        else:
            cleaned = []
            skip_next = False
            for a in argv:
                if skip_next:
                    skip_next = False
                    continue
                if a == "__REF__":
                    if cleaned and cleaned[-1] in ("--ref", "-r"):
                        cleaned.pop()
                    continue
                if a in ("--ref", "-r"):
                    skip_next = True
                    continue
                cleaned.append(a)
            argv = cleaned
        try:
            r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout_s, cwd=str(self.cwd) if self.cwd else None)
            self.last_stderr = (r.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            runs = {f"{c}/{s}": RunMetrics(c, s, 0, 0, valid=False, fail_reason="evaluation timeout") for c in self.cases for s in self.seeds}
            return EvalResult(cid, runs, wall_s=time.time() - t0)
        if not out.exists():
            runs = {f"{c}/{s}": RunMetrics(c, s, 0, 0, valid=False, fail_reason=f"no result file (rc={r.returncode})") for c in self.cases for s in self.seeds}
            return EvalResult(cid, runs, stage_logs={"stderr": self.last_stderr}, wall_s=time.time() - t0)
        try:
            d = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            runs = {f"{c}/{s}": RunMetrics(c, s, 0, 0, valid=False, fail_reason=f"bad result json: {ex}") for c in self.cases for s in self.seeds}
            return EvalResult(cid, runs, wall_s=time.time() - t0)
        res = parse_result_json(cid, d, self.cases, self.seeds)
        res.wall_s = time.time() - t0
        return res

    def baseline(self) -> EvalResult:
        if self.baseline_json.exists():
            d = json.loads(self.baseline_json.read_text(encoding="utf-8"))
            return parse_result_json("baseline", d, self.cases, self.seeds)
        if self.baseline_root is None:
            raise RuntimeError(f"baseline result {self.baseline_json} missing and no baseline_root to run it")
        res = self._run("baseline", self.baseline_root, self.baseline_json, None)
        return res

    def evaluate(self, candidate_id: str, workdir: Path) -> EvalResult:
        out = workdir / self.workdir_out
        if out.exists():
            out.unlink()
        return self._run(candidate_id, workdir, out, self.baseline_json if self.baseline_json.exists() else None)


# ---------------------------------------------------------------------------
class ProxyRunner:
    """只跑切分、不跑物理流程的本地代理。返回 EvalResult（wns/tns 为 0，valid 表示切分成功），stats 里是代理指标。
    缓存键 = 候选目标文件内容 hash（同源码同结果）。"""

    def __init__(self, cases, seeds, cmd: str, baseline_json: Path, cache_dir: Path, timeout_s: float = 1800, cwd: Optional[Path] = None):
        self.cases, self.seeds, self.cmd = cases, seeds, cmd
        self.cwd = cwd
        self.baseline_json = baseline_json
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.enabled = bool(cmd.strip())

    @staticmethod
    def source_hash(files: Dict[str, str], extra: str = "") -> str:
        h = hashlib.sha256()
        for k in sorted(files):
            h.update(k.encode()); h.update(files[k].encode("utf-8"))
        h.update(extra.encode("utf-8"))
        return h.hexdigest()[:16]

    def run(self, cid: str, workdir: Path, files: Dict[str, str]) -> Optional[EvalResult]:
        if not self.enabled:
            return None
        ref = ""
        if self.baseline_json.exists():
            ref = hashlib.sha256(self.baseline_json.read_bytes()).hexdigest()[:8]
        key = self.source_hash(files, "|" + ",".join(self.cases) + "|" + ",".join(map(str, self.seeds)) + "|" + self.cmd + "|" + ref)
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            d = json.loads(cached.read_text(encoding="utf-8"))
            res = parse_result_json(cid, d, self.cases, self.seeds)
            res.raw["cached"] = True
            return res
        out = workdir / "proxy_result.json"
        ev = CommandEvaluator(self.cases, self.seeds, self.cmd, self.baseline_json, timeout_s=self.timeout_s, workdir_out="proxy_result.json", cwd=self.cwd)
        res = ev.evaluate(cid, workdir)
        if out.exists():
            cached.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        return res


# ---------------------------------------------------------------------------
class MockEvaluator(Evaluator):
    """秒级确定性伪评估器（单元测试用）：候选文件内容哈希 → 伪随机改善/退化；fail_rate 概率整个候选失败。"""

    BASE = {"case_small": (-0.6, -22.0), "case_mid": (-1.3, -50.0), "case_large": (-3.0, -130.0)}

    def __init__(self, cases, seeds, fail_rate: float = 0.2, sporadic_rate: float = 0.0):
        super().__init__(cases, seeds)
        self.fail_rate, self.sporadic_rate = fail_rate, sporadic_rate

    def baseline(self) -> EvalResult:
        runs, parts = {}, {}
        for c in self.cases:
            w, t = self.BASE.get(c, (-1.0, -40.0))
            for s in self.seeds:
                j = 1 + 0.03 * ((s % 3) - 1)
                runs[f"{c}/{s}"] = RunMetrics(c, s, w * j, t * j, stats={"cut": 100.0, "wcut": 300.0, "crit_cross": 0.1, "repair_flips": 50.0})
                parts[f"{c}/{s}"] = [i % 2 for i in range(50)]
        return EvalResult("baseline", runs, parts)

    def evaluate(self, candidate_id: str, workdir: Path) -> EvalResult:
        t0 = time.time()
        blob = b"".join(p.read_bytes() for p in sorted(workdir.rglob("*.py")))
        h = int(hashlib.sha256(blob).hexdigest(), 16)
        rng = random.Random(h)
        text = blob.decode("utf-8", "ignore").lower()
        bias = sum(b for kw, b in (("slack", 0.06), ("crit", 0.05), ("rank", 0.04), ("hinge", 0.03), ("clamp", 0.02)) if kw in text)
        base = self.baseline().runs
        runs, parts = {}, {}
        cand_fails = rng.random() < self.fail_rate
        for key, b in base.items():
            if cand_fails or rng.random() < self.sporadic_rate:
                runs[key] = RunMetrics(b.case, b.seed, 0, 0, valid=False, fail_reason=rng.choice(["loss nan", "repair failed", "timeout"]))
                continue
            dw, dt = rng.gauss(bias, 0.12), rng.gauss(bias * 1.5, 0.15)
            runs[key] = RunMetrics(b.case, b.seed, b.wns * (1 - dw), b.tns * (1 - dt),
                                   stats={"cut": 100 * (1 - dt * 0.5), "wcut": 300 * (1 - dt), "crit_cross": max(0.0, 0.1 - dt * 0.1), "repair_flips": 50.0})
            parts[key] = [(i + h) % 2 for i in range(50)]
        return EvalResult(candidate_id, runs, parts, wall_s=time.time() - t0)
