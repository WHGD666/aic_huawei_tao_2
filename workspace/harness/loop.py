"""
loop.py -- 演化主循环：提出修改 → 生成代码 → 检查代码 → （本地代理筛选）→ 提交评估 → 分析反馈 → 选择方案。

用法:
    python -m harness.loop [--baseline prob|embed] [--evaluator command|mock]
                           [--max-evaluations 30 --max-cost 15] [--llm-mode live|record|replay]
                           [--run-dir DIR] [--max-rounds N]

预算语义（题面 §十一(二)）：max_evaluations 次候选评估、max_cost_cny 元 API 费，任一触顶即停。
保证：evaluator.evaluate() 与 budget.charge_evaluation() 严格配对；所有 LLM 调用经 _chat 计费；
每个候选在拿到 id 的瞬间就落 meta.json 与血统事件。

每轮流程：
  ① Proposer 一次调用产出 k 个候选（同一父代）
  ② 每个候选：应用 → gate（AST 边界 / 禁用符号 / 单位 / 可微冒烟）→ 失败则一次修复
  ③ 过 gate 的候选跑本地代理（只切分不跑物理流程，若配置了）：拿切分统计 + 切分指纹
       - 指纹与已评估候选相同 → duplicate（继承分数，不花评估）
       - 与基线几乎相同 / 代理显示退化 → screened（不提交）
  ④ 剩下的按 selection_rule 挑 1 个提交官方评估；其余 screened
  ⑤ 分析（规则 + LLM）；更新代理↔真实秩相关
  ⑥ selector 记录结果
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .analyzer import build_analyze_messages, proxy_correlation, render_proxy_correlation, rule_digest
from .budget import Budget, Pricing
from .codebase import Codebase, apply_edits, apply_unified_diff, git_diff, patch_changed_lines, patch_fingerprint
from .evaluator import CommandEvaluator, EvalResult, Evaluator, MockEvaluator, ProxyRunner
from .gate import Gate, Target
from .llm import LLMClient, LLMError
from .profile import case_aliases, proxy_vector, render_baseline_profile
from .proposer import Proposal, Proposer, parse_proposals
from .scoring import RunMetrics, candidate_score
from .selector import Selector
from .store import CandidateMeta, CandidateStore

WS = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    pass


def _candidate_segment(text: str, slot: int) -> str:
    """从多候选响应里取出第 slot 段（0 起），并保留 CANDIDATE 标记。"""
    import re as _re
    marks = list(_re.finditer(r"^\**\s*CANDIDATE\s*(\d+)\s*\**\s*:?\s*$", text, _re.MULTILINE | _re.IGNORECASE))
    if not marks or slot >= len(marks):
        return text
    start = marks[slot].start()
    end = marks[slot + 1].start() if slot + 1 < len(marks) else len(text)
    return text[start:end].strip()


def _env_expand(v):
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return os.environ.get(v[2:-1], "")
    return v


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cfg["llm"]["model"] = _env_expand(cfg["llm"]["model"]) or "deepseek-flash"
    return cfg


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _sha256_tree(root: Path, rels: List[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(rels):
        p = root / rel
        if p.exists():
            h.update(rel.encode()); h.update(b"\0"); h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:16]


def _git_head(path: Path) -> str:
    for p in [path] + list(path.parents):
        if (p / ".git").exists():
            try:
                r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=p, capture_output=True, text=True, timeout=10)
                return r.stdout.strip() if r.returncode == 0 else ""
            except Exception:
                return ""
    return ""


class Harness:
    def __init__(self, cfg: dict, llm_mode: str = "live", run_dir: Optional[Path] = None):
        self.cfg = cfg
        self.baseline_kind = cfg["baseline"]                      # prob | embed
        bl = cfg["baselines"][self.baseline_kind]
        self.run_dir = run_dir or (WS / "logs" / f"run_{self.baseline_kind}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.run_dir / "harness.log", "a", encoding="utf-8")

        # ---- 预算 ----
        pricing = Pricing(WS / "config" / "pricing.yaml")
        b = cfg["budget"]
        self.budget = Budget(b["max_evaluations"], b["max_cost_cny"], pricing, state_file=self.run_dir / "budget.json", reserve_evaluations=0)
        self.per_round_cap = b.get("per_round_cost_cap_cny", 0.5)
        self.max_consecutive_gate_failures = int(b.get("max_consecutive_gate_failures", 6))
        self.max_consecutive_duplicates = int(b.get("max_consecutive_duplicates", 4))
        self.max_consecutive_no_submit = int(b.get("max_consecutive_no_submit", 6))

        # ---- LLM ----
        l = cfg["llm"]
        self.llm = LLMClient(model=l["model"], timeout_s=l["timeout_s"], max_retries=l["max_retries"],
                             record_dir=WS / "logs" / "llm_records", mode=llm_mode)
        self.lcfg = l

        # ---- 演化目标 / 代码库 / 门禁 ----
        self.base_root = Path(bl.get("root") or (WS / "baseline" / self.baseline_kind))
        if not self.base_root.is_absolute():
            self.base_root = WS / self.base_root
        self.targets = [Target(t["file"], t["func"], t["kind"]) for t in bl["targets"]]
        self.cb = Codebase(self.base_root, [t.file for t in self.targets], [])
        missing = [t.file for t in self.targets if not (self.base_root / t.file).exists()]
        if missing:
            raise ConfigError(f"baseline files missing under {self.base_root}: {missing}")
        e = cfg["evaluator"]
        g = cfg.get("gate", {})
        fn_names = g.get("feature_names")
        if isinstance(fn_names, str):
            fn_names = [x.strip() for x in fn_names.split(",")]
        self.gate = Gate(self.targets, self.base_root, case_names=list(e.get("cases", [])),
                         smoke_enabled=bool(g.get("smoke", True)), smoke_time_limit_s=float(g.get("smoke_time_limit_s", 10)),
                         feature_names=fn_names)

        # ---- 评估器（fail-fast：非 mock 必须有命令） ----
        mode = e["mode"]
        self.baseline_json = self.run_dir / "baseline_eval.json"
        if mode == "mock":
            self.evaluator: Evaluator = MockEvaluator(e["cases"], e["seeds"], fail_rate=float(e.get("mock_fail_rate", 0.2)))
        elif mode == "command":
            cmd = str(bl.get("eval_cmd") or e.get("cmd", "")).strip()
            if not cmd:
                raise ConfigError("evaluator.mode=command but no eval_cmd configured — refuse to burn evaluations")
            shared = Path(e.get("baseline_json") or (WS / "baseline" / f"baseline_eval_{self.baseline_kind}.json"))
            if shared.exists() and not self.baseline_json.exists():
                shutil.copy2(shared, self.baseline_json)
            self.evaluator = CommandEvaluator(e["cases"], e["seeds"], cmd, self.baseline_json, baseline_root=self.base_root,
                                              timeout_s=float(e.get("timeout_s", 6 * 3600)), cwd=WS)
        else:
            raise ConfigError(f"unknown evaluator.mode {mode!r}")
        self.baseline_costs_evaluation = bool(e.get("baseline_costs_evaluation", True))

        # ---- 本地代理（可选） ----
        p = cfg.get("proxy", {})
        pcmd = str(bl.get("proxy_cmd") or p.get("cmd", "")).strip() if p.get("enabled", True) else ""
        self.proxy = ProxyRunner(e["cases"], e["seeds"], pcmd, self.baseline_json, self.run_dir / "proxy_cache",
                                 timeout_s=float(p.get("timeout_s", 1800)), cwd=WS)
        self.proxy_min_change = float(p.get("min_change_vs_baseline", 0.005))
        self.selection_rule = str(p.get("selection_rule", "proxy_best"))

        # ---- 历史 / 选择 ----
        self.store = CandidateStore(self.run_dir / "candidates")
        s = cfg["search"]
        self.aliases: Dict[str, str] = {}
        self.selector = Selector(self.store, s["elite_size"], s["robustness_metric"], s["max_consecutive_failures_per_lineage"],
                                 s.get("phase_plan"), seed=cfg.get("seed", 0), kappa_seed=float(s.get("kappa_seed", 0.5)))
        if sum(self.selector.phase_plan.values()) != self.budget.max_evaluations:
            self.log(f"phase_plan sum {sum(self.selector.phase_plan.values())} != max_evaluations {self.budget.max_evaluations}; rescaling")
            self.selector.rescale_phase_plan(self.budget.max_evaluations)
        self.propose_k = int(s.get("propose_k", 2))
        self.proposer = Proposer(self.cb, self.baseline_kind, self._targets_desc(), propose_k=self.propose_k)
        self.code_chars = int(l.get("code_chars", 8000))
        self.abl = {k: bool(v) for k, v in (cfg.get("ablation") or {}).items()}
        self.proposer.use_knowledge = self.abl.get("knowledge_block", True)
        self.proposer.rebuild_system()

        self.baseline_eval: Optional[EvalResult] = None
        self.profile = ""
        # 续跑：上次运行被中断时停在 proposed 的候选（无评估记录）标为 discarded，避免血统/历史里出现悬空状态
        for m in self.store.all():
            if m.status == "proposed":
                m.status = "discarded"
                m.fail_reasons = ["run interrupted before gate/evaluation"]
                self.store.update(m, "interrupted")
        self._patch_hashes: Dict[str, str] = {}
        self._part_hashes: Dict[str, str] = {}         # 切分指纹 -> 候选 id（去重）
        self._proxy_samples: List[Tuple[Dict[str, float], float]] = []
        for m in self.store.all():
            pth = self.store.dir(m.id) / "candidate.patch"
            if pth.exists() and m.status in ("evaluated", "invalid") and pth.stat().st_size > 0:
                self._patch_hashes.setdefault(patch_fingerprint(pth.read_text(encoding="utf-8")), m.id)
            if m.status == "evaluated" and m.proxy and m.score_total is not None:
                self._proxy_samples.append((m.proxy, m.score_total))
            ev = self.store.dir(m.id) / "eval.json"
            if m.status == "evaluated" and ev.exists():
                try:
                    d = json.loads(ev.read_text(encoding="utf-8"))
                    if d.get("partition_hash"):
                        self._part_hashes.setdefault(d["partition_hash"], m.id)
                except json.JSONDecodeError:
                    pass

    def _targets_desc(self) -> str:
        return "\n".join(f"  - {t.file}::{t.func}  ({'training loss, must stay differentiable' if t.kind == 'loss' else 'deterministic decoder: embeddings -> 0/1 tier vector'})"
                         for t in self.targets)

    # ---------- util ----------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            self.log_f.write(line + "\n"); self.log_f.flush()
        except Exception:
            pass

    def _chat(self, msgs, temperature, max_tokens, tag, cid: Optional[str] = None):
        if not self.budget.can_call_llm():
            raise LLMError("cost budget exhausted")
        text, usage = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens, tag=tag)
        cost = self.budget.charge_llm(self.lcfg["model"], usage)
        if cid:
            self.store.save_llm(cid, tag, msgs, text, {**vars(usage), "cost_cny": cost})
        self.log(f"  llm[{tag}] in={usage.prompt_tokens} hit={usage.cache_hit_tokens} out={usage.completion_tokens} cost={cost:.4f}")
        return text, cost

    # ---------- init ----------
    def init_baseline(self) -> None:
        had = self.baseline_json.exists()
        self.baseline_eval = self.evaluator.baseline()
        if not had and self.baseline_costs_evaluation and isinstance(self.evaluator, CommandEvaluator):
            self.budget.charge_evaluation()
            self.log("baseline evaluation charged to the budget (configured)")
        if not self.baseline_eval.all_valid:
            bad = [k for k, r in self.baseline_eval.runs.items() if not r.valid]
            raise ConfigError(f"baseline itself has invalid runs {bad}; cannot score candidates")
        if isinstance(self.evaluator, CommandEvaluator) and not self.baseline_json.exists():
            self.baseline_json.write_text(json.dumps(self.baseline_eval.to_dict()), encoding="utf-8")
        (self.store.root / "baseline").mkdir(exist_ok=True)
        self.store.save_eval("baseline", {**self.baseline_eval.to_dict(with_partitions=False), "partition_hash": self.baseline_eval.partition_hash()})
        self.aliases = case_aliases(self.evaluator.cases, self.baseline_eval)
        self.selector.aliases = self.aliases
        self.profile = render_baseline_profile(self.baseline_eval, self.aliases) if self.abl.get("problem_profile", True) else \
            "Baseline loaded (profile disabled)."
        bph = self.baseline_eval.partition_hash()
        if bph is not None:
            self._part_hashes.setdefault(bph, "baseline")
        else:
            self.log("baseline has no partition data: duplicate detection is disabled for this run")
        self.log("baseline loaded\n" + self.profile)

    # ---------- work dir ----------
    def _prepare_work(self, cid: str, parent_id: Optional[str]) -> Path:
        work = self.run_dir / "work" / cid
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for t in self.targets:                       # 工作副本始终是完整的目标文件集合（评估命令要能直接读）
            (work / t.file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.base_root / t.file, work / t.file)
        if parent_id:
            patch = self.store.dir(parent_id) / "candidate.patch"
            if not patch.exists():
                raise RuntimeError(f"parent {parent_id} has no candidate.patch; lineage would be wrong")
            if patch.stat().st_size > 0:
                self._apply_patch(patch, work, parent_id)
        return work

    def _apply_patch(self, patch: Path, work: Path, parent_id: str) -> None:
        """把父代 patch 打到 work。先用纯 Python 的 unified-diff 应用（不依赖 git、不受"work 在某个 git 仓库里"影响），
        失败再退回复制父代工作副本。"""
        from .codebase import apply_unified_diff
        before = {t.file: (work / t.file).read_bytes() for t in self.targets if (work / t.file).exists()}
        ok, err = apply_unified_diff(patch.read_text(encoding="utf-8"), work)
        after = {t.file: (work / t.file).read_bytes() for t in self.targets if (work / t.file).exists()}
        if not ok or before == after:
            self.log(f"  applying parent patch failed ({err or 'no file changed'}); falling back to parent work copy")
            pw = self.run_dir / "work" / parent_id
            if not pw.exists():
                raise RuntimeError(f"cannot rebuild parent {parent_id}: patch does not apply and work dir missing")
            for t in self.targets:
                if (pw / t.file).exists():
                    shutil.copy2(pw / t.file, work / t.file)

    def _changed_files_in(self, work: Path) -> List[str]:
        out = []
        for t in self.targets:
            p = work / t.file
            if p.exists() and p.read_bytes() != (self.base_root / t.file).read_bytes():
                out.append(t.file)
        return out

    def _stray_files(self, work: Path) -> List[str]:
        """work 目录里**不是目标文件**、且内容与基线不同的文件（候选越界写入）。
        edit 的路径若不在白名单，apply_edits 会把它写进 work —— 必须在这里拦下，否则门禁只看 targets、
        patch 也只含 targets，越界改动会被评估命令真实使用。"""
        allowed = {t.file for t in self.targets}
        stray = []
        for p in work.rglob("*"):
            if not p.is_file() or p.name == "candidate.patch" or p.name in ("eval_result.json", "proxy_result.json"):
                continue
            rel = str(p.relative_to(work)).replace("\\", "/")
            if rel in allowed:
                continue
            base = self.base_root / rel
            if not base.exists() or base.read_bytes() != p.read_bytes():
                stray.append(rel)
        return sorted(stray)

    def _target_files(self, work: Path) -> Dict[str, str]:
        return {t.file: (work / t.file).read_text(encoding="utf-8", errors="ignore") for t in self.targets if (work / t.file).exists()}

    def _apply_and_gate(self, prop: Proposal, work: Path, parent_id: Optional[str], shown_chunks):
        """返回 (errors, gate_report, patch)。"""
        if prop.parse_errors:
            return list(prop.parse_errors), None, ""
        pre = self.proposer.precheck(prop, shown_chunks) if self.abl.get("precheck", True) else []
        if pre:
            return pre, None, ""
        _, errs = apply_edits(work, work, prop.edits, fallback_root=self.base_root)
        if errs:
            return errs, None, ""
        changed = self._changed_files_in(work)
        patch = git_diff(self.base_root, work, changed) if changed else ""
        full = {f: (work / f).read_text(encoding="utf-8", errors="ignore") for f in changed}
        base = {f: (self.base_root / f).read_text(encoding="utf-8", errors="ignore") for f in changed}
        rep = self.gate.check(full, base)
        stray = self._stray_files(work)
        if stray:
            rep.errors.append(f"files outside the evolvable targets were modified or created: {stray[:6]}; "
                              f"only the body of the target function(s) may change — helpers must be nested inside it")
            rep.ok = False
        if rep.ok:
            parent_patch_text = (self.store.dir(parent_id) / "candidate.patch").read_text(encoding="utf-8") if parent_id else ""
            if not patch.strip() or patch == parent_patch_text:
                rep.errors.append("no effective change relative to parent/baseline (REPLACE equals SEARCH or edit reverted)"); rep.ok = False
            else:
                h = patch_fingerprint(patch)
                if h in self._patch_hashes:
                    rep.errors.append(f"identical patch already evaluated as {self._patch_hashes[h]}; propose something different"); rep.ok = False
        return [], rep, patch

    # ---------- one round ----------
    def round(self, idx: int) -> List[CandidateMeta]:
        evals_done = self.budget.state.evaluations
        phase = self.selector.phase_for(evals_done)
        parent_id = self.selector.choose_parent(phase)
        parent_patch = (self.store.dir(parent_id) / "candidate.patch").read_text(encoding="utf-8") if parent_id else None
        k = 1 if phase == "final" else self.propose_k
        self.log(f"=== round {idx} | phase={phase} | parent={parent_id or 'base'} | k={k} | {self.budget.summary()}")

        # 提示词上下文：父代代码叠加
        parent_work = self._prepare_work("_parent_view", parent_id)
        self.proposer.cb = Codebase.overlay(self.cb, parent_work)
        boost = patch_changed_lines(parent_patch) if (parent_patch and self.abl.get("boost_parent_lines", True)) else None
        hist = self.selector.history_digest(with_analysis=self.abl.get("analysis_in_history", True))
        proxy_info = render_proxy_correlation(proxy_correlation(self._proxy_samples)) if self.abl.get("proxy_feedback", True) else ""
        msgs, shown = self.proposer.build_messages(phase, self.profile, hist, parent_patch, proxy_info, code_chars=self.code_chars, boost_lines=boost, k=k)

        # 登记 k 个候选（规则：任何之后的崩溃都有 meta.json 与血统事件）
        metas: List[CandidateMeta] = []
        for slot in range(k):
            cid = self.store.new_id()
            m = CandidateMeta(cid, parent_id, "(proposing)", "", [], phase)
            m.slot = slot
            self.store.create(m, "")
            metas.append(m)
        round_cost = 0.0
        try:
            text, c = self._chat(msgs, self.lcfg["temperature_propose"], self.lcfg["max_tokens_propose"], "propose", metas[0].id)
            round_cost += c
            props = parse_proposals(text, k)
            # 每个候选：应用 + gate（一次修复）
            works: Dict[str, Path] = {}
            results: Dict[str, Tuple[List[str], object, str]] = {}
            for m in metas:
                prop = props[m.slot] if m.slot < len(props) else Proposal("", "", [], [], "", ["candidate slot not produced by the model"], m.slot)
                m.hypothesis = prop.hypothesis or "(unparsed)"; m.expected = prop.expected; m.tags = prop.keywords[:8]
                self.store.update(m, "proposed")
                work = self._prepare_work(m.id, parent_id)
                errs, rep, patch = self._apply_and_gate(prop, work, parent_id, shown)
                if (errs or (rep and not rep.ok)) and self.abl.get("fix_round", True) and self.budget.can_call_llm(0.05):
                    fb = f"[CANDIDATE {m.slot + 1}]\n" + ("\n".join(errs) if errs else rep.text())
                    self.log(f"  {m.id} gate/apply failed -> fix attempt: {fb[:200]}")
                    seg = text
                    if len(props) > 1 and m.slot < len(props):
                        # 只把该候选自己的段落当作 previous_response：模型修 slot n 时不应看到/重发其它候选
                        seg = _candidate_segment(text, m.slot)
                    msgs2, shown2 = self.proposer.build_messages(phase, self.profile, hist, parent_patch, proxy_info, feedback_for_fix=fb,
                                                                previous_response=seg, code_chars=self.code_chars, boost_lines=boost, k=1)
                    text2, c2 = self._chat(msgs2, self.lcfg["temperature_fix"], self.lcfg["max_tokens_fix"], "fix", m.id)
                    round_cost += c2
                    prop2 = parse_proposals(text2, 1)[0]
                    if prop2.hypothesis:
                        m.hypothesis = prop2.hypothesis; m.expected = prop2.expected; m.tags = prop2.keywords[:8]
                    work = self._prepare_work(m.id, parent_id)
                    errs, rep, patch = self._apply_and_gate(prop2, work, parent_id, shown2)
                works[m.id] = work
                results[m.id] = (errs, rep, patch)
        finally:
            per = round_cost / max(len(metas), 1)
            for m in metas:
                m.llm_cost_cny += per
                if m.status == "proposed":      # propose 阶段抛异常：不留悬空状态
                    m.status = "discarded"
                    m.fail_reasons = ["proposal stage crashed before gate"]
                    self.store.update(m, "proposal_crash")
            self._adapt_code_chars(round_cost)

        # gate 结论
        passed: List[CandidateMeta] = []
        # 冒烟子进程连 import 都失败 = 环境/移植问题（不是候选的错）：直接停，别把每个候选都判 gate_failed
        for m in metas:
            errs, rep, patch = results.get(m.id, (["harness error before gate"], None, ""))
            msg = ("\n".join(errs) if errs else (rep.text() if rep and not rep.ok else ""))
            if "smoke subprocess failed" in msg or "cannot load candidate function" in msg:
                raise ConfigError(f"smoke environment is broken (porting/config issue, not a candidate issue): {msg[:300]}")
            work = works.get(m.id)
            m.files_changed = self._changed_files_in(work) if work else []
            self.store.write_patch(m.id, patch)
            if work:
                (work / "candidate.patch").write_text(patch, encoding="utf-8")
            if errs or (rep and not rep.ok):
                m.status = "gate_failed"
                m.gate_report = "\n".join(errs) if errs else rep.text()
                m.fail_reasons = [m.gate_report[:200]]
                self.store.update(m, "gate_failed")
                self.log(f"  {m.id} gate_failed: {m.gate_report[:140]}")
                continue
            m.gate_report = rep.text() if rep else "OK"
            m._smoke_metrics = dict(rep.smoke) if (rep and rep.smoke) else {}   # 冒烟指标单独保存，代理统计在前
            passed.append(m)
        if not passed:
            return metas

        # ③ 本地代理筛选
        submit: List[CandidateMeta] = []
        for m in passed:
            res = self.proxy.run(m.id, works[m.id], self._target_files(works[m.id]))
            if res is None:
                submit.append(m); continue
            self.store.save_eval(m.id, {"proxy": res.to_dict(with_partitions=False), "partition_hash": res.partition_hash()})
            if not res.all_valid:
                m.status = "invalid"; m.all_valid = False
                m.fail_reasons = [f"{k2}:{r.fail_reason}" for k2, r in res.runs.items() if not r.valid][:6]
                m.analysis = "Failed already in the local partition run (no evaluation spent): " + "; ".join(m.fail_reasons)[:200]
                self.store.update(m, "proxy_invalid")
                self.log(f"  {m.id} invalid in proxy run (no evaluation consumed): {m.fail_reasons[:2]}")
                continue
            smoke = getattr(m, "_smoke_metrics", {}) or {}
            m.proxy.update({**proxy_vector(res), **{f"smoke_{k2}": v for k2, v in smoke.items()}})
            ph = res.partition_hash()
            if ph is not None and ph in self._part_hashes:
                dup = self._part_hashes[ph]
                m.status = "duplicate"; m.duplicate_of = dup
                if dup != "baseline":
                    src = self.store.get(dup)
                    m.score_total, m.score_robust, m.per_case, m.all_valid, m.seed_std = src.score_total, src.score_robust, dict(src.per_case), src.all_valid, src.seed_std
                else:
                    m.score_total, m.score_robust, m.all_valid = 0.0, 0.0, True
                self.store.update(m, "duplicate", of=dup)
                self.log(f"  {m.id} partition identical to {dup}; no evaluation consumed")
                continue
            ham = m.proxy.get("hamming_ref")
            if ham is not None and ham < self.proxy_min_change:
                m.status = "screened"
                m.analysis = f"Partition differs from the baseline on only {ham:.2%} of nodes; expected score ~0, not submitted."
                self.store.update(m, "screened", reason="negligible change")
                self.log(f"  {m.id} screened: negligible change vs baseline ({ham:.2%})")
                continue
            submit.append(m)
        if not submit:
            return metas
        chosen = self._pick_for_submission(submit, phase)
        for m in submit:
            if m is not chosen:
                m.status = "screened"
                self.store.update(m, "screened", reason="another candidate of this round was submitted")

        # ④ 提交评估
        if not self.budget.can_evaluate():
            self.log("  evaluation budget exhausted; candidate discarded")
            chosen.status = "discarded"; self.store.update(chosen, "no_budget")
            return metas
        self._evaluate_and_analyze(chosen, works[chosen.id])
        return metas

    def _pick_for_submission(self, cands: List[CandidateMeta], phase: str) -> CandidateMeta:
        if len(cands) == 1 or self.selection_rule == "first":
            return cands[0]
        corr = proxy_correlation(self._proxy_samples)
        # 有相关性证据：用符号加权的代理指标 z-score 打分；没有：探针阶段选"变化最大"（信息量），其它阶段选 wcut 最小
        def score(m: CandidateMeta) -> float:
            if corr:
                s = 0.0
                for k2, (rho, n) in corr.items():
                    if rho is None or k2 not in m.proxy:
                        continue
                    vals = [x.proxy[k2] for x in cands if k2 in x.proxy]
                    mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
                    s += rho * (m.proxy[k2] - mu) / sd
                return s
            if phase == "probe":
                return m.proxy.get("hamming_ref", 0.0)
            return -m.proxy.get("wcut", 0.0)
        return max(cands, key=score)

    def _evaluate_and_analyze(self, meta: CandidateMeta, work: Path) -> None:
        try:
            self.budget.begin_evaluation()          # 预扣：中途被杀也不会漏记
            try:
                res = self.evaluator.evaluate(meta.id, work)
            except Exception as ex:  # noqa: BLE001
                self.log(f"  evaluator raised: {ex!r}")
                res = EvalResult(meta.id, {k: RunMetrics(r.case, r.seed, 0, 0, valid=False, fail_reason=f"evaluator error: {ex!r}"[:120])
                                           for k, r in self.baseline_eval.runs.items()})
            self.budget.charge_evaluation()
            d = res.to_dict(with_partitions=False)
            d["partition_hash"] = res.partition_hash()
            prev = self.store.dir(meta.id) / "eval.json"
            if prev.exists():
                try:
                    d["proxy"] = json.loads(prev.read_text(encoding="utf-8")).get("proxy")
                except json.JSONDecodeError:
                    pass
            self.store.save_eval(meta.id, d)
            if res.partitions:
                (self.store.dir(meta.id) / "partitions.json").write_text(json.dumps(res.partitions), encoding="utf-8")
            meta.eval_wall_s = res.wall_s
            # 只有真正被评估过的 patch 才登记（screened/discarded 的必须允许以后再提）
            pth = self.store.dir(meta.id) / "candidate.patch"
            if pth.exists():
                self._patch_hashes[patch_fingerprint(pth.read_text(encoding="utf-8"))] = meta.id
            score = candidate_score(self.baseline_eval.runs, res.runs)
            digest = rule_digest(self.baseline_eval, res, score, self.aliases)
            meta.all_valid = bool(score.all_valid and res.all_valid)
            meta.per_case = score.per_case
            meta.fail_reasons = [f"{self.aliases.get(r.case, r.case)}/{r.seed}:{r.fail_reason}" for r in res.runs.values() if not r.valid][:6]
            if not meta.proxy or all(k2.startswith("smoke_") for k2 in meta.proxy):
                meta.proxy.update(proxy_vector(res))
            if meta.all_valid:
                meta.status = "evaluated"
                meta.score_total = score.total
                meta.seed_std = score.seed_std_mean
                meta.score_robust = score.robust(self.selector.robustness, self.selector.kappa_seed)
                ph = res.partition_hash()
                if ph is not None:
                    self._part_hashes.setdefault(ph, meta.id)
                pv = {k2: v for k2, v in meta.proxy.items() if not k2.startswith("smoke_")}
                if pv:
                    self._proxy_samples.append((pv, score.total))
            else:
                meta.status = "invalid"
            if self.budget.can_call_llm(0.02) and self.abl.get("llm_analysis", True):
                try:
                    a, c = self._chat(build_analyze_messages(meta.hypothesis, digest), self.lcfg["temperature_analyze"],
                                      self.lcfg["max_tokens_analyze"], "analyze", meta.id)
                    meta.analysis = a.strip(); meta.llm_cost_cny += c
                except LLMError as ex:
                    self.log(f"  analyze skipped: {ex}")
        finally:
            if meta.status == "proposed":
                meta.status = "invalid"
                meta.fail_reasons = meta.fail_reasons or ["harness error during evaluation/analysis"]
            self.store.update(meta, meta.status)
        self.selector.record_outcome(meta)
        self.log(f"  {meta.id} {meta.status} total={meta.score_total} robust={meta.score_robust}\n{digest}")

    def _adapt_code_chars(self, round_cost: float) -> None:
        if round_cost > self.per_round_cap:
            self.code_chars = max(3000, int(self.code_chars * 0.7))
            self.log(f"  round cost {round_cost:.3f} > cap {self.per_round_cap}; code_chars -> {self.code_chars}")

    # ---------- run ----------
    def run(self, max_rounds: Optional[int] = None) -> Optional[CandidateMeta]:
        if max_rounds is None:
            max_rounds = 3 * self.budget.max_evaluations
        best = None
        try:
            self.init_baseline()
            idx = 0
            consecutive_errors = consecutive_gate = consecutive_dup = consecutive_nosub = 0
            while self.budget.can_evaluate() and idx < max_rounds:
                idx += 1
                try:
                    metas = self.round(idx)
                    consecutive_errors = 0
                    consecutive_gate = consecutive_gate + 1 if all(m.status == "gate_failed" for m in metas) else 0
                    consecutive_dup = consecutive_dup + 1 if any(m.status == "duplicate" for m in metas) and not any(m.status == "evaluated" for m in metas) else 0
                    if consecutive_gate >= self.max_consecutive_gate_failures:
                        self.log(f"  {consecutive_gate} consecutive all-gate-failed rounds; stopping to protect cost budget"); break
                    consecutive_nosub = consecutive_nosub + 1 if not any(m.status == "evaluated" for m in metas) else 0
                    if consecutive_nosub >= self.max_consecutive_no_submit:
                        self.log(f"  {consecutive_nosub} consecutive rounds produced no submitted evaluation "
                                 f"(everything screened / invalid / gate_failed); stopping to protect cost budget")
                        break
                    if consecutive_dup >= self.max_consecutive_duplicates:
                        self.log(f"  {consecutive_dup} consecutive rounds whose partitions were identical to already-evaluated candidates "
                                 f"(the objective is insensitive at this granularity); stopping to protect cost budget")
                        break
                except LLMError as e:
                    consecutive_errors += 1
                    self.log(f"  LLM error ({consecutive_errors}): {e}")
                    if "budget exhausted" in str(e) or consecutive_errors >= 3:
                        break
                    time.sleep(5 * consecutive_errors)
                except Exception as e:  # noqa: BLE001
                    consecutive_errors += 1
                    import traceback
                    self.log(f"  round {idx} crashed: {e!r}\n{traceback.format_exc()[-800:]}")
                    if consecutive_errors >= 3:
                        break
            best = self.selector.final_pick()
        except KeyboardInterrupt:
            self.log("interrupted; finalizing")
            best = self.selector.final_pick()
        finally:
            self.finalize(best)
        return best

    def _holdout_check(self, best: CandidateMeta) -> Tuple[bool, str]:
        """最终候选的泛化自检（免费）：用与训练集不同形态的合成输入（无宏 / 不同工艺单位）跑一遍目标函数。
        对应题面"验证/隐藏算例上任一种子失败 → 整个最终候选无效"。返回 (ok, 说明)。
        没有 torch 环境或替身不可用时返回 (True, "skipped")。"""
        try:
            from .smoke import smoke_test
            from .codebase import apply_unified_diff
        except Exception:
            return True, "skipped (smoke unavailable)"
        import tempfile as _tf
        with _tf.TemporaryDirectory() as td:
            tw = Path(td)
            for t in self.targets:
                (tw / t.file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.base_root / t.file, tw / t.file)
            ok, err = apply_unified_diff((self.store.dir(best.id) / "candidate.patch").read_text(encoding="utf-8"), tw)
            if not ok:
                return True, f"skipped (patch rebuild failed: {err})"
            for t in self.targets:
                src = (tw / t.file).read_text(encoding="utf-8")
                e, _w, _m = smoke_test(t.kind, src, t.func, time_limit_s=float(self.cfg.get("gate", {}).get("smoke_time_limit_s", 10)))
                # smoke 内部已经覆盖"无宏 + 不同单位（asap7 ps）"两种退化输入
                if e:
                    return False, f"{t.func}: {e[0][:200]}"
        return True, "ok (no-macro and ps-unit synthetic inputs passed the differentiability feasibility smoke)"

    def finalize(self, best: Optional[CandidateMeta]) -> None:
        self.log("=== done | " + self.budget.summary())
        tree = self.store.export_tree_text()
        (self.run_dir / "lineage_tree.txt").write_text(tree, encoding="utf-8")
        self.log("lineage:\n" + tree)
        corr = proxy_correlation(self._proxy_samples)
        summary = {
            "baseline": self.baseline_kind, "run_dir": str(self.run_dir), "budget": self.budget.state.to_dict(),
            "algorithms_dir": str(Path(self.cfg.get("algorithms_dir") or (WS / "algorithms"))),
            "n_candidates": len(self.store.all()), "n_evaluated": len(self.store.evaluated()),
            "n_gate_failed": sum(1 for m in self.store.all() if m.status == "gate_failed"),
            "n_screened": sum(1 for m in self.store.all() if m.status == "screened"),
            "n_duplicate": sum(1 for m in self.store.all() if m.status == "duplicate"),
            "n_invalid": sum(1 for m in self.store.all() if m.status == "invalid"),
            "proxy_correlation": {k: {"rho": v[0], "n": v[1]} for k, v in corr.items()},
            "case_aliases": self.aliases,
            "best": vars(best) if best else None,
            "code_version": {"harness_git": _git_head(WS), "baseline_targets_sha256": _sha256_tree(self.base_root, self.cb.files),
                             "model": self.lcfg["model"], "evaluator_mode": self.cfg["evaluator"]["mode"], "targets": [vars(t) for t in self.targets]},
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        if best:
            ok_hold, why_hold = self._holdout_check(best)
            summary["holdout_check"] = {"ok": ok_hold, "detail": why_hold}
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            if not ok_hold:
                self.log(f"  WARNING holdout check failed ({why_hold}); exporting anyway but the candidate may be invalid on unseen circuits")
        if best:
            out = Path(self.cfg.get("algorithms_dir") or (WS / "algorithms")) / self.baseline_kind
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.store.dir(best.id) / "candidate.patch", out / "candidate.patch")
            # 导出完整的最终函数文件：从基线 + candidate.patch 重建（work/ 可能已被清理；绝不能静默回退基线代码）
            import tempfile as _tf
            with _tf.TemporaryDirectory() as td:
                tw = Path(td)
                for t in self.targets:
                    (tw / t.file).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.base_root / t.file, tw / t.file)
                ok, err = apply_unified_diff((self.store.dir(best.id) / "candidate.patch").read_text(encoding="utf-8"), tw)
                if not ok:
                    self.log(f"  WARNING: could not rebuild final files from candidate.patch ({err}); falling back to the work copy")
                    for t in self.targets:
                        w = self.run_dir / "work" / best.id / t.file
                        if not w.exists():
                            raise RuntimeError(f"export: neither patch rebuild nor work copy available for {t.file}")
                        shutil.copy2(w, out / Path(t.file).name)
                else:
                    for t in self.targets:
                        shutil.copy2(tw / t.file, out / Path(t.file).name)
            (out / "SELECTED.json").write_text(json.dumps(vars(best), indent=1, ensure_ascii=False), encoding="utf-8")
            self.log(f"best = {best.id} total={best.score_total:+.3f} -> {out}")
        else:
            self.log("no valid candidate produced")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agent harness: 提出修改→生成代码→检查→评估→分析→选择")
    ap.add_argument("--config", default=str(WS / "config" / "harness.yaml"))
    ap.add_argument("--baseline", default=None, choices=[None, "prob", "embed"])
    ap.add_argument("--llm-mode", default="live", choices=["live", "record", "replay"])
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--max-evaluations", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=None)
    ap.add_argument("--evaluator", default=None, choices=[None, "mock", "command"])
    ap.add_argument("--eval-cmd", default=None)
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    load_dotenv(WS / ".env")
    cfg = load_config(Path(args.config))
    if args.baseline:
        cfg["baseline"] = args.baseline
    if args.max_evaluations is not None:
        cfg["budget"]["max_evaluations"] = args.max_evaluations
    if args.max_cost is not None:
        cfg["budget"]["max_cost_cny"] = args.max_cost
    if args.evaluator:
        cfg["evaluator"]["mode"] = args.evaluator
    if args.eval_cmd:
        cfg["evaluator"]["cmd"] = args.eval_cmd
        for b in cfg["baselines"].values():
            b["eval_cmd"] = args.eval_cmd
    if args.no_proxy:
        cfg.setdefault("proxy", {})["enabled"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    try:
        h = Harness(cfg, llm_mode=args.llm_mode, run_dir=Path(args.run_dir) if args.run_dir else None)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    h.run(max_rounds=args.max_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
