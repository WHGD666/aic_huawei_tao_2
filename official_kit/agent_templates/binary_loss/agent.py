"""Participant agent for the binary-loss baseline (probability-prediction GNN).

Flow per round:
  1. ask the model for one candidate program (small prompt: rules + offline
     knowledge + parent program + memory + feedback digest)
  2. extract the EVOLVE-BLOCK program, run the organizer's static validator
     locally (free), and repair once if it fails
  3. submit the source string, wait for the 9-cell evaluation, record memory
  4. keep the best scored candidate; stop after one new submission so the next
     full-matrix result can be reviewed before another session starts

The organizer owns budgets, evaluation and scoring; this file only decides
what to try next.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import time
import uuid

from contest2.agent_api import AgentAPIError

from common.agent_common import (
    LocalCheck, append_memory, extract_program, feedback_digest, knowledge_for,
    read_memory, recover_history, write_program,
)

BASELINE = "binary_loss"
TASK = "tp_gnn_binary_loss_timing"
MAX_NEW_SUBMISSIONS_PER_SESSION = 1

SYSTEM = """You improve one Python function that trains a graph neural network to split a 3D chip's netlist into two tiers. Only the body of compute_binary_partition_loss may change. Return the complete program between the EVOLVE-BLOCK markers and nothing else.

Hard rules (violations are rejected and still cost an evaluation):
- keep the exact function name, parameter names and order; no imports, no extra functions, no decorators, no type annotations, no default arguments
- no loops, comprehensions, lambdas, try/except, randomness, `detach`, `.backward`, `.item()`, `.numpy()`, `.to()`, `.cpu()`, `.cuda()`, `.data`, `.grad`, or any name ending with `_`
- only these torch helpers: abs, clamp, exp, isfinite, log, log1p, mean, ones_like, sigmoid, softmax, sqrt, square, std, sum, tanh, var, where, zeros_like
- only these F helpers: cosine_similarity, logsigmoid, mse_loss, normalize, relu, softplus; scalar helpers abs, float, int, len, max, min
- return one finite differentiable scalar tensor; no dense N-by-N tensors; work must stay linear
- every input may be read but never rebound or mutated

Change ONE mechanism per candidate and keep the rest of the parent intact."""

REPAIR = """Your program failed the organizer's static validator with:

{error}

Fix only what caused this. Return the complete corrected program between the EVOLVE-BLOCK markers, nothing else."""


def _user_prompt(knowledge, parent_source, memory, digest, history_brief, remaining, attempt_note):
    parts = [
        "Offline expert knowledge (authoritative, within the 200-unit rule):",
        knowledge,
        "",
        f"Evaluations remaining: {remaining}.",
    ]
    if history_brief:
        parts += ["", "Recent outcomes (this run):", history_brief]
    if memory:
        parts += ["", "What you learned earlier in this run:", memory]
    if digest:
        parts += ["", "Feedback of the last evaluation:", digest]
    if attempt_note:
        parts += ["", attempt_note]
    parts += [
        "",
        "Current program (the parent; keep what works):",
        "```python",
        parent_source.rstrip(),
        "```",
        "",
        "Return the complete new program between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.",
    ]
    return "\n".join(parts)


def _ask(context, messages, max_tokens=6000):
    payload = {
        "model": "deepseek-flash",
        "input": messages,
        "temperature": 0.8,
        "max_output_tokens": max_tokens,
    }
    return context.model.responses_create(payload)


def _reply_text(response) -> str:
    """Extract assistant text from a Responses-API reply (several shapes seen)."""
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"]
    chunks = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            chunks.append(item["text"])
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "\n".join(chunks)


def _same_program(left: str, right: str) -> bool:
    """Ignore formatting and comments when checking for a wasted repeat."""
    try:
        return ast.dump(ast.parse(left), include_attributes=False) == ast.dump(
            ast.parse(right), include_attributes=False)
    except SyntaxError:
        return left.strip() == right.strip()


def _native_metrics(result: dict, case: str, seed: str) -> dict:
    """Read the organizer-published native summary when it is available."""
    root = result.get("artifacts_dir")
    if not isinstance(root, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case) or not re.fullmatch(r"[0-9]+", seed):
        return {}
    path = Path(root) / "cells" / case / seed / "result" / "evaluation_result.json"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        return metrics if isinstance(metrics, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _detailed_feedback(result: dict) -> str:
    """Give the model measured timing and partition diagnostics for every seed."""
    lines = [feedback_digest(result, limit=1600)]
    for cell in (result.get("feedback") or {}).get("cells") or []:
        case, seed = cell.get("case"), str(cell.get("seed", ""))
        if not isinstance(case, str):
            continue
        metrics = cell.get("metrics") or {}
        native = _native_metrics(result, case, seed)
        parts = [f"  {case}/{seed}"]
        score = cell.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"score={score:+.1f}")
        for key, label, precision in (
            ("final_global_wns_setup_ns", "WNS_ns", 3),
            ("final_global_tns_setup_ns", "TNS_ns", 1),
            ("timing_weighted_cut", "weighted_cut", 1),
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.{precision}f}")
        for key, label in (
            ("binary_raw_top_area_ratio", "raw_top_area"),
            ("binary_closure_moved_area_fraction", "repair_moved_area"),
            ("binary_probability_uncertain_area_fraction", "uncertain_area"),
        ):
            value = native.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.1%}")
        lines.append(" ".join(parts))
    return "\n".join(lines)[:4400]


def _best_feedback(context, state: dict, best: dict):
    for row in reversed(state.get("history") or []):
        if row.get("status") == "succeeded" and row.get("candidate_id") == best["candidate_id"]:
            return context.evaluation_status(row["request_id"])
    raise RuntimeError("Best scored candidate has no successful evaluation receipt")


class Agent:
    def run(self, context):
        folder = Path(__file__).parent
        knowledge = knowledge_for(BASELINE)
        checker = LocalCheck()
        if not checker.available:
            raise RuntimeError("Organizer static validator is unavailable; refusing to spend an evaluation")
        smoke = getattr(context, "evaluation_mode", "formal") == "smoke"

        # Finish any accepted evaluation and restore the actual best source.
        pending_at_start = any(
            row.get("status") in ("accepted", "uncertain", "reconciling")
            for row in context.status().get("history", []))
        best, parent_source, last_result = recover_history(context, folder, TASK)
        best_source = parent_source
        state = context.status()
        history_brief = ""
        if best is not None:
            history_brief = f"best so far: candidate {best['candidate_id']} score {best.get('score')}"

        digest = ""
        if best is not None:
            best_result = _best_feedback(context, state, best)
            digest = "Best verified result:\n" + _detailed_feedback(best_result)
        if last_result and (best is None or last_result.get("candidate_id") != best["candidate_id"]):
            digest += "\nLatest attempt:\n" + feedback_digest(last_result)
        if last_result:
            append_memory(folder, f"[recovered] {feedback_digest(last_result)}")
        if pending_at_start:
            append_memory(folder, "[pause] previous accepted matrix was reconciled; review its result before a new submission")
            if best is None:
                raise RuntimeError("No verified candidate after reconciling the previous matrix")
            if not smoke:
                write_program(folder, best_source)
            return best["candidate_id"]

        attempts = 0
        model_failures = 0
        submissions = 0
        while submissions < MAX_NEW_SUBMISSIONS_PER_SESSION:
            state = context.status()
            if not state.get("dispatch_allowed", True) or state.get("evaluations_remaining", 0) <= 0:
                break
            remaining = state.get("evaluations_remaining", 0)

            # ---- 1. propose ----
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_prompt(
                    knowledge, parent_source, read_memory(folder), digest, history_brief,
                    remaining, f"Attempt {attempts + 1}." if attempts else "")},
            ]
            try:
                reply = _reply_text(_ask(context, messages))
            except AgentAPIError as error:
                model_failures += 1
                append_memory(folder, f"[model] call failed: {str(error)[:160]}")
                if model_failures >= 3:
                    break
                time.sleep(5)
                continue

            source = extract_program(reply)
            if not source:
                attempts += 1
                append_memory(folder, "[propose] reply did not contain a complete program; retrying")
                if attempts >= 3:
                    break
                continue

            # ---- 2. local static check (+ one repair) ----
            ok, why = checker.check(source, BASELINE)
            if not ok:
                append_memory(folder, f"[check] rejected locally: {why[:200]}")
                try:
                    repair = _reply_text(_ask(context, [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": _user_prompt(
                            knowledge, parent_source, read_memory(folder), digest, history_brief,
                            remaining, "")},
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": REPAIR.format(error=why[:1200])},
                    ]))
                    fixed = extract_program(repair)
                    ok2, why2 = checker.check(fixed, BASELINE) if fixed else (False, "no program in repair reply")
                except AgentAPIError:
                    ok2, why2 = False, "model call failed during repair"
                if not ok2:
                    attempts += 1
                    append_memory(folder, f"[check] repair failed: {str(why2)[:200]}")
                    if attempts >= 3:
                        break
                    continue
                source = fixed

            # Avoid spending a full matrix on a formatting-only rewrite.
            if _same_program(source, parent_source):
                attempts += 1
                append_memory(folder, "[check] proposed program is identical to the scored parent")
                if attempts >= 3:
                    break
                continue

            write_program(folder, source)

            # ---- 3. submit + wait ----
            request_id = uuid.uuid4().hex
            try:
                context.submit_evaluation(source, request_id=request_id)
            except AgentAPIError as error:
                known = any(r.get("request_id") == request_id for r in context.status().get("history", []))
                if not known:
                    append_memory(folder, f"[submit] rejected: {str(error)[:200]}")
                    attempts += 1
                    if attempts >= 3:
                        break
                    continue
            submissions += 1
            try:
                result = context.wait_evaluation(request_id)
            except AgentAPIError as error:
                append_memory(folder, f"[wait] communication failed: {str(error)[:160]}")
                break

            digest = feedback_digest(result)
            append_memory(folder, f"[eval] {digest}")
            (folder / "feedback.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

            if result.get("status") == "succeeded":
                score = (result.get("feedback") or {}).get("score")
                if score is not None and (best is None or score > best.get("score", float("-inf"))):
                    best = {"candidate_id": result.get("candidate_id"), "score": score}
                    best_source = source
                    append_memory(folder, f"[decision] new best score {score:.3f}; this program becomes the parent")
                else:
                    append_memory(folder, f"[decision] score {score}; keep the earlier best as parent")
                parent_source = best_source
                history_brief = f"best so far: candidate {best['candidate_id']} score {best['score']:.3f}"
            else:
                append_memory(folder, "[decision] candidate did not succeed; parent unchanged")

            attempts = 0
            if smoke:
                break

        if best is None:
            raise RuntimeError("no candidate completed a verified training evaluation")
        if not smoke:
            write_program(folder, best_source)
        return best["candidate_id"]
