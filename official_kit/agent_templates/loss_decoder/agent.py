"""Participant agent for the loss+decoder baseline (representation-learning TP-GNN).

Same loop as the binary-loss agent, but the candidate program contains two
functions: `compute_tp_gnn_loss` and `decode_tp_gnn_embeddings`. A candidate may
change either or both; the template below tells the model which one it is
probing this round, which keeps each evaluation attributable to one mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from contest2.agent_api import AgentAPIError

from common.agent_common import (
    LocalCheck, append_memory, extract_program, feedback_digest, knowledge_for,
    local_partition_change, read_memory, read_program, write_program,
)

BASELINE = "loss_decoder"
TASK = "tp_gnn_loss_decoder_timing"

SYSTEM = """You improve two Python functions of a graph neural network that partitions a 3D chip netlist: the embedding training loss and the clustering decoder that turns embeddings into tier labels. Return one complete program containing both functions, in that order, between the EVOLVE-BLOCK markers and nothing else.

Hard rules (violations are rejected and still cost an evaluation):
- keep both exact function names, parameter names and order; no imports, no extra functions, no decorators, no annotations, no default arguments
- loss: no loops, comprehensions, lambdas, try/except, randomness, `detach`, `.backward`, `.item()`, `.numpy()`, `.to()`, `.cpu()`, `.cuda()`, `.data`, `.grad`, or names ending with `_`. Only torch: abs, clamp, exp, isfinite, log, log1p, mean, ones_like, sigmoid, softmax, sqrt, square, std, sum, tanh, var, where, zeros_like. Only F: cosine_similarity, logsigmoid, mse_loss, normalize, relu, softplus. Scalars: abs, float, int, len, max, min.
- decoder: only `np` and scalar helpers; no array element assignment (`a[i] = x`), no `np.zeros`/`np.ones`/`np.full`; build labels with np.where / zeros_like / argsort / cumsum / take / concatenate / lexsort / clip / maximum / minimum. At most ONE `for ... in range(max_iterations)` loop, no nesting, no np.linalg, no N-by-N distances, no input mutation. Either call `reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)` exactly once with unchanged arguments and no loop, or write your own bounded loop without that call. Labels: 1 = top tier, 0 = bottom; both tiers must be non-empty.

Change ONE mechanism per candidate and keep the rest of the parent intact."""

REPAIR = """Your program failed the organizer's static validator with:

{error}

Fix only what caused this. Return the complete corrected program between the EVOLVE-BLOCK markers, nothing else."""

_ROTATION = [
    "This round: change ONLY the decoder. Keep compute_tp_gnn_loss exactly as in the parent.",
    "This round: change ONLY the loss. Keep decode_tp_gnn_embeddings exactly as in the parent.",
]


def _user_prompt(knowledge, parent_source, memory, digest, history_brief, remaining, round_note):
    parts = [
        "Offline expert knowledge (authoritative, within the 200-unit rule):",
        knowledge,
        "",
        f"Evaluations remaining: {remaining}.",
    ]
    if round_note:
        parts += ["", round_note]
    if history_brief:
        parts += ["", "Recent outcomes (this run):", history_brief]
    if memory:
        parts += ["", "What you learned earlier in this run:", memory]
    if digest:
        parts += ["", "Feedback of the last evaluation:", digest]
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


def _ask(context, messages):
    return context.model.responses_create({
        "model": "deepseek-flash",
        "input": messages,
        "temperature": 0.8,
    })


def _reply_text(response) -> str:
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


def decode_is_plausible(source: str) -> tuple[bool, str]:
    """Cheap local guard for the failure modes the static validator cannot see:
    a decoder that can put every node on one tier, or that returns floats."""
    if "reference_global_two_way" in source:
        return True, "delegates to the reference helper"
    if "np.where" not in source and "zeros_like" not in source:
        return False, "decoder builds labels without np.where/zeros_like; it may return one tier or floats"
    if "cumsum" not in source and "argsort" not in source and "clip" not in source:
        return False, "decoder has no area/embedding ordering; it probably cannot hit the balance target"
    return True, "ok"


class Agent:
    def run(self, context):
        folder = Path(__file__).parent
        knowledge = knowledge_for(BASELINE)
        checker = LocalCheck()
        smoke = getattr(context, "evaluation_mode", "formal") == "smoke"

        state = context.status()
        last_result = None
        for row in state.get("history", []):
            if row.get("status") in ("accepted", "uncertain", "reconciling"):
                try:
                    last_result = context.wait_evaluation(row["request_id"])
                except AgentAPIError:
                    last_result = None

        state = context.status()
        succeeded = [r for r in state.get("history", []) if r.get("status") == "succeeded"]
        best = max(succeeded, key=lambda r: r.get("score", float("-inf"))) if succeeded else None
        parent_source = read_program(folder)
        history_brief = f"best so far: candidate {best['candidate_id']} score {best.get('score')}" if best else ""
        digest = feedback_digest(last_result) if last_result else ""
        if last_result:
            append_memory(folder, f"[eval] {digest}")

        attempts = 0
        model_failures = 0
        round_index = sum(1 for r in state.get("history", []) if r.get("status") in ("succeeded", "failed"))
        while True:
            state = context.status()
            if not state.get("dispatch_allowed", True) or state.get("evaluations_remaining", 0) <= 0:
                break
            remaining = state.get("evaluations_remaining", 0)

            note = _ROTATION[round_index % 2] if remaining > 1 else \
                "This round: refine the best parent conservatively (small coefficient change), do not introduce a new mechanism."
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_prompt(
                    knowledge, parent_source, read_memory(folder), digest, history_brief, remaining, note)},
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
                parent_source = read_program(folder)
                if attempts >= 3:
                    break
                continue

            ok, why = checker.check(source, BASELINE)
            if ok:
                ok, why = decode_is_plausible(source)
            if not ok:
                append_memory(folder, f"[check] rejected locally: {why[:200]}")
                try:
                    repair = _reply_text(_ask(context, [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": _user_prompt(
                            knowledge, parent_source, read_memory(folder), digest, history_brief, remaining, note)},
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": REPAIR.format(error=why[:1200])},
                    ]))
                    fixed = extract_program(repair)
                    ok2, why2 = checker.check(fixed, BASELINE) if fixed else (False, "no program in repair reply")
                    if ok2:
                        ok2, why2 = decode_is_plausible(fixed)
                except AgentAPIError:
                    ok2, why2 = False, "model call failed during repair"
                if not ok2:
                    attempts += 1
                    append_memory(folder, f"[check] repair failed: {str(why2)[:200]}")
                    if attempts >= 3:
                        break
                    continue
                source = fixed

            changed, why_change = local_partition_change(source, BASELINE)
            if changed is not None and changed < 0.01:
                attempts += 1
                append_memory(folder, f"[smoke] partition changed only {changed:.2%} vs the baseline ({why_change}); "
                                      f"try a different mechanism")
                if attempts >= 3:
                    break
                parent_source = read_program(folder)
                continue

            write_program(folder, source)
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
            try:
                result = context.wait_evaluation(request_id)
            except AgentAPIError as error:
                append_memory(folder, f"[wait] communication failed: {str(error)[:160]}")
                break

            round_index += 1
            digest = feedback_digest(result)
            append_memory(folder, f"[eval] {digest}")
            (folder / "feedback.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

            if result.get("status") == "succeeded":
                score = (result.get("feedback") or {}).get("score")
                if score is not None and (best is None or score > best.get("score", float("-inf"))):
                    best = {"candidate_id": result.get("candidate_id"), "score": score}
                    parent_source = source
                    append_memory(folder, f"[decision] new best score {score:.3f}; this program becomes the parent")
                else:
                    append_memory(folder, f"[decision] score {score}; keep the earlier best as parent")
                history_brief = f"best so far: candidate {best['candidate_id']} score {best['score']:.3f}"
            else:
                append_memory(folder, "[decision] candidate did not succeed; parent unchanged")

            attempts = 0
            if smoke:
                break

        if best is None:
            raise RuntimeError("no candidate completed a verified training evaluation")
        return best["candidate_id"]
