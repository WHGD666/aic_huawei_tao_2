"""The binary-loss Agent reviews a full matrix before another submission."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "official_kit"),
                str(ROOT / "official_kit" / "agent_templates" / "binary_loss")]
from official_kit.agent_templates.binary_loss import agent as binary_agent


class FakeCheck:
    available = True

    def check(self, source, baseline):
        return True, "ok"


class FakeContext:
    evaluation_mode = "formal"

    def __init__(self, best_result, next_result):
        self.best_result = best_result
        self.next_result = next_result
        self.submissions = []
        self.request_ids = []

    def status(self):
        return {"dispatch_allowed": True, "evaluations_remaining": 8,
                "history": [{"request_id": "first", "candidate_id": "first-candidate",
                             "status": "succeeded"},
                            {"request_id": "interrupted", "candidate_id": "interrupted-candidate",
                             "status": "failed"}]}

    def evaluation_status(self, request_id):
        assert request_id == "first"
        return self.best_result

    def submit_evaluation(self, source, request_id):
        self.submissions.append(source)
        self.request_ids.append(request_id)
        return {"status": "accepted", "request_id": request_id}

    def wait_evaluation(self, request_id):
        return self.next_result


def scored_receipt(artifacts_dir=None):
    return {"request_id": "first", "candidate_id": "first-candidate",
            "status": "succeeded", "source": "def compute_binary_partition_loss(): return 1\n",
            "artifacts_dir": str(artifacts_dir) if artifacts_dir else "",
            "feedback": {"score": 28.97, "cells": [
                {"case": "bp-ng45", "seed": "1001", "score": -15.4,
                 "valid": True, "metrics": {
                     "final_global_wns_setup_ns": -9.767,
                     "final_global_tns_setup_ns": -7763.8,
                     "timing_weighted_cut": 5552.8}}]}}


class BinaryAgentCycleTests(unittest.TestCase):
    def test_response_excludes_reasoning_text(self):
        reply = binary_agent._reply_text({"output": [
            {"type": "reasoning", "text": "This is not Python"},
            {"type": "message", "content": [{"type": "output_text", "text":
                "def compute_binary_partition_loss(): return 2"}]},
        ]})
        self.assertEqual(reply, "def compute_binary_partition_loss(): return 2")

    def test_measured_comparison_reports_regression_and_code_change(self):
        best = scored_receipt()
        best["feedback"]["cases"] = [{"case": "bp-ng45", "mean": -6.35}]
        latest = {"status": "succeeded", "source": "def compute_binary_partition_loss(): return 4\n",
                  "feedback": {"score": 7.25, "cases": [{"case": "bp-ng45", "mean": -13.48}]}}
        text = binary_agent._measured_comparison(best, latest, best["source"])
        self.assertIn("-21.72", text)
        self.assertIn("bp-ng45: -7.13", text)
        self.assertIn("+def compute_binary_partition_loss(): return 4", text)

    def test_clipped_negative_score_is_labeled(self):
        receipt = scored_receipt()
        receipt["feedback"]["cells"][0].update(score=-100.0, raw_score=-150.1)
        self.assertIn("raw_score_before_floor=-150.1", binary_agent._detailed_feedback(receipt))

    def test_feedback_includes_native_partition_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "cells/bp-ng45/1001/result"
            folder.mkdir(parents=True)
            (folder / "evaluation_result.json").write_text(json.dumps({"metrics": {
                "binary_raw_top_area_ratio": 0.61,
                "binary_closure_moved_area_fraction": 0.41,
                "binary_probability_uncertain_area_fraction": 0.07,
            }}), encoding="utf-8")
            digest = binary_agent._detailed_feedback(scored_receipt(temp))
        self.assertIn("score=-15.4", digest)
        self.assertIn("WNS_ns=-9.767", digest)
        self.assertIn("raw_top_area=61.0%", digest)
        self.assertIn("repair_moved_area=41.0%", digest)

    def test_recovered_feedback_precedes_exactly_one_new_submission(self):
        context = FakeContext(scored_receipt(), {
            "request_id": "next", "candidate_id": "next-candidate",
            "status": "succeeded", "source": "def compute_binary_partition_loss(): return 2\n",
            "feedback": {"score": 35.0, "cells": []}})
        proposal = "def compute_binary_partition_loss(): return 2\n"
        seen = []

        def ask(_context, messages):
            seen.append(messages[1]["content"])
            return {"output_text": proposal}

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(binary_agent, "__file__", str(Path(temp) / "agent.py")), \
                 patch.object(binary_agent, "LocalCheck", FakeCheck), \
                 patch.object(binary_agent, "recover_history", return_value=(
                     {"candidate_id": "first-candidate", "score": 28.97},
                     "def compute_binary_partition_loss(): return 1\n",
                     {"request_id": "interrupted", "candidate_id": "interrupted-candidate",
                      "status": "failed", "reason": "interrupted"})), \
                 patch.object(binary_agent, "_ask", side_effect=ask), \
                 patch.object(binary_agent, "write_program"), \
                 patch.object(binary_agent, "append_memory"), \
                 patch.object(binary_agent, "read_memory", return_value=""), \
                 patch.object(binary_agent, "knowledge_for", return_value=""):
                chosen = binary_agent.Agent().run(context)

        self.assertEqual(chosen, "next-candidate")
        self.assertEqual(context.submissions, [proposal])
        self.assertEqual(len(seen), 1)
        self.assertIn("bp-ng45/1001 score=-15.4", seen[0])
        self.assertIn("Latest attempt", seen[0])

    def test_identical_program_does_not_spend_an_evaluation(self):
        context = FakeContext(scored_receipt(), None)
        with patch.object(binary_agent, "LocalCheck", FakeCheck), \
             patch.object(binary_agent, "recover_history", return_value=(
                 {"candidate_id": "first-candidate", "score": 28.97},
                 "def compute_binary_partition_loss(): return 1\n", None)), \
             patch.object(binary_agent, "_ask", return_value={
                 "output_text": "def compute_binary_partition_loss():  # same body\n    return 1\n"}), \
             patch.object(binary_agent, "write_program"), \
             patch.object(binary_agent, "append_memory"), \
             patch.object(binary_agent, "read_memory", return_value=""), \
             patch.object(binary_agent, "knowledge_for", return_value=""):
            chosen = binary_agent.Agent().run(context)
        self.assertEqual(chosen, "first-candidate")
        self.assertEqual(context.submissions, [])

    def test_reviewed_proposal_submits_once_without_model_call(self):
        class ReceiptContext(FakeContext):
            def status(self):
                state = super().status()
                if self.request_ids:
                    state["history"].append({"request_id": self.request_ids[0],
                                             "candidate_id": "prepared-candidate",
                                             "status": "succeeded"})
                return state

            def evaluation_status(self, request_id):
                if self.request_ids and request_id == self.request_ids[0]:
                    return self.next_result
                return super().evaluation_status(request_id)

        proposal = "def compute_binary_partition_loss(): return 2\n"
        context = ReceiptContext(scored_receipt(), {
            "request_id": "prepared", "candidate_id": "prepared-candidate",
            "status": "succeeded", "source": proposal,
            "feedback": {"score": 35.0, "cells": []}})
        parent = "def compute_binary_partition_loss(): return 1\n"
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "NEXT_CANDIDATE.py").write_text(proposal, encoding="utf-8")
            with patch.object(binary_agent, "__file__", str(Path(temp) / "agent.py")), \
                 patch.object(binary_agent, "LocalCheck", FakeCheck), \
                 patch.object(binary_agent, "recover_history", side_effect=[
                     ({"candidate_id": "first-candidate", "score": 28.97}, parent, None),
                     ({"candidate_id": "prepared-candidate", "score": 35.0},
                      proposal, context.next_result)]), \
                 patch.object(binary_agent, "_ask", side_effect=AssertionError("unexpected model call")), \
                 patch.object(binary_agent, "write_program"), \
                 patch.object(binary_agent, "append_memory"), \
                 patch.object(binary_agent, "read_memory", return_value=""), \
                 patch.object(binary_agent, "knowledge_for", return_value=""):
                chosen = binary_agent.Agent().run(context)
                repeated = binary_agent.Agent().run(context)

        self.assertEqual(chosen, "prepared-candidate")
        self.assertEqual(repeated, "prepared-candidate")
        self.assertEqual(context.submissions, [proposal])
        self.assertEqual(len(context.request_ids), 1)

    def test_reconciliation_only_session_does_not_submit(self):
        class PendingContext(FakeContext):
            checks = 0

            def status(self):
                state = super().status()
                self.checks += 1
                if self.checks == 1:
                    state["history"][-1]["status"] = "accepted"
                return state

        context = PendingContext(scored_receipt(), None)
        with patch.object(binary_agent, "LocalCheck", FakeCheck), \
             patch.object(binary_agent, "recover_history", return_value=(
                 {"candidate_id": "first-candidate", "score": 28.97},
                 "def compute_binary_partition_loss(): return 1\n",
                 {"request_id": "interrupted", "candidate_id": "interrupted-candidate",
                  "status": "failed", "reason": "interrupted"})), \
             patch.object(binary_agent, "_ask", side_effect=AssertionError("model call after recovery")), \
             patch.object(binary_agent, "write_program"), \
             patch.object(binary_agent, "append_memory"):
            chosen = binary_agent.Agent().run(context)
        self.assertEqual(chosen, "first-candidate")
        self.assertEqual(context.submissions, [])


if __name__ == "__main__":
    unittest.main()
