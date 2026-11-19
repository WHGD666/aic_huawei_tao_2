"""Recovery keeps the best evaluated source even after an unscored draft."""
from pathlib import Path
import tempfile
import unittest

from official_kit.agent_templates.common.agent_common import recover_history


class FakeContext:
    def __init__(self, rows, receipts):
        self.rows = rows
        self.receipts = receipts
        self.waited = []

    def status(self):
        return {"history": self.rows}

    def evaluation_status(self, request_id):
        return self.receipts[request_id]

    def wait_evaluation(self, request_id):
        self.waited.append(request_id)
        return self.receipts[request_id]


def receipt(request_id, candidate_id, score, source):
    return {"request_id": request_id, "candidate_id": candidate_id,
            "status": "succeeded", "source": source,
            "feedback": {"score": score}}


class RecoveryTests(unittest.TestCase):
    def test_restores_best_receipt_and_reconciles_pending_once(self):
        rows = [{"request_id": "a", "status": "succeeded"},
                {"request_id": "b", "status": "succeeded"},
                {"request_id": "c", "status": "accepted"}]
        receipts = {"a": receipt("a", "candidate-a", 5.0, "SOURCE_A"),
                    "b": receipt("b", "candidate-b", -1.0, "SOURCE_B"),
                    "c": receipt("c", "candidate-c", 8.0, "SOURCE_C")}
        context = FakeContext(rows, receipts)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            Path(temp, "program.py").write_text("UNSCORED_DRAFT", encoding="utf-8")
            best, source, last = recover_history(context, Path(temp), "missing_local_task")
        self.assertEqual(best, {"candidate_id": "candidate-c", "score": 8.0})
        self.assertEqual(source, "SOURCE_C")
        self.assertEqual(last["request_id"], "c")
        self.assertEqual(context.waited, ["c"])

    def test_success_without_frozen_source_fails_before_new_submission(self):
        context = FakeContext([{"request_id": "a", "status": "succeeded"}],
                              {"a": receipt("a", "candidate-a", 5.0, None)})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            Path(temp, "program.py").write_text("UNSCORED_DRAFT", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen score or source"):
                recover_history(context, Path(temp), "missing_local_task")


if __name__ == "__main__":
    unittest.main()
