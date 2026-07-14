from __future__ import annotations

import importlib.util
import sys
import unittest
import unittest.mock
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("openclaw_eval_watch_resume.py")
SPEC = importlib.util.spec_from_file_location("openclaw_eval_watch_resume", MODULE_PATH)
assert SPEC and SPEC.loader
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def test_selects_only_unnotified_workflows(self):
        state = {
            "mlxp/copy": {"request_id": "a", "outcome": "copying_checkpoint"},
            "skt/eval": {"request_id": "b", "eval_job_id": "123"},
            "mlxp/uncertain": {
                "request_id": "e",
                "outcome": "eval_submit_uncertain",
            },
            "mlxp/replacing": {
                "request_id": "f",
                "outcome": "eval_replacing",
                "eval_job_id": None,
            },
            "mlxp/wait": {"request_id": "c", "outcome": "awaiting_eval_target"},
            "skt/done": {
                "request_id": "d",
                "eval_job_id": "456",
                "eval_terminal_notified_at": "now",
            },
            "skt/legacy-timeout": {
                "request_id": "g",
                "eval_job_id": "789",
                "eval_state": "TIMEOUT",
                "outcome": "eval_failed",
                "eval_terminal_notified_at": "old-watcher",
            },
            "skt/stalled": {
                "request_id": "h",
                "eval_job_id": "790",
                "eval_state": "TIMEOUT",
                "outcome": "eval_resume_stalled",
                "eval_terminal_notified_at": "now",
            },
        }
        self.assertEqual(
            [key for key, _ in resume.pending_workflows(state)],
            [
                "mlxp/copy",
                "skt/eval",
                "mlxp/uncertain",
                "mlxp/replacing",
                "skt/legacy-timeout",
            ],
        )

    def test_direct_eval_uses_synthetic_copy_id(self):
        entry = {
            "request_id": "req1",
            "variant": "variant",
            "dest_cluster": "skt",
            "dest_partition": "l40s-gpu_background",
            "eval_job_id": "123",
        }
        command = resume.build_worker_command(
            worker=Path("/tmp/worker.py"),
            state_file=Path("/tmp/state.json"),
            state_key="skt/train1",
            entry=entry,
            api_base="http://127.0.0.1:8000",
            slack_channel="channel:test",
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--copy-id") + 1], "direct-req1")
        self.assertNotIn("--delete-source", command)
        self.assertIn("--resume-timed-out", command)

    def test_incomplete_direct_eval_resume_preserves_skip_copy_checkpoint(self):
        entry = {
            "request_id": "req2",
            "variant": "variant",
            "dest_cluster": "skt",
            "dest_partition": "l40s-gpu_background",
            "copy_skipped": True,
            "checkpoint_path": "/fsx/checkpoint/direct",
            "outcome": "eval_submitting",
        }
        command = resume.build_worker_command(
            worker=Path("/tmp/worker.py"),
            state_file=Path("/tmp/state.json"),
            state_key="skt/train2",
            entry=entry,
            api_base="http://127.0.0.1:8000",
            slack_channel="channel:test",
        )
        self.assertIn("--skip-copy", command)
        self.assertEqual(
            command[command.index("--checkpoint-path") + 1],
            "/fsx/checkpoint/direct",
        )

    def test_watcher_owned_ids_cover_chains_and_stalled_entries(self):
        state = {
            "mlxp/a": {
                "request_id": "a",
                "eval_job_id": "300",
                "eval_resume_parent_job_id": "200",
                "eval_resume_chain": [
                    {"old_job_id": "100", "new_job_id": "200"},
                    {"old_job_id": "200", "new_job_id": "300"},
                ],
                "outcome": "eval_resume_stalled",
            },
            "mlxp/b": {"request_id": "b", "eval_job_id": "400"},
            "mlxp/c": "not-a-dict",
        }
        self.assertEqual(
            resume.watcher_owned_job_ids(state), {"100", "200", "300", "400"}
        )

    def test_sweep_resumes_only_unowned_timed_out_evals(self):
        rows = [
            # Should be swept: TIMEOUT eval, right prefix, unowned, no child.
            {"job_id": "500", "job_name": "youngwoong_eval_v_20260714_010101",
             "state": "TIMEOUT", "phase": "eval"},
            # Watcher-owned: skipped.
            {"job_id": "501", "job_name": "youngwoong_eval_v_20260714_020202",
             "state": "TIMEOUT", "phase": "eval"},
            # Already resumed (502 is 503's resume_of): skipped.
            {"job_id": "502", "job_name": "youngwoong_eval_v_20260714_030303",
             "state": "TIMEOUT", "phase": "eval"},
            {"job_id": "503", "job_name": "youngwoong_eval_v_20260714_040404",
             "state": "RUNNING", "phase": "eval", "resume_of": "502"},
            # Wrong phase / state / prefix: skipped.
            {"job_id": "504", "job_name": "youngwoong_train_v_20260714_050505",
             "state": "TIMEOUT", "phase": "train"},
            {"job_id": "505", "job_name": "youngwoong_eval_v_20260714_060606",
             "state": "FAILED", "phase": "eval"},
            {"job_id": "506", "job_name": "other_eval_v_20260714_070707",
             "state": "TIMEOUT", "phase": "eval"},
        ]
        calls = []

        def fake_api(base, method, path):
            calls.append((method, path))
            if method == "GET":
                return {"jobs": rows}
            return {"job_id": "600", "recovered": False}

        with unittest.mock.patch.object(resume, "_api_request", fake_api):
            swept, errors = resume.sweep_timed_out_evals(
                api_base="http://unused",
                clusters=["kakao"],
                hours=48,
                name_prefix="youngwoong_eval_",
                max_chain=3,
                owned_job_ids={"501"},
            )

        self.assertEqual(errors, [])
        self.assertEqual(
            swept,
            [{"cluster": "kakao", "job_id": "500", "resumed_as": "600",
              "recovered": False}],
        )
        self.assertEqual(
            [c for c in calls if c[0] == "POST"],
            [("POST", "/api/jobs/kakao/500/resume")],
        )

    def test_sweep_stops_at_chain_cap(self):
        rows = [
            {"job_id": "700", "job_name": "youngwoong_eval_v_20260714_010101",
             "state": "TIMEOUT", "phase": "eval"},
            {"job_id": "701", "job_name": "youngwoong_eval_v_20260714_020202",
             "state": "TIMEOUT", "phase": "eval", "resume_of": "700"},
            {"job_id": "702", "job_name": "youngwoong_eval_v_20260714_030303",
             "state": "TIMEOUT", "phase": "eval", "resume_of": "701"},
        ]

        def fake_api(base, method, path):
            if method == "GET":
                return {"jobs": rows}
            raise AssertionError(f"unexpected POST {path}")

        with unittest.mock.patch.object(resume, "_api_request", fake_api):
            swept, errors = resume.sweep_timed_out_evals(
                api_base="http://unused",
                clusters=["kakao"],
                hours=48,
                name_prefix="youngwoong_eval_",
                max_chain=2,
                owned_job_ids=set(),
            )

        # 700 and 701 already have children; 702 is at depth 2 (the cap).
        self.assertEqual(swept, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["job_id"], "702")
        self.assertIn("chain reached 2", errors[0]["error"])

    def test_persisted_dexjoco_task_is_forwarded(self):
        entry = {
            "request_id": "req3",
            "variant": "variant",
            "dest_cluster": "skt",
            "dest_partition": "l40s-gpu_background",
            "outcome": "eval_submitting",
            "dexjoco_task": "pick_and_place",
        }
        command = resume.build_worker_command(
            worker=Path("/tmp/worker.py"),
            state_file=Path("/tmp/state.json"),
            state_key="skt/train3",
            entry=entry,
            api_base="http://127.0.0.1:8000",
            slack_channel="channel:test",
        )
        self.assertEqual(
            command[command.index("--dexjoco-task") + 1], "pick_and_place"
        )


if __name__ == "__main__":
    unittest.main()
