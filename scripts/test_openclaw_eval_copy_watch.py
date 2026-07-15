from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("openclaw_eval_copy_watch.py")
SPEC = importlib.util.spec_from_file_location("openclaw_eval_copy_watch", MODULE_PATH)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


class FakeApi:
    def __init__(
        self,
        *,
        source_exists=False,
        dest_exists=True,
        eval_state="COMPLETED",
        eval_restarts=0,
        eval_exit_code="0:0",
        eval_states=None,
        eval_runs=None,
        eval_episodes=None,
        episode_total=300,
        resumes=None,
        resume_job_ids=None,
        lose_resume_response=False,
        jobs=None,
        lose_first_post_response=False,
        reject_posts=False,
    ):
        self.source_exists = source_exists
        self.dest_exists = dest_exists
        self.eval_state = eval_state
        self.eval_restarts = eval_restarts
        self.eval_exit_code = eval_exit_code
        self.eval_states = dict(eval_states or {})
        self.eval_runs = dict(eval_runs or {})
        self.eval_episodes = {str(k): int(v) for k, v in (eval_episodes or {}).items()}
        self.episode_total = episode_total
        self.resumes = {str(key): list(value) for key, value in (resumes or {}).items()}
        self.resume_job_ids = list(resume_job_ids or ["12346"])
        self.lose_resume_response = lose_resume_response
        self.posts = []
        self.deletes = []
        self.history_reads = 0
        self.jobs = list(jobs or [])
        self.lose_first_post_response = lose_first_post_response
        self.reject_posts = reject_posts

    def get(self, path):
        if path.startswith("/api/copy-jobs/"):
            return {"status": "done", "error": None}
        if path.endswith("/checkpoint-copies"):
            self.history_reads += 1
            return [{
                "copy_id": "copy123",
                "dest_cluster": "skt",
                "dest_path": "/fsx/checkpoint/run",
                "source_exists": self.source_exists,
                "dest_exists": self.dest_exists,
            }]
        if "/path-exists?" in path:
            return {"exists": True, "kind": "dir"}
        if path.startswith("/api/jobs?"):
            return {"jobs": list(self.jobs)}
        if path.endswith("/eval-runs"):
            job_id = path.split("/")[-2]
            count = int(self.eval_runs.get(job_id, 0))
            return {"eval_runs": [{"run": index} for index in range(count)]}
        if path.endswith("/progress"):
            job_id = path.split("/")[-2]
            return {
                "progress": {
                    "current_step": self.eval_episodes.get(job_id, 0),
                    "max_steps": self.episode_total,
                }
            }
        if path.endswith("/resumes"):
            job_id = path.split("/")[-2]
            return list(self.resumes.get(job_id, []))
        if path.startswith("/api/variants/"):
            return {
                "vars": {"N_RUNS": "3"},
                "arrays": {"TASKS": ["task_a", "task_b"], "EVAL_SETS": ["rand_obj"]},
            }
        if path.startswith("/api/jobs/skt/"):
            job_id = path.rsplit("/", 1)[-1]
            state = self.eval_states.get(job_id, self.eval_state)
            if isinstance(state, list):
                state = state.pop(0) if len(state) > 1 else state[0]
            return {
                "JobID": job_id,
                "State": state,
                "Restarts": self.eval_restarts,
                "ExitCode": self.eval_exit_code,
                "End": "2026-07-13T17:00:00+09:00",
            }
        raise AssertionError(path)

    def post(self, path, payload):
        self.posts.append((path, payload))
        if path.endswith("/resume"):
            parent_id = path.split("/")[-2]
            child_id = self.resume_job_ids.pop(0)
            child = {
                "job_id": child_id,
                "job_name": f"resumed-{child_id}",
                "partition": "l40s-gpu_background",
            }
            self.resumes.setdefault(parent_id, []).append(child)
            if self.lose_resume_response:
                self.lose_resume_response = False
                raise watcher.ApiError(None, "POST resume: timed out")
            return child
        if self.reject_posts:
            raise watcher.ApiError(400, "POST /api/submit: dexjoco_task is required")
        if self.lose_first_post_response and len(self.posts) == 1:
            self.jobs.append({
                "cluster": payload["cluster"],
                "job_id": "12345",
                "job_name": payload["job_name"],
                "partition": payload["partition"],
                "state": "PENDING",
            })
            raise watcher.ApiError(None, "POST /api/submit: timed out")
        return {"job_id": "12345", "job_name": payload["job_name"]}

    def delete(self, path):
        self.deletes.append(path)
        return {"status": "cancelled"}


class WatcherTests(unittest.TestCase):
    def make_args(self, state_file):
        return argparse.Namespace(
            copy_id="copy123",
            request_id="req1",
            state_key="mlxp/train1",
            source_cluster="mlxp",
            source_job_id="train1",
            variant="smoke_variant",
            dest_cluster="skt",
            partition="l40s-gpu_background",
            skip_copy=False,
            checkpoint_path=None,
            dexjoco_task=None,
            delete_source=True,
            api_base="http://unused",
            state_file=str(state_file),
            slack_channel="channel:test",
            poll_seconds=0,
            timeout_seconds=1,
            eval_poll_seconds=0,
            eval_timeout_seconds=1,
            api_timeout_seconds=1,
            submit_reconcile_poll_seconds=0,
            submit_reconcile_seconds=1,
            submit_retry_grace_seconds=300,
            slack_timeout_seconds=1,
            replace_existing_eval=False,
            resume_timed_out=False,
        )

    def write_state(self, path, **extra):
        entry = {"request_id": "req1", "outcome": "awaiting_copy_choice"}
        entry.update(extra)
        path.write_text(json.dumps({"mlxp/train1": entry}))

    def test_verified_copy_submits_exactly_one_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi()
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda channel, message: notices.append((channel, message)),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(result["eval_job_id"], "12345")
            self.assertIn("eval_terminal_notified_at", result)
            self.assertEqual(len(api.posts), 1)
            payload = api.posts[0][1]
            self.assertEqual(payload["cluster"], "skt")
            self.assertEqual(payload["partition"], "l40s-gpu_background")
            self.assertEqual(payload["phase"], "eval")
            self.assertEqual(payload["idempotency_key"], "openclaw:req1")
            self.assertNotIn("eval_num_gpus", payload)
            self.assertEqual(payload["checkpoint_path"], "/fsx/checkpoint/run")
            self.assertEqual(len(notices), 2)

    def test_existing_eval_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="already")
            api = FakeApi()
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda channel, message: notices.append(message),
            )
            self.assertEqual(result["eval_job_id"], "already")
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(api.posts, [])
            self.assertEqual(len(notices), 1)

    def test_lowercase_job_list_recovers_existing_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            job_name = "youngwoong_eval_smoke_variant_20260713_120000"
            self.write_state(state_file, eval_job_name=job_name)
            api = FakeApi(jobs=[{
                "cluster": "skt",
                "job_id": "222",
                "job_name": job_name,
                "partition": "l40s-gpu_background",
                "state": "PENDING",
            }])
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["eval_job_id"], "222")
            self.assertTrue(result["eval_submit_recovered"])
            self.assertEqual(api.posts, [])

    def test_lost_submit_response_reconciles_without_second_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi(lose_first_post_response=True)
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["eval_job_id"], "12345")
            self.assertTrue(result["eval_submit_recovered"])
            self.assertEqual(len(api.posts), 1)

    def test_recent_uncertain_submit_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_name="youngwoong_eval_smoke_variant_20260713_120000",
                outcome="eval_submit_uncertain",
                eval_submit_uncertain_at=datetime.now().astimezone().isoformat(),
            )
            api = FakeApi()
            with self.assertRaisesRegex(RuntimeError, "refusing to resubmit"):
                watcher.run_workflow(
                    self.make_args(state_file),
                    api=api,
                    notifier=lambda _channel, _message: None,
                    sleep=lambda _: None,
                )
            self.assertEqual(api.posts, [])

    def test_duplicate_history_recovers_oldest_job(self):
        name = "youngwoong_eval_smoke_variant_20260713_120000"
        api = FakeApi(jobs=[
            {"job_id": "153069", "job_name": name, "state": "CANCELLED"},
            {"job_id": "153064", "job_name": name, "state": "PENDING"},
        ])
        existing = watcher.find_existing_eval(api, "skt", name)
        self.assertIsNotNone(existing)
        self.assertEqual(watcher._job_id(existing), "153064")

    def test_same_cluster_direct_eval_uses_watcher_and_skips_copy_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi()
            args = self.make_args(state_file)
            args.source_cluster = "skt"
            args.state_key = "skt/train1"
            args.skip_copy = True
            args.delete_source = False
            args.checkpoint_path = "/fsx/checkpoint/direct"
            state_file.write_text(json.dumps({
                "skt/train1": {
                    "request_id": "req1",
                    "outcome": "awaiting_eval_target",
                }
            }))
            result = watcher.run_workflow(
                args,
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["eval_job_id"], "12345")
            self.assertEqual(api.history_reads, 0)
            self.assertEqual(api.posts[0][1]["checkpoint_path"], "/fsx/checkpoint/direct")
            state = json.loads(state_file.read_text())["skt/train1"]
            self.assertTrue(state["copy_skipped"])

    def test_existing_eval_can_be_replaced_with_gpu_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_id="wrong-gpu-job",
                eval_job_name="wrong-gpu-name",
            )
            api = FakeApi()
            notices = []
            args = self.make_args(state_file)
            args.replace_existing_eval = True
            result = watcher.run_workflow(
                args,
                api=api,
                notifier=lambda channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(
                api.deletes, ["/api/jobs/skt/wrong-gpu-job"]
            )
            self.assertEqual(len(api.posts), 1)
            self.assertNotIn("eval_num_gpus", api.posts[0][1])
            self.assertEqual(result["eval_job_id"], "12345")
            self.assertEqual(result["replaced_eval_job_id"], "wrong-gpu-job")

    def test_failed_eval_is_reported_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="failed-job")
            api = FakeApi(eval_state="FAILED")
            notices = []
            args = self.make_args(state_file)
            result = watcher.run_workflow(
                args,
                api=api,
                notifier=lambda channel, message: notices.append(message),
            )
            self.assertEqual(result["outcome"], "eval_failed")
            self.assertEqual(result["eval_state"], "FAILED")
            self.assertIn("failed", notices[0].lower())
            again = watcher.run_workflow(
                args,
                api=api,
                notifier=lambda channel, message: notices.append(message),
            )
            self.assertEqual(again["outcome"], "eval_failed")
            self.assertEqual(len(notices), 1)

    def test_timed_out_eval_resumes_then_reports_child_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_id="100",
                eval_state="TIMEOUT",
                eval_terminal_notified_at="old-watcher",
            )
            api = FakeApi(
                eval_states={"100": "TIMEOUT", "12346": "COMPLETED"},
                eval_runs={"100": 2},
            )
            notices = []
            args = self.make_args(state_file)
            args.resume_timed_out = True
            result = watcher.run_workflow(
                args,
                api=api,
                notifier=lambda _channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(result["eval_job_id"], "12346")
            self.assertEqual(result["eval_resume_last_completed_runs"], 2)
            self.assertEqual(result["eval_total_runs"], 6)
            self.assertEqual(len(result["eval_resume_chain"]), 1)
            self.assertEqual([path for path, _ in api.posts], ["/api/jobs/skt/100/resume"])
            self.assertIn("2/6", notices[0])
            self.assertIn("completed", notices[1].lower())

    def test_exhausted_automatic_requeue_is_resumed_not_treated_as_user_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="100")
            api = FakeApi(
                eval_states={"100": "CANCELLED", "12346": "COMPLETED"},
                eval_restarts=5,
                eval_runs={"100": 2},
            )
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(result["eval_job_id"], "12346")
            self.assertEqual([path for path, _ in api.posts], ["/api/jobs/skt/100/resume"])

    def test_requeue_exhaustion_resumes_even_when_previous_window_made_no_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_id="200",
                eval_resume_parent_job_id="100",
                eval_resume_last_completed_runs=2,
                eval_total_runs=6,
            )
            api = FakeApi(
                eval_states={"200": "CANCELLED", "12346": "COMPLETED"},
                eval_restarts=5,
                eval_runs={"200": 2},
            )
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(result["eval_job_id"], "12346")
            self.assertEqual(
                [path for path, _ in api.posts], ["/api/jobs/skt/200/resume"]
            )

    def test_intentional_cancel_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="100")
            api = FakeApi(eval_state="CANCELLED by 501", eval_restarts=5)
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_cancelled")
            self.assertEqual(api.posts, [])

    def test_preempted_is_transient_not_terminal(self):
        # A preemptible-partition job goes PREEMPTED and slurm requeues it under
        # the same id; the watcher must wait through it, not report failure.
        self.assertIsNone(watcher.eval_terminal_kind("PREEMPTED"))

    def test_preemption_then_requeue_completes_without_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="100")
            # Preempted -> requeued (RUNNING) -> COMPLETED: one clean finish.
            api = FakeApi(eval_states={"100": ["PREEMPTED", "RUNNING", "COMPLETED"]})
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(api.posts, [])
            self.assertTrue(all("failed" not in m.lower() for m in notices))

    def test_second_timeout_without_episode_progress_stops_instead_of_resuming(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_id="200",
                eval_resume_parent_job_id="100",
                eval_resume_last_completed_runs=0,
                eval_resume_last_completed_episodes=40,
                eval_total_runs=6,
            )
            # Same 40 episodes as the previous window: genuine stall.
            api = FakeApi(
                eval_states={"200": "TIMEOUT"},
                eval_runs={"200": 0},
                eval_episodes={"200": 40},
            )
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_resume_stalled")
            self.assertEqual(api.posts, [])
            self.assertIn("no new completed episodes", notices[0])

    def test_timeout_resumes_when_episodes_advanced_without_completed_runs(self):
        # The regression that made 153556 restart from scratch: a big eval can
        # burn a whole window advancing episodes without finishing a single
        # run. Run-based stall detection would stop here; episode-based must not.
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(
                state_file,
                eval_job_id="200",
                eval_resume_parent_job_id="100",
                eval_resume_last_completed_runs=0,
                eval_resume_last_completed_episodes=40,
                eval_total_runs=6,
            )
            # 0 completed runs, but episodes climbed 40 -> 95: real progress.
            api = FakeApi(
                eval_states={"200": ["TIMEOUT", "TIMEOUT"], "12346": "COMPLETED"},
                eval_runs={"200": 0},
                eval_episodes={"200": 95},
            )
            notices = []
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, message: notices.append(message),
                sleep=lambda _: None,
            )
            self.assertEqual(result["outcome"], "eval_completed")
            self.assertEqual(result["eval_job_id"], "12346")
            self.assertEqual(result["eval_resume_last_completed_episodes"], 95)
            self.assertEqual([p for p, _ in api.posts], ["/api/jobs/skt/200/resume"])

    def test_lost_resume_response_recovers_existing_child_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="100")
            api = FakeApi(
                eval_states={"100": "TIMEOUT", "12346": "COMPLETED"},
                eval_runs={"100": 1},
                lose_resume_response=True,
            )
            result = watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(result["eval_job_id"], "12346")
            self.assertEqual(len(api.posts), 1)

    def test_monitor_timeout_warning_is_reported_once_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, eval_job_id="pending-job")
            api = FakeApi(eval_state="PENDING")
            notices = []
            args = self.make_args(state_file)
            args.eval_timeout_seconds = 0
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    watcher.run_workflow(
                        args,
                        api=api,
                        notifier=lambda _channel, message: notices.append(message),
                        sleep=lambda _: None,
                    )
            self.assertEqual(len(notices), 1)
            self.assertIn("monitoring failed", notices[0].lower())

    def test_dexjoco_task_is_forwarded_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi()
            args = self.make_args(state_file)
            args.dexjoco_task = "pick_and_place"
            watcher.run_workflow(
                args,
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(api.posts[0][1]["dexjoco_task"], "pick_and_place")
            state = json.loads(state_file.read_text())["mlxp/train1"]
            self.assertEqual(state["dexjoco_task"], "pick_and_place")

    def test_resumed_run_reuses_persisted_dexjoco_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file, dexjoco_task="pick_and_place")
            api = FakeApi()
            watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            self.assertEqual(api.posts[0][1]["dexjoco_task"], "pick_and_place")

    def test_generated_job_name_is_unique_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi()
            watcher.run_workflow(
                self.make_args(state_file),
                api=api,
                notifier=lambda _channel, _message: None,
                sleep=lambda _: None,
            )
            job_name = api.posts[0][1]["job_name"]
            self.assertRegex(
                job_name,
                r"^youngwoong_eval_smoke_variant_[0-9a-f]{6}_\d{8}_\d{6}$",
            )
            token = hashlib.sha256(b"req1").hexdigest()[:6]
            self.assertIn(f"_{token}_", job_name)

    def test_submit_rejection_is_notified_once_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi(reject_posts=True)
            notices = []
            args = self.make_args(state_file)
            for _ in range(2):
                with self.assertRaises(watcher.ApiError):
                    watcher.run_workflow(
                        args,
                        api=api,
                        notifier=lambda _channel, message: notices.append(message),
                        sleep=lambda _: None,
                    )
            self.assertEqual(len(notices), 1)
            self.assertIn("rejected", notices[0].lower())
            state = json.loads(state_file.read_text())["mlxp/train1"]
            self.assertEqual(state["outcome"], "eval_submit_failed")

    def test_delete_source_must_be_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            self.write_state(state_file)
            api = FakeApi(source_exists=True)
            notices = []
            args = self.make_args(state_file)
            args.timeout_seconds = 0.01
            with self.assertRaises(TimeoutError):
                watcher.run_workflow(
                    args,
                    api=api,
                    notifier=lambda channel, message: notices.append(message),
                    sleep=lambda _: None,
                )
            state = json.loads(state_file.read_text())["mlxp/train1"]
            self.assertEqual(state["outcome"], "copy_failed")
            self.assertEqual(api.posts, [])
            self.assertGreater(api.history_reads, 0)
            self.assertEqual(len(notices), 1)


if __name__ == "__main__":
    unittest.main()
