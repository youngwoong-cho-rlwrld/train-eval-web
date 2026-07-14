from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import job_resume, jobs, main, submit  # noqa: E402
from app.ssh import SSHResult  # noqa: E402


def request(*, key: str | None = "openclaw:req1") -> submit.SubmitRequest:
    return submit.SubmitRequest(
        cluster="skt",
        variant="dexjoco_physixel_bimanual_5tasks_224",
        phase="eval",
        partition="l40s-gpu_background",
        checkpoint_path="/fsx/checkpoint/run",
        job_name="youngwoong_eval_dexjoco_physixel_bimanual_5tasks_224_20260713_212351",
        idempotency_key=key,
    )


def job(job_id: str = "153064") -> jobs.Job:
    return jobs.Job(
        cluster="skt",
        job_id=job_id,
        job_name=request().job_name or "",
        partition="l40s-gpu_background",
        state="PENDING",
        elapsed="00:00",
        nodelist="(Priority)",
        phase="eval",
        variant="dexjoco_physixel_bimanual_5tasks_224",
    )


class SubmitIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main._eval_submit_locks.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            "os.environ",
            {"TRAIN_EVAL_SUBMIT_LOCK_DIR": self.tmp.name},
        )
        self.env.start()
        self.real_recover_submission_metadata = submit.recover_submission_metadata
        self.transaction_patch = patch.object(
            main.submit,
            "read_submission_transaction",
            AsyncMock(return_value={}),
        )
        self.read_transaction = self.transaction_patch.start()
        self.recovery_patch = patch.object(
            main.submit,
            "recover_submission_metadata",
            AsyncMock(return_value=False),
        )
        self.recover_metadata = self.recovery_patch.start()

    async def asyncTearDown(self):
        self.recovery_patch.stop()
        self.transaction_patch.stop()
        self.env.stop()
        self.tmp.cleanup()
        main._eval_submit_locks.clear()

    async def test_existing_named_job_is_recovered_without_submit_or_notification(self):
        list_jobs = AsyncMock(return_value=[job()])
        do_submit = AsyncMock()
        notify = AsyncMock()
        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
            patch.object(main.notifications, "note_submitted", notify),
        ):
            response = await main.post_submit(request())

        self.assertEqual(response.job_id, "153064")
        self.assertTrue(response.recovered)
        do_submit.assert_not_awaited()
        notify.assert_not_awaited()

    async def test_exhausted_requeue_classifier_excludes_user_cancel(self):
        self.assertTrue(
            jobs.is_exhausted_requeue(
                {"State": "CANCELLED", "Restarts": "5", "ExitCode": "0:0"}
            )
        )
        self.assertFalse(
            jobs.is_exhausted_requeue(
                {
                    "State": "CANCELLED by 501",
                    "Restarts": "5",
                    "ExitCode": "0:0",
                }
            )
        )

    async def test_active_and_history_parsers_keep_independent_field_counts(self):
        active = "315|eval-name|background|PENDING|0:00|(Resources)|2:00:00|Unknown"
        history = (
            "315|eval-name|background|CANCELLED|00:32:20|start|end|node|5"
        )
        self.assertEqual(len(jobs._squeue_parts(active) or []), 8)
        self.assertEqual(len(jobs._sacct_list_parts(history) or []), 9)
        self.assertIsNone(jobs._squeue_parts("too|short"))
        self.assertIsNone(jobs._sacct_list_parts("too|short"))

    async def test_concurrent_retries_create_only_one_job(self):
        visible: list[jobs.Job] = []
        submit_calls = 0

        async def list_jobs(*_args, **_kwargs):
            return list(visible)

        async def do_submit(req):
            nonlocal submit_calls
            submit_calls += 1
            await asyncio.sleep(0.05)
            visible.append(job())
            return submit.SubmitResponse(
                job_id="153064",
                job_name=req.job_name,
                partition=req.partition,
                sbatch_cmd="sbatch",
                rsync_stdout="",
                sbatch_stdout="Submitted batch job 153064",
            )

        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
        ):
            first, second = await asyncio.gather(
                main._submit_slurm_once(request(key="caller:a")),
                main._submit_slurm_once(request(key="caller:b")),
            )

        self.assertEqual(submit_calls, 1)
        self.assertEqual(sorted((first[1], second[1])), [False, True])
        self.assertEqual({first[0].job_id, second[0].job_id}, {"153064"})

    async def test_restart_reconciliation_does_not_depend_on_memory_lock(self):
        do_submit = AsyncMock()
        with (
            patch.object(main.jobs, "list_jobs", AsyncMock(return_value=[job()])),
            patch.object(main.submit, "submit", do_submit),
        ):
            main._eval_submit_locks.clear()  # equivalent to a fresh process
            response, created = await main._submit_slurm_once(request())

        self.assertFalse(created)
        self.assertTrue(response.recovered)
        do_submit.assert_not_awaited()

    async def test_durable_transaction_is_recovered_before_job_listing_or_restage(self):
        transaction = {
            "job_id": "153065",
            "job_name": request().job_name or "",
            "partition": "l40s-gpu_background",
            "transaction_status": "submitted",
        }
        self.read_transaction.return_value = transaction
        list_jobs = AsyncMock()
        do_submit = AsyncMock()
        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
        ):
            response, created = await main._submit_slurm_once(request())

        self.assertFalse(created)
        self.assertTrue(response.recovered)
        self.assertEqual(response.job_id, "153065")
        list_jobs.assert_not_awaited()
        do_submit.assert_not_awaited()
        self.recover_metadata.assert_awaited_with(request(), "153065")

    async def test_in_progress_transaction_fails_closed_before_restage(self):
        self.read_transaction.return_value = {"transaction_status": "submitting"}
        list_jobs = AsyncMock(return_value=[])
        do_submit = AsyncMock()
        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
        ):
            with self.assertRaisesRegex(RuntimeError, "still in progress"):
                await main._submit_slurm_once(request())
        list_jobs.assert_awaited_once()
        do_submit.assert_not_awaited()

    async def test_in_progress_transaction_recovers_named_job_after_host_restart(self):
        self.read_transaction.return_value = {"transaction_status": "submitting"}
        list_jobs = AsyncMock(return_value=[job("153066")])
        do_submit = AsyncMock()
        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
        ):
            response, created = await main._submit_slurm_once(request())

        self.assertFalse(created)
        self.assertTrue(response.recovered)
        self.assertEqual(response.job_id, "153066")
        do_submit.assert_not_awaited()
        self.recover_metadata.assert_awaited_with(request(), "153066")

    async def test_reconciliation_failure_fails_closed(self):
        do_submit = AsyncMock()
        with (
            patch.object(
                main.jobs,
                "list_jobs",
                AsyncMock(side_effect=RuntimeError("sacct unavailable")),
            ),
            patch.object(main.submit, "submit", do_submit),
        ):
            with self.assertRaisesRegex(RuntimeError, "sacct unavailable"):
                await main._submit_slurm_once(request())

        do_submit.assert_not_awaited()

    async def test_keyless_submission_with_job_name_is_not_deduplicated(self):
        # The submit UI sends explicit (memoized, possibly stale) job names
        # without an idempotency_key; those must keep plain submit semantics.
        do_submit = AsyncMock(
            return_value=submit.SubmitResponse(
                job_id="153070",
                job_name=request().job_name or "",
                partition="l40s-gpu_background",
                sbatch_cmd="sbatch",
                rsync_stdout="",
                sbatch_stdout="Submitted batch job 153070",
            )
        )
        list_jobs = AsyncMock(return_value=[job()])
        with (
            patch.object(main.jobs, "list_jobs", list_jobs),
            patch.object(main.submit, "submit", do_submit),
        ):
            response, created = await main._submit_slurm_once(request(key=None))

        self.assertTrue(created)
        self.assertFalse(response.recovered)
        self.assertEqual(response.job_id, "153070")
        do_submit.assert_awaited_once()
        list_jobs.assert_not_awaited()

    async def test_cancelled_original_is_still_idempotent(self):
        cancelled = job()
        cancelled.state = "CANCELLED"
        do_submit = AsyncMock()
        with (
            patch.object(
                main.jobs,
                "list_jobs",
                AsyncMock(return_value=[cancelled]),
            ),
            patch.object(main.submit, "submit", do_submit),
        ):
            response, created = await main._submit_slurm_once(request())

        self.assertFalse(created)
        self.assertEqual(response.job_id, "153064")
        do_submit.assert_not_awaited()

    async def test_existing_resume_child_is_recovered_without_resubmit(self):
        child = job("153065")
        do_resume = AsyncMock()
        with (
            patch.object(
                main.job_resume,
                "list_resumed_jobs",
                AsyncMock(return_value=[child]),
            ),
            patch.object(main.job_resume, "resume_timed_out_job", do_resume),
        ):
            response, created = await main._resume_slurm_once("skt", "153064")

        self.assertFalse(created)
        self.assertTrue(response.recovered)
        self.assertEqual(response.job_id, "153065")
        do_resume.assert_not_awaited()

    async def test_concurrent_resume_requests_create_only_one_child(self):
        visible: list[jobs.Job] = []
        calls = 0

        async def list_children(*_args, **_kwargs):
            return list(visible)

        async def do_resume(_request):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            child = job("153065")
            visible.append(child)
            return submit.SubmitResponse(
                job_id=child.job_id,
                job_name=child.job_name,
                partition=child.partition,
                sbatch_cmd="sbatch",
                rsync_stdout="",
                sbatch_stdout="Submitted batch job 153065",
            )

        with (
            patch.object(main.job_resume, "list_resumed_jobs", list_children),
            patch.object(
                main.job_resume,
                "build_resubmit_request",
                AsyncMock(return_value=request()),
            ),
            patch.object(main.jobs, "list_jobs", AsyncMock(return_value=[])),
            patch.object(main.submit, "submit", do_resume),
        ):
            first, second = await asyncio.gather(
                main._resume_slurm_once("skt", "153064"),
                main._resume_slurm_once("skt", "153064"),
            )

        self.assertEqual(calls, 1)
        self.assertEqual(sorted((first[1], second[1])), [False, True])
        self.assertEqual({first[0].job_id, second[0].job_id}, {"153065"})

    async def test_resume_job_name_is_stable_and_chain_suffix_does_not_grow(self):
        original = "youngwoong_eval_dexjoco_physixel_variant_20260713_212351"
        first = job_resume._resume_job_name(original, "153064")
        self.assertEqual(
            first,
            "youngwoong_eval_dexjoco_physixel_variant_r153064_20260713_212351",
        )
        self.assertEqual(job_resume._resume_job_name(first, "153065"),
                         "youngwoong_eval_dexjoco_physixel_variant_r153065_20260713_212351")
        self.assertEqual(job_resume._resume_job_name(original, "153064"), first)
        self.assertGreaterEqual(len(job_resume._resume_job_name(None, "1")), 50)

    async def test_resume_recovers_exact_name_when_sidecar_is_missing(self):
        resumed_request = request(key="resume:skt:153064")
        named_child = job("153065")
        named_child.job_name = resumed_request.job_name or ""
        do_submit = AsyncMock()
        with (
            patch.object(
                main.job_resume, "list_resumed_jobs", AsyncMock(return_value=[])
            ),
            patch.object(
                main.job_resume,
                "build_resubmit_request",
                AsyncMock(return_value=resumed_request),
            ),
            patch.object(main.jobs, "list_jobs", AsyncMock(return_value=[named_child])),
            patch.object(main.submit, "submit", do_submit),
        ):
            response, created = await main._resume_slurm_once("skt", "153064")

        self.assertFalse(created)
        self.assertEqual(response.job_id, "153065")
        do_submit.assert_not_awaited()
        self.recover_metadata.assert_awaited_with(resumed_request, "153065")

    async def test_remote_transaction_returns_persisted_job_id(self):
        transaction = (
            "job_id=153065\n"
            "transaction_status=submitted\n"
            "sbatch_stdout_b64=U3VibWl0dGVkIGJhdGNoIGpvYiAxNTMwNjU=\n"
        )
        remote = AsyncMock(return_value=SSHResult(0, transaction, ""))
        with patch.object(submit, "ssh_run", remote):
            job_id, output = await submit._run_idempotent_sbatch(
                host="skt",
                req=request(),
                job_name=request().job_name or "",
                sbatch_cmd="sbatch test.sh",
                meta="phase=eval\nvariant=test\n",
            )

        self.assertEqual(job_id, "153065")
        self.assertEqual(output, "Submitted batch job 153065")
        command = remote.await_args.args[1]
        script = remote.await_args.kwargs["input_text"]
        self.assertIn("nohup bash", command)
        self.assertIn("script=$HOME/", command)
        self.assertIn(".sh.$$", command)
        install_syntax = subprocess.run(
            ["/bin/bash", "-n", "-c", command], text=True, capture_output=True
        )
        self.assertEqual(install_syntax.returncode, 0, install_syntax.stderr)
        self.assertIn("flock -x", script)
        self.assertIn("transaction_status=submitted", script)
        self.assertIn("jobs/${job_id}.meta", script)
        syntax = subprocess.run(
            ["/bin/bash", "-n"], input=script, text=True, capture_output=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            bin_dir = home / "bin"
            bin_dir.mkdir()
            flock = bin_dir / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n")
            flock.chmod(0o755)
            sbatch = bin_dir / "sbatch"
            sbatch.write_text(
                "#!/bin/sh\n"
                "echo call >> \"$HOME/sbatch.calls\"\n"
                "echo 'Submitted batch job 153065'\n"
            )
            sbatch.chmod(0o755)
            env = {
                "HOME": str(home),
                "PATH": f"{bin_dir}:/usr/bin:/bin",
            }
            for _ in range(2):
                ran = subprocess.run(
                    ["/bin/bash"],
                    input=script,
                    text=True,
                    capture_output=True,
                    env=env,
                )
                self.assertEqual(ran.returncode, 0, ran.stderr)
            self.assertEqual((home / "sbatch.calls").read_text().splitlines(), ["call"])
            sidecar = home / ".train-eval-web/jobs/153065.meta"
            self.assertIn("job_id=153065", sidecar.read_text())
            self.assertIn("phase=eval", sidecar.read_text())

    async def test_resume_reconciliation_fails_closed_on_remote_error(self):
        with (
            patch.object(
                job_resume,
                "load_cluster",
                AsyncMock(return_value=SimpleNamespace(ssh_alias="skt")),
            ),
            patch.object(
                job_resume,
                "ssh_run",
                AsyncMock(return_value=SSHResult(255, "", "connection failed")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to reconcile"):
                await job_resume.list_resumed_jobs("skt", "153064")

    async def test_recovery_backfills_sidecar_only_from_existing_transaction(self):
        remote = AsyncMock(return_value=SSHResult(0, "RECOVERED\n", ""))
        with (
            patch.object(
                submit,
                "load_cluster",
                AsyncMock(return_value=SimpleNamespace(ssh_alias="skt")),
            ),
            patch.object(submit, "ssh_run", remote),
        ):
            recovered = await self.real_recover_submission_metadata(
                request(), "153065"
            )

        self.assertTrue(recovered)
        command = remote.await_args.args[1]
        syntax = subprocess.run(
            ["/bin/bash", "-n", "-c", command], text=True, capture_output=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertIn("if [ ! -s", command)
        self.assertIn("transaction job id mismatch", command)


if __name__ == "__main__":
    unittest.main()
