from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import jobs, main, submit  # noqa: E402


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

    async def asyncTearDown(self):
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

        async def do_resume(_cluster, _job_id):
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
            patch.object(main.job_resume, "resume_timed_out_job", do_resume),
        ):
            first, second = await asyncio.gather(
                main._resume_slurm_once("skt", "153064"),
                main._resume_slurm_once("skt", "153064"),
            )

        self.assertEqual(calls, 1)
        self.assertEqual(sorted((first[1], second[1])), [False, True])
        self.assertEqual({first[0].job_id, second[0].job_id}, {"153065"})


if __name__ == "__main__":
    unittest.main()
