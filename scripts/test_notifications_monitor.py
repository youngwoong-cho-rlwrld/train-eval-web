from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import jobs, notifications  # noqa: E402


def job(cluster: str, state: str) -> jobs.Job:
    return jobs.Job(
        cluster=cluster,
        job_id="153064",
        job_name=f"youngwoong_eval_{cluster}_20260713_212351",
        partition="background",
        state=state,
        elapsed="00:01",
        nodelist="node",
        phase="eval",
        variant=f"variant_{cluster}",
    )


class NotificationMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_numeric_id_on_two_clusters_does_not_collide(self):
        async def list_jobs(cluster_names, **_kwargs):
            cluster = cluster_names[0]
            return [job(cluster, "RUNNING")]

        monitor = notifications._Monitor()
        with (
            patch.object(
                notifications.clusters,
                "list_clusters",
                return_value=["kakao", "skt"],
            ),
            patch.object(notifications.jobs, "list_jobs", list_jobs),
        ):
            current = await monitor._collect()

        self.assertEqual(set(current), {"kakao/153064", "skt/153064"})

    async def test_both_cluster_transitions_are_notified(self):
        async def collect():
            return {
                "kakao/153064": {
                    "cluster": "kakao",
                    "event": "completed",
                    "_job": job("kakao", "COMPLETED"),
                },
                "skt/153064": {
                    "cluster": "skt",
                    "event": "failed",
                    "_job": job("skt", "FAILED"),
                },
            }

        monitor = notifications._Monitor()
        monitor.primed = True
        monitor.state = {
            "kakao/153064": {"cluster": "kakao", "event": "running"},
            "skt/153064": {"cluster": "skt", "event": "running"},
        }
        monitor._collect = collect
        monitor._persist = Mock()
        post = AsyncMock()
        with (
            patch.object(
                notifications.notifications_config,
                "get_settings",
                return_value=SimpleNamespace(enabled=True, configured=True),
            ),
            patch.object(
                notifications.notifications_config,
                "event_enabled",
                return_value=True,
            ),
            patch.object(notifications, "_post", post),
        ):
            await monitor._tick()

        self.assertEqual(post.await_count, 2)
        messages = "\n".join(call.args[0] for call in post.await_args_list)
        self.assertIn("kakao/153064", messages)
        self.assertIn("skt/153064", messages)


if __name__ == "__main__":
    unittest.main()
