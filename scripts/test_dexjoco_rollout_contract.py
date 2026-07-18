from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.dexjoco_rollout import rollout_for_variant  # noqa: E402
from app.eval_harness import harness_for  # noqa: E402
from app.submission_snapshot import snapshot_metadata  # noqa: E402
from app.variants import load_variant  # noqa: E402


EXPERIMENTS = ROOT / "configs" / "experiments"


def dexjoco_config_paths() -> list[Path]:
    return sorted(EXPERIMENTS.glob("dexjoco_*/config.sh"))


class DexjocoRolloutContractTests(unittest.TestCase):
    def test_unset_mode_is_new_true_sync_default(self):
        rollout = rollout_for_variant(SimpleNamespace(vars={}))
        self.assertEqual(rollout.inference_mode, "sync")
        self.assertEqual(rollout.action_horizon, "auto")
        self.assertEqual(rollout.replan_ratio, "0.8")

    def test_gam_eval_task_subsets(self):
        single = {
            "click_mouse",
            "fold_glasses",
            "hammer_nail",
            "pick_bucket",
            "pinch_tongs",
            "water_plant",
        }
        bimanual = {
            "bimanual_assembly",
            "bimanual_hanoi",
            "bimanual_microwave_cook",
            "bimanual_photograph",
            "bimanual_unlock_ipad",
        }
        expected = {
            "dexjoco_gam_multitask_11tasks_224": single | bimanual,
            "dexjoco_gam_single_arm_6tasks_224": single,
            "dexjoco_gam_bimanual_5tasks_224": bimanual,
        }
        for name, wanted in expected.items():
            variant = asyncio.run(load_variant(name))
            self.assertEqual(
                variant.vars["TRAIN_GIT_COMMIT"],
                "69afa536658198a22750b5618322edf68fdea93a",
                name,
            )
            actual = {entry.split("|", 1)[0] for entry in variant.arrays["TASKS"]}
            self.assertEqual(actual, wanted, name)
            self.assertEqual(variant.vars["EVAL_NUM_GPUS"], "4", name)
            self.assertEqual(variant.vars["N_RUNS"], "3", name)
            self.assertEqual(variant.vars["N_EPISODES"], "50", name)

    def test_all_dexjoco_configs_have_compatibility_rollout_defaults(self):
        configs = dexjoco_config_paths()
        self.assertEqual(len(configs), 32)

        for path in configs:
            variant = asyncio.run(load_variant(path.parent.name))
            rollout = rollout_for_variant(variant)
            self.assertEqual(rollout.inference_mode, "blocking_overlap", path)

            model_id = variant.vars["MODEL_ID"]
            if model_id in {"dexjoco-gam", "dexjoco-physixel"}:
                expected = ("16", "0.5")
            elif model_id == "dexjoco-pi05":
                expected = ("30", "0.8")
            elif model_id in {"dexjoco-n16", "dexjoco-n17"}:
                expected = ("16", "0.8")
            else:  # pragma: no cover - makes newly added families fail loudly
                self.fail(f"unclassified DexJoCo model {model_id!r}: {path}")
            self.assertEqual(
                (rollout.action_horizon, rollout.replan_ratio), expected, path
            )

            flags = dict(harness_for(variant).eval_flags(variant))
            self.assertEqual(flags["--inference-mode"], "blocking_overlap", path)
            self.assertEqual(flags["--action-horizon"], expected[0], path)
            self.assertEqual(flags["--replan-ratio"], expected[1], path)

    def test_gam_checkpoint_chunk_contract_matches_eval_horizon(self):
        for yaml_path in sorted(EXPERIMENTS.glob("dexjoco_gam_*/gam_config.yaml")):
            data = yaml.safe_load(yaml_path.read_text())
            variant = asyncio.run(load_variant(yaml_path.parent.name))
            rollout = rollout_for_variant(variant)
            self.assertEqual(data["action_head"]["chunk_size"], 16, yaml_path)
            self.assertEqual(data["dataset"]["chunk_size"], 16, yaml_path)
            self.assertEqual(rollout.action_horizon, "16", yaml_path)

    def test_action_horizon_accepts_positive_integer_or_auto(self):
        base = {
            "DEXJOCO_INFERENCE_MODE": "sync",
            "DEXJOCO_REPLAN_RATIO": "0.5",
        }
        for value in ("1", "01", "16", "auto"):
            rollout = rollout_for_variant(
                SimpleNamespace(vars={**base, "DEXJOCO_ACTION_HORIZON": value})
            )
            self.assertEqual(rollout.action_horizon, value)
        for value in ("", "0", "-1", "1.5", "AUTO"):
            vars_ = {**base, "DEXJOCO_ACTION_HORIZON": value}
            if value == "":
                # Unset/blank resolves to the canonical client auto mode.
                self.assertEqual(
                    rollout_for_variant(SimpleNamespace(vars=vars_)).action_horizon,
                    "auto",
                )
            else:
                with self.assertRaisesRegex(ValueError, "positive integer or 'auto'"):
                    rollout_for_variant(SimpleNamespace(vars=vars_))

    def test_ratio_and_mode_validation(self):
        base = {"DEXJOCO_ACTION_HORIZON": "16"}
        for ratio in ("0", "0.5", "1"):
            for mode in ("async", "sync", "blocking_overlap"):
                rollout_for_variant(
                    SimpleNamespace(
                        vars={
                            **base,
                            "DEXJOCO_REPLAN_RATIO": ratio,
                            "DEXJOCO_INFERENCE_MODE": mode,
                        }
                    )
                )
        for ratio in ("-0.1", "1.1", "nan", "abc"):
            with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
                rollout_for_variant(
                    SimpleNamespace(vars={**base, "DEXJOCO_REPLAN_RATIO": ratio})
                )
        with self.assertRaisesRegex(ValueError, "blocking_overlap"):
            rollout_for_variant(
                SimpleNamespace(vars={**base, "DEXJOCO_INFERENCE_MODE": "blocking"})
            )

    def test_snapshot_metadata_records_effective_rollout(self):
        rollout = {
            "inference_mode": "blocking_overlap",
            "action_horizon": "16",
            "replan_ratio": "0.5",
        }
        meta = snapshot_metadata(
            job_name="eval_test",
            cluster="skt",
            phase="eval",
            variant="dexjoco_gam_multitask_11tasks_224",
            path="/config.sh",
            meta_path="/config.meta.json",
            eval_rollout=rollout,
        )
        self.assertEqual(meta["eval_rollout"], rollout)

    def test_shell_harness_contains_client_capability_preflight_and_gam_cleanup(self):
        body = (ROOT / "lib" / "eval_body_dexjoco.sh").read_text()
        self.assertLess(
            body.index("\nresolve_exp_and_config\n"),
            body.index("\nresolve_eval_output_paths\n"),
            "DexJoCo eval must initialize EXP_DIR and OUT_DIR before output paths",
        )
        for required in (
            "dexjoco-openpi-eval --help",
            'commit="${DEXJOCO_GIT_COMMIT:-}"',
            'DEXJOCO_CLIENT_PYTHONPATH="$repo_src/dexjoco:$repo_src/openpi/packages/openpi-client/src"',
            'DEXJOCO_CLIENT_PYTHONPATH="$worktree/dexjoco:$worktree/openpi/packages/openpi-client/src"',
            '--inference-mode="$DEXJOCO_INFERENCE_MODE"',
            '--action-horizon="$DEXJOCO_ACTION_HORIZON"',
            '--replan-ratio="$DEXJOCO_REPLAN_RATIO"',
            "blocking_overlap",
            "gam_dexjoco_server.py.*--port $PORT",
        ):
            self.assertIn(required, body)


if __name__ == "__main__":
    unittest.main()
