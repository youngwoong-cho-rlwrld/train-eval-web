"""Completion checks for eval jobs whose Slurm state is misleading."""

import shlex

from .paths import CLUSTER_STAGING_REL
from .ssh import ssh_run
from .variants import Variant, load_variant


def eval_total(eval_sets: list[str], n_runs: int, tasks: list[str]) -> int:
    return max(len(tasks) * len(eval_sets) * n_runs, 0)


async def expected_eval_runs(variant: str, overrides: dict[str, str] | None = None) -> int:
    v = await load_variant(variant)
    eval_sets, n_runs, _, tasks = eval_shape(v, overrides)
    return eval_total(eval_sets, n_runs, tasks)


def eval_shape(
    variant: Variant,
    overrides: dict[str, str] | None = None,
) -> tuple[list[str], int, int, list[str]]:
    overrides = overrides or {}
    eval_sets = _override_list(overrides.get("eval_sets")) or variant.arrays.get("EVAL_SETS", [])
    n_runs = _override_int(overrides.get("eval_n_runs"), variant.vars.get("N_RUNS", "0"))
    n_eps = _override_int(overrides.get("eval_n_episodes"), variant.vars.get("N_EPISODES", "0"))
    tasks = variant.arrays.get("TASKS") or ["__single__"]
    return eval_sets, n_runs, n_eps, tasks


def exp_dir_rel_candidates(variant: str) -> list[str]:
    return [
        f"{CLUSTER_STAGING_REL}/experiments/{variant}",
        f"train-eval-scripts/experiments/{variant}",
    ]


async def eval_job_completed(
    host: str,
    stdout_path: str,
    eval_dir: str,
    variant: str,
    overrides: dict[str, str] | None = None,
) -> bool:
    expected = await expected_eval_runs(variant, overrides)
    if expected <= 0:
        return False

    stdout_q = shlex.quote(stdout_path)
    eval_dir_q = remote_path_expr(eval_dir)
    cmd = (
        f"stdout_path={stdout_q}; eval_dir={eval_dir_q}; expected={expected}; "
        "saved=$(grep -h '^Results saved to:' \"$stdout_path\" 2>/dev/null | wc -l); "
        "skipped=$(grep -h 'SKIP (results.json already exists):' \"$stdout_path\" 2>/dev/null | wc -l); "
        "done_count=$(grep -h '^DONE[[:space:]]' \"$stdout_path\" 2>/dev/null | wc -l); "
        "files=$(find \"$eval_dir\" -type f -name results.json 2>/dev/null | wc -l); "
        "echo \"$saved $skipped $done_count $files\""
    )
    r = await ssh_run(host, cmd, timeout=10.0)
    return _parse_completion_probe(r.stdout, expected)


async def eval_job_completed_from_log_dir(
    host: str,
    log_dir: str,
    job_id: str,
    variant: str,
    overrides: dict[str, str] | None = None,
) -> bool:
    expected = await expected_eval_runs(variant, overrides)
    if expected <= 0:
        return False

    log_dir_q = shlex.quote(log_dir)
    job_id_q = shlex.quote(job_id)
    if overrides and overrides.get("eval_dir"):
        eval_dirs = remote_path_expr(overrides["eval_dir"])
    else:
        eval_dirs = " ".join(
            remote_path_expr(f"$HOME/{rel}/eval_results")
            for rel in exp_dir_rel_candidates(variant)
        )
    cmd = (
        f"stdout_path=$(ls -1 {log_dir_q}/*_{job_id_q}.out 2>/dev/null | head -1); "
        "if [ -z \"$stdout_path\" ]; then echo '0 0 0 0'; exit 0; fi; "
        "saved=$(grep -h '^Results saved to:' \"$stdout_path\" 2>/dev/null | wc -l); "
        "skipped=$(grep -h 'SKIP (results.json already exists):' \"$stdout_path\" 2>/dev/null | wc -l); "
        "done_count=$(grep -h '^DONE[[:space:]]' \"$stdout_path\" 2>/dev/null | wc -l); "
        "files=0; "
        f"for d in {eval_dirs}; do "
        'c=$(find "$d" -type f -name results.json 2>/dev/null | wc -l); '
        'case "$c" in ""|*[!0-9]*) c=0;; esac; '
        'if [ "$c" -gt "$files" ]; then files="$c"; fi; '
        "done; "
        "echo \"$saved $skipped $done_count $files\""
    )
    r = await ssh_run(host, cmd, timeout=10.0)
    return _parse_completion_probe(r.stdout, expected)


def _parse_completion_probe(stdout: str, expected: int) -> bool:
    try:
        parts = stdout.strip().split()
        saved = int(parts[0]) if len(parts) > 0 else 0
        skipped = int(parts[1]) if len(parts) > 1 else 0
        done_count = int(parts[2]) if len(parts) > 2 else 0
        files = int(parts[3]) if len(parts) > 3 else 0
    except (ValueError, IndexError):
        return False

    stdout_complete = done_count > 0 or (saved + skipped) >= expected
    files_complete = files >= expected
    return stdout_complete and files_complete


def remote_path_expr(path: str) -> str:
    return path if path.startswith("$HOME/") else shlex.quote(path)


def _override_int(value: str | None, fallback: str) -> int:
    raw = (value or fallback or "").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _override_list(value: str | None) -> list[str]:
    return [part for part in (value or "").split() if part]
