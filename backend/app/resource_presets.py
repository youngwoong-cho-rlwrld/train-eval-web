"""Default scheduler resource requests for submitted jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SlurmResources:
    cpus_per_task: int
    memory: str


_SKT_RLWRLD_GPU_TRAIN: dict[int, SlurmResources] = {
    1: SlurmResources(cpus_per_task=22, memory="250G"),
    2: SlurmResources(cpus_per_task=44, memory="500G"),
    4: SlurmResources(cpus_per_task=88, memory="1000G"),
    7: SlurmResources(cpus_per_task=154, memory="1700G"),
    8: SlurmResources(cpus_per_task=176, memory="0"),
}

_SKT_L40S_GPU_TRAIN: dict[int, SlurmResources] = {
    1: SlurmResources(cpus_per_task=12, memory="90G"),
    2: SlurmResources(cpus_per_task=24, memory="180G"),
    3: SlurmResources(cpus_per_task=36, memory="270G"),
    4: SlurmResources(cpus_per_task=44, memory="350G"),
}

_SLURM_TRAIN_FALLBACK = SlurmResources(cpus_per_task=16, memory="180G")
_SLURM_EVAL_DEFAULT = SlurmResources(cpus_per_task=4, memory="40G")


def slurm_resources_for(
    *,
    cluster: str,
    partition: str,
    phase: Literal["train", "eval"],
    num_gpus: int,
) -> SlurmResources | None:
    """Resource request for the sbatch command, or None to send no flags.

    kakao's slurmctld rejects an explicit --cpus-per-task ("파티션 기본값
    (DefCpuPerGPU)이 자동 적용됩니다") and derives CPU/memory from the
    partition's per-GPU defaults, so kakao submissions must not carry
    resource flags at all.
    """
    cluster_key = cluster.strip().lower()
    if cluster_key == "kakao":
        return None

    if phase == "eval":
        return _SLURM_EVAL_DEFAULT

    partition_key = partition.strip().lower()
    if cluster_key == "skt" and partition_key == "rlwrld-gpu":
        return _SKT_RLWRLD_GPU_TRAIN.get(num_gpus, _SLURM_TRAIN_FALLBACK)
    if cluster_key == "skt" and partition_key == "l40s-gpu":
        return _SKT_L40S_GPU_TRAIN.get(num_gpus, _SLURM_TRAIN_FALLBACK)
    return _SLURM_TRAIN_FALLBACK
