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

_KAKAO_RLWRLD_TRAIN: dict[int, SlurmResources] = {
    1: SlurmResources(cpus_per_task=16, memory="180G"),
    2: SlurmResources(cpus_per_task=32, memory="360G"),
    4: SlurmResources(cpus_per_task=64, memory="720G"),
    8: SlurmResources(cpus_per_task=120, memory="1300G"),
}

_SLURM_TRAIN_FALLBACK = SlurmResources(cpus_per_task=16, memory="180G")
_SLURM_EVAL_DEFAULT = SlurmResources(cpus_per_task=4, memory="40G")


def slurm_resources_for(
    *,
    cluster: str,
    partition: str,
    phase: Literal["train", "eval"],
    num_gpus: int,
) -> SlurmResources:
    if phase == "eval":
        return _SLURM_EVAL_DEFAULT

    cluster_key = cluster.strip().lower()
    partition_key = partition.strip().lower()
    if cluster_key == "skt" and partition_key == "rlwrld-gpu":
        return _SKT_RLWRLD_GPU_TRAIN.get(num_gpus, _SLURM_TRAIN_FALLBACK)
    if cluster_key == "skt" and partition_key == "l40s-gpu":
        return _SKT_L40S_GPU_TRAIN.get(num_gpus, _SLURM_TRAIN_FALLBACK)
    if cluster_key == "kakao" and partition_key == "rlwrld":
        return _KAKAO_RLWRLD_TRAIN.get(num_gpus, _SLURM_TRAIN_FALLBACK)
    return _SLURM_TRAIN_FALLBACK
