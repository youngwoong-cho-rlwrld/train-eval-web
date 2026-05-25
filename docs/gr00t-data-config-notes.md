# GR00T Data Config Notes

This note summarizes how GR00T decides which data modalities and timesteps to read, how embodiment tags are used, and how dataset normalization behaves for single-dataset, multi-dataset, and pretrained-checkpoint cases.

## 1. Modalities And Timesteps

GR00T has two data-config styles depending on the model generation.

### N1.5 Style

N1.5 commonly uses a YAML dataset config shaped like this:

Actual sample directory: `/fsx/rlwrld/youngwoong_cho/workspace/gr00t/configs/` (for example, `test_hs_singletrain.yaml`).

```yaml
train:
  datasets:
    - path: /path/to/dataset
      embodiment_tag: new_embodiment
      data_config: allex_thetwo_ck40_egostereo
      weight: 1.0
```

The YAML does not directly list every modality key and timestep. Instead, `data_config` names a registered config in GR00T's data config registry. That registered config defines the actual modality streams, timestep windows, transforms, and action/state processing.

Actual registered config excerpt from `/fsx/rlwrld/youngwoong_cho/workspace/gr00t/gr00t/experiment/data_config.py`:

```python
class allex_thetwo_ck40_egostereo_config(BaseDataConfig):
    video_keys = ["video.camera_ego_left", "video.camera_ego_right"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(40))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={key: "q99" for key in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={key: "q99" for key in self.action_keys}),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
                torque_horizon=0,
                max_torque_dim=0,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "allex_thetwo_ck40_egostereo": allex_thetwo_ck40_egostereo_config(),
}
```

N1.5 timestep rule:

Path: `/fsx/rlwrld/youngwoong_cho/workspace/gr00t/gr00t/data/dataset.py`

N1.5 also converts `delta_indices` into concrete episode frames by adding them to the sampled `base_index`. Unlike N1.6, the implementation is split across modality-specific helpers.

```python
class ModalityConfig(BaseModel):
    delta_indices: list[int]
    """Delta indices to sample relative to the current index.
    The returned data will correspond to the original data at a sampled base index + delta indices."""


class LeRobotSingleDataset(Dataset):
    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        return self.transforms(self.get_step_data(trajectory_id, base_index))

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        ...

    def get_state_or_action(self, trajectory_id: int, modality: str, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        ...

    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        step_indices = self.delta_indices[key] + base_index
        ...
```

Practical implication: for N1.5, the important follow-up is to inspect the registered config behind the YAML `data_config` value.

### N1.6 / Physixel Style

N1.6 / Physixel uses a Python modality config that directly registers modality streams for an embodiment tag. A typical ALLEX config looks like this:

Actual sample directory: `/fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/configs/` (for example, `allex_egostereo_ck40_config_absolute.py`).

```python
allex_egostereo_ck8_config_absolute = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera_ego_left", "camera_ego_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm_joints",
            "left_arm_joints",
            "right_hand_joints",
            "left_hand_joints",
            "neck_joints",
            "waist_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(8)),
        modality_keys=[
            "right_arm_joints",
            "left_arm_joints",
            "right_hand_joints",
            "left_hand_joints",
            "neck_joints",
            "waist_joints",
        ],
        action_configs=[...],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(allex_egostereo_ck8_config_absolute, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

The core timestep rule is inside the single-step dataset loader. This function converts each modality's relative `delta_indices` into concrete episode frame indices for the sampled step.

Path: `/fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/sharded_single_step_dataset.py`

```python
def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = False,
) -> VLAStepData:
    step_data = {}

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # Sample timesteps according to delta indices configuration
        indices_to_load = [step_index + delta_index for delta_index in config.delta_indices]
        if allow_padding:
            indices_to_load = [max(0, min(idx, len(episode_data) - 1)) for idx in indices_to_load]
        for key in config.modality_keys:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            if modality in ["state", "action"]:
                step_data[modality][key] = np.vstack(
                    [
                        np.array(modality_data.iloc[i]).astype(np.float32)
                        for i in range(len(modality_data))
                    ]
                )
            else:
                step_data[modality][key] = modality_data.tolist()

    video_data = step_data.get("video", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]

    vla_step_data = VLAStepData(
        images=video_data,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return vla_step_data
```

Examples:

- `video.delta_indices=[0]` means current image only.
- `state.delta_indices=[0]` means current state only.
- `action.delta_indices=list(range(8))` means an 8-step action chunk from offsets 0 through 7.
- `language.delta_indices=[0]` means current task text.

If padding is allowed, out-of-range indices near episode boundaries are clamped to valid frame indices. If padding is not allowed, the usable episode length is shortened so future action indices remain valid.

For N1.6 / Physixel, action horizon is therefore defined by `action.delta_indices`. ck8 means 8 predicted action steps, ck16 means 16, ck40 means 40.

### Dataset Metadata Mapping

Dataset metadata mapping connects logical modality names in a config to the actual parquet columns and slice ranges inside a LeRobot dataset. For example, if `state.right_hand_joints` is stored as a slice of a larger `observation.state` vector, `meta/modality.json` defines the `original_key`, `start`, and `end` used to extract it.

The modality config keys must match the dataset's LeRobot metadata.

N1.5 mapping happens in `/fsx/rlwrld/youngwoong_cho/workspace/gr00t/gr00t/data/dataset.py`:

```python
def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
    modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_meta_path, "r") as f:
        modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
    return modality_meta

def get_state_or_action(self, trajectory_id: int, modality: str, key: str, base_index: int) -> np.ndarray:
    assert key.startswith(modality + ".")
    key = key.replace(modality + ".", "")

    le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
    le_key = le_state_or_action_cfg[key].original_key
    if le_key is None:
        le_key = key

    data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])
    le_indices = np.arange(
        le_state_or_action_cfg[key].start,
        le_state_or_action_cfg[key].end,
    )
    data_array = data_array[:, le_indices]
```

N1.6 mapping happens in `/fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/lerobot_episode_loader.py`:

```python
def _extract_joint_groups(self, df: pd.DataFrame, joint_groups: list[str], modality_type: str) -> pd.DataFrame:
    modality_info = self.modality_meta.get(modality_type, {})
    joint_data = pd.DataFrame()

    for group_name in joint_groups:
        if group_name in modality_info:
            group_info = modality_info[group_name]
            start_idx = group_info["start"]
            end_idx = group_info["end"]
            original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])

            if isinstance(df[original_key].iloc[0], np.ndarray):
                joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
            else:
                joint_data[group_name] = df[original_key]

    return joint_data

def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
    original_df = pd.read_parquet(parquet_path)
    loaded_df = pd.DataFrame()

    for modality_type in ["state", "action"]:
        if modality_type not in self.modality_configs:
            continue
        joint_groups_df = self._extract_joint_groups(
            original_df, self.modality_configs[modality_type].modality_keys, modality_type
        )
        for joint_group in joint_groups_df.columns:
            loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]
```

```text
modality key
  -> dataset/meta/modality.json
  -> original_key + start/end slice
  -> parquet column slice
```

For example, `right_hand_joints` may be a named slice of a larger `observation.state` or `action` vector. The episode loader uses `meta/modality.json` to map the configured group name to the raw column and slice range.

## 2. Embodiment Tags

Embodiment tags identify which robot/body schema a sample belongs to. Put simply, this is the label that tells GR00T which robot state/action format the data follows. GR00T stores modality configs and normalization statistics under embodiment tags.

Implementation differs between N1.5 and N1.6.

N1.5:

- `embodiment_tag` is read from the YAML dataset entry and passed into the dataset.
- Modality/timestep/transform are selected separately by `data_config`.
- The tag also maps to an action-expert category id through `EMBODIMENT_TAG_MAPPING`.

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t/scripts/gr00t_finetune.py
ds_entries = (train_cfg.get("train") or {}).get("datasets") or []

dataset_paths = [str(e["path"]) for e in ds_entries]
data_configs = [str(e["data_config"]) for e in ds_entries]
embodiment_tags = [str(e["embodiment_tag"]) for e in ds_entries]

data_config_cls = [load_data_config(config) for config in data_configs]
modality_configs = [config.modality_config() for config in data_config_cls]
transforms = [config.transform() for config in data_config_cls]

dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs[dataset_idx],
    transforms=transforms[dataset_idx],
    embodiment_tag=embodiment_tags[dataset_idx],
    video_backend=config.video_backend,
)


# /fsx/rlwrld/youngwoong_cho/workspace/gr00t/gr00t/data/embodiment_tags.py
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.EGODEX.value: 30,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
}
```

N1.6 / Physixel:

- `embodiment_tag` is parsed from fine-tune config.
- The Python modality config registers itself under that tag.
- Dataset construction looks up modality config by that tag.
- Statistics are merged by that tag and installed into the processor.

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/experiment/launch_finetune.py
embodiment_tag = ft_config.embodiment_tag.value
if ft_config.modality_config_path is not None:
    load_modality_config(ft_config.modality_config_path)

config = get_default_config().load_dict({
    "data": {
        "datasets": [{
            "dataset_paths": ft_config.dataset_path,
            "mix_ratio": 1.0,
            "embodiment_tag": embodiment_tag,
        }],
    }
})


# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/configs/data/embodiment_configs.py
def register_modality_config(config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT):
    assert embodiment_tag.value not in MODALITY_CONFIGS
    MODALITY_CONFIGS[embodiment_tag.value] = config


# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/factory.py
dataset = ShardedSingleStepDataset(
    dataset_path=dataset_path,
    embodiment_tag=EmbodimentTag(embodiment_tag),
    modality_configs=self.config.data.modality_configs[embodiment_tag],
    ...
)


# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/sharded_mixture_dataset.py
emb = ds.embodiment_tag.value
all_stats_by_emb[emb].append(ds.get_dataset_statistics())
weights_by_emb[emb].append(w)

self.global_stats = stats_by_emb
self.processor.set_statistics(self.global_stats, override=self.override_pretraining_statistics)
```

Important consequences:

- In N1.5, changing only `embodiment_tag` does not change modality/timestep behavior; changing `data_config` does.
- In N1.6, dataset specs and modality config registration must use the same tag.
- Multiple datasets with the same tag share one modality schema and one merged normalization statistic set.
- Datasets with different tags keep separate modality configs and statistics.
- Checkpoints save processor metadata by tag, including modality configs, statistics, and embodiment id mappings.

The `EmbodimentTag` enum is the source of valid tag values. The common fallback for new robot setups is `NEW_EMBODIMENT`, whose value is `new_embodiment`.

A tag should be shared only when datasets truly share the same robot schema, state/action grouping, and normalization semantics. If morphology or state/action layout differs, use separate tags and register separate modality configs.

## 3. Dataset Normalization

Normalization is controlled by dataset statistics plus the active modality config.

### Per-Dataset Statistics

Each LeRobot dataset carries metadata such as:

```text
meta/info.json
meta/modality.json
meta/stats.json
meta/episodes.jsonl
meta/tasks.jsonl
```

Dataset construction ensures statistics exist:

```text
DatasetFactory
  -> generate_stats(dataset_path)
  -> generate_rel_stats(dataset_path, embodiment_tag)
```

`meta/stats.json` stores raw feature statistics. The episode loader slices these raw statistics into the configured joint groups using `meta/modality.json`.

For absolute action configs, normal action stats are used. For relative action configs, GR00T can generate and use `relative_stats.json` for action normalization.

### Single Dataset

Even for a single dataset, GR00T routes statistics through `ShardedMixtureDataset`, builds `global_stats`, and installs them into the processor under the embodiment tag. With one dataset, the merged stats are effectively that dataset's stats.

```text
single dataset stats
  -> stats_by_emb[embodiment_tag]
  -> processor.set_statistics(stats_by_emb)
  -> StateActionProcessor computes norm_params
  -> apply_state / apply_action normalize active modality keys
```

Relevant code:

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/sharded_mixture_dataset.py
for ds, w in zip(self.datasets, self.weights):
    emb = ds.embodiment_tag.value
    stats = ds.get_dataset_statistics()
    all_stats_by_emb[emb].append(stats)
    weights_by_emb[emb].append(w)

self.global_stats = stats_by_emb
self.processor.set_statistics(self.global_stats, override=self.override_pretraining_statistics)
```

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/state_action/state_action_processor.py
def set_statistics(self, statistics, override: bool = False) -> None:
    for key in statistics:
        if key not in self.statistics or override:
            self.statistics[key] = deepcopy(statistics[key])
        else:
            print(f"Embodiment tag {key} already in statistics, skipping updating")
    self._compute_normalization_parameters()

def _compute_normalization_parameters(self) -> None:
    for embodiment_tag in self.statistics:
        for modality in ["state", "action"]:
            for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                min_vals = np.array(stats["q01"] if self.use_percentiles else stats["min"])
                max_vals = np.array(stats["q99"] if self.use_percentiles else stats["max"])
                mean_vals = np.array(stats["mean"])
                std_vals = np.array(stats["std"])
                self.norm_params[embodiment_tag][modality][joint_group] = {
                    "min": min_vals,
                    "max": max_vals,
                    "mean": mean_vals,
                    "std": std_vals,
                }
```

State/action normalization usually maps values to `[-1, 1]` using min/max and clipping. If a configured key is marked for mean/std normalization, that key uses mean/std. If percentile mode is enabled, q01/q99 are used instead of full min/max bounds.

### Multi-Dataset

For multiple datasets under the same embodiment tag, GR00T groups stats by `embodiment_tag`, merges stats within each group, and then keeps one normalization space per embodiment. It does not keep separate per-dataset normalization at runtime.

```text
per-dataset stats
  -> group by embodiment_tag
  -> merge stats within each embodiment group
  -> processor.statistics[embodiment_tag]
  -> processor.norm_params[embodiment_tag]
```

Relevant code:

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/data/dataset/sharded_mixture_dataset.py
all_stats_by_emb: dict[str, list] = {}
weights_by_emb: dict[str, list[float]] = {}
for ds, w in zip(self.datasets, self.weights):
    emb = ds.embodiment_tag.value
    stats = ds.get_dataset_statistics()
    all_stats_by_emb[emb].append(stats)
    weights_by_emb[emb].append(w)

stats_by_emb = {}
for emb, stats in all_stats_by_emb.items():
    stats_by_emb[emb] = {}
    for modality in ["state", "action", "relative_action"]:
        if modality in stats[0]:
            modality_stats = [s[modality] for s in stats]
            stats_by_emb[emb][modality] = merge_statistics(
                per_dataset_stats=modality_stats,
                dataset_sampling_weights=weights_by_emb[emb],
                is_relative_stats=(modality == "relative_action"),
            )
```

The weighted variance merge uses the standard second-moment formula:

```text
merged_mean = sum(weight_i * mean_i)
merged_var = sum(weight_i * (std_i^2 + mean_i^2)) - merged_mean^2
```

The merged stats are then stored under the embodiment tag and used by the processor for all datasets with that tag.

Min/max style bounds are merged to cover the dataset ranges. The final result is one merged statistic set for each embodiment tag.

### Pretrained Checkpoints

When fine-tuning from a pretrained N1.6 checkpoint, the checkpoint processor is loaded first. That processor already contains `processor_config.json`, `statistics.json`, and `embodiment_id.json`.

```text
checkpoint processor
  -> load processor_config/statistics/embodiment_id
  -> merge current modality_configs into loaded processor config
  -> dataset build computes current dataset stats
  -> processor.set_statistics(current_stats, override=override_pretraining_statistics)
```

Relevant code:

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/model/gr00t_n1d6/setup.py
if self.config.training.start_from_checkpoint is not None:
    processor = AutoProcessor.from_pretrained(
        self.config.training.start_from_checkpoint,
        modality_configs=self.config.data.modality_configs,
        max_action_horizon=self.model_config.action_horizon,
        use_relative_action=self.model_config.use_relative_action,
        **self.transformers_loading_kwargs,
    )
```

```python
# /fsx/rlwrld/youngwoong_cho/workspace/gr00t-n16/gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py
with open(config_file, "r") as f:
    config = json.load(f)
with open(statistics_file, "r") as f:
    statistics = json.load(f)

processor_kwargs = config["processor_kwargs"]
processor_kwargs["statistics"] = statistics
processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

modality_configs = kwargs.pop("modality_configs", {})
for embodiment_tag, modality_config in modality_configs.items():
    processor_kwargs["modality_configs"][embodiment_tag] = modality_config
```

Important distinction:

- Current modality config is merged/overwritten into the loaded processor config for the matching embodiment tag.
- Statistics follow `processor.set_statistics(..., override=override_pretraining_statistics)`.
- If `override_pretraining_statistics=False` and checkpoint stats already contain the same embodiment tag, current dataset stats are skipped.
- If `override_pretraining_statistics=True`, current dataset stats replace checkpoint stats for that tag.

At evaluation time, the policy loads the trained checkpoint processor. It uses the modality configs and statistics saved in that checkpoint rather than recomputing normalization from evaluation data.

## Code Landmarks

N1.6 / Physixel:

```text
gr00t/experiment/launch_finetune.py
gr00t/configs/finetune_config.py
gr00t/configs/data/embodiment_configs.py
gr00t/data/embodiment_tags.py
gr00t/data/dataset/factory.py
gr00t/data/dataset/sharded_single_step_dataset.py
gr00t/data/dataset/lerobot_episode_loader.py
gr00t/data/dataset/sharded_mixture_dataset.py
gr00t/data/state_action/state_action_processor.py
gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py
```

N1.5:

```text
scripts/gr00t_finetune.py
gr00t/experiment/data_config.py
DATA_CONFIG_MAP
```

## Practical Checklist

1. Identify whether the run uses the N1.5 registered-data-config path or the N1.6 Python-modality-config path.
2. For N1.5, inspect the registered config behind the YAML `data_config` name.
3. For N1.6 / Physixel, inspect the Python modality config directly.
4. Confirm every modality key exists in `meta/modality.json`.
5. Confirm `action.delta_indices` matches the intended action horizon.
6. Confirm datasets sharing an embodiment tag should really share one normalization statistic set.
7. Inspect the pretrained processor's `statistics.json` to see whether the same embodiment tag already exists.
8. Use `override_pretraining_statistics=True` when current dataset stats should replace pretrained stats for an existing tag.
