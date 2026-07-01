const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export type LogStream = "out" | "err" | "isaac";

export type ApiJobPhase = "train" | "resume" | "eval" | "unknown";

export function logStreamUrl(cluster: string, jobId: string, stream: LogStream) {
  return `${API_BASE}/api/jobs/${cluster}/${jobId}/logs?stream=${stream}`;
}

// ── types matching backend Pydantic models ──

export type Cluster = { name: string; vars: Record<string, string> };

export type Variant = {
  name: string;
  raw: string;
  vars: Record<string, string>;
  arrays: Record<string, string[]>;
};

export type Job = {
  cluster: string;
  job_id: string;
  job_name: string;
  partition: string;
  state: string;
  elapsed: string;
  nodelist: string;
  reason?: string;
  time_left?: string | null;
  queue_position?: number | null;
  start?: string | null;
  end?: string | null;
  phase?: string | null;
  variant?: string | null;
  resume_of?: string | null;
  resubmit_action?: string | null;
};

export type SubmitResponse = {
  job_id: string;
  job_name: string;
  partition: string;
  sbatch_cmd: string;
  rsync_stdout: string;
  sbatch_stdout: string;
};

export type ConfigPreviewFlag = {
  flag: string;
  value: string;
};

export type DataInterfaceSummary = {
  variant: string;
  source: string | null;
  path: string | null;
  text: string | null;
  config_name: string | null;
  embodiment_tag: string | null;
  action_horizon: number | null;
  error: string | null;
};

export type ExperimentFile = {
  kind: string;
  label: string;
  title: string;
  path: string;
  content: string;
  exists: boolean;
  purpose: string;
};

export type ExperimentFileVersion = {
  created_at: string;
  path: string;
  files: string[];
};

export type ExperimentFiles = {
  variant: string;
  config: ExperimentFile;
  second_file: ExperimentFile;
  versions: ExperimentFileVersion[];
};

export type SaveExperimentFilesResponse = ExperimentFiles & {
  saved_version_path: string | null;
};

export type SubmitConfigPreview = {
  path: string | null;
  model_id: string | null;
  model_label: string | null;
  model_repo_path: string | null;
  model_repo_error: string | null;
  text: string;
  flags: ConfigPreviewFlag[];
};

export type Paths = {
  stdout: string;
  stderr: string;
  exp_dir: string;
  ckpt_dir: string | null;
  eval_checkpoint: string | null;
  eval_dir: string | null;
  isaac_logs_glob: string | null;
};

export type PathExistence = {
  exists: boolean;
  kind: string | null;
};

export type CheckpointEntry = {
  path: string;
  job_name: string;
  step: number;
};

export type CheckpointCopyRecord = {
  copy_id: string;
  source_cluster: string;
  source_job: string;
  source_path: string;
  dest_cluster: string;
  dest_path: string;
  copied_at: number;
  delete_source: boolean;
  source_exists: boolean | null;
  dest_exists: boolean | null;
};

export type CopyJobStatus = {
  copy_id: string;
  status: "running" | "done" | "error";
  error: string | null;
  phase: string | null;
  copies_total: number;
  copies_done: number;
  current_source: string | null;
  current_dest: string | null;
  src_size_bytes: number | null;
  dest_size_bytes: number | null;
  started_at: number;
  finished_at: number | null;
};

export type Progress = {
  phase: string;
  current_step: number | null;
  max_steps: number | null;
  completed_runs: number | null;
  total_runs: number | null;
  current_label: string | null;
  percent: number | null;
};

export type GpuDeviceUsage = {
  index: number;
  name: string | null;
  utilization_gpu_percent: number | null;
  used_gb: number;
  total_gb: number;
  used_mib: number;
  total_mib: number;
};

export type GpuUsage = {
  node: string | null;
  utilization_gpu_percent: number | null;
  used_gb: number | null;
  total_gb: number | null;
  devices: GpuDeviceUsage[];
  error: string | null;
};

export type ConfigSnapshot = {
  path: string | null;
  meta_path: string | null;
  text: string | null;
  extra_args_path: string | null;
  extra_args: string[];
  wandb_project: string | null;
  git_repo_path: string | null;
  git_repo_label: string | null;
  git_branch: string | null;
  git_commit: string | null;
  git_dirty_at_submit: boolean | null;
  git_committed_dirty: boolean | null;
  error: string | null;
};

export type EvalRun = {
  task: string | null;
  eval_set: string;
  run: string;
  seed: number | null;
  success_count: number | null;
  total_episodes: number | null;
  success_rate: number | null;
  path: string;
};

export type TrainingJobRef = {
  cluster: string;
  job_id: string;
  job_name: string | null;
};

export type Dataset = {
  name: string;
  path: string;
  height: number | null;
  width: number | null;
  episodes: number | null;
  codec: string | null;
};

export type DexjocoTasks = {
  families: Array<{ family: string; tasks: string[] }>;
  tasks: string[];
};

export type MlxpNode = {
  name: string;
  gpu_used: number;
  gpu_total: number;
  gpu_free: number;
  queued_jobs: number;
  queued_gpus: number;
  gpu_type: string | null;
};

export type MlxpSettings = {
  user: string;
  namespace: string;
  owner_label: string;
  tool_label: string;
  default_node: string;
  gpu_node_prefix: string;
  gpu_type: string;
  gpus_per_node: number;
  ddn_mount: string;
  ddn_user_home: string;
  datasets_dir: string;
  experiments_dir: string;
  hf_home: string;
  workspace_dir: string;
  isaac_dir: string;
  data_pod_name: string;
  ddn_pvc: string;
  image: string;
  image_pull_secret: string;
  zone: string;
  wandb_secret: string;
};

export type ClusterEnvSettings = {
  name: string;
  env_text: string;
  path: string | null;
};

export type WandbStatus = {
  logged_in: boolean;
  entity: string | null;
  project: string;
  error: string | null;
};

export type Partition = {
  name: string;
  is_default: boolean;
  is_background: boolean;
  total_nodes: number;
  idle_nodes: number;
  gpu_total: number;
  gpu_free: number;
  queued_jobs: number;
  queued_gpus: number;
  gpu_type: string | null;
  states: Record<string, number>;
};

export type GpuQueueNode = {
  name: string;
  gpu_type: string | null;
  gpu_total: number;
  gpu_used: number;
  state: string | null;
  reason: string | null;
};

export type GpuQueueJob = {
  job_id: string;
  requested_gpus: number;
  reason: string | null;
  name: string | null;
};

export type GpuQueueSnapshot = {
  cluster: string;
  partition: string;
  nodes: GpuQueueNode[];
  queue: GpuQueueJob[];
};

export type JobDetails = {
  cluster: string;
  job_id: string;
  job_name: string;
  phase: ApiJobPhase;
  variant: string | null;
  resume_of: string | null;
  resubmit_action: string | null;
  training_job: TrainingJobRef | null;
  train_note: string | null;
  state: string;
  elapsed: string;
  wandb_project: string | null;
  wandb_url: string | null;
  paths: Paths;
  progress: Progress;
  gpu: GpuUsage | null;
  config_snapshot: ConfigSnapshot | null;
  data_interface: DataInterfaceSummary | null;
  eval_runs: EvalRun[];
};

export type JobMetadata = {
  wandb_project: string | null;
  config_snapshot: ConfigSnapshot | null;
  data_interface: DataInterfaceSummary | null;
};

export type JobGpu = {
  gpu: GpuUsage | null;
};

export type JobProgress = {
  cluster: string;
  job_id: string;
  phase: ApiJobPhase;
  state: string;
  elapsed: string;
  wandb_url: string | null;
  progress: Progress;
};

export type JobEvalRuns = {
  eval_runs: EvalRun[];
};

export type GitStatus = {
  repo_path: string | null;
  repo_label: string | null;
  commit: string | null;
  short_commit: string | null;
  commit_subject: string | null;
  branch: string | null;
  dirty: boolean;
  files: string[];
  error: string | null;
};

export type GitCommitOption = {
  commit: string;
  short_commit: string;
  subject: string;
};

export type ResultCell = {
  eval_set: string;
  mean_success_rate: number | null;
  std_success_rate: number | null;
  per_run_success_rate: number[];
  success_counts: Array<number | null>;
  episode_counts: Array<number | null>;
  completed_runs: number;
  expected_runs?: number | null;
  source?: string | null;
};

export type ResultTask = {
  task: string;
  task_name?: string | null;
  instruction?: string | null;
  eval_sets: ResultCell[];
};

export type ResultVariant = {
  cluster: string;
  job_id?: string | null;
  job_name?: string | null;
  job_state?: string | null;
  checkpoint_job_cluster?: string | null;
  checkpoint_job_id?: string | null;
  checkpoint_job_name?: string | null;
  variant: string;
  experiment?: string | null;
  model_version?: string | null;
  note?: string | null;
  checkpoint?: string | null;
  n_episodes?: number | null;
  n_runs?: number | null;
  num_envs_per_gpu?: number | null;
  total_num_envs?: number | null;
  source?: string | null;
  completed_at?: number | null;
  tasks: ResultTask[];
};

export type ResultsResponse = {
  clusters: string[];
  variants: ResultVariant[];
  errors: Array<{ cluster: string; error: string }>;
};
