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
  start?: string | null;
  end?: string | null;
};

export type SubmitResponse = {
  job_id: string;
  job_name: string;
  partition: string;
  sbatch_cmd: string;
  rsync_stdout: string;
  sbatch_stdout: string;
};

export type Paths = {
  stdout: string;
  stderr: string;
  exp_dir: string;
  ckpt_dir: string | null;
  eval_dir: string | null;
  isaac_logs_glob: string | null;
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

export type Dataset = {
  name: string;
  path: string;
  height: number | null;
  width: number | null;
  episodes: number | null;
  codec: string | null;
};

export type MlxpNode = {
  name: string;
  gpu_used: number;
  gpu_total: number;
  gpu_free: number;
  sanctioned: boolean;
};

export type Partition = {
  name: string;
  is_default: boolean;
  is_background: boolean;
  total_nodes: number;
  idle_nodes: number;
  gpu_total: number;
  gpu_idle: number;
  states: Record<string, number>;
};

export type JobDetails = {
  cluster: string;
  job_id: string;
  job_name: string;
  phase: "train" | "resume" | "eval" | "unknown";
  variant: string | null;
  state: string;
  elapsed: string;
  wandb_url: string | null;
  paths: Paths;
  progress: Progress;
};
