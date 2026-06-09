export function jobDetailHref(cluster: string, jobId?: string | null) {
  if (!jobId) return undefined;
  return `/jobs/${encodeURIComponent(cluster)}/${encodeURIComponent(jobId)}`;
}
