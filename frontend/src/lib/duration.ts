/** Parse slurm-style durations: SS, MM:SS, HH:MM:SS, D-HH:MM:SS. */
export function parseSlurmDuration(s: string): number {
  if (!s) return 0;
  let days = 0;
  let rest = s;
  if (s.includes("-")) {
    const [d, r] = s.split("-");
    days = parseInt(d, 10) || 0;
    rest = r;
  }
  const parts = rest.split(":").map((p) => parseInt(p, 10) || 0);
  let h = 0,
    m = 0,
    sec = 0;
  if (parts.length === 3) [h, m, sec] = parts;
  else if (parts.length === 2) [m, sec] = parts;
  else [sec] = parts;
  return days * 86400 + h * 3600 + m * 60 + sec;
}

/** Humanize seconds: "45s", "12m", "3h 7m", "2d 4h". */
export function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const days = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.round((seconds % 3600) / 60);
  if (days > 0) return h > 0 ? `${days}d ${h}h` : `${days}d`;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}
