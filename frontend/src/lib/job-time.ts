export function parseJobTimestampMs(value?: string | null, cluster?: string | null): number {
  const normalized = normalizeJobTimestamp(value, cluster);
  if (!normalized) return 0;
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? 0 : ms;
}

export function formatJobTimestamp(
  value?: string | null,
  cluster?: string | null,
): { short: string; full: string } | null {
  const raw = value?.trim();
  const normalized = normalizeJobTimestamp(raw, cluster);
  if (!raw || !normalized) return null;

  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return { short: raw, full: raw };
  }

  const short = new Intl.DateTimeFormat(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
  const full = new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
    hour12: false,
  }).format(date);

  return { short, full };
}

function normalizeJobTimestamp(value?: string | null, cluster?: string | null): string | null {
  const raw = value?.trim();
  if (!raw || raw === "Unknown" || raw === "None" || raw === "N/A") {
    return null;
  }
  if (hasTimeZone(raw)) {
    return raw;
  }
  // SKT Slurm reports UTC timestamps without a zone; Kakao reports local time.
  if (cluster === "skt" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$/.test(raw)) {
    return `${raw}Z`;
  }
  return raw;
}

function hasTimeZone(value: string): boolean {
  return /(?:Z|[+-]\d{2}:?\d{2})$/i.test(value);
}
