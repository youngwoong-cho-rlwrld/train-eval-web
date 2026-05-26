export function parseJobTimestampMs(value?: string | null, _cluster?: string | null): number {
  void _cluster;
  const normalized = normalizeJobTimestamp(value);
  if (!normalized) return 0;
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? 0 : ms;
}

export function formatJobTimestamp(
  value?: string | null,
  _cluster?: string | null,
): { short: string; full: string } | null {
  void _cluster;
  const raw = value?.trim();
  const normalized = normalizeJobTimestamp(raw);
  if (!raw || !normalized) return null;

  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return { short: raw, full: raw };
  }

  const short = new Intl.DateTimeFormat(undefined, {
    timeZone: "Asia/Seoul",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
  const full = new Intl.DateTimeFormat(undefined, {
    timeZone: "Asia/Seoul",
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

function normalizeJobTimestamp(value?: string | null): string | null {
  const raw = value?.trim();
  if (!raw || raw === "Unknown" || raw === "None" || raw === "N/A") {
    return null;
  }
  return raw;
}
