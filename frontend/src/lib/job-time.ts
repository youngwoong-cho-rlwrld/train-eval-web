// Every job timestamp in the UI renders in KST regardless of the viewer's
// locale; results/checkpoint-history formatters share these instead of
// re-declaring the timezone and field set.
export const KST_TIME_ZONE = "Asia/Seoul";

const KST_SHORT_FIELDS: Intl.DateTimeFormatOptions = {
  timeZone: KST_TIME_ZONE,
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
};

export function formatKstShort(ms: number): string {
  return new Intl.DateTimeFormat(undefined, KST_SHORT_FIELDS).format(new Date(ms));
}

export function formatKstShortWithYear(ms: number): string {
  return new Intl.DateTimeFormat(undefined, {
    ...KST_SHORT_FIELDS,
    year: "numeric",
  }).format(new Date(ms));
}

export function parseJobTimestampMs(value?: string | null): number {
  const normalized = normalizeJobTimestamp(value);
  if (!normalized) return 0;
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? 0 : ms;
}

export function formatJobTimestamp(
  value?: string | null,
): { short: string; full: string } | null {
  const normalized = normalizeJobTimestamp(value);
  if (!normalized) return null;

  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return { short: normalized, full: normalized };
  }

  const short = formatKstShort(date.getTime());
  const full = new Intl.DateTimeFormat(undefined, {
    timeZone: KST_TIME_ZONE,
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
