export function formatPct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

/** Last non-empty path segment, falling back to the original string. */
export function basename(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}
