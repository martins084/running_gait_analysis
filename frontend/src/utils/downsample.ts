/** Max points to send to charts for smooth rendering on long clips. */
export const CHART_MAX_POINTS = 500;

/**
 * Returns indices 0..len-1 downsampled to at most maxPoints (keeps first and last).
 */
export function downsampleIndices(len: number, maxPoints: number): number[] {
  if (len <= 0) return [];
  if (len <= maxPoints) return Array.from({ length: len }, (_, i) => i);
  const step = (len - 1) / (maxPoints - 1);
  const out: number[] = [];
  for (let i = 0; i < maxPoints; i++) {
    out.push(Math.round(i * step));
  }
  // Deduplicate while preserving order
  return [...new Set(out)].sort((a, b) => a - b);
}
