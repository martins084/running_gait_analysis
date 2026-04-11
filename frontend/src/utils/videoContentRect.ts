/**
 * Map normalized landmark / COM coordinates (video pixel space 0–1) onto the
 * visible `<video>` rectangle when `object-fit: contain` letterboxes the frame.
 */

export type VideoContentRect = { offsetX: number; offsetY: number; contentW: number; contentH: number };

/**
 * Returns the inner rectangle where the decoded video frame is painted, in **CSS pixels**
 * relative to the video element's content box (top-left of element = 0,0).
 */
export function getVideoContentRect(video: HTMLVideoElement): VideoContentRect | null {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const elW = video.clientWidth;
  const elH = video.clientHeight;
  if (vw <= 0 || vh <= 0 || elW <= 0 || elH <= 0) return null;

  const scale = Math.min(elW / vw, elH / vh);
  const contentW = vw * scale;
  const contentH = vh * scale;
  const offsetX = (elW - contentW) / 2;
  const offsetY = (elH - contentH) / 2;

  return { offsetX, offsetY, contentW, contentH };
}

/** Normalized COM (0–1 in source frame) → canvas pixel position. */
export function normalizedToContentPixel(
  nx: number,
  ny: number,
  r: VideoContentRect,
): { x: number; y: number } {
  return {
    x: r.offsetX + nx * r.contentW,
    y: r.offsetY + ny * r.contentH,
  };
}

