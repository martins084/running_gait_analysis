import { useCallback, useEffect, useRef } from "react";

export type UseVideoFrameSyncOptions = {
  videoRef: React.RefObject<HTMLVideoElement>;
  /** Analysis frame count (joint_angles.length). */
  frameCount: number;
  /** From API pose pipeline when available; else derived from duration. */
  fps?: number | null;
  onFrame: (frameIndex: number) => void;
};

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

/**
 * Resolves frames-per-second for mapping video time ↔ analysis frame index.
 */
function resolveFps(
  video: HTMLVideoElement,
  frameCount: number,
  fps: number | null | undefined,
): number | null {
  if (fps != null && fps > 0 && Number.isFinite(fps)) {
    return fps;
  }
  const dur = video.duration;
  if (Number.isFinite(dur) && dur > 0 && frameCount > 0) {
    return frameCount / dur;
  }
  return null;
}

/**
 * Maps `video.currentTime` to a clamped analysis frame index (0..frameCount-1).
 */
export function timeToFrameIndex(
  currentTime: number,
  video: HTMLVideoElement,
  frameCount: number,
  fps: number | null | undefined,
): number {
  if (frameCount <= 0) return 0;
  const rate = resolveFps(video, frameCount, fps);
  if (rate == null || !Number.isFinite(rate)) return 0;
  return clamp(Math.round(currentTime * rate), 0, frameCount - 1);
}

/**
 * Seeks the video so the nearest analysis frame matches `frameIndex`.
 */
export function seekVideoToFrame(
  video: HTMLVideoElement,
  frameIndex: number,
  frameCount: number,
  fps: number | null | undefined,
): void {
  if (frameCount <= 0) return;
  const rate = resolveFps(video, frameCount, fps);
  if (rate == null || !Number.isFinite(rate)) return;
  const f = clamp(Math.round(frameIndex), 0, frameCount - 1);
  video.currentTime = f / rate;
}

/**
 * Subscribes to video playback and maps `currentTime` to analysis frame indices.
 * Uses `timeupdate` / `seeked` / `loadedmetadata` always; adds a rAF loop while
 * playing only when the user has not requested reduced motion (smoother playhead).
 */
export function useVideoFrameSync({
  videoRef,
  frameCount,
  fps,
  onFrame,
}: UseVideoFrameSyncOptions): void {
  const lastEmitted = useRef<number>(-1);
  const rafId = useRef<number | null>(null);
  const prefersReducedMotion = useRef(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    prefersReducedMotion.current = mq.matches;
    const onChange = () => {
      prefersReducedMotion.current = mq.matches;
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const emitFrame = useCallback(() => {
    const el = videoRef.current;
    if (!el || frameCount <= 0) return;
    const f = timeToFrameIndex(el.currentTime, el, frameCount, fps);
    if (f !== lastEmitted.current) {
      lastEmitted.current = f;
      onFrame(f);
    }
  }, [videoRef, frameCount, fps, onFrame]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || frameCount <= 0) return;

    const onTimeUpdate = () => emitFrame();
    const onSeeked = () => emitFrame();
    const onLoadedMetadata = () => {
      lastEmitted.current = -1;
      emitFrame();
    };

    const cancelRaf = () => {
      if (rafId.current != null) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    };

    const rafLoop = () => {
      emitFrame();
      const v = videoRef.current;
      if (v && !v.paused && !v.ended && !prefersReducedMotion.current) {
        rafId.current = requestAnimationFrame(rafLoop);
      } else {
        rafId.current = null;
      }
    };

    const onPlay = () => {
      cancelRaf();
      if (!prefersReducedMotion.current) {
        rafId.current = requestAnimationFrame(rafLoop);
      }
    };

    const onPause = () => {
      cancelRaf();
      emitFrame();
    };

    el.addEventListener("timeupdate", onTimeUpdate);
    el.addEventListener("seeked", onSeeked);
    el.addEventListener("loadedmetadata", onLoadedMetadata);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);

    emitFrame();

    return () => {
      el.removeEventListener("timeupdate", onTimeUpdate);
      el.removeEventListener("seeked", onSeeked);
      el.removeEventListener("loadedmetadata", onLoadedMetadata);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      cancelRaf();
    };
  }, [videoRef, frameCount, fps, emitFrame]);
}
