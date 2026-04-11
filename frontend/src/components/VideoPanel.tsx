import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { fetchAuthenticatedBlob, getPosesJson, resolveApiUrl } from "../api/client";
import type { ComXY } from "../api/types";
import { useI18n } from "../i18n";
import { computeComSeriesFromPosesJson } from "../utils/comSegmentation";
import { getVideoContentRect, normalizedToContentPixel } from "../utils/videoContentRect";

const TRAIL_LEN = 20;

type Props = {
  analysisId: string;
  /** `null` / missing when minimal profile — server did not retain annotated MP4. */
  annotatedVideoPath: string | null | undefined;
  videoRef: RefObject<HTMLVideoElement>;
  /** Synced with charts / `useVideoFrameSync`. */
  currentFrameIndex: number;
  /** From API (smoothed pipeline); if missing, client may fetch `/poses` and recompute. */
  com_xy_per_frame?: (ComXY | null)[] | null;
};

export function VideoPanel({
  analysisId,
  annotatedVideoPath,
  videoRef,
  currentFrameIndex,
  com_xy_per_frame,
}: Props) {
  const { t } = useI18n();
  const downloadHref = resolveApiUrl(`/download/${analysisId}`);
  /** Blob URL — `<video src>` cannot send `Authorization`; fetch with token then play locally. */
  const [mediaObjectUrl, setMediaObjectUrl] = useState<string | null>(null);
  /**
   * `fetch_blob` → API blob received → `decode` (video element mounted, waiting for metadata) → `ready`.
   * Hiding the player until `loadedmetadata` avoids an empty/black frame that feels like “video didn’t load”.
   */
  const [videoPhase, setVideoPhase] = useState<
    "unavailable" | "fetch_blob" | "decode" | "ready" | "error"
  >(() => (annotatedVideoPath ? "fetch_blob" : "unavailable"));
  const mediaUrlRef = useRef<string | null>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** COM series from `/poses` when API did not include `com_xy_per_frame` (legacy analyses). */
  const [clientComSeries, setClientComSeries] = useState<(ComXY | null)[] | null>(null);
  const [posesError, setPosesError] = useState(false);
  const posesFetchKeyRef = useRef<string | null>(null);

  // Need at least one valid point; otherwise fall back to /poses (e.g. legacy rows or all-null series).
  const hasApiCom =
    Array.isArray(com_xy_per_frame) &&
    com_xy_per_frame.length > 0 &&
    com_xy_per_frame.some((p) => p != null);
  const comSeries = hasApiCom ? com_xy_per_frame! : clientComSeries;

  useEffect(() => {
    setClientComSeries(null);
    setPosesError(false);
    posesFetchKeyRef.current = null;
  }, [analysisId]);

  useEffect(() => {
    let cancelled = false;

    if (!annotatedVideoPath) {
      setVideoPhase("unavailable");
      setMediaObjectUrl(null);
      if (mediaUrlRef.current) {
        URL.revokeObjectURL(mediaUrlRef.current);
        mediaUrlRef.current = null;
      }
      return () => {
        cancelled = true;
      };
    }

    setVideoPhase("fetch_blob");
    setMediaObjectUrl(null);
    if (mediaUrlRef.current) {
      URL.revokeObjectURL(mediaUrlRef.current);
      mediaUrlRef.current = null;
    }

    void (async () => {
      try {
        const blob = await fetchAuthenticatedBlob(`/download/${encodeURIComponent(analysisId)}`);
        if (cancelled) return;
        const u = URL.createObjectURL(blob);
        mediaUrlRef.current = u;
        setMediaObjectUrl(u);
        setVideoPhase("decode");
      } catch {
        if (!cancelled) setVideoPhase("error");
      }
    })();

    return () => {
      cancelled = true;
      if (mediaUrlRef.current) {
        URL.revokeObjectURL(mediaUrlRef.current);
        mediaUrlRef.current = null;
      }
    };
  }, [analysisId, annotatedVideoPath]);

  useEffect(() => {
    if (hasApiCom) return;
    const key = analysisId;
    if (posesFetchKeyRef.current === key) return;
    posesFetchKeyRef.current = key;

    let cancelled = false;
    void getPosesJson(analysisId)
      .then((data) => {
        if (!cancelled) setClientComSeries(computeComSeriesFromPosesJson(data));
      })
      .catch(() => {
        if (!cancelled) setPosesError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId, hasApiCom]);

  const drawComOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const shell = shellRef.current;
    if (!video || !canvas || !shell || !comSeries || comSeries.length === 0) return;

    const idx = Math.max(0, Math.min(comSeries.length - 1, Math.round(currentFrameIndex)));
    const pt = comSeries[idx];
    if (!pt) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const rect = getVideoContentRect(video);
    if (!rect) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = shell.clientWidth;
    const ch = shell.clientHeight;
    if (cw <= 0 || ch <= 0) return;

    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const start = Math.max(0, idx - TRAIL_LEN);
    const trailPts: { x: number; y: number }[] = [];
    for (let i = start; i <= idx; i++) {
      const c = comSeries[i];
      if (!c) continue;
      trailPts.push(normalizedToContentPixel(c.x, c.y, rect));
    }

    if (trailPts.length >= 2) {
      ctx.strokeStyle = "rgba(255, 160, 40, 0.55)";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(trailPts[0]!.x, trailPts[0]!.y);
      for (let k = 1; k < trailPts.length; k++) {
        ctx.lineTo(trailPts[k]!.x, trailPts[k]!.y);
      }
      ctx.stroke();
    }

    const { x: cx, y: cy } = normalizedToContentPixel(pt.x, pt.y, rect);
    const r = Math.max(5, Math.min(cw, ch) / 90);
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(30, 30, 30, 0.85)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 200, 60, 0.95)";
    ctx.fill();

    const d = r + 4;
    ctx.strokeStyle = "rgba(30, 30, 30, 0.75)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - d, cy);
    ctx.lineTo(cx + d, cy);
    ctx.moveTo(cx, cy - d);
    ctx.lineTo(cx, cy + d);
    ctx.stroke();
  }, [videoRef, comSeries, currentFrameIndex]);

  useEffect(() => {
    drawComOverlay();
  }, [drawComOverlay]);

  useEffect(() => {
    const shell = shellRef.current;
    const video = videoRef.current;
    if (!shell) return;

    const ro = new ResizeObserver(() => drawComOverlay());
    ro.observe(shell);
    video?.addEventListener("loadedmetadata", drawComOverlay);

    return () => {
      ro.disconnect();
      video?.removeEventListener("loadedmetadata", drawComOverlay);
    };
  }, [drawComOverlay, videoRef]);

  const showCom =
    Boolean(annotatedVideoPath) &&
    comSeries &&
    comSeries.length > 0 &&
    videoPhase === "ready" &&
    Boolean(mediaObjectUrl);

  return (
    <section className="card">
      <h2 className="card__title">{t("video.title")}</h2>
      <p className="video-shell__caption">{t("video.caption")}</p>
      <p className="card__note">{t("video.note")}</p>
      {posesError && !hasApiCom && (
        <p className="card__note video-com__warn" role="status">
          {t("video.comFallbackWarn")}
        </p>
      )}
      <div ref={shellRef} className="video-shell video-shell--com">
        {!annotatedVideoPath ? (
          <div className="video-fallback">
            <p className="video-fallback__text">{t("video.notStored")}</p>
          </div>
        ) : videoPhase === "error" ? (
          <div className="video-fallback">
            <p className="video-fallback__text">{t("video.fallback")}</p>
            <a className="link-button" href={downloadHref}>
              {t("video.openDownload")}
            </a>
          </div>
        ) : !mediaObjectUrl ? (
          <div className="video-fallback video-fallback--loading">
            <p className="video-fallback__text mono">{t("video.loading")}</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              className="video-shell__el"
              controls
              playsInline
              preload="auto"
              src={mediaObjectUrl}
              onLoadedMetadata={() => setVideoPhase("ready")}
              onError={() => setVideoPhase("error")}
            />
            {videoPhase === "decode" && (
              <div className="video-fallback video-fallback--loading video-fallback--overlay" aria-busy>
                <p className="video-fallback__text mono">{t("video.loading")}</p>
              </div>
            )}
            {showCom && (
              <canvas
                ref={canvasRef}
                className="video-shell__com-canvas"
                aria-hidden
                role="presentation"
              />
            )}
          </>
        )}
      </div>
      {showCom && <p className="video-com__legend">{t("video.comLegend")}</p>}
      {annotatedVideoPath ? (
        <a className="link-button" href={mediaObjectUrl ?? downloadHref} download="annotated.mp4">
          {t("video.downloadMp4")}
        </a>
      ) : null}
    </section>
  );
}
