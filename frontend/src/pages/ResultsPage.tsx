import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getResults } from "../api/client";
import type { AnalyzeResponse } from "../api/types";
import { ComOscillationChart } from "../components/ComOscillationChart";
import { JointAngleChart } from "../components/JointAngleChart";
import { MetricsGrid } from "../components/MetricsGrid";
import { MlPhasePlaceholder } from "../components/MlPhasePlaceholder";
import { SymmetryChart } from "../components/SymmetryChart";
import { VideoPanel } from "../components/VideoPanel";
import { useI18n } from "../i18n";
import { seekVideoToFrame, useVideoFrameSync } from "../hooks/useVideoFrameSync";
import { formatNumber } from "../utils/format";

function ResultsSkeleton() {
  const { t } = useI18n();
  return (
    <div className="page page--results results-skeleton" aria-busy="true" aria-label={t("results.loadingAria")}>
      <div className="skeleton-block" style={{ height: 72 }} />
      <div className="skeleton-block" />
      <div className="skeleton-block skeleton-block--tall" />
    </div>
  );
}

function ResultsJumpNav() {
  const { t } = useI18n();
  return (
    <nav className="results-jump-nav" aria-label={t("results.jumpNavAria")}>
      <span className="results-jump-nav__label">{t("results.jumpLabel")}</span>
      <ul className="results-jump-nav__list">
        <li>
          <a href="#section-stride">{t("results.stride")}</a>
        </li>
        <li>
          <a href="#section-vertical">{t("results.vertical")}</a>
        </li>
        <li>
          <a href="#section-com">{t("results.com")}</a>
        </li>
        <li>
          <a href="#section-angles">{t("results.angles")}</a>
        </li>
        <li>
          <a href="#section-symmetry">{t("results.symmetry")}</a>
        </li>
        <li>
          <a href="#section-ml">{t("results.mlPhases")}</a>
        </li>
      </ul>
    </nav>
  );
}

function statusLabel(status: string, t: (key: string) => string): string {
  if (status === "completed") return t("status.completed");
  if (status === "pending") return t("status.pending");
  return status;
}

export function ResultsPage() {
  const { t } = useI18n();
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  useEffect(() => {
    if (!id) {
      setError(t("results.missingId"));
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    getResults(id)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : t("results.loadFailed"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // Only analysis id should trigger reload — not locale (avoids duplicate GET on LV/EN switch).
    // eslint-disable-next-line react-hooks/exhaustive-deps -- t() for errors is read when effect runs
  }, [id]);

  const copyId = useCallback(() => {
    if (!data?.id) return;
    const tid = data.id;
    void navigator.clipboard.writeText(tid).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 2000);
      },
      () => {
        /* fallback: ignore */
      },
    );
  }, [data?.id]);

  const fps = data?.features?.fps ?? null;
  /** Prefer full video length (COM / pose JSON); fall back to angle series for legacy rows. */
  const frameCount =
    data?.features?.video_frame_count ??
    data?.features?.com_xy_per_frame?.length ??
    data?.features?.joint_angles?.length ??
    0;

  useVideoFrameSync({
    videoRef,
    frameCount,
    fps,
    onFrame: setCurrentFrameIndex,
  });

  const seekToFrame = useCallback(
    (frame: number) => {
      const v = videoRef.current;
      if (!v) return;
      seekVideoToFrame(v, frame, frameCount, fps);
    },
    [frameCount, fps],
  );

  if (loading) {
    return <ResultsSkeleton />;
  }

  if (error || !data) {
    return (
      <div className="page">
        <div className="panel panel--error" role="alert">
          {error ?? t("results.noData")}
        </div>
        <Link className="link-back" to="/">
          {t("results.backUpload")}
        </Link>
      </div>
    );
  }

  const { features } = data;
  const vOsc = features.vertical_oscillation_px;

  return (
    <div className="page page--results">
      <div className="results-workbench">
        <aside className="results-rail" aria-label={t("results.sessionAria")}>
          <Link className="link-back" to="/">
            {t("results.newAnalysis")}
          </Link>
          <h1 className="results-header__title">{t("results.title")}</h1>
          <div className="results-rail__meta">
            <p className="results-header__id mono">{data.id}</p>
            <button type="button" className="copy-id-btn" onClick={copyId}>
              {copied ? t("results.copied") : t("results.copyId")}
            </button>
          </div>
          <span className={`badge badge--${data.status === "completed" ? "ok" : "pending"}`}>
            {statusLabel(data.status, t)}
          </span>
        </aside>

        <div id="section-video" className="results-video-col">
          <VideoPanel
            analysisId={data.id}
            annotatedVideoPath={data.annotated_video}
            videoRef={videoRef}
            currentFrameIndex={currentFrameIndex}
            com_xy_per_frame={features.com_xy_per_frame ?? null}
          />
        </div>

        <div className="results-analysis-col">
          <div className="results-analysis-scroll" role="region" aria-label={t("results.regionAria")}>
            <ResultsJumpNav />

            <div id="section-stride" className="results-anchor">
              <MetricsGrid stride={features.stride_metrics} />
            </div>

            <section id="section-vertical" className="card results-anchor">
              <h2 className="card__title">{t("results.verticalTitle")}</h2>
              <p className="card__note">{t("results.verticalNote")}</p>
              <p className="metric-single">
                {vOsc === null || vOsc === undefined ? "—" : formatNumber(vOsc, 4)}
              </p>
            </section>

            <ComOscillationChart
              com_xy_per_frame={features.com_xy_per_frame}
              currentFrameIndex={currentFrameIndex}
              onSeekFrame={seekToFrame}
            />

            <div id="section-angles" className="results-anchor">
              <JointAngleChart
                jointAngles={features.joint_angles}
                currentFrameIndex={currentFrameIndex}
                onSeekFrame={seekToFrame}
              />
            </div>
            <div id="section-symmetry" className="results-anchor">
              <SymmetryChart
                symmetry={features.symmetry}
                currentFrameIndex={currentFrameIndex}
                onSeekFrame={seekToFrame}
              />
            </div>
            <div id="section-ml" className="results-anchor">
              <MlPhasePlaceholder
                frameCount={features.joint_angles.length}
                ml={features.ml}
                currentFrameIndex={currentFrameIndex}
              />
            </div>

            <footer className="results-footer results-footer--compact">
              <button
                type="button"
                className="link-button link-button--ghost"
                disabled
                title={t("results.compareTitle")}
              >
                {t("results.compareSoon")}
              </button>
            </footer>
          </div>
        </div>
      </div>
    </div>
  );
}
