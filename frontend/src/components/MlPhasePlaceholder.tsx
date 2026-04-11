import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GaitPhaseLabel, MlGaitBlock } from "../api/types";
import { useI18n } from "../i18n";
import { CHART_MAX_POINTS, downsampleIndices } from "../utils/downsample";

type Props = {
  frameCount: number;
  ml?: MlGaitBlock;
  currentFrameIndex?: number | null;
  onSeekFrame?: (frame: number) => void;
};

const PHASE_COLORS: Record<GaitPhaseLabel, string> = {
  stance: "var(--phase-stance)",
  swing: "var(--phase-swing)",
  push: "var(--phase-push)",
};

const PHASE_I18N_KEY: Record<GaitPhaseLabel, "ml.phaseStance" | "ml.phaseSwing" | "ml.phasePush"> = {
  stance: "ml.phaseStance",
  swing: "ml.phaseSwing",
  push: "ml.phasePush",
};

type ChartClickState = {
  activeTooltipIndex?: number;
  xValue?: number;
};

function clampFrame(n: number, maxIdx: number): number {
  return Math.max(0, Math.min(maxIdx, Math.round(n)));
}

function handleChartSeek(
  state: ChartClickState | undefined,
  data: { frame: number }[],
  frameCount: number,
  onSeekFrame?: (frame: number) => void,
): void {
  if (!onSeekFrame || frameCount <= 0) return;
  const maxIdx = frameCount - 1;
  const idx = state?.activeTooltipIndex;
  if (typeof idx === "number" && idx >= 0 && data[idx]) {
    onSeekFrame(data[idx].frame);
    return;
  }
  if (typeof state?.xValue === "number" && Number.isFinite(state.xValue)) {
    onSeekFrame(clampFrame(state.xValue, maxIdx));
  }
}

function buildMockPhases(n: number): { phases: GaitPhaseLabel[]; confidence: number[] } {
  const cycle: GaitPhaseLabel[] = ["stance", "swing", "push", "swing"];
  const phases = Array.from({ length: n }, (_, i) => cycle[i % cycle.length]!);
  const confidence = phases.map((_, i) =>
    Math.min(0.98, 0.52 + 0.38 * (0.5 + 0.5 * Math.sin(i * 0.15))),
  );
  return { phases, confidence };
}

function playheadPercent(frame: number, totalFrames: number): number {
  if (totalFrames <= 1) return 0;
  return (frame / (totalFrames - 1)) * 100;
}

export function MlPhasePlaceholder({
  frameCount,
  ml,
  currentFrameIndex = null,
  onSeekFrame,
}: Props) {
  const { t } = useI18n();
  const devMock =
    import.meta.env.DEV &&
    import.meta.env.VITE_DEV_ML_MOCK === "true";

  const { phases, confidence, modelVersion, isGeometry } = useMemo(() => {
    if (ml?.phases_per_frame && ml.phases_per_frame.length > 0) {
      const mv = ml.model_version ?? "";
      return {
        phases: ml.phases_per_frame,
        confidence: ml.confidence ?? null,
        modelVersion: mv,
        isGeometry: mv.includes("geometry"),
      };
    }
    if (devMock && frameCount > 0) {
      const n = Math.max(0, Math.min(frameCount, 400));
      const mock = buildMockPhases(n);
      return {
        phases: mock.phases,
        confidence: mock.confidence,
        modelVersion: "dev-mock",
        isGeometry: false,
      };
    }
    return { phases: null, confidence: null, modelVersion: "", isGeometry: false };
  }, [ml, devMock, frameCount]);

  const showPlayhead =
    currentFrameIndex != null &&
    frameCount > 0 &&
    currentFrameIndex >= 0 &&
    currentFrameIndex < frameCount;

  const playheadLeft = showPlayhead ? playheadPercent(currentFrameIndex, frameCount) : 0;

  const currentPhase: GaitPhaseLabel | null =
    phases && showPlayhead
      ? phases[Math.max(0, Math.min(phases.length - 1, currentFrameIndex))] ?? null
      : null;

  const confChartData = useMemo(() => {
    if (!confidence || confidence.length === 0) return [];
    const idx = downsampleIndices(confidence.length, CHART_MAX_POINTS);
    return idx.map((i) => ({
      frame: i,
      confidence: confidence[i] ?? null,
    }));
  }, [confidence]);

  const xAxisLabel = useMemo(
    () => ({
      value: t("charts.frameAxis"),
      position: "insideBottom" as const,
      offset: -4,
      fill: "var(--text-muted)",
    }),
    [t],
  );

  if (!phases) {
    return (
      <section className="card card--muted">
        <h2 className="card__title">{t("ml.title")}</h2>
        <p className="card__note">{t("ml.placeholderNote")}</p>
        <div className="phase-placeholder-box phase-placeholder-box--track">
          <span className="phase-placeholder-box__text">{t("ml.waiting")}</span>
          {showPlayhead && (
            <div
              className="phase-playhead"
              style={{ left: `${playheadLeft}%` }}
              aria-hidden
            />
          )}
        </div>
      </section>
    );
  }

  const maxSeg = 120;
  const step = Math.max(1, Math.ceil(phases.length / maxSeg));
  const segments = phases.filter((_, i) => i % step === 0);

  const noteText = (() => {
    if (isGeometry) return t("ml.geometryNote");
    if (devMock && modelVersion === "dev-mock") return t("ml.devMock");
    if (ml?.model_version) return `${t("ml.modelPrefix")} ${ml.model_version}`;
    return t("ml.perFrame");
  })();

  return (
    <section className="card">
      <div className="ml-phase-header">
        <h2 className="card__title">{t("ml.title")}</h2>
        {currentPhase && (
          <span
            className="ml-phase-pill"
            style={{ borderColor: PHASE_COLORS[currentPhase] }}
          >
            <span
              className="ml-phase-pill__dot"
              style={{ background: PHASE_COLORS[currentPhase] }}
            />
            {t(PHASE_I18N_KEY[currentPhase])}
          </span>
        )}
      </div>
      <p className="card__note">{noteText}</p>

      <div className="phase-timeline-wrap">
        <div className="phase-timeline" role="img" aria-label={t("ml.timelineAria")}>
          {segments.map((p, i) => (
            <div
              key={i}
              className="phase-timeline__seg"
              style={{ background: PHASE_COLORS[p] ?? "var(--border)" }}
              title={t(PHASE_I18N_KEY[p])}
            />
          ))}
        </div>
        {showPlayhead && (
          <div
            className="phase-playhead"
            style={{ left: `${playheadLeft}%` }}
            aria-hidden
          />
        )}
      </div>

      {confidence && confChartData.length > 0 && (
        <div className={onSeekFrame ? "chart-wrap chart-wrap--seekable ml-confidence-wrap" : "chart-wrap ml-confidence-wrap"}>
          <p className="ml-confidence-caption">{t("ml.confidenceCaption")}</p>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart
              data={confChartData}
              margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
              onClick={(s) =>
                handleChartSeek(s as ChartClickState, confChartData, frameCount, onSeekFrame)
              }
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                dataKey="frame"
                type="number"
                domain={["dataMin", "dataMax"]}
                allowDataOverflow
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                label={xAxisLabel}
              />
              <YAxis
                domain={[0, 1]}
                width={40}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(1)}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--surface-elevated)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                }}
                formatter={(v: number | string) => [
                  typeof v === "number" ? v.toFixed(2) : v,
                  t("ml.confidenceLegend"),
                ]}
              />
              <Area
                name={t("ml.confidenceLegend")}
                type="monotone"
                dataKey="confidence"
                stroke="var(--accent)"
                fill="var(--accent-muted)"
                strokeWidth={1.5}
                connectNulls={false}
              />
              {showPlayhead && (
                <ReferenceLine
                  x={currentFrameIndex ?? 0}
                  stroke="var(--text-muted)"
                  strokeWidth={1.5}
                  strokeOpacity={0.9}
                  ifOverflow="extendDomain"
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <ul className="phase-legend">
        {(Object.keys(PHASE_COLORS) as GaitPhaseLabel[]).map((k) => (
          <li key={k}>
            <span className="phase-legend__swatch" style={{ background: PHASE_COLORS[k] }} />
            {t(PHASE_I18N_KEY[k])}
          </li>
        ))}
      </ul>
    </section>
  );
}
