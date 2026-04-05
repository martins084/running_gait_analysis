import type { GaitPhaseLabel, MlGaitBlock } from "../api/types";
import { useI18n } from "../i18n";

type Props = {
  frameCount: number;
  ml?: MlGaitBlock;
  currentFrameIndex?: number | null;
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

function buildMockPhases(n: number): GaitPhaseLabel[] {
  const cycle: GaitPhaseLabel[] = ["stance", "swing", "push", "swing"];
  return Array.from({ length: n }, (_, i) => cycle[i % cycle.length]);
}

function playheadPercent(frame: number, totalFrames: number): number {
  if (totalFrames <= 1) return 0;
  return (frame / (totalFrames - 1)) * 100;
}

export function MlPhasePlaceholder({ frameCount, ml, currentFrameIndex = null }: Props) {
  const { t } = useI18n();
  const devMock =
    import.meta.env.DEV &&
    import.meta.env.VITE_DEV_ML_MOCK === "true";

  const phases =
    ml?.phases_per_frame && ml.phases_per_frame.length > 0
      ? ml.phases_per_frame
      : devMock
        ? buildMockPhases(Math.max(0, Math.min(frameCount, 400)))
        : null;

  const showPlayhead =
    currentFrameIndex != null &&
    frameCount > 0 &&
    currentFrameIndex >= 0 &&
    currentFrameIndex < frameCount;

  const playheadLeft = showPlayhead ? playheadPercent(currentFrameIndex, frameCount) : 0;

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

  return (
    <section className="card">
      <h2 className="card__title">{t("ml.title")}</h2>
      <p className="card__note">
        {devMock && !ml?.phases_per_frame?.length
          ? t("ml.devMock")
          : ml?.model_version
            ? `${t("ml.modelPrefix")} ${ml.model_version}`
            : t("ml.perFrame")}
      </p>
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
