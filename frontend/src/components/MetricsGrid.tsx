import type { StrideMetrics } from "../api/types";
import { useI18n } from "../i18n";
import { formatNumber } from "../utils/format";

type Props = {
  stride: StrideMetrics;
};

function optNum(n: number | undefined, digits: number): string {
  return n === undefined ? "—" : formatNumber(n, digits);
}

export function MetricsGrid({ stride }: Props) {
  const { t } = useI18n();

  return (
    <section className="card">
      <h2 className="card__title">{t("metrics.title")}</h2>
      <p className="card__note">{t("metrics.note")}</p>
      <dl className="metrics-grid">
        <div className="metrics-grid__item">
          <dt>{t("metrics.cadenceMerged")}</dt>
          <dd>
            {optNum(stride.cadence_steps_per_min_merged, 1)} {t("metrics.stepsPerMin")}
          </dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.cadenceLeft")}</dt>
          <dd>
            {optNum(stride.cadence_steps_per_min, 1)} {t("metrics.stepsPerMin")}
          </dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.strideTime")}</dt>
          <dd>{stride.stride_time_sec != null ? `${formatNumber(stride.stride_time_sec, 2)} s` : "—"}</dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.strideLength")}</dt>
          <dd>{optNum(stride.stride_length_px, 4)}</dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.strideHipRatio")}</dt>
          <dd>{optNum(stride.stride_length_over_hip_width, 2)}</dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.footStrikes")}</dt>
          <dd>{stride.num_foot_strikes_merged ?? "—"}</dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.leftContacts")}</dt>
          <dd>{stride.num_same_foot_contacts_left ?? "—"}</dd>
        </div>
        <div className="metrics-grid__item">
          <dt>{t("metrics.stridesDetected")}</dt>
          <dd>{stride.num_strides_detected ?? "—"}</dd>
        </div>
      </dl>
    </section>
  );
}
