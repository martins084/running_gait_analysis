import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ComXY } from "../api/types";
import { useI18n } from "../i18n";
import { CHART_MAX_POINTS, downsampleIndices } from "../utils/downsample";

type Row = {
  frame: number;
  cx: number | null;
  cy: number | null;
};

type ChartClickState = {
  activeTooltipIndex?: number;
  xValue?: number;
};

type Props = {
  com_xy_per_frame: (ComXY | null)[] | null | undefined;
  currentFrameIndex?: number;
  onSeekFrame?: (frame: number) => void;
};

function clampFrame(n: number, maxIdx: number): number {
  return Math.max(0, Math.min(maxIdx, Math.round(n)));
}

function handleChartSeek(
  state: ChartClickState | undefined,
  data: Row[],
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

export function ComOscillationChart({
  com_xy_per_frame,
  currentFrameIndex = 0,
  onSeekFrame,
}: Props) {
  const { t } = useI18n();

  const frameCount = com_xy_per_frame?.length ?? 0;
  const idx = downsampleIndices(frameCount, CHART_MAX_POINTS);
  const data: Row[] = idx.map((i) => {
    const p = com_xy_per_frame?.[i];
    if (p == null) {
      return { frame: i, cx: null, cy: null };
    }
    return { frame: i, cx: p.x, cy: p.y };
  });

  const showPlayhead = frameCount > 0;

  const xAxisLabel = useMemo(
    () => ({
      value: t("charts.frameAxis"),
      position: "insideBottom" as const,
      offset: -4,
      fill: "var(--text-muted)",
    }),
    [t],
  );

  if (frameCount === 0) {
    return (
      <section className="card results-anchor" id="section-com">
        <h2 className="card__title">{t("charts.comTitle")}</h2>
        <p className="card__note">{t("charts.comEmpty")}</p>
      </section>
    );
  }

  return (
    <section className="card results-anchor" id="section-com">
      <h2 className="card__title">{t("charts.comTitle")}</h2>
      <p className="card__note">{t("charts.comNote")}</p>
      <div className={onSeekFrame ? "chart-wrap chart-wrap--seekable" : "chart-wrap"}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={data}
            margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
            onClick={(s) => handleChartSeek(s as ChartClickState, data, frameCount, onSeekFrame)}
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
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              label={{
                value: t("charts.comYAxisLabel"),
                angle: -90,
                position: "insideLeft",
                fill: "var(--text-muted)",
                fontSize: 11,
              }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface-elevated)",
                border: "1px solid var(--border)",
                borderRadius: 8,
              }}
              formatter={(value: number | string) =>
                typeof value === "number" ? value.toFixed(4) : value
              }
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="cy"
              name={t("charts.comY")}
              stroke="var(--series-1)"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="cx"
              name={t("charts.comX")}
              stroke="var(--series-2)"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            {showPlayhead && (
              <ReferenceLine
                x={currentFrameIndex}
                stroke="var(--text-muted)"
                strokeWidth={1.5}
                strokeOpacity={0.9}
                ifOverflow="extendDomain"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
