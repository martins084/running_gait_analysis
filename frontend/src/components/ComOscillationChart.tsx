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

/**
 * Recharts can emit duplicate / overlapping Y tick labels when the domain span is
 * tiny (COM y barely moves while COM x sweeps a wider range). Force a minimum
 * vertical span and round tick text so the axis stays readable.
 */
function comPlotYDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  const minSpan = 0.1;
  if (span < 1e-9) {
    const mid = (lo + hi) / 2;
    return [mid - minSpan / 2, mid + minSpan / 2];
  }
  if (span < minSpan) {
    const pad = (minSpan - span) / 2;
    return [lo - pad, hi + pad];
  }
  const pad = Math.max(0.02, span * 0.08);
  return [lo - pad, hi + pad];
}

function formatComYTick(v: number, domain: [number, number]): string {
  if (!Number.isFinite(v)) return "";
  const span = domain[1] - domain[0];
  const decimals = span > 0.25 ? 2 : span > 0.06 ? 3 : 4;
  return v.toFixed(decimals);
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

  const yDomain = useMemo(() => {
    const vals: number[] = [];
    if (com_xy_per_frame) {
      for (const p of com_xy_per_frame) {
        if (!p) continue;
        if (Number.isFinite(p.x)) vals.push(p.x);
        if (Number.isFinite(p.y)) vals.push(p.y);
      }
    }
    const raw = comPlotYDomain(vals);
    const clamped: [number, number] = [
      Math.max(-0.08, raw[0]),
      Math.min(1.08, raw[1]),
    ];
    if (clamped[1] <= clamped[0]) {
      return [0, 1] as [number, number];
    }
    return clamped;
  }, [com_xy_per_frame]);

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
              domain={yDomain}
              width={52}
              allowDecimals
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              tickFormatter={(v: number | string) =>
                formatComYTick(typeof v === "number" ? v : Number(v), yDomain)
              }
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
                borderRadius: 4,
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
