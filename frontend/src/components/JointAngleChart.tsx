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
import type { JointAnglesSeries } from "../api/types";
import { useI18n } from "../i18n";
import { CHART_MAX_POINTS, downsampleIndices } from "../utils/downsample";

type Row = {
  frame: number;
  left_hip: number | null;
  right_hip: number | null;
  left_knee: number | null;
  right_knee: number | null;
};

type ChartClickState = {
  activeTooltipIndex?: number;
  xValue?: number;
};

type Props = {
  jointAngles: JointAnglesSeries;
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

export function JointAngleChart({ jointAngles, currentFrameIndex = 0, onSeekFrame }: Props) {
  const { t } = useI18n();
  const frameCount = jointAngles.length;
  const idx = downsampleIndices(jointAngles.length, CHART_MAX_POINTS);
  const data: Row[] = idx.map((i) => {
    const ja = jointAngles[i];
    if (!ja) {
      return { frame: i, left_hip: null, right_hip: null, left_knee: null, right_knee: null };
    }
    return {
      frame: i,
      left_hip: ja.left_hip,
      right_hip: ja.right_hip,
      left_knee: ja.left_knee,
      right_knee: ja.right_knee,
    };
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

  return (
    <section className="card">
      <h2 className="card__title">{t("charts.jointTitle")}</h2>
      <p className="card__note">{t("charts.jointNote")}</p>
      <div className={onSeekFrame ? "chart-wrap chart-wrap--seekable" : "chart-wrap"}>
        <ResponsiveContainer width="100%" height={320}>
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
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              domain={["auto", "auto"]}
              label={{ value: "°", angle: -90, position: "insideLeft", fill: "var(--text-muted)" }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface-elevated)",
                border: "1px solid var(--border)",
                borderRadius: 8,
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="left_hip"
              name={t("charts.leftHip")}
              stroke="var(--series-1)"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="right_hip"
              name={t("charts.rightHip")}
              stroke="var(--series-2)"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="left_knee"
              name={t("charts.leftKnee")}
              stroke="var(--series-3)"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="right_knee"
              name={t("charts.rightKnee")}
              stroke="var(--series-4)"
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
