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
import { useI18n } from "../i18n";
import { CHART_MAX_POINTS, downsampleIndices } from "../utils/downsample";

type Row = { frame: number; symmetry: number };

type ChartClickState = {
  activeTooltipIndex?: number;
  xValue?: number;
};

type Props = {
  symmetry: number[];
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

export function SymmetryChart({ symmetry, currentFrameIndex = 0, onSeekFrame }: Props) {
  const { t } = useI18n();
  const frameCount = symmetry.length;
  const idx = downsampleIndices(symmetry.length, CHART_MAX_POINTS);
  const data = idx.map((i) => ({ frame: i, symmetry: symmetry[i] }));
  const showPlayhead = frameCount > 0;

  return (
    <section className="card">
      <h2 className="card__title">{t("charts.symTitle")}</h2>
      <p className="card__note">{t("charts.symNote")}</p>
      <div className={onSeekFrame ? "chart-wrap chart-wrap--seekable" : "chart-wrap"}>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart
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
            />
            <YAxis domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                background: "var(--surface-elevated)",
                border: "1px solid var(--border)",
                borderRadius: 8,
              }}
            />
            <Area
              type="monotone"
              dataKey="symmetry"
              stroke="var(--accent)"
              fill="var(--accent-muted)"
              strokeWidth={1.5}
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
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
