/**
 * Segment-weighted whole-body COM in normalized image coordinates.
 * Mirrors `utils/com_segmentation.py` for client-side use with `/poses/{id}` JSON.
 */

import type { ComXY, PosesJson } from "../api/types";

const SEGMENT_MASS = {
  head: 8.1,
  trunk: 46.7,
  l_upper_arm: 2.8,
  r_upper_arm: 2.8,
  l_forearm: 1.6,
  r_forearm: 1.6,
  l_hand: 0.6,
  r_hand: 0.6,
  l_thigh: 10.0,
  r_thigh: 10.0,
  l_shank: 4.65,
  r_shank: 4.65,
  l_foot: 1.43,
  r_foot: 1.43,
} as const;

const FRAC_TRUNK = 0.43;
const FRAC_UPPER_ARM = 0.436;
const FRAC_FOREARM = 0.43;
const FRAC_HAND = 0.5;
const FRAC_THIGH = 0.433;
const FRAC_SHANK = 0.433;
const FRAC_FOOT = 0.5;
const VIS_FLOOR = 0.18;

function vis(lm: number[][], i: number): number {
  const row = lm[i];
  if (row.length > 3) return Math.max(0, Math.min(1, row[3]!));
  return 1;
}

function segWeight(lm: number[][], idx: number[]): number {
  let v = 1;
  for (const i of idx) {
    v = Math.min(v, vis(lm, i));
  }
  return Math.max(0, v);
}

function comOnSegment(lm: number[][], a: number, b: number, frac: number): [number, number] {
  const pax = lm[a]![0]!;
  const pay = lm[a]![1]!;
  const pbx = lm[b]![0]!;
  const pby = lm[b]![1]!;
  return [pax + frac * (pbx - pax), pay + frac * (pby - pay)];
}

/**
 * Approximate whole-body COM in normalized [0,1] image coordinates.
 */
export function computeSegmentWeightedComXY(landmarks: number[][]): ComXY | null {
  if (landmarks.length < 33) return null;

  const lm = landmarks;

  const earMidX = (lm[7]![0]! + lm[8]![0]!) / 2;
  const earMidY = (lm[7]![1]! + lm[8]![1]!) / 2;
  const headComX = (lm[0]![0]! + earMidX) / 2;
  const headComY = (lm[0]![1]! + earMidY) / 2;
  const wHead = segWeight(lm, [0, 7, 8]) * SEGMENT_MASS.head;

  const hipMidX = (lm[23]![0]! + lm[24]![0]!) / 2;
  const hipMidY = (lm[23]![1]! + lm[24]![1]!) / 2;
  const shoulderMidX = (lm[11]![0]! + lm[12]![0]!) / 2;
  const shoulderMidY = (lm[11]![1]! + lm[12]![1]!) / 2;
  const trunkVecX = shoulderMidX - hipMidX;
  const trunkVecY = shoulderMidY - hipMidY;
  const trunkComX = hipMidX + FRAC_TRUNK * trunkVecX;
  const trunkComY = hipMidY + FRAC_TRUNK * trunkVecY;
  const wTrunk = segWeight(lm, [11, 12, 23, 24]) * SEGMENT_MASS.trunk;

  const segments: [number, [number, number]][] = [
    [wHead, [headComX, headComY]],
    [wTrunk, [trunkComX, trunkComY]],
    [segWeight(lm, [11, 13]) * SEGMENT_MASS.l_upper_arm, comOnSegment(lm, 11, 13, FRAC_UPPER_ARM)],
    [segWeight(lm, [12, 14]) * SEGMENT_MASS.r_upper_arm, comOnSegment(lm, 12, 14, FRAC_UPPER_ARM)],
    [segWeight(lm, [13, 15]) * SEGMENT_MASS.l_forearm, comOnSegment(lm, 13, 15, FRAC_FOREARM)],
    [segWeight(lm, [14, 16]) * SEGMENT_MASS.r_forearm, comOnSegment(lm, 14, 16, FRAC_FOREARM)],
    [segWeight(lm, [15, 19]) * SEGMENT_MASS.l_hand, comOnSegment(lm, 15, 19, FRAC_HAND)],
    [segWeight(lm, [16, 20]) * SEGMENT_MASS.r_hand, comOnSegment(lm, 16, 20, FRAC_HAND)],
    [segWeight(lm, [23, 25]) * SEGMENT_MASS.l_thigh, comOnSegment(lm, 23, 25, FRAC_THIGH)],
    [segWeight(lm, [24, 26]) * SEGMENT_MASS.r_thigh, comOnSegment(lm, 24, 26, FRAC_THIGH)],
    [segWeight(lm, [25, 27]) * SEGMENT_MASS.l_shank, comOnSegment(lm, 25, 27, FRAC_SHANK)],
    [segWeight(lm, [26, 28]) * SEGMENT_MASS.r_shank, comOnSegment(lm, 26, 28, FRAC_SHANK)],
    [segWeight(lm, [27, 31]) * SEGMENT_MASS.l_foot, comOnSegment(lm, 27, 31, FRAC_FOOT)],
    [segWeight(lm, [28, 32]) * SEGMENT_MASS.r_foot, comOnSegment(lm, 28, 32, FRAC_FOOT)],
  ];

  const weighted: [number, [number, number]][] = [];
  for (const [mass, pt] of segments) {
    if (mass < VIS_FLOOR) continue;
    weighted.push([mass, pt]);
  }
  if (weighted.length === 0) return null;

  let totalW = 0;
  let sx = 0;
  let sy = 0;
  for (const [m, [px, py]] of weighted) {
    totalW += m;
    sx += m * px;
    sy += m * py;
  }
  if (totalW < 1e-8) return null;

  const x = sx / totalW;
  const y = sy / totalW;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

/** Build a per-frame COM series from raw pose JSON (uncertainties differ from API-smoothed series). */
export function computeComSeriesFromPosesJson(data: PosesJson): (ComXY | null)[] {
  return data.poses.map((p) => {
    if (!p.landmarks || p.landmarks.length < 33) return null;
    return computeSegmentWeightedComXY(p.landmarks);
  });
}
