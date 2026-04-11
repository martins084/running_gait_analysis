/**
 * Types aligned with docs/API_Response_Schema.md and optional future ML fields.
 */

export type GaitPhaseLabel = "stance" | "swing" | "push";

/** Populated by geometry-cadence estimate and, later, optional CNN-LSTM weights. */
export interface MlGaitBlock {
  model_version?: string;
  phases_per_frame?: GaitPhaseLabel[];
  /** 0–1 heuristic confidence per frame (geometry pipeline); optional on legacy rows. */
  confidence?: (number | null)[];
}

/** Stride block may be partial when the detector finds too few foot contacts. */
export interface StrideMetrics {
  stride_length_px?: number;
  stride_time_sec?: number;
  cadence_steps_per_min?: number;
  cadence_steps_per_min_merged?: number;
  stride_length_over_hip_width?: number;
  num_same_foot_contacts_left?: number;
  num_foot_strikes_merged?: number;
  num_strides_detected?: number;
}

export interface JointAnglesFrame {
  left_hip: number;
  right_hip: number;
  left_knee: number;
  right_knee: number;
}

/** Per video frame; null when pose missing for that frame (keeps index aligned with video). */
export type JointAnglesSeries = (JointAnglesFrame | null)[];
export type SymmetrySeries = (number | null)[];

/** Normalized image-plane COM (0–1); 2D proxy, not metres. */
export interface ComXY {
  x: number;
  y: number;
}

export interface FeaturePayload {
  stride_metrics: StrideMetrics;
  joint_angles: JointAnglesSeries;
  symmetry: SymmetrySeries;
  vertical_oscillation_px: number | null;
  /** Frames per second from pose pipeline; omitted on legacy stored results. */
  fps?: number;
  /** Redundant with joint_angles.length; useful for validation. */
  frame_count?: number;
  /**
   * Segment-weighted COM per video frame (aligned with pose JSON length).
   * Omitted on analyses computed before this field existed.
   */
  com_xy_per_frame?: (ComXY | null)[] | null;
  /** Full video frame count (= pose JSON length); use for video ↔ metrics sync. */
  video_frame_count?: number;
  /** Optional ML block when backend exposes classifier output. */
  ml?: MlGaitBlock;
}

/** BlazePose export from `GET /poses/{id}` (large). */
export interface PosesJson {
  fps: number;
  frame_count?: number;
  poses: Array<{
    frame: number;
    timestamp: number;
    landmarks: number[][] | null;
  }>;
}

export interface AnalyzeResponse {
  id: string;
  status: string;
  features: FeaturePayload;
  /** `null` when minimal profile or file removed — no GET /download/{id}. */
  annotated_video: string | null;
  /** `null` when poses JSON not retained — no GET /poses/{id}. */
  poses: string | null;
  storage_profile?: "minimal" | "standard" | "full";
}

export interface HealthResponse {
  status: string;
}

export interface AuthResponse {
  token: string;
  user_id: string;
  email: string;
}

export interface DsarResponse {
  request_id: string;
  status: string;
  download?: string;
  deleted_ids?: string[];
}
