/**
 * TypeScript types matching the FastAPI backend models
 */

export type RunState =
  | 'created'
  | 'uploading'
  | 'queued'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface RunConfig {
  device: string;
  detection_backend: 'detectron2' | 'yolo';
  yolo_weights?: string;
  confidence_threshold: number;
  pose_confidence_threshold: number;
  max_inference_width: number;
  tracker_iou_threshold: number;
  tracker_max_lost: number;
  density_low_sigma: number;
  density_high_sigma: number;
  agitation_threshold_sigma: number;
  video_extract_fps?: number | null;
  save_annotated: boolean;
  save_heatmaps: boolean;
  generate_plots: boolean;
  save_database: boolean;
  video_file?: string | null;
}

export interface ProgressInfo {
  total_frames: number;
  processed_frames: number;
  current_frame: string | null;
  current_stage: string | null;
  message?: string | null;
  eta_seconds?: number | null;
  per_stage_timings: Record<string, number>;
}

export interface RunStatus {
  run_id: string;
  state: RunState;
  progress: ProgressInfo;
  config: RunConfig;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message?: string | null;
}

export interface RunSummary {
  run_id: string;
  state: RunState;
  total_frames: number;
  created_at: string;
  finished_at: string | null;
  has_events: boolean;
}

export interface RunListResponse {
  runs: RunSummary[];
  total: number;
}

export interface RunCreateResponse {
  run_id: string;
  status: RunState;
  message: string;
}

export interface FileInfo {
  name: string;
  path: string;
  type: string;
  size_bytes: number;
  modified_at: string;
}

export interface FileListResponse {
  files: FileInfo[];
  categories: Record<string, FileInfo[]>;
}

export interface HealthResponse {
  status: string;
  device_available: string;
  cuda_available: boolean;
  backends_available: string[];
  version: string;
}

export interface EventTimelineResponse {
  events: EscalationEvent[];
  total_events: number;
}

export interface EscalationEvent {
  frame_name: string;
  frame_index: number;
  agitation_score: number;
  density_ratio: number;
  classification: string;
  timestamp: string;
}

// Metrics and Summary types
export interface FrameMetrics {
  frame_name: string;
  frame_index: number;
  people_count: number;
  head_count?: number | null;
  persons_in_region?: number | null;
  persons_per_m2?: number | null;
  space_per_person?: number | null;
  los_class?: string | null;
  mean_speed?: number | null;
  speed_std?: number | null;
  speed_unit?: string | null;
  crowd_pressure?: number | null;
  velocity_variance?: number | null;
  directional_entropy?: number | null;
  accel_event_rate?: number | null;
  crowd_state?: string | null;
  inference_time_det: number;
  inference_time_pose: number;
  average_confidence: number;
  density_ratio: number;
  agitation_index: number;
  classification: string;
}

export interface SummaryStats {
  total_frames: number;
  // Ground metrics — present only for calibrated runs
  mean_persons_per_m2?: number;
  peak_persons_per_m2?: number;
  los_distribution?: Record<string, number>;
  worst_los?: string | null;
  mean_speed?: number | null;
  speed_unit?: string | null;
  crowd_state_distribution?: Record<string, number>;
  dominant_state?: string;
  mean_crowd_pressure?: number;
  peak_crowd_pressure?: number;
  state_classifier?: string | null;
  mean_density: number;
  peak_density_frame: string;
  peak_density_value: number;
  mean_agitation: number;
  highest_agitation_frame: string;
  highest_agitation_value: number;
  total_escalation_events: number;
  crowd_classification_distribution: Record<string, number>;
  average_crowd_count: number;
  std_crowd_count: number;
  density_moving_average: number[];
}
