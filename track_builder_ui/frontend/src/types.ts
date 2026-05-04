export type Point = {
  x: number;
  y: number;
};

export type GridConfig = {
  type: "square";
  spacing: number;
  offsetX: number;
  offsetY: number;
  visible: boolean;
};

export type TrackPath = {
  id: string;
  name: string;
  points: Point[];
  keyframes?: Point[][];
  trackMode?: "moving" | "static";
  pointMode?: "polyline" | "points";
  closed: boolean;
  color: string;
};

export type TrackDocument = {
  version: "0.1";
  image: {
    src: string;
    width: number;
    height: number;
  };
  grid: GridConfig;
  paths: TrackPath[];
  meta: {
    createdAt: string;
    updatedAt: string;
  };
};

export type EditorTool = "pointCloudEdit";

export type GenerationMode = "motion_only" | "text_only" | "joint_tm";

export type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "canceled"
  | "interrupted";

export type JobRecord = {
  job_id: string;
  status: JobStatus;
  mode: GenerationMode;
  prompt: string;
  seed: number;
  text_guidance_weight: number;
  motion_guidance_weight: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  input: {
    image: string;
    tracks: string;
    preview?: string | null;
  };
  outputs: {
    video?: string | null;
    overlay_video?: string | null;
  };
  log_path: string;
  error_message?: string | null;
  return_code?: number | null;
  source_job_id?: string | null;
};
