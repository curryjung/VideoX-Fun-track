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
