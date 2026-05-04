import React, { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";
import ArchivePanel from "./components/ArchivePanel";
import GenerationControls from "./components/GenerationControls";
import QueuePanel from "./components/QueuePanel";
import TrackCanvas from "./components/TrackCanvas";
import TrackPreview from "./components/TrackPreview";
import type { GenerationMode, GridConfig, JobRecord, TrackDocument, TrackPath } from "./types";

const FRAME_COUNT = 81;
const DEFAULT_IMAGE_WIDTH = 832;
const DEFAULT_IMAGE_HEIGHT = 480;
const BACKEND_BASE_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined)?.trim() ?? "";
const DEFAULT_FLORENCE_TASK = "<MORE_DETAILED_CAPTION>";

type TrackPackageArtifacts = {
  trackNpzBlob: Blob;
  firstFrameBlob: Blob;
  previewBlob: Blob | null;
  pointCount: number;
  pathCount: number;
};

function buildApiUrl(path: string): string {
  if (!BACKEND_BASE_URL) {
    return path;
  }
  const base = BACKEND_BASE_URL.endsWith("/")
    ? BACKEND_BASE_URL.slice(0, -1)
    : BACKEND_BASE_URL;
  return `${base}${path}`;
}

function encodeRelativeUrlPath(path: string): string {
  return path.split("/").map((part) => encodeURIComponent(part)).join("/");
}

function generatePathId(): string {
  return `path-${Math.random().toString(36).slice(2, 10)}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });
}

function drawCenterCropCover(
  ctx: CanvasRenderingContext2D,
  source: HTMLImageElement,
  width: number,
  height: number
): void {
  const sourceWidth = source.width;
  const sourceHeight = source.height;
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    return;
  }

  const sourceAspect = sourceWidth / sourceHeight;
  const targetAspect = width / height;

  let sx = 0;
  let sy = 0;
  let sw = sourceWidth;
  let sh = sourceHeight;

  if (sourceAspect > targetAspect) {
    sw = sourceHeight * targetAspect;
    sx = (sourceWidth - sw) / 2;
  } else {
    sh = sourceWidth / targetAspect;
    sy = (sourceHeight - sh) / 2;
  }

  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, width, height);
}

function createSquareGridPoints(
  rows: number,
  cols: number,
  spacing: number,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number }[] {
  const totalWidth = (cols - 1) * spacing;
  const totalHeight = (rows - 1) * spacing;
  const startX = canvasWidth / 2 - totalWidth / 2;
  const startY = canvasHeight / 2 - totalHeight / 2;

  const points: { x: number; y: number }[] = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      points.push({
        x: startX + c * spacing,
        y: startY + r * spacing
      });
    }
  }
  return points;
}

function createCirclePoints(
  pointCount: number,
  spacing: number,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number }[] {
  const count = Math.max(1, pointCount);
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;
  const minSide = Math.min(canvasWidth, canvasHeight);
  const baseRadius = minSide * 0.35;
  const scaledRadius = baseRadius * (spacing / 50);
  const radius = Math.max(16, Math.min(minSide * 0.48, scaledRadius));

  // Phyllotaxis layout gives stable, even circular point cloud density.
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    const r = radius * Math.sqrt(t);
    const theta = i * goldenAngle;
    points.push({
      x: centerX + r * Math.cos(theta),
      y: centerY + r * Math.sin(theta)
    });
  }
  return points;
}

function clampGridCount(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.max(1, Math.min(64, Math.round(value)));
}

function clonePoints(points: { x: number; y: number }[]): { x: number; y: number }[] {
  return points.map((point) => ({ x: point.x, y: point.y }));
}

function getTrackMode(path: TrackPath | null | undefined): "moving" | "static" {
  return path?.trackMode === "static" ? "static" : "moving";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isPoint(value: unknown): value is { x: number; y: number } {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { x?: unknown; y?: unknown };
  return isFiniteNumber(candidate.x) && isFiniteNumber(candidate.y);
}

function isPointArray(value: unknown): value is { x: number; y: number }[] {
  return Array.isArray(value) && value.every((point) => isPoint(point));
}

function parseTrackPath(value: unknown): TrackPath | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const candidate = value as Partial<TrackPath> & {
    pointMode?: unknown;
    keyframes?: unknown;
    trackMode?: unknown;
  };
  if (
    typeof candidate.id !== "string" ||
    typeof candidate.name !== "string" ||
    !isPointArray(candidate.points) ||
    typeof candidate.closed !== "boolean" ||
    typeof candidate.color !== "string"
  ) {
    return null;
  }
  const pointMode =
    candidate.pointMode === "points" || candidate.pointMode === "polyline"
      ? candidate.pointMode
      : "points";
  const trackMode = candidate.trackMode === "static" ? "static" : "moving";
  const keyframes = Array.isArray(candidate.keyframes)
    ? candidate.keyframes.filter((frame): frame is { x: number; y: number }[] => isPointArray(frame))
    : undefined;

  return {
    id: candidate.id,
    name: candidate.name,
    points: clonePoints(candidate.points),
    keyframes,
    trackMode,
    pointMode,
    closed: candidate.closed,
    color: candidate.color
  };
}

function parseTrackDocument(raw: unknown): TrackDocument | null {
  if (typeof raw !== "object" || raw === null) {
    return null;
  }

  const candidate = raw as {
    version?: unknown;
    image?: unknown;
    grid?: unknown;
    paths?: unknown;
    meta?: unknown;
  };
  if (candidate.version !== "0.1") {
    return null;
  }
  if (typeof candidate.image !== "object" || candidate.image === null) {
    return null;
  }
  if (typeof candidate.grid !== "object" || candidate.grid === null) {
    return null;
  }

  const image = candidate.image as { src?: unknown; width?: unknown; height?: unknown };
  const grid = candidate.grid as {
    type?: unknown;
    spacing?: unknown;
    offsetX?: unknown;
    offsetY?: unknown;
    visible?: unknown;
  };
  if (
    typeof image.src !== "string" ||
    !isFiniteNumber(image.width) ||
    !isFiniteNumber(image.height) ||
    grid.type !== "square" ||
    !isFiniteNumber(grid.spacing) ||
    !isFiniteNumber(grid.offsetX) ||
    !isFiniteNumber(grid.offsetY) ||
    typeof grid.visible !== "boolean" ||
    !Array.isArray(candidate.paths)
  ) {
    return null;
  }

  const parsedPaths = candidate.paths
    .map((path) => parseTrackPath(path))
    .filter((path): path is TrackPath => path !== null);
  if (parsedPaths.length !== candidate.paths.length) {
    return null;
  }

  return {
    version: "0.1",
    image: {
      src: image.src,
      width: image.width,
      height: image.height
    },
    grid: {
      type: "square",
      spacing: grid.spacing,
      offsetX: grid.offsetX,
      offsetY: grid.offsetY,
      visible: grid.visible
    },
    paths: parsedPaths,
    meta: {
      createdAt: nowIso(),
      updatedAt: nowIso()
    }
  };
}

function scalePointsAboutCentroid(
  points: { x: number; y: number }[],
  scale: number
): { x: number; y: number }[] {
  if (points.length === 0) {
    return [];
  }
  const centerX =
    points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const centerY =
    points.reduce((sum, point) => sum + point.y, 0) / points.length;

  return points.map((point) => ({
    x: centerX + (point.x - centerX) * scale,
    y: centerY + (point.y - centerY) * scale
  }));
}

function ensureKeyframes(
  path: TrackPath,
  frameCount: number
): { x: number; y: number }[][] {
  const safeCount = Math.max(1, frameCount);
  if (!path.keyframes || path.keyframes.length === 0) {
    return Array.from({ length: safeCount }, () => clonePoints(path.points));
  }

  const resized = path.keyframes.slice(0, safeCount).map((frame) => clonePoints(frame));
  const fallback = resized[resized.length - 1] ?? clonePoints(path.points);
  while (resized.length < safeCount) {
    resized.push(clonePoints(fallback));
  }
  return resized;
}

function createStaticKeyframes(
  points: { x: number; y: number }[],
  frameCount: number
): { x: number; y: number }[][] {
  return Array.from({ length: Math.max(1, frameCount) }, () =>
    clonePoints(points)
  );
}

function makeNpyBuffer(
  descr: "<f4" | "<i8",
  shape: number[],
  rawData: Uint8Array
): Uint8Array {
  const encoder = new TextEncoder();
  const shapeLiteral =
    shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")},)`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeLiteral}, }`;
  const baseLength = 10;
  const prePadLength = baseLength + header.length + 1;
  const padLength = (16 - (prePadLength % 16)) % 16;
  header = `${header}${" ".repeat(padLength)}\n`;
  const headerBytes = encoder.encode(header);

  const buffer = new Uint8Array(baseLength + headerBytes.length + rawData.length);
  buffer[0] = 0x93;
  buffer[1] = 0x4e;
  buffer[2] = 0x55;
  buffer[3] = 0x4d;
  buffer[4] = 0x50;
  buffer[5] = 0x59;
  buffer[6] = 0x01;
  buffer[7] = 0x00;
  buffer[8] = headerBytes.length & 0xff;
  buffer[9] = (headerBytes.length >> 8) & 0xff;
  buffer.set(headerBytes, baseLength);
  buffer.set(rawData, baseLength + headerBytes.length);
  return buffer;
}

function typedArrayBytes(
  values: Float32Array | BigInt64Array
): Uint8Array {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to encode PNG blob"));
        return;
      }
      resolve(blob);
    }, "image/png");
  });
}

function createDemoImageDataUrl(
  width = DEFAULT_IMAGE_WIDTH,
  height = DEFAULT_IMAGE_HEIGHT
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return "";
  }

  const grad = ctx.createLinearGradient(0, 0, width, height);
  grad.addColorStop(0, "#202a3f");
  grad.addColorStop(1, "#0f1522");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  const block = 48;
  for (let y = 0; y < height; y += block) {
    for (let x = 0; x < width; x += block) {
      const even = (x / block + y / block) % 2 === 0;
      ctx.fillStyle = even ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)";
      ctx.fillRect(x, y, block, block);
    }
  }

  ctx.fillStyle = "rgba(210, 226, 255, 0.9)";
  ctx.font = "600 28px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText("Track Builder Demo Canvas", width / 2, height / 2 - 10);
  ctx.font = "15px Segoe UI";
  ctx.fillStyle = "rgba(182, 204, 244, 0.95)";
  ctx.fillText("Load Image to replace this demo background", width / 2, height / 2 + 20);

  return canvas.toDataURL("image/png");
}

function dataUrlToBlob(dataUrl: string): Blob | null {
  const parts = dataUrl.split(",");
  if (parts.length !== 2) {
    return null;
  }

  const mimeMatch = parts[0].match(/data:(.*?);base64/);
  if (!mimeMatch) {
    return null;
  }
  const mimeType = mimeMatch[1];
  const byteString = atob(parts[1]);
  const buffer = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i += 1) {
    buffer[i] = byteString.charCodeAt(i);
  }
  return new Blob([buffer], { type: mimeType });
}

export default function App(): JSX.Element {
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<"builder" | "archive">("builder");
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageSrc, setImageSrc] = useState<string>("");
  const [activeObjectUrl, setActiveObjectUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [grid, setGrid] = useState<GridConfig>({
    type: "square",
    spacing: 50,
    offsetX: 0,
    offsetY: 0,
    visible: false
  });

  const [paths, setPaths] = useState<TrackPath[]>([]);
  const [selectedPathId, setSelectedPathId] = useState<string | null>(null);
  const [gridRows, setGridRows] = useState<number>(8);
  const [gridCols, setGridCols] = useState<number>(8);
  const [pointCloudShape, setPointCloudShape] = useState<"square" | "circle">("square");
  const [isStaticPointCloud, setIsStaticPointCloud] = useState<boolean>(false);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isRecordingCompleted, setIsRecordingCompleted] = useState<boolean>(false);
  const [pointSpacingScale, setPointSpacingScale] = useState<number>(1.0);
  const pointSpacingScaleRef = useRef<number>(1.0);
  // const [captionTask, setCaptionTask] = useState<string>(DEFAULT_FLORENCE_TASK);
  // const [generatedCaption, setGeneratedCaption] = useState<string>("");
  // const [isCaptioning, setIsCaptioning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>(
    "Ready - Create point cloud and drag it across 81 frames"
  );
  const [serverExportPath, setServerExportPath] = useState<string>("/data/project-vilab/jaeseok/VideoX-Fun/asset/track_samples/");
  const [serverExportSubDir, setServerExportSubDir] = useState<string>("");
  const [localDownload, setLocalDownload] = useState<boolean>(true);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("motion_only");
  const [generationPrompt, setGenerationPrompt] = useState<string>("a video");
  const [generationSeed, setGenerationSeed] = useState<number>(42);
  const [textGuidanceWeight, setTextGuidanceWeight] = useState<number>(0.0);
  const [motionGuidanceWeight, setMotionGuidanceWeight] = useState<number>(3.0);
  const [isQueueing, setIsQueueing] = useState<boolean>(false);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJobLog, setSelectedJobLog] = useState<string>("");

  useEffect(() => {
    if (image) {
      return;
    }

    const demoSrc = createDemoImageDataUrl();
    if (!demoSrc) {
      return;
    }

    const demoImage = new Image();
    demoImage.onload = () => {
      setImage(demoImage);
      setImageSrc(demoSrc);
      setStatus("Loaded demo canvas - Upload an image to start editing real input");
    };
    demoImage.src = demoSrc;
  }, [image]);

  const loadJobs = async (): Promise<void> => {
    try {
      const response = await fetch(buildApiUrl("/api/jobs"));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as { jobs?: JobRecord[] };
      setJobs(Array.isArray(payload.jobs) ? payload.jobs : []);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "unknown error";
      setStatus(`Failed to refresh jobs: ${reason}`);
    }
  };

  const loadSelectedJobLog = async (jobId: string): Promise<void> => {
    try {
      const response = await fetch(buildApiUrl(`/api/jobs/${jobId}/log`));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as { text?: string };
      setSelectedJobLog(payload.text ?? "");
    } catch {
      setSelectedJobLog("");
    }
  };

  useEffect(() => {
    void loadJobs();
    const interval = window.setInterval(() => {
      void loadJobs();
    }, 4000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJobLog("");
      return;
    }
    void loadSelectedJobLog(selectedJobId);
    const interval = window.setInterval(() => {
      void loadSelectedJobLog(selectedJobId);
    }, 3000);
    return () => window.clearInterval(interval);
  }, [selectedJobId]);

  const buildJobFileUrl = (job: JobRecord, relPath: string): string =>
    buildApiUrl(`/api/jobs/${job.job_id}/file/${encodeRelativeUrlPath(relPath)}`);

  const handleGenerationModeChange = (mode: GenerationMode): void => {
    setGenerationMode(mode);
    if (mode === "text_only") {
      setTextGuidanceWeight(3.0);
      setMotionGuidanceWeight(0.0);
    } else if (mode === "joint_tm") {
      setTextGuidanceWeight(3.0);
      setMotionGuidanceWeight(1.5);
    } else {
      setTextGuidanceWeight(0.0);
      setMotionGuidanceWeight(3.0);
    }
  };

  const displayPaths = useMemo(
    () =>
      paths.map((path) => ({
        ...path,
        points:
          path.keyframes?.[currentFrame] !== undefined
            ? path.keyframes[currentFrame]
            : path.points
      })),
    [currentFrame, paths]
  );

  const selectedPath = useMemo(
    () => paths.find((path) => path.id === selectedPathId) ?? null,
    [paths, selectedPathId]
  );
  const selectedTrackMode = getTrackMode(selectedPath);

  const selectedDisplayPath = useMemo(
    () => displayPaths.find((path) => path.id === selectedPathId) ?? null,
    [displayPaths, selectedPathId]
  );

  const targetPointCount = useMemo(
    () => clampGridCount(gridRows) * clampGridCount(gridCols),
    [gridCols, gridRows]
  );

  const totalPointCount = useMemo(
    () =>
      paths.reduce(
        (sum, path) => sum + (path.keyframes?.[0]?.length ?? path.points.length),
        0
      ),
    [paths]
  );
  const canAddToQueue = Boolean(image) && totalPointCount > 0 && !isRecording;
  const archiveJobCount = useMemo(
    () =>
      jobs.filter((job) =>
        job.status === "done" ||
        job.status === "failed" ||
        job.status === "canceled" ||
        job.status === "interrupted"
      ).length,
    [jobs]
  );
  const activeJobCount = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "running").length,
    [jobs]
  );

  useEffect(() => {
    if (currentFrame > FRAME_COUNT - 1) {
      setCurrentFrame(FRAME_COUNT - 1);
    }
  }, [currentFrame]);

  const loadImageFile = async (file: File): Promise<void> => {
    if (!file.type.startsWith("image/")) {
      setStatus(`Skipped non-image file: ${file.name}`);
      return;
    }

    const tempObjectUrl = URL.createObjectURL(file);
    try {
      const original = await loadHtmlImage(tempObjectUrl);
      const canvas = document.createElement("canvas");
      canvas.width = DEFAULT_IMAGE_WIDTH;
      canvas.height = DEFAULT_IMAGE_HEIGHT;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setStatus("Failed to process image");
        return;
      }

      drawCenterCropCover(
        ctx,
        original,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_IMAGE_HEIGHT
      );
      const processedSrc = canvas.toDataURL("image/png");
      const processedImage = await loadHtmlImage(processedSrc);

      if (activeObjectUrl) {
        URL.revokeObjectURL(activeObjectUrl);
        setActiveObjectUrl(null);
      }
      setImage(processedImage);
      setImageSrc(processedSrc);
      // setGeneratedCaption("");
      setStatus(
        `Loaded and resized to ${DEFAULT_IMAGE_WIDTH}x${DEFAULT_IMAGE_HEIGHT} (source ${original.width}x${original.height})`
      );
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to load image: ${error.message}`
          : "Failed to load image"
      );
    } finally {
      URL.revokeObjectURL(tempObjectUrl);
    }
  };

  useEffect(() => {
    return () => {
      if (activeObjectUrl) {
        URL.revokeObjectURL(activeObjectUrl);
      }
    };
  }, [activeObjectUrl]);

  const onImageUpload = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await loadImageFile(file);
  };

  const onCanvasDragOver = (event: React.DragEvent<HTMLElement>): void => {
    event.preventDefault();
    if (!isDragOver) {
      setIsDragOver(true);
    }
  };

  const onCanvasDragLeave = (event: React.DragEvent<HTMLElement>): void => {
    event.preventDefault();
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && event.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsDragOver(false);
  };

  const onCanvasDrop = async (event: React.DragEvent<HTMLElement>): Promise<void> => {
    event.preventDefault();
    setIsDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) {
      return;
    }
    await loadImageFile(file);
  };

  const deleteSelectedPath = (): void => {
    if (!selectedPathId) {
      return;
    }

    setPaths((prev) => prev.filter((path) => path.id !== selectedPathId));
    setSelectedPathId(null);
    setIsRecording(false);
    setIsRecordingCompleted(false);
    setCurrentFrame(0);
    setGridRows(8);
    setGridCols(8);
    setPointCloudShape("square");
    setIsStaticPointCloud(false);
    pointSpacingScaleRef.current = 1.0;
    setPointSpacingScale(1.0);
    setGrid((prev) => ({ ...prev, spacing: 50 }));
    setStatus("Deleted selected path");
  };

  const applyStaticPoseToSelectedPath = (statusMessage?: string): void => {
    if (!selectedPathId) {
      setStatus("Select a point cloud first");
      return;
    }

    setPaths((prevPaths) =>
      prevPaths.map((path) => {
        if (path.id !== selectedPathId) {
          return path;
        }
        const keyframes = ensureKeyframes(path, FRAME_COUNT);
        const source = clonePoints(keyframes[currentFrame] ?? keyframes[0] ?? path.points);
        return {
          ...path,
          trackMode: "static",
          points: source,
          keyframes: createStaticKeyframes(source, FRAME_COUNT)
        };
      })
    );
    setCurrentFrame(0);
    setIsRecording(false);
    setIsRecordingCompleted(true);
    setStatus(statusMessage ?? "Made selected point cloud static across all 81 frames");
  };

  const handleStaticPointCloudToggle = (checked: boolean): void => {
    setIsStaticPointCloud(checked);
    setStatus(
      checked
        ? "New point-cloud tracks will be static"
        : "New point-cloud tracks will be moving"
    );
  };

  const makeSelectedPathMoving = (): void => {
    if (!selectedPathId) {
      setStatus("Select a point cloud first");
      return;
    }

    setPaths((prevPaths) =>
      prevPaths.map((path) =>
        path.id === selectedPathId ? { ...path, trackMode: "moving" } : path
      )
    );
    setIsRecordingCompleted(false);
    setStatus("Selected track is now moving - use Recording ON to create motion over time");
  };

  const createOrUpdatePointCloudPath = (forceNew = false): void => {
    const rows = clampGridCount(gridRows);
    const cols = clampGridCount(gridCols);
    const spacing = Math.max(4, Math.round(grid.spacing));
    const width = image?.width ?? 960;
    const height = image?.height ?? 540;
    const points =
      pointCloudShape === "circle"
        ? createCirclePoints(rows * cols, spacing, width, height)
        : createSquareGridPoints(rows, cols, spacing, width, height);
    const shapeLabel = pointCloudShape === "circle" ? "circle" : "square";
    const newTrackMode = isStaticPointCloud ? "static" : "moving";

    if (selectedPathId && !forceNew) {
      const updatedTrackMode = selectedTrackMode;
      setPaths((prev) =>
        prev.map((path) =>
          path.id === selectedPathId
            ? {
                ...path,
                points,
                keyframes: createStaticKeyframes(points, FRAME_COUNT),
                trackMode: updatedTrackMode,
                pointMode: "points"
              }
            : path
        )
      );
      pointSpacingScaleRef.current = 1.0;
      setPointSpacingScale(1.0);
      setStatus(
        updatedTrackMode === "static"
          ? `Updated selected path as static ${shapeLabel} point cloud (${points.length} points)`
          : `Updated selected path as ${shapeLabel} point cloud (${points.length} points)`
      );
      setCurrentFrame(0);
      setIsRecording(false);
      setIsRecordingCompleted(updatedTrackMode === "static");
      return;
    }

    const newPath: TrackPath = {
      id: generatePathId(),
      name:
        newTrackMode === "static"
          ? pointCloudShape === "circle"
            ? `Static Circle ${rows * cols}pts`
            : `Static Grid ${rows}x${cols}`
          : pointCloudShape === "circle"
          ? `Circle ${rows * cols}pts`
          : `Grid ${rows}x${cols}`,
      points,
      keyframes: createStaticKeyframes(points, FRAME_COUNT),
      trackMode: newTrackMode,
      pointMode: "points",
      closed: false,
      color: "#00d7ff"
    };
    setPaths((prev) => [...prev, newPath]);
    setSelectedPathId(newPath.id);
    setPointSpacingScale(1.0);
    setCurrentFrame(0);
    setIsRecording(false);
    setIsRecordingCompleted(newTrackMode === "static");
    setStatus(
      newTrackMode === "static"
        ? `Created static ${shapeLabel} point cloud (${points.length} points)`
        : `Created ${shapeLabel} point cloud (${points.length} points)`
    );
  };

  const handlePointSpacingScaleChange = (nextScale: number): void => {
    const clamped = Math.max(0.2, Math.min(3.0, nextScale));
    if (!selectedPathId) {
      pointSpacingScaleRef.current = clamped;
      setPointSpacingScale(clamped);
      return;
    }

    const prevScale = Math.max(0.2, Math.min(3.0, pointSpacingScaleRef.current));
    const ratio = clamped / prevScale;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      pointSpacingScaleRef.current = clamped;
      setPointSpacingScale(clamped);
      return;
    }

    setPaths((prev) =>
      prev.map((path) => {
        if (path.id !== selectedPathId) {
          return path;
        }
        const keyframes = ensureKeyframes(path, FRAME_COUNT).map((frame) =>
          scalePointsAboutCentroid(frame, ratio)
        );
        return {
          ...path,
          keyframes,
          points: keyframes[currentFrame] ?? path.points
        };
      })
    );
    pointSpacingScaleRef.current = clamped;
    setPointSpacingScale(clamped);
  };

  useEffect(() => {
    setPaths((prev) =>
      prev.map((path) => ({
        ...path,
        keyframes: ensureKeyframes(path, FRAME_COUNT)
      }))
    );
  }, []);

  const setCurrentAsStart = (): void => {
    if (!selectedPathId) {
      setStatus("Select a point cloud first");
      return;
    }
    setPaths((prevPaths) =>
      prevPaths.map((path) => {
        if (path.id !== selectedPathId) {
          return path;
        }
        const keyframes = ensureKeyframes(path, FRAME_COUNT);
        const source = clonePoints(keyframes[currentFrame] ?? keyframes[0] ?? []);
        const isPathStatic = getTrackMode(path) === "static";
        const nextKeyframes = isPathStatic
          ? createStaticKeyframes(source, FRAME_COUNT)
          : keyframes;
        nextKeyframes[0] = source;
        return {
          ...path,
          points: source,
          keyframes: nextKeyframes
        };
      })
    );
    setCurrentFrame(0);
    setIsRecordingCompleted(selectedTrackMode === "static");
    setStatus(
      selectedTrackMode === "static"
        ? "Set current point cloud pose as static track across all frames"
        : "Set current point cloud pose as start frame (frame 1)"
    );
  };

  const handlePathDrag = (pathId: string, dx: number, dy: number): void => {
    if (dx === 0 && dy === 0) {
      return;
    }

    setPaths((prevPaths) =>
      prevPaths.map((path) => {
        if (path.id !== pathId) {
          return path;
        }
        const keyframes = ensureKeyframes(path, FRAME_COUNT);
        const sourcePoints = keyframes[currentFrame] ?? keyframes[0] ?? [];
        const movedPoints = sourcePoints.map((point) => ({
          x: point.x + dx,
          y: point.y + dy
        }));
        const nextKeyframes = getTrackMode(path) === "static"
          ? createStaticKeyframes(movedPoints, FRAME_COUNT)
          : keyframes;
        nextKeyframes[currentFrame] = movedPoints;
        return {
          ...path,
          points: movedPoints,
          keyframes: nextKeyframes
        };
      })
    );
  };

  const handleStopRecording = (): void => {
    setPaths((prevPaths) =>
      prevPaths.map((path) => {
        if (path.id !== selectedPathId) {
          return path;
        }
        const kf = ensureKeyframes(path, FRAME_COUNT);
        const lastPos = kf[currentFrame] ?? kf[0] ?? [];
        for (let f = currentFrame + 1; f < FRAME_COUNT; f++) {
          kf[f] = clonePoints(lastPos);
        }
        return { ...path, keyframes: kf };
      })
    );
    setIsRecording(false);
    setIsRecordingCompleted(true);
    setStatus("Recording complete");
  };

  const handleRecordTick = (pathId: string): void => {
    if (!isRecording) {
      return;
    }

    setCurrentFrame((prevFrame) => {
      const nextFrame = prevFrame < FRAME_COUNT - 1 ? prevFrame + 1 : prevFrame;

      setPaths((prevPaths) =>
        prevPaths.map((path) => {
          if (path.id !== pathId) {
            return path;
          }
          if (getTrackMode(path) === "static") {
            return path;
          }

          const keyframes = ensureKeyframes(path, FRAME_COUNT);
          const sourcePoints = keyframes[prevFrame] ?? keyframes[0] ?? [];
          keyframes[nextFrame] = clonePoints(sourcePoints);

          return {
            ...path,
            points: keyframes[nextFrame] ?? sourcePoints,
            keyframes
          };
        })
      );

      if (nextFrame >= FRAME_COUNT - 1) {
        setIsRecording(false);
        setIsRecordingCompleted(true);
        setStatus("Reached final frame. Recording turned OFF automatically.");
      }

      return nextFrame;
    });
  };

  const exportJson = (): void => {
    const width = image?.width ?? 960;
    const height = image?.height ?? 540;

    const doc: TrackDocument = {
      version: "0.1",
      image: {
        src: imageSrc,
        width,
        height
      },
      grid,
      paths,
      meta: {
        createdAt: nowIso(),
        updatedAt: nowIso()
      }
    };

    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "track_document.json";
    link.click();
    URL.revokeObjectURL(url);
    setStatus("Exported track_document.json");
  };

  const buildTrackPackageArtifacts = async (): Promise<TrackPackageArtifacts> => {
    const exportPaths = paths
      .map((path) => {
        const keyframes = ensureKeyframes(path, FRAME_COUNT);
        return {
          path,
          keyframes,
          pointCount: keyframes[0]?.length ?? 0
        };
      })
      .filter((item) => item.pointCount > 0);

    if (exportPaths.length === 0) {
      throw new Error("No path available to export");
    }
    if (!image) {
      throw new Error("No image available to export");
    }

    const pointCount = exportPaths.reduce((sum, item) => sum + item.pointCount, 0);
    if (pointCount <= 0) {
      throw new Error("No point-cloud points available to export");
    }

    const tracks = new Float32Array(FRAME_COUNT * pointCount * 2);
    const visibility = new Float32Array(FRAME_COUNT * pointCount);
    let pointOffset = 0;
    for (const item of exportPaths) {
      for (let frame = 0; frame < FRAME_COUNT; frame += 1) {
        const framePoints = item.keyframes[frame] ?? [];
        for (let localPointIndex = 0; localPointIndex < item.pointCount; localPointIndex += 1) {
          const globalPointIndex = pointOffset + localPointIndex;
          const point = framePoints[localPointIndex];
          const baseTrackIdx = (frame * pointCount + globalPointIndex) * 2;
          if (point) {
            tracks[baseTrackIdx] = point.x;
            tracks[baseTrackIdx + 1] = point.y;
            visibility[frame * pointCount + globalPointIndex] = 1;
          } else {
            tracks[baseTrackIdx] = 0;
            tracks[baseTrackIdx + 1] = 0;
            visibility[frame * pointCount + globalPointIndex] = 0;
          }
        }
      }
      pointOffset += item.pointCount;
    }

    const validIdx = new BigInt64Array(pointCount);
    for (let i = 0; i < pointCount; i += 1) {
      validIdx[i] = BigInt(i);
    }

    const trackNpzZip = new JSZip();
    trackNpzZip.file(
      "tracks_compressed.npy",
      makeNpyBuffer("<f4", [FRAME_COUNT, pointCount, 2], typedArrayBytes(tracks))
    );
    trackNpzZip.file(
      "visibility_compressed.npy",
      makeNpyBuffer("<f4", [FRAME_COUNT, pointCount], typedArrayBytes(visibility))
    );
    trackNpzZip.file(
      "valid_idx.npy",
      makeNpyBuffer("<i8", [pointCount], typedArrayBytes(validIdx))
    );
    const trackNpzBlob = await trackNpzZip.generateAsync({
      type: "blob",
      compression: "DEFLATE"
    });

    const firstFrameCanvas = document.createElement("canvas");
    firstFrameCanvas.width = DEFAULT_IMAGE_WIDTH;
    firstFrameCanvas.height = DEFAULT_IMAGE_HEIGHT;
    const firstFrameCtx = firstFrameCanvas.getContext("2d");
    if (!firstFrameCtx) {
      throw new Error("Failed to export first frame image");
    }
    drawCenterCropCover(firstFrameCtx, image, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
    const firstFrameBlob = await canvasToPngBlob(firstFrameCanvas);

    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = DEFAULT_IMAGE_WIDTH;
    previewCanvas.height = DEFAULT_IMAGE_HEIGHT;
    const previewCtx = previewCanvas.getContext("2d");
    let previewBlob: Blob | null = null;
    if (previewCtx) {
      drawCenterCropCover(previewCtx, image, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
      let globalPointIndex = 0;
      for (const item of exportPaths) {
        for (let pi = 0; pi < item.pointCount; pi += 1) {
          const hue = (globalPointIndex * 137.508) % 360;
          const h = hue / 60;
          const s = 0.9, l = 0.6;
          const c = (1 - Math.abs(2 * l - 1)) * s;
          const x = c * (1 - Math.abs(h % 2 - 1));
          const m = l - c / 2;
          let r = 0, g = 0, b = 0;
          if (h < 1) { r = c; g = x; } else if (h < 2) { r = x; g = c; } else if (h < 3) { g = c; b = x; } else if (h < 4) { g = x; b = c; } else if (h < 5) { r = x; b = c; } else { r = c; b = x; }
          const cr = Math.round((r + m) * 255), cg = Math.round((g + m) * 255), cb = Math.round((b + m) * 255);
          previewCtx.save();
          previewCtx.strokeStyle = `rgb(${cr},${cg},${cb})`;
          previewCtx.lineWidth = 1.5;
          previewCtx.globalAlpha = getTrackMode(item.path) === "static" ? 0.35 : 0.75;
          previewCtx.beginPath();
          const firstPt = item.keyframes[0][pi];
          if (firstPt) {
            previewCtx.moveTo(firstPt.x, firstPt.y);
            for (let t = 1; t < FRAME_COUNT; t += 1) {
              const p = item.keyframes[t][pi];
              if (p) {
                previewCtx.lineTo(p.x, p.y);
              }
            }
          }
          previewCtx.stroke();
          previewCtx.restore();
          const endPt = item.keyframes[FRAME_COUNT - 1][pi];
          if (endPt) {
            previewCtx.beginPath();
            previewCtx.fillStyle = `rgb(${cr},${cg},${cb})`;
            previewCtx.arc(endPt.x, endPt.y, 3.5, 0, Math.PI * 2);
            previewCtx.fill();
          }
          globalPointIndex += 1;
        }
      }
      previewBlob = await canvasToPngBlob(previewCanvas);
    }

    return {
      trackNpzBlob,
      firstFrameBlob,
      previewBlob,
      pointCount,
      pathCount: exportPaths.length
    };
  };

  const exportTrackPackage = async (): Promise<void> => {
    try {
      const {
        trackNpzBlob,
        firstFrameBlob,
        previewBlob,
        pointCount,
        pathCount
      } = await buildTrackPackageArtifacts();

      // const captionFileContent = [
      //   `task_prompt: ${captionTask.trim() || DEFAULT_FLORENCE_TASK}`,
      //   `generated_at: ${new Date().toISOString()}`,
      //   "",
      //   generatedCaption.trim() || "(empty caption)"
      // ].join("\n");

      // Server export
      const baseDir = serverExportPath.trim();
      if (baseDir) {
        const userSubDir = serverExportSubDir.trim();
        let subDir: string;
        if (userSubDir) {
          subDir = userSubDir;
        } else {
          const now = new Date();
          subDir = `track_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;
        }
        const exportDir = baseDir.endsWith("/") ? `${baseDir}${subDir}` : `${baseDir}/${subDir}`;
        const formData = new FormData();
        formData.append("directory", exportDir);
        formData.append("image", firstFrameBlob, "first_frame.png");
        formData.append("tracks_npz", trackNpzBlob, "transformed_tracks_grid50_survived.npz");
        // formData.append("caption", captionFileContent);
        if (previewBlob) {
          formData.append("preview_png", previewBlob, "track_preview.png");
        }
        const resp = await fetch(buildApiUrl("/api/export/package"), {
          method: "POST",
          body: formData
        });
        if (!resp.ok) {
          const errText = await resp.text();
          setStatus(`Server export failed: ${errText}`);
          return;
        }
        const result = await resp.json() as { directory: string };
        setStatus(`Saved to server: ${result.directory} (${FRAME_COUNT} frames, ${pointCount} pts, ${pathCount} tracks)`);
      }

      // Local download
      if (localDownload) {
        const packageZip = new JSZip();
        const exportDir = "processed_832x480_fps16";
        packageZip.file(`${exportDir}/first_frame.png`, firstFrameBlob);
        packageZip.file(`${exportDir}/transformed_tracks_grid50_survived.npz`, trackNpzBlob);
        // packageZip.file(`${exportDir}/image_caption.txt`, captionFileContent);
        if (previewBlob) {
          packageZip.file(`${exportDir}/track_preview.png`, previewBlob);
        }
        const packageBlob = await packageZip.generateAsync({
          type: "blob",
          compression: "DEFLATE"
        });
        const url = URL.createObjectURL(packageBlob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "processed_832x480_fps16.zip";
        link.click();
        URL.revokeObjectURL(url);
        if (!baseDir) {
          setStatus(`Exported track package: ${FRAME_COUNT} frames, ${pointCount} points, ${pathCount} tracks`);
        }
      }

      if (!baseDir && !localDownload) {
        setStatus("Nothing exported - set a server path or enable local download.");
      }
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to export track package: ${error.message}`
          : "Failed to export track package"
      );
    }
  };

  const addCurrentToQueue = async (): Promise<void> => {
    setIsQueueing(true);
    try {
      const {
        trackNpzBlob,
        firstFrameBlob,
        previewBlob,
        pointCount,
        pathCount
      } = await buildTrackPackageArtifacts();
      const formData = new FormData();
      formData.append("image", firstFrameBlob, "first_frame.png");
      formData.append("tracks_npz", trackNpzBlob, "transformed_tracks_grid50_survived.npz");
      if (previewBlob) {
        formData.append("preview_png", previewBlob, "track_preview.png");
      }
      formData.append("mode", generationMode);
      formData.append("prompt", generationPrompt.trim() || "a video");
      formData.append("seed", String(Number.isFinite(generationSeed) ? generationSeed : 42));
      formData.append("text_guidance_weight", String(textGuidanceWeight));
      formData.append("motion_guidance_weight", String(motionGuidanceWeight));

      const response = await fetch(buildApiUrl("/api/jobs"), {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }
      const payload = (await response.json()) as { job?: JobRecord };
      if (payload.job) {
        setSelectedJobId(payload.job.job_id);
        setStatus(
          `Queued ${payload.job.job_id}: ${pathCount} tracks, ${pointCount} points`
        );
      } else {
        setStatus(`Queued generation: ${pathCount} tracks, ${pointCount} points`);
      }
      await loadJobs();
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to queue generation: ${error.message}`
          : "Failed to queue generation"
      );
    } finally {
      setIsQueueing(false);
    }
  };

  const cancelJob = async (jobId: string): Promise<void> => {
    try {
      const response = await fetch(buildApiUrl(`/api/jobs/${jobId}/cancel`), {
        method: "POST"
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }
      setStatus(`Canceled ${jobId}`);
      await loadJobs();
      if (selectedJobId === jobId) {
        await loadSelectedJobLog(jobId);
      }
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to cancel job: ${error.message}`
          : "Failed to cancel job"
      );
    }
  };

  const retryJob = async (jobId: string): Promise<void> => {
    try {
      const response = await fetch(buildApiUrl(`/api/jobs/${jobId}/retry`), {
        method: "POST"
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }
      const payload = (await response.json()) as { job?: JobRecord };
      if (payload.job) {
        setSelectedJobId(payload.job.job_id);
        setStatus(`Queued retry ${payload.job.job_id}`);
      } else {
        setStatus("Queued retry");
      }
      await loadJobs();
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to retry job: ${error.message}`
          : "Failed to retry job"
      );
    }
  };

  const importJson = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const parsedJson = JSON.parse(await file.text()) as unknown;
      const parsed = parseTrackDocument(parsedJson);
      if (!parsed) {
        setStatus("Invalid JSON schema. Import aborted.");
        return;
      }

      setGrid({ ...parsed.grid, visible: false });
      setPaths(
        parsed.paths.map((path) => ({
          ...path,
          trackMode: getTrackMode(path),
          pointMode: "points",
          keyframes: ensureKeyframes(path, FRAME_COUNT)
        }))
      );
      setSelectedPathId(parsed.paths[0]?.id ?? null);
      setCurrentFrame(0);
      setIsRecordingCompleted(false);

      if (activeObjectUrl) {
        URL.revokeObjectURL(activeObjectUrl);
        setActiveObjectUrl(null);
      }
      if (parsed.image?.src) {
        const img = new Image();
        img.onload = () => setImage(img);
        img.src = parsed.image.src;
        setImageSrc(parsed.image.src);
        // setGeneratedCaption("");
      }

      setStatus("Imported track document");
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Failed to import JSON: ${error.message}`
          : "Failed to import JSON"
      );
    }
  };

  // const generateImageCaption = async (): Promise<void> => {
  //   if (!imageSrc) {
  //     setStatus("Load image first");
  //     return;
  //   }
  //   const imageBlob = dataUrlToBlob(imageSrc);
  //   if (!imageBlob) {
  //     setStatus("Unsupported image source for captioning");
  //     return;
  //   }
  //
  //   setIsCaptioning(true);
  //   setStatus("Generating Florence caption...");
  //   try {
  //     const formData = new FormData();
  //     formData.append("file", imageBlob, "track_builder_input.png");
  //     formData.append("task", captionTask.trim() || DEFAULT_FLORENCE_TASK);
  //
  //     const response = await fetch(buildApiUrl("/api/images/caption"), {
  //       method: "POST",
  //       body: formData
  //     });
  //     if (!response.ok) {
  //       const errorText = await response.text();
  //       throw new Error(errorText || `HTTP ${response.status}`);
  //     }
  //     const payload = (await response.json()) as {
  //       text?: string;
  //       task?: string;
  //     };
  //     const nextCaption = payload.text?.trim() ?? "";
  //     if (!nextCaption) {
  //       setGeneratedCaption("");
  //       setStatus("Caption response is empty");
  //       return;
  //     }
  //     setGeneratedCaption(nextCaption);
  //     setStatus(`Caption generated with task ${payload.task ?? captionTask}`);
  //   } catch (error) {
  //     const reason = error instanceof Error ? error.message : "unknown error";
  //     setStatus(
  //       `Caption failed: ${reason} (check backend on :8000 or set VITE_BACKEND_URL)`
  //     );
  //   } finally {
  //     setIsCaptioning(false);
  //   }
  // };

  return (
    <div className="app-root">
      <main
        className={isDragOver ? "main-drop active" : "main-drop"}
        onDragOver={onCanvasDragOver}
        onDragEnter={onCanvasDragOver}
        onDragLeave={onCanvasDragLeave}
        onDrop={onCanvasDrop}
      >
        {isDragOver ? (
          <div className="drop-overlay">Drop image here</div>
        ) : null}

        <div className="workspace-tabs" role="tablist" aria-label="Track builder workspace">
          <button
            type="button"
            className={activeWorkspaceTab === "builder" ? "workspace-tab active" : "workspace-tab"}
            onClick={() => setActiveWorkspaceTab("builder")}
          >
            Builder
            {activeJobCount > 0 ? <span>{activeJobCount}</span> : null}
          </button>
          <button
            type="button"
            className={activeWorkspaceTab === "archive" ? "workspace-tab active" : "workspace-tab"}
            onClick={() => setActiveWorkspaceTab("archive")}
          >
            Archive
            {archiveJobCount > 0 ? <span>{archiveJobCount}</span> : null}
          </button>
        </div>

        {activeWorkspaceTab === "builder" ? (
          <div className="app-layout">
            <aside className="control-panel">
              <div className="panel-title">Track Builder Controls</div>
              <div className="status">
                {isRecording
                  ? `Recording... ${currentFrame + 1}/${FRAME_COUNT}`
                  : status}
              </div>

              <section className="control-section">
                <h3>1) Input</h3>
                <div className="control-row">
                  <label className="btn">
                    Load Image
                    <input type="file" accept="image/*" onChange={onImageUpload} hidden />
                  </label>
                  <label className="btn">
                    Import JSON
                    <input type="file" accept="application/json" onChange={importJson} hidden />
                  </label>
                  <button type="button" onClick={exportJson}>Export JSON</button>
                  <button type="button" onClick={() => void exportTrackPackage()}>
                    Export Track Package
                  </button>
                </div>
                <label className="slider-field">
                  Server Export Directory
                  <input
                    type="text"
                    value={serverExportPath}
                    onChange={(event) => setServerExportPath(event.target.value)}
                    placeholder="/path/to/root/dir (empty = local only)"
                  />
                </label>
                <label className="slider-field">
                  Export Subdirectory Name
                  <input
                    type="text"
                    value={serverExportSubDir}
                    onChange={(event) => setServerExportSubDir(event.target.value)}
                    placeholder="e.g. scene_001 (empty = auto timestamp)"
                  />
                </label>
                <div className="control-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={localDownload}
                      onChange={(event) => setLocalDownload(event.target.checked)}
                    />
                    {" "}Also download locally
                  </label>
                </div>
                <div className="path-info">
                  Image: {image ? `${image.width}x${image.height}` : "None"}
                </div>
              {/* <label className="slider-field">
                Florence Task Prompt
                <input
                  type="text"
                  value={captionTask}
                  onChange={(event) => setCaptionTask(event.target.value)}
                  placeholder="<MORE_DETAILED_CAPTION>"
                  disabled={isCaptioning}
                />
              </label>
              <div className="control-row">
                <button
                  type="button"
                  onClick={() => void generateImageCaption()}
                  disabled={isCaptioning || !image}
                >
                  {isCaptioning ? "Generating Caption..." : "Generate Caption (Florence)"}
                </button>
              </div>
              <label className="slider-field">
                Image Caption
                <textarea
                  value={generatedCaption}
                  onChange={(event) => setGeneratedCaption(event.target.value)}
                  placeholder="Caption will appear here"
                  rows={4}
                />
              </label> */}
              </section>

              <section className="control-section">
                <h3>2) Point Cloud</h3>
              <div className="control-row">
                <label>
                  Shape
                  <select
                    value={pointCloudShape}
                    onChange={(event) =>
                      setPointCloudShape(
                        event.target.value === "circle" ? "circle" : "square"
                      )
                    }
                    disabled={isRecording}
                  >
                    <option value="square">Square</option>
                    <option value="circle">Circle</option>
                  </select>
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={isStaticPointCloud}
                    onChange={(event) => handleStaticPointCloudToggle(event.target.checked)}
                    disabled={isRecording}
                  />
                  {" "}New Track Static
                </label>
                <label>
                  Rows
                  <input
                    className="numeric-input"
                    type="number"
                    min={1}
                    max={64}
                    value={gridRows}
                    onChange={(event) => setGridRows(clampGridCount(Number(event.target.value)))}
                  />
                </label>
                <label>
                  Cols
                  <input
                    className="numeric-input"
                    type="number"
                    min={1}
                    max={64}
                    value={gridCols}
                    onChange={(event) => setGridCols(clampGridCount(Number(event.target.value)))}
                  />
                </label>
              </div>
              <div className="path-info">Target per new track: {targetPointCount} points</div>
              <div className="control-row">
                <button
                  type="button"
                  onClick={() => createOrUpdatePointCloudPath(true)}
                  disabled={isRecording}
                >
                  Add Point Cloud Track
                </button>
                <button
                  type="button"
                  onClick={() => createOrUpdatePointCloudPath(false)}
                  disabled={!selectedPathId || isRecording}
                >
                  Update Selected Track
                </button>
                <button
                  type="button"
                  onClick={setCurrentAsStart}
                  disabled={!selectedPathId || isRecording}
                >
                  Use Current Pose as Frame 1
                </button>
                <button
                  type="button"
                  onClick={() => applyStaticPoseToSelectedPath()}
                  disabled={!selectedPathId || isRecording}
                >
                  Make Current Pose Static
                </button>
                <button
                  type="button"
                  onClick={makeSelectedPathMoving}
                  disabled={!selectedPathId || isRecording}
                >
                  Make Selected Moving
                </button>
              </div>
              {paths.length > 0 ? (
                <div className="track-list">
                  {paths.map((path, index) => {
                    const mode = getTrackMode(path);
                    const pointCount = path.keyframes?.[0]?.length ?? path.points.length;
                    return (
                      <button
                        key={path.id}
                        type="button"
                        className={path.id === selectedPathId ? "track-item active" : "track-item"}
                        onClick={() => setSelectedPathId(path.id)}
                        disabled={isRecording}
                      >
                        <span>{index + 1}. {path.name}</span>
                        <span>{mode} / {pointCount} pts</span>
                      </button>
                    );
                  })}
                </div>
              ) : null}
              <label className="slider-field">
                Point Spacing Scale: x{pointSpacingScale.toFixed(2)}
                <input
                  type="range"
                  min={0.2}
                  max={3}
                  step={0.05}
                  value={pointSpacingScale}
                  onChange={(event) =>
                    handlePointSpacingScaleChange(Number(event.target.value))
                  }
                  disabled={!selectedPathId || isRecording}
                />
              </label>
              </section>

              <section className="control-section">
                <h3>3) Recording</h3>
              <button
                type="button"
                className={isRecording ? "recording-button recording primary" : "recording-button primary"}
                onClick={() => {
                  if (isRecording) {
                    handleStopRecording();
                  } else {
                    setIsRecordingCompleted(false);
                    setIsRecording(true);
                  }
                }}
                disabled={!selectedPathId || selectedTrackMode === "static"}
              >
                {isRecording ? "Recording" : "Recording ON"}
              </button>
              <label className="slider-field">
                Time: {currentFrame + 1}/{FRAME_COUNT}
                <input
                  type="range"
                  min={0}
                  max={FRAME_COUNT - 1}
                  value={Math.min(currentFrame, FRAME_COUNT - 1)}
                  onChange={(event) => setCurrentFrame(Number(event.target.value))}
                />
              </label>
              <div className="control-row">
                <button type="button" onClick={() => setCurrentFrame(0)}>Reset to Frame 1</button>
              </div>
              </section>

              <GenerationControls
                mode={generationMode}
                prompt={generationPrompt}
                seed={generationSeed}
                textGuidanceWeight={textGuidanceWeight}
                motionGuidanceWeight={motionGuidanceWeight}
                isQueueing={isQueueing}
                canSubmit={canAddToQueue}
                onModeChange={handleGenerationModeChange}
                onPromptChange={setGenerationPrompt}
                onSeedChange={setGenerationSeed}
                onTextGuidanceWeightChange={setTextGuidanceWeight}
                onMotionGuidanceWeightChange={setMotionGuidanceWeight}
                onAddToQueue={() => void addCurrentToQueue()}
              />

              <section className="control-section danger">
                <h3>5) Cleanup</h3>
                <button type="button" onClick={deleteSelectedPath} disabled={!selectedPathId}>
                  Delete Selected Path
                </button>
              </section>

              <div className="path-info">
                Selected: {selectedPath ? `${selectedPath.name} / ${selectedTrackMode}` : "None"} (
                {selectedDisplayPath?.points.length ?? 0} pts @ frame {currentFrame + 1})
                {" "}Total: {paths.length} tracks / {totalPointCount} pts
              </div>
            </aside>

            <div className="canvas-stage">
              <TrackCanvas
                image={image}
                grid={{ ...grid, visible: false }}
                paths={displayPaths}
                selectedPathId={selectedPathId}
                onMovePathPoints={handlePathDrag}
                onRecordTick={handleRecordTick}
                onSelectPath={setSelectedPathId}
                onDragComplete={() => undefined}
              />
            </div>

            <aside className="preview-stage">
              <TrackPreview
                image={image}
                path={selectedPath}
                frameCount={FRAME_COUNT}
                currentFrame={currentFrame}
              />
              <QueuePanel
                jobs={jobs}
                selectedJobId={selectedJobId}
                selectedLog={selectedJobLog}
                onSelectJob={setSelectedJobId}
                onRefresh={() => void loadJobs()}
                onCancel={(jobId) => void cancelJob(jobId)}
                onRetry={(jobId) => void retryJob(jobId)}
                buildFileUrl={buildJobFileUrl}
              />
            </aside>
          </div>
        ) : (
          <div className="archive-workspace">
            <ArchivePanel
              jobs={jobs}
              onRetry={(jobId) => void retryJob(jobId)}
              buildFileUrl={buildJobFileUrl}
            />
          </div>
        )}
      </main>
    </div>
  );
}
