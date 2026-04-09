import React, { useEffect, useMemo, useRef } from "react";
import type { GridConfig, Point, TrackPath } from "../types";

type Props = {
  image: HTMLImageElement | null;
  grid: GridConfig;
  paths: TrackPath[];
  selectedPathId: string | null;
  onMovePathPoints: (pathId: string, dx: number, dy: number) => void;
  onRecordTick: (pathId: string) => void;
  onSelectPath: (pathId: string | null) => void;
  onDragComplete: () => void;
};

type DragState =
  | { kind: "none" }
  | { kind: "path"; pathId: string; lastX: number; lastY: number };

const RECORD_TOTAL_MS = 5000;
const RECORD_FRAME_COUNT = 81;
const RECORD_TICK_MS = Math.max(16, Math.round(RECORD_TOTAL_MS / RECORD_FRAME_COUNT));

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  grid: GridConfig
): void {
  if (!grid.visible) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = "rgba(80, 255, 210, 0.45)";
  ctx.lineWidth = 1;

  const spacing = Math.max(grid.spacing, 4);
  const startX = ((grid.offsetX % spacing) + spacing) % spacing;
  const startY = ((grid.offsetY % spacing) + spacing) % spacing;

  for (let x = startX; x <= width; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = startY; y <= height; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawPaths(
  ctx: CanvasRenderingContext2D,
  paths: TrackPath[],
  selectedPathId: string | null
): void {
  for (const path of paths) {
    if (path.points.length === 0) {
      continue;
    }

    const isSelected = path.id === selectedPathId;
    const shouldDrawPolyline = (path.pointMode ?? "polyline") === "polyline";

    ctx.save();
    ctx.strokeStyle = path.color;
    ctx.lineWidth = isSelected ? 3 : 2;
    if (shouldDrawPolyline) {
      ctx.beginPath();
      ctx.moveTo(path.points[0].x, path.points[0].y);
      for (let i = 1; i < path.points.length; i += 1) {
        ctx.lineTo(path.points[i].x, path.points[i].y);
      }
      if (path.closed && path.points.length >= 3) {
        ctx.closePath();
      }
      ctx.stroke();
    }

    for (let i = 0; i < path.points.length; i += 1) {
      const point = path.points[i];
      const hue = (i * 137.508) % 360;
      const fillColor = `hsl(${hue}, 90%, 60%)`;

      if (isSelected) {
        // Outer ring for visibility on bright/dark backgrounds.
        ctx.beginPath();
        ctx.fillStyle = "rgba(8, 12, 20, 0.95)";
        ctx.arc(point.x, point.y, 6.4, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.fillStyle = fillColor;
      ctx.arc(point.x, point.y, isSelected ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
}

function drawEmptyOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
): void {
  ctx.save();
  ctx.fillStyle = "rgba(8, 10, 14, 0.55)";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#dce7ff";
  ctx.font = "600 20px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText("Track Builder UI", width / 2, height / 2 - 28);

  ctx.font = "14px Segoe UI";
  ctx.fillStyle = "#b4c5ec";
  ctx.fillText("1) Create Square Point Cloud (e.g., 8x8)", width / 2, height / 2 + 4);
  ctx.fillText("2) Drag to move the whole point-cloud group", width / 2, height / 2 + 28);
  ctx.restore();
}

function getMousePos(canvas: HTMLCanvasElement, event: MouseEvent): Point {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function distanceToSegment(point: Point, a: Point, b: Point): number {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const apx = point.x - a.x;
  const apy = point.y - a.y;
  const ab2 = abx * abx + aby * aby;
  const t = ab2 === 0 ? 0 : Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  const cx = a.x + abx * t;
  const cy = a.y + aby * t;
  const dx = point.x - cx;
  const dy = point.y - cy;
  return Math.sqrt(dx * dx + dy * dy);
}

function pickPath(paths: TrackPath[], point: Point, threshold = 12): string | null {
  let bestId: string | null = null;
  let bestDist = Number.POSITIVE_INFINITY;

  for (const path of paths) {
    if (path.points.length === 0) {
      continue;
    }

    if ((path.pointMode ?? "polyline") === "points" || path.points.length === 1) {
      for (const p of path.points) {
        const dx = p.x - point.x;
        const dy = p.y - point.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist && d <= threshold) {
          bestDist = d;
          bestId = path.id;
        }
      }
      continue;
    }

    if (path.points.length === 1) {
      const dx = path.points[0].x - point.x;
      const dy = path.points[0].y - point.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < bestDist && d <= threshold) {
        bestDist = d;
        bestId = path.id;
      }
      continue;
    }

    for (let i = 0; i < path.points.length - 1; i += 1) {
      const d = distanceToSegment(point, path.points[i], path.points[i + 1]);
      if (d < bestDist && d <= threshold) {
        bestDist = d;
        bestId = path.id;
      }
    }
  }

  return bestId;
}

export default function TrackCanvas({
  image,
  grid,
  paths,
  selectedPathId,
  onMovePathPoints,
  onRecordTick,
  onSelectPath,
  onDragComplete
}: Props): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragStateRef = useRef<DragState>({ kind: "none" });
  const movedRef = useRef<boolean>(false);
  const holdTickTimerRef = useRef<number | null>(null);
  const pathsRef = useRef<TrackPath[]>(paths);
  const selectedPathIdRef = useRef<string | null>(selectedPathId);
  const onMovePathPointsRef = useRef<Props["onMovePathPoints"]>(onMovePathPoints);
  const onRecordTickRef = useRef<Props["onRecordTick"]>(onRecordTick);
  const onSelectPathRef = useRef<Props["onSelectPath"]>(onSelectPath);
  const onDragCompleteRef = useRef<Props["onDragComplete"]>(onDragComplete);

  const canvasSize = useMemo(() => {
    if (!image) {
      return { width: 960, height: 540 };
    }
    return { width: image.width, height: image.height };
  }, [image]);

  useEffect(() => {
    pathsRef.current = paths;
    selectedPathIdRef.current = selectedPathId;
    onMovePathPointsRef.current = onMovePathPoints;
    onRecordTickRef.current = onRecordTick;
    onSelectPathRef.current = onSelectPath;
    onDragCompleteRef.current = onDragComplete;
  }, [
    onDragComplete,
    onMovePathPoints,
    onRecordTick,
    onSelectPath,
    paths,
    selectedPathId
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#1b2230";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    drawGrid(ctx, canvas.width, canvas.height, grid);
    drawPaths(ctx, paths, selectedPathId);
    if (!image) {
      drawEmptyOverlay(ctx, canvas.width, canvas.height);
    }
  }, [canvasSize, grid, image, paths, selectedPathId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const startHoldTickIfNeeded = (): void => {
      if (dragStateRef.current.kind !== "path") {
        return;
      }
      if (holdTickTimerRef.current !== null) {
        return;
      }
      holdTickTimerRef.current = window.setInterval(() => {
        if (dragStateRef.current.kind === "path") {
          onRecordTickRef.current(dragStateRef.current.pathId);
        }
      }, RECORD_TICK_MS);
    };

    const onMouseDown = (event: MouseEvent): void => {
      if (event.button !== 0) {
        return;
      }
      const pos = getMousePos(canvas, event);
      movedRef.current = false;

      const pickedPath = pickPath(pathsRef.current, pos, 20) ?? selectedPathIdRef.current;
      onSelectPathRef.current(pickedPath);
      if (pickedPath) {
        dragStateRef.current = {
          kind: "path",
          pathId: pickedPath,
          lastX: pos.x,
          lastY: pos.y
        };
        if (holdTickTimerRef.current !== null) {
          window.clearInterval(holdTickTimerRef.current);
          holdTickTimerRef.current = null;
        }
        startHoldTickIfNeeded();
      }
    };

    const onMouseMove = (event: MouseEvent): void => {
      const dragState = dragStateRef.current;
      if (dragState.kind === "none") {
        return;
      }

      const pos = getMousePos(canvas, event);

      if (dragState.kind === "path") {
        const dx = pos.x - dragState.lastX;
        const dy = pos.y - dragState.lastY;
        if (dx !== 0 || dy !== 0) {
          movedRef.current = true;
        }
        onMovePathPointsRef.current(dragState.pathId, dx, dy);
        dragStateRef.current = {
          ...dragState,
          lastX: pos.x,
          lastY: pos.y
        };
      }

    };

    const onMouseUp = (): void => {
      dragStateRef.current = { kind: "none" };
      if (holdTickTimerRef.current !== null) {
        window.clearInterval(holdTickTimerRef.current);
        holdTickTimerRef.current = null;
      }
      if (movedRef.current) {
        onDragCompleteRef.current();
      }
      movedRef.current = false;
    };

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    startHoldTickIfNeeded();

    return () => {
      if (holdTickTimerRef.current !== null) {
        window.clearInterval(holdTickTimerRef.current);
        holdTickTimerRef.current = null;
      }
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  return (
    <div className="canvas-wrapper">
      <canvas ref={canvasRef} className="editor-canvas" />
    </div>
  );
}
