import React, { useEffect, useMemo, useRef, useState } from "react";
import type { Point, TrackPath } from "../types";

type Props = {
  image: HTMLImageElement | null;
  path: TrackPath | null;
  frameCount: number;
  currentFrame: number;
};

const PREVIEW_TAIL_LENGTH = 8;
const PREVIEW_TOTAL_MS = 5000;
const PREVIEW_FRAME_COUNT = 81;
const PREVIEW_TICK_MS = Math.max(
  16,
  Math.round(PREVIEW_TOTAL_MS / PREVIEW_FRAME_COUNT)
);

function getPointColor(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue}, 90%, 60%)`;
}

function getSafeKeyframes(path: TrackPath, frameCount: number): Point[][] {
  const base = path.keyframes ?? [path.points];
  const clipped = base.slice(0, Math.max(1, frameCount));
  const last = clipped[clipped.length - 1] ?? path.points;
  while (clipped.length < frameCount) {
    clipped.push(last.map((point) => ({ x: point.x, y: point.y })));
  }
  return clipped;
}

export default function TrackPreview({
  image,
  path,
  frameCount,
  currentFrame
}: Props): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [previewFrame, setPreviewFrame] = useState<number>(0);
  const keyframes = useMemo(
    () => (path ? getSafeKeyframes(path, frameCount) : []),
    [frameCount, path]
  );
  const maxFrame = Math.max(0, Math.min(currentFrame, frameCount - 1));

  useEffect(() => {
    setPreviewFrame(0);
  }, [maxFrame, path?.id]);

  useEffect(() => {
    if (!path || keyframes.length === 0 || keyframes[0]?.length === 0 || maxFrame <= 0) {
      return;
    }
    const timer = window.setInterval(() => {
      setPreviewFrame((prev) => (prev >= maxFrame ? 0 : prev + 1));
    }, PREVIEW_TICK_MS);
    return () => window.clearInterval(timer);
  }, [keyframes, maxFrame, path]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const width = image?.width ?? 960;
    const height = image?.height ?? 540;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, width, height);
    if (image) {
      ctx.drawImage(image, 0, 0, width, height);
    } else {
      ctx.fillStyle = "#111824";
      ctx.fillRect(0, 0, width, height);
    }

    if (!path || keyframes.length === 0 || keyframes[0].length === 0) {
      ctx.fillStyle = "rgba(195, 210, 235, 0.9)";
      ctx.font = "600 20px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText("No completed recording preview", width / 2, height / 2);
      return;
    }

    const pointCount = keyframes[0].length;
    const activeFrame = Math.min(previewFrame, maxFrame);
    const tailStart = Math.max(0, activeFrame - (PREVIEW_TAIL_LENGTH - 1));

    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      const color = getPointColor(pointIndex);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      const first = keyframes[tailStart][pointIndex];
      ctx.moveTo(first.x, first.y);
      for (let frame = tailStart + 1; frame <= activeFrame; frame += 1) {
        const p = keyframes[frame][pointIndex];
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.restore();

      const current = keyframes[activeFrame][pointIndex];
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(current.x, current.y, 4.2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(220, 235, 255, 0.9)";
    ctx.font = "600 14px Segoe UI";
    ctx.textAlign = "left";
    ctx.fillText(`GIF Preview ${activeFrame + 1}/${maxFrame + 1} (tail: ${PREVIEW_TAIL_LENGTH})`, 12, 22);
  }, [frameCount, image, keyframes, maxFrame, path, previewFrame]);

  return (
    <div className="preview-panel">
      <div className="preview-title">Track Preview On Image</div>
      <canvas ref={canvasRef} className="preview-canvas" />
    </div>
  );
}
