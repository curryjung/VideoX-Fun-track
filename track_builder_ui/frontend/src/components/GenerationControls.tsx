import React from "react";
import type { GenerationMode } from "../types";

type GenerationControlsProps = {
  mode: GenerationMode;
  prompt: string;
  seed: number;
  textGuidanceWeight: number;
  motionGuidanceWeight: number;
  isQueueing: boolean;
  canSubmit: boolean;
  onModeChange: (mode: GenerationMode) => void;
  onPromptChange: (prompt: string) => void;
  onSeedChange: (seed: number) => void;
  onTextGuidanceWeightChange: (weight: number) => void;
  onMotionGuidanceWeightChange: (weight: number) => void;
  onAddToQueue: () => void;
};

function modeHint(mode: GenerationMode): string {
  if (mode === "text_only") {
    return "Track is saved for preview/archive, but generation uses text guidance only.";
  }
  if (mode === "joint_tm") {
    return "Uses both prompt and track motion guidance.";
  }
  return "Uses track motion guidance with null text guidance.";
}

export default function GenerationControls({
  mode,
  prompt,
  seed,
  textGuidanceWeight,
  motionGuidanceWeight,
  isQueueing,
  canSubmit,
  onModeChange,
  onPromptChange,
  onSeedChange,
  onTextGuidanceWeightChange,
  onMotionGuidanceWeightChange,
  onAddToQueue
}: GenerationControlsProps): JSX.Element {
  return (
    <section className="control-section">
      <h3>4) Generation</h3>
      <label className="slider-field">
        Mode
        <select
          value={mode}
          onChange={(event) => onModeChange(event.target.value as GenerationMode)}
          disabled={isQueueing}
        >
          <option value="motion_only">Motion-only</option>
          <option value="text_only">Text-only</option>
          <option value="joint_tm">Joint</option>
        </select>
      </label>
      <div className="path-info">{modeHint(mode)}</div>
      <label className="slider-field">
        Prompt
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder="a video"
          rows={3}
          disabled={isQueueing}
        />
      </label>
      <div className="control-row">
        <label>
          Seed
          <input
            className="numeric-input"
            type="number"
            value={seed}
            onChange={(event) => onSeedChange(Number(event.target.value))}
            disabled={isQueueing}
          />
        </label>
      </div>
      {mode !== "motion_only" ? (
        <label className="slider-field">
          Text Weight: {textGuidanceWeight.toFixed(2)}
          <input
            type="range"
            min={0}
            max={8}
            step={0.1}
            value={textGuidanceWeight}
            onChange={(event) => onTextGuidanceWeightChange(Number(event.target.value))}
            disabled={isQueueing}
          />
        </label>
      ) : null}
      {mode !== "text_only" ? (
        <label className="slider-field">
          Motion Weight: {motionGuidanceWeight.toFixed(2)}
          <input
            type="range"
            min={0}
            max={8}
            step={0.1}
            value={motionGuidanceWeight}
            onChange={(event) => onMotionGuidanceWeightChange(Number(event.target.value))}
            disabled={isQueueing}
          />
        </label>
      ) : null}
      <button
        type="button"
        className="recording-button primary"
        onClick={onAddToQueue}
        disabled={!canSubmit || isQueueing}
      >
        {isQueueing ? "Adding..." : "Add to Queue"}
      </button>
    </section>
  );
}
