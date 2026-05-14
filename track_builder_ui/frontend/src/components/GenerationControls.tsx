import React from "react";
import type { GenerationMode, RunnerConfig } from "../types";

type GenerationControlsProps = {
  mode: GenerationMode;
  prompt: string;
  seed: number;
  textGuidanceWeight: number;
  motionGuidanceWeight: number;
  trackLatentFirstFrameScale: number;
  trackLatentRestFrameScale: number;
  runnerConfig: RunnerConfig | null;
  runnerConfigError: string;
  isQueueing: boolean;
  canSubmit: boolean;
  onModeChange: (mode: GenerationMode) => void;
  onPromptChange: (prompt: string) => void;
  onSeedChange: (seed: number) => void;
  onTextGuidanceWeightChange: (weight: number) => void;
  onMotionGuidanceWeightChange: (weight: number) => void;
  onTrackLatentFirstFrameScaleChange: (scale: number) => void;
  onTrackLatentRestFrameScaleChange: (scale: number) => void;
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

function trackConditionModeLabel(mode: RunnerConfig["track_condition_mode"]): string {
  return mode === "wan_move" ? "Wan-Move" : "Track Head";
}

function wanMoveStrideLabel(value: string): string {
  const trimmed = value.trim();
  return trimmed && trimmed !== "0" ? trimmed : "auto";
}

export default function GenerationControls({
  mode,
  prompt,
  seed,
  textGuidanceWeight,
  motionGuidanceWeight,
  trackLatentFirstFrameScale,
  trackLatentRestFrameScale,
  runnerConfig,
  runnerConfigError,
  isQueueing,
  canSubmit,
  onModeChange,
  onPromptChange,
  onSeedChange,
  onTextGuidanceWeightChange,
  onMotionGuidanceWeightChange,
  onTrackLatentFirstFrameScaleChange,
  onTrackLatentRestFrameScaleChange,
  onAddToQueue
}: GenerationControlsProps): JSX.Element {
  const trackScaleDisabled = isQueueing || mode === "text_only";

  return (
    <section className="control-section">
      <h3>4) Generation</h3>
      <div className="runner-config">
        {runnerConfig ? (
          <>
            <div className="runner-config-row">
              <span className="runner-config-label">Backend</span>
              <span className="runner-config-value">
                {trackConditionModeLabel(runnerConfig.track_condition_mode)}
              </span>
            </div>
            <div className="runner-config-row">
              <span className="runner-config-label">Checkpoint</span>
              <span
                className="runner-config-value monospace"
                title={runnerConfig.transformer_checkpoint_path}
              >
                {runnerConfig.checkpoint_label || runnerConfig.transformer_checkpoint_path}
              </span>
            </div>
            <div className="runner-config-row">
              <span className="runner-config-label">Runtime</span>
              <span className="runner-config-value">
                {runnerConfig.runner_mode} / GPU {runnerConfig.cuda_visible_devices || "-"}
                {" / "}
                {runnerConfig.track_max_points === "-1"
                  ? "all pts"
                  : `${runnerConfig.track_max_points} pts`}
                {runnerConfig.track_condition_mode === "wan_move"
                  ? ` / stride ${wanMoveStrideLabel(runnerConfig.wan_move_temporal_stride)}`
                  : null}
              </span>
            </div>
          </>
        ) : (
          <div className="path-info">
            {runnerConfigError ? `Runner config unavailable: ${runnerConfigError}` : "Loading runner config..."}
          </div>
        )}
      </div>
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
      <div className="control-row">
        <label>
          First Scale
          <input
            className="numeric-input"
            type="number"
            min={0}
            step={0.1}
            value={trackLatentFirstFrameScale}
            onChange={(event) => onTrackLatentFirstFrameScaleChange(Number(event.target.value))}
            disabled={trackScaleDisabled}
          />
        </label>
        <label>
          Rest Scale
          <input
            className="numeric-input"
            type="number"
            min={0}
            step={0.1}
            value={trackLatentRestFrameScale}
            onChange={(event) => onTrackLatentRestFrameScaleChange(Number(event.target.value))}
            disabled={trackScaleDisabled}
          />
        </label>
      </div>
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
