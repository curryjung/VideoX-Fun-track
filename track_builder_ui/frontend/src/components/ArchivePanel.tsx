import React from "react";
import type { JobRecord } from "../types";

type ArchivePanelProps = {
  jobs: JobRecord[];
  onRetry: (jobId: string) => void;
  onDelete: (jobId: string) => void;
  buildFileUrl: (job: JobRecord, relPath: string) => string;
};

function modeLabel(mode: JobRecord["mode"]): string {
  if (mode === "joint_tm") {
    return "Joint";
  }
  return mode === "motion_only" ? "Motion-only" : "Text-only";
}

function formatTime(value?: string | null): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function scaleLabel(job: JobRecord): string {
  return `f${job.track_latent_first_frame_scale} / r${job.track_latent_rest_frame_scale}`;
}

export default function ArchivePanel({
  jobs,
  onRetry,
  onDelete,
  buildFileUrl
}: ArchivePanelProps): JSX.Element {
  const archiveJobs = jobs.filter((job) =>
    job.status === "done" ||
    job.status === "failed" ||
    job.status === "canceled" ||
    job.status === "interrupted"
  );

  return (
    <section className="archive-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Archive</div>
          <div className="path-info">{archiveJobs.length} saved jobs</div>
        </div>
      </div>

      {archiveJobs.length === 0 ? (
        <div className="empty-state">Completed jobs will stay here after restart</div>
      ) : (
        <div className="archive-list">
          {archiveJobs.map((job) => {
            const videoPath = job.outputs.video ?? null;
            const overlayPath = job.outputs.overlay_video ?? null;
            const hasVideo = Boolean(videoPath || overlayPath);
            return (
              <article key={job.job_id} className="archive-item">
                {hasVideo ? (
                  <div className="archive-media-grid">
                    {videoPath ? (
                      <div className="archive-media-block">
                        <div className="archive-media-label">Generated</div>
                        <video
                          className="archive-video"
                          src={buildFileUrl(job, videoPath)}
                          controls
                          preload="metadata"
                        />
                      </div>
                    ) : null}
                    {overlayPath ? (
                      <div className="archive-media-block">
                        <div className="archive-media-label">Overlap</div>
                        <video
                          className="archive-video"
                          src={buildFileUrl(job, overlayPath)}
                          controls
                          preload="metadata"
                        />
                      </div>
                    ) : null}
                  </div>
                ) : job.input.preview ? (
                  <img
                    className="archive-preview"
                    src={buildFileUrl(job, job.input.preview)}
                    alt=""
                  />
                ) : null}
                <div className="archive-meta">
                  <div className="archive-title">
                    <span>{modeLabel(job.mode)}</span>
                    <span className={`status-pill ${job.status}`}>{job.status}</span>
                  </div>
                  <div className="path-info">
                    {formatTime(job.finished_at ?? job.created_at)} / seed {job.seed} / scale {scaleLabel(job)}
                  </div>
                  <div className="archive-prompt">{job.prompt || "a video"}</div>
                  {job.error_message ? (
                    <div className="error-text">{job.error_message}</div>
                  ) : null}
                  <div className="control-row">
                    <button type="button" onClick={() => onRetry(job.job_id)}>
                      Re-run
                    </button>
                    <button
                      type="button"
                      className="danger-button"
                      onClick={() => onDelete(job.job_id)}
                    >
                      Delete
                    </button>
                    {videoPath ? (
                      <a
                        className="btn"
                        href={buildFileUrl(job, videoPath)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Video
                      </a>
                    ) : null}
                    {job.outputs.overlay_video ? (
                      <a
                        className="btn"
                        href={buildFileUrl(job, job.outputs.overlay_video)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Overlap
                      </a>
                    ) : null}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
