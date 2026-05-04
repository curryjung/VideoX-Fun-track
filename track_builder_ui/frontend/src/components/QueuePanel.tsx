import React from "react";
import type { JobRecord, JobStatus } from "../types";

type QueuePanelProps = {
  jobs: JobRecord[];
  selectedJobId: string | null;
  selectedLog: string;
  onSelectJob: (jobId: string) => void;
  onRefresh: () => void;
  onCancel: (jobId: string) => void;
  onRetry: (jobId: string) => void;
  buildFileUrl: (job: JobRecord, relPath: string) => string;
};

function statusLabel(status: JobStatus): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function modeLabel(mode: JobRecord["mode"]): string {
  if (mode === "joint_tm") {
    return "Joint";
  }
  return mode === "motion_only" ? "Motion-only" : "Text-only";
}

function shortPrompt(prompt: string): string {
  const trimmed = prompt.trim();
  if (trimmed.length <= 72) {
    return trimmed || "a video";
  }
  return `${trimmed.slice(0, 72)}...`;
}

export default function QueuePanel({
  jobs,
  selectedJobId,
  selectedLog,
  onSelectJob,
  onRefresh,
  onCancel,
  onRetry,
  buildFileUrl
}: QueuePanelProps): JSX.Element {
  const activeJobs = jobs.filter((job) => job.status === "queued" || job.status === "running");
  const selectedJob = jobs.find((job) => job.job_id === selectedJobId) ?? activeJobs[0] ?? null;

  return (
    <section className="queue-panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Queue</div>
          <div className="path-info">{activeJobs.length} waiting/running</div>
        </div>
        <button type="button" onClick={onRefresh}>Refresh</button>
      </div>

      {activeJobs.length === 0 ? (
        <div className="empty-state">No queued jobs</div>
      ) : (
        <div className="job-list">
          {activeJobs.map((job) => (
            <button
              key={job.job_id}
              type="button"
              className={job.job_id === selectedJob?.job_id ? "job-item active" : "job-item"}
              onClick={() => onSelectJob(job.job_id)}
            >
              {job.input.preview ? (
                <img
                  src={buildFileUrl(job, job.input.preview)}
                  alt=""
                  className="job-thumb"
                />
              ) : null}
              <span className="job-main">
                <span className="job-title">{modeLabel(job.mode)} / seed {job.seed}</span>
                <span className="job-subtitle">{shortPrompt(job.prompt)}</span>
              </span>
              <span className={`status-pill ${job.status}`}>{statusLabel(job.status)}</span>
            </button>
          ))}
        </div>
      )}

      {selectedJob ? (
        <div className="job-detail">
          <div className="job-detail-title">{selectedJob.job_id}</div>
          <div className="control-row">
            {(selectedJob.status === "queued" || selectedJob.status === "running") ? (
              <button type="button" onClick={() => onCancel(selectedJob.job_id)}>
                Cancel
              </button>
            ) : null}
            {selectedJob.status !== "queued" && selectedJob.status !== "running" ? (
              <button type="button" onClick={() => onRetry(selectedJob.job_id)}>
                Retry
              </button>
            ) : null}
          </div>
          <pre className="log-tail">{selectedLog || "Log will appear after the job starts."}</pre>
        </div>
      ) : null}
    </section>
  );
}
