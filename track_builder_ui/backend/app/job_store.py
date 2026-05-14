import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .schemas import JobInputPaths, JobOutputPaths, JobRecord


ARCHIVE_STATUSES = {"done", "failed", "canceled", "interrupted"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_job_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"job_{timestamp}_{uuid4().hex[:8]}"


class JobStore:
    def __init__(self, jobs_root: Path):
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _validate_job_id(self, job_id: str) -> None:
        if not job_id or job_id in {".", ".."}:
            raise ValueError("Invalid job_id")
        if "/" in job_id or "\\" in job_id:
            raise ValueError("Invalid job_id")

    def job_dir(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        return self.jobs_root / job_id

    def job_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def save_job(self, job: JobRecord) -> JobRecord:
        with self._lock:
            job_dir = self.job_dir(job.job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.job_json_path(job.job_id)
            tmp_path = output_path.with_suffix(".json.tmp")
            tmp_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp_path, output_path)
        return job

    def read_job(self, job_id: str) -> JobRecord:
        with self._lock:
            path = self.job_json_path(job_id)
            if not path.exists():
                raise FileNotFoundError(f"Job not found: {job_id}")
            return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs: list[JobRecord] = []
            for path in sorted(self.jobs_root.glob("*/job.json")):
                try:
                    jobs.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
            jobs.sort(key=lambda item: item.created_at, reverse=True)
            return jobs

    def list_archive_jobs(self) -> list[JobRecord]:
        return [job for job in self.list_jobs() if job.status in ARCHIVE_STATUSES]

    def next_queued_job(self) -> JobRecord | None:
        queued = [job for job in self.list_jobs() if job.status == "queued"]
        queued.sort(key=lambda item: item.created_at)
        return queued[0] if queued else None

    def claim_next_queued_job(self) -> JobRecord | None:
        with self._lock:
            queued: list[JobRecord] = []
            for path in sorted(self.jobs_root.glob("*/job.json")):
                try:
                    job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if job.status == "queued":
                    queued.append(job)
            queued.sort(key=lambda item: item.created_at)
            if not queued:
                return None
            job = queued[0]
            job.status = "running"
            job.started_at = now_iso()
            job.finished_at = None
            job.error_message = None
            job.return_code = None
            job.outputs = JobOutputPaths()
            return self.save_job(job)

    def create_job(
        self,
        *,
        image_bytes: bytes,
        tracks_bytes: bytes,
        preview_bytes: bytes | None,
        mode: str,
        prompt: str,
        seed: int,
        text_guidance_weight: float,
        motion_guidance_weight: float,
        track_latent_first_frame_scale: float,
        track_latent_rest_frame_scale: float,
    ) -> JobRecord:
        job_id = make_job_id()
        job_dir = self.job_dir(job_id)
        input_dir = job_dir / "input"
        (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        image_rel = "input/first_frame.png"
        tracks_rel = "input/transformed_tracks_grid50_survived.npz"
        preview_rel = "input/track_preview.png" if preview_bytes else None

        (job_dir / image_rel).write_bytes(image_bytes)
        (job_dir / tracks_rel).write_bytes(tracks_bytes)
        if preview_bytes and preview_rel:
            (job_dir / preview_rel).write_bytes(preview_bytes)

        prompt_text = prompt.strip() or "a video"
        job = JobRecord(
            job_id=job_id,
            status="queued",
            mode=mode,  # type: ignore[arg-type]
            prompt=prompt_text,
            seed=int(seed),
            text_guidance_weight=float(text_guidance_weight),
            motion_guidance_weight=float(motion_guidance_weight),
            track_latent_first_frame_scale=float(track_latent_first_frame_scale),
            track_latent_rest_frame_scale=float(track_latent_rest_frame_scale),
            created_at=now_iso(),
            input=JobInputPaths(
                image=image_rel,
                tracks=tracks_rel,
                preview=preview_rel,
            ),
            outputs=JobOutputPaths(),
            log_path="logs/run.log",
        )
        return self.save_job(job)

    def retry_job(self, source_job_id: str) -> JobRecord:
        source = self.read_job(source_job_id)
        source_dir = self.job_dir(source_job_id)
        job_id = make_job_id()
        job_dir = self.job_dir(job_id)
        input_dir = job_dir / "input"
        (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        def copy_input(rel_path: str | None) -> str | None:
            if not rel_path:
                return None
            src = source_dir / rel_path
            dst = job_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
                return rel_path
            return None

        image_rel = copy_input(source.input.image)
        tracks_rel = copy_input(source.input.tracks)
        preview_rel = copy_input(source.input.preview)
        if not image_rel or not tracks_rel:
            raise FileNotFoundError("Source job input files are missing")

        job = JobRecord(
            job_id=job_id,
            status="queued",
            mode=source.mode,
            prompt=source.prompt,
            seed=source.seed,
            text_guidance_weight=source.text_guidance_weight,
            motion_guidance_weight=source.motion_guidance_weight,
            track_latent_first_frame_scale=source.track_latent_first_frame_scale,
            track_latent_rest_frame_scale=source.track_latent_rest_frame_scale,
            created_at=now_iso(),
            input=JobInputPaths(
                image=image_rel,
                tracks=tracks_rel,
                preview=preview_rel,
            ),
            outputs=JobOutputPaths(),
            log_path="logs/run.log",
            source_job_id=source.job_id,
        )
        return self.save_job(job)

    def mark_running(self, job_id: str) -> JobRecord:
        job = self.read_job(job_id)
        job.status = "running"
        job.started_at = now_iso()
        job.finished_at = None
        job.error_message = None
        job.return_code = None
        job.outputs = JobOutputPaths()
        return self.save_job(job)

    def discover_outputs(self, job_id: str) -> JobOutputPaths:
        job_dir = self.job_dir(job_id)
        output_dir = job_dir / "outputs"
        videos = sorted(
            output_dir.glob("*.mp4"),
            key=lambda path: path.stat().st_mtime,
        )
        plain = None
        overlay = None
        for path in videos:
            rel = path.relative_to(job_dir).as_posix()
            if path.stem.endswith("_overlay"):
                overlay = rel
            else:
                plain = rel
        return JobOutputPaths(video=plain, overlay_video=overlay)

    def mark_done(self, job_id: str, return_code: int) -> JobRecord:
        job = self.read_job(job_id)
        job.status = "done"
        job.finished_at = now_iso()
        job.return_code = int(return_code)
        job.error_message = None
        job.outputs = self.discover_outputs(job_id)
        return self.save_job(job)

    def mark_failed(self, job_id: str, return_code: int | None, message: str) -> JobRecord:
        job = self.read_job(job_id)
        job.status = "failed"
        job.finished_at = now_iso()
        job.return_code = return_code
        job.error_message = message
        job.outputs = self.discover_outputs(job_id)
        return self.save_job(job)

    def mark_canceled(self, job_id: str, message: str = "Canceled by user") -> JobRecord:
        job = self.read_job(job_id)
        job.status = "canceled"
        job.finished_at = now_iso()
        job.error_message = message
        job.outputs = self.discover_outputs(job_id)
        return self.save_job(job)

    def delete_archived_job(self, job_id: str) -> None:
        with self._lock:
            job = self.read_job(job_id)
            if job.status not in ARCHIVE_STATUSES:
                raise ValueError("Only archived jobs can be deleted.")
            job_dir = self.job_dir(job_id)
            if not job_dir.exists():
                raise FileNotFoundError(f"Job not found: {job_id}")
            shutil.rmtree(job_dir)

    def recover_after_restart(self) -> None:
        for job in self.list_jobs():
            if job.status != "running":
                continue
            job.status = "interrupted"
            job.finished_at = now_iso()
            job.error_message = "Backend restarted while this job was running."
            self.save_job(job)

    def resolve_job_file(self, job_id: str, rel_path: str) -> Path:
        job_dir = self.job_dir(job_id).resolve()
        target = (job_dir / rel_path).resolve()
        if target != job_dir and job_dir not in target.parents:
            raise ValueError("Invalid job file path")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Job file not found: {rel_path}")
        return target

    def read_log_tail(self, job_id: str, max_bytes: int = 40000) -> str:
        job = self.read_job(job_id)
        log_path = self.resolve_job_file(job_id, job.log_path)
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode("utf-8", errors="replace")
