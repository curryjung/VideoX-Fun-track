import os
import subprocess
import threading
import time
from pathlib import Path

from .job_store import JobStore
from .schemas import JobRecord


DEFAULT_EXP_NAME = (
    "wan_track_patch-copy_first_gain-1.0_scale-4.0_"
    "track_local-point-id_bs64_train_78k_h_dim_64_all-dropouts_0p1"
)
DEFAULT_CKPT = "10000"


class JobRunner:
    def __init__(self, store: JobStore, repo_root: Path):
        self.store = store
        self.repo_root = Path(repo_root)
        self.mode = os.environ.get("TRACK_BUILDER_RUNNER_MODE", "subprocess").strip().lower()
        self._condition = threading.Condition()
        self._started = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._current_job_id: str | None = None
        self._cancel_requested: set[str] = set()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            if self.mode in {"external", "persistent", "persistent_external"}:
                print(
                    "[job_runner] external runner mode enabled; "
                    "FastAPI will enqueue jobs but will not execute generation."
                )
                self._condition.notify_all()
                return
            self.store.recover_after_restart()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()
            self._condition.notify_all()

    def notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def cancel(self, job_id: str) -> JobRecord:
        job = self.store.read_job(job_id)
        if job.status == "queued":
            canceled = self.store.mark_canceled(job_id)
            self.notify()
            return canceled
        if job.status != "running":
            return job

        if self.mode in {"external", "persistent", "persistent_external"}:
            canceled = self.store.mark_canceled(
                job_id,
                "Cancel requested by user. External worker may finish its current generation.",
            )
            self.notify()
            return canceled

        with self._condition:
            self._cancel_requested.add(job_id)
            if self._current_job_id == job_id and self._process is not None:
                if self._process.poll() is None:
                    self._process.terminate()
            canceled = self.store.mark_canceled(job_id, "Cancel requested by user")
            self._condition.notify_all()
            return canceled

    def _worker_loop(self) -> None:
        while True:
            job = self.store.next_queued_job()
            if job is None:
                with self._condition:
                    self._condition.wait(timeout=2.0)
                continue
            self._run_job(job)

    def _checkpoint_path(self) -> str:
        explicit = os.environ.get("TRACK_BUILDER_TRANSFORMER_CHECKPOINT_PATH", "").strip()
        if explicit:
            return explicit
        exp_name = os.environ.get("TRACK_BUILDER_EXP_NAME", DEFAULT_EXP_NAME).strip() or DEFAULT_EXP_NAME
        ckpt = os.environ.get("TRACK_BUILDER_CKPT", DEFAULT_CKPT).strip() or DEFAULT_CKPT
        return str(self.repo_root / "checkpoints" / exp_name / f"checkpoint-{ckpt}")

    def _build_env(self, job: JobRecord) -> dict[str, str]:
        job_dir = self.store.job_dir(job.job_id)
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": os.environ.get("TRACK_BUILDER_PYTHON_BIN", env.get("PYTHON_BIN", "python")),
                "CUDA_VISIBLE_DEVICES": os.environ.get("TRACK_BUILDER_CUDA_VISIBLE_DEVICES", "6"),
                "TRANSFORMER_CHECKPOINT_PATH": self._checkpoint_path(),
                "VALIDATION_IMAGE_START": str(job_dir / job.input.image),
                "TRACK_FILE_PATH": str(job_dir / job.input.tracks),
                "PROMPT": job.prompt.strip() or "a video",
                "SAVE_DIR": str(job_dir / "outputs"),
                "OUTPUT_NAME_SUFFIX": f"{job.mode}_seed{job.seed}_{job.job_id}",
                "GUIDANCE_MODE": job.mode,
                "TEXT_GUIDANCE_WEIGHT": str(job.text_guidance_weight),
                "MOTION_GUIDANCE_WEIGHT": str(job.motion_guidance_weight),
                "TRACK_NORMALIZE": "true",
                "TRACK_HEAD_HIDDEN_DIM": os.environ.get("TRACK_BUILDER_TRACK_HEAD_HIDDEN_DIM", "64"),
                "TRACK_LATENT_SCALE": os.environ.get("TRACK_BUILDER_TRACK_LATENT_SCALE", "4.0"),
                "TRACK_MAX_POINTS": os.environ.get("TRACK_BUILDER_TRACK_MAX_POINTS", "2000"),
                "TRACK_POINT_SAMPLE_MODE": os.environ.get("TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE", "random"),
                "TRACK_SORT_SELECTED_INDICES": os.environ.get("TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES", "false"),
                "TRACK_POINT_ID_MODE": os.environ.get("TRACK_BUILDER_TRACK_POINT_ID_MODE", "local"),
                "TRACK_POINT_SAMPLE_SEED": str(job.seed),
                "SEED": str(job.seed),
                "DEBUG_TRACK_CONDITION": os.environ.get("TRACK_BUILDER_DEBUG_TRACK_CONDITION", "false"),
                "TRACK_ANALYSIS": os.environ.get("TRACK_BUILDER_TRACK_ANALYSIS", "false"),
                "OVERLAY_PAD_VALUE": os.environ.get("TRACK_BUILDER_OVERLAY_PAD_VALUE", "0"),
                "OVERLAY_LINEWIDTH": os.environ.get("TRACK_BUILDER_OVERLAY_LINEWIDTH", "1"),
                "OVERLAY_TRACE_FRAMES": os.environ.get("TRACK_BUILDER_OVERLAY_TRACE_FRAMES", "8"),
                "COTRACKER_ROOT": os.environ.get(
                    "TRACK_BUILDER_COTRACKER_ROOT",
                    "/data/project-vilab/jaeseok/co-tracker",
                ),
            }
        )
        return env

    def _run_job(self, job: JobRecord) -> None:
        try:
            job = self.store.mark_running(job.job_id)
            job_dir = self.store.job_dir(job.job_id)
            (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
            log_path = job_dir / job.log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = ["bash", "examples/wan2.1_fun_track/run_predict_i2v_track.sh"]
            env = self._build_env(job)

            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"[job_runner] job_id={job.job_id}\n")
                log_file.write(f"[job_runner] mode={job.mode}\n")
                log_file.write(f"[job_runner] command={' '.join(command)}\n")
                log_file.write(f"[job_runner] checkpoint={env.get('TRANSFORMER_CHECKPOINT_PATH', '')}\n")
                log_file.write(f"[job_runner] save_dir={env.get('SAVE_DIR', '')}\n")
                log_file.flush()

                process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._condition:
                    self._process = process
                    self._current_job_id = job.job_id

                while process.poll() is None:
                    time.sleep(1.0)

                return_code = int(process.returncode or 0)

            with self._condition:
                self._process = None
                self._current_job_id = None

            latest = self.store.read_job(job.job_id)
            if latest.status == "canceled" or job.job_id in self._cancel_requested:
                self._cancel_requested.discard(job.job_id)
                self.store.mark_canceled(job.job_id)
            elif return_code == 0:
                self.store.mark_done(job.job_id, return_code)
            else:
                self.store.mark_failed(
                    job.job_id,
                    return_code,
                    f"Generation process exited with code {return_code}.",
                )
        except Exception as error:  # noqa: BLE001
            with self._condition:
                self._process = None
                self._current_job_id = None
            self.store.mark_failed(job.job_id, None, str(error))
        finally:
            self.notify()
