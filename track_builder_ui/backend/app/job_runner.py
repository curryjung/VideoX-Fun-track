import os
import subprocess
import threading
import time
from pathlib import Path

from .job_store import JobStore
from .schemas import JobRecord


DEFAULT_TRACK_HEAD_EXP_NAME = (
    "wan_track_init-proud-sea-57-ckpt10000_ff-scale-0.5_rf-scale-1.8_openvid-0p6m_wisa-80k"
)
DEFAULT_TRACK_HEAD_CKPT = "12600"
DEFAULT_WAN_MOVE_EXP_NAME = (
    "wan_track_wan_move_condition_bin8_train_78k_dropout_first-frame_0p1_text_0p1_track_0p1"
)
DEFAULT_WAN_MOVE_CKPT = "11800"
DEFAULT_EXP_NAME = DEFAULT_TRACK_HEAD_EXP_NAME
DEFAULT_CKPT = DEFAULT_TRACK_HEAD_CKPT
VALID_TRACK_CONDITION_MODES = {"track_head", "wan_move"}


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

    def _track_condition_mode(self) -> str:
        mode = (
            os.environ.get("TRACK_BUILDER_TRACK_CONDITION_MODE", "track_head")
            .strip()
            .lower()
            .replace("-", "_")
        )
        mode = mode or "track_head"
        if mode not in VALID_TRACK_CONDITION_MODES:
            valid_modes = ", ".join(sorted(VALID_TRACK_CONDITION_MODES))
            raise ValueError(
                "TRACK_BUILDER_TRACK_CONDITION_MODE must be one of "
                f"{valid_modes}; got {mode!r}."
            )
        return mode

    def _checkpoint_path(self, track_condition_mode: str) -> str:
        explicit = os.environ.get("TRACK_BUILDER_TRANSFORMER_CHECKPOINT_PATH", "").strip()
        if explicit:
            return explicit

        if track_condition_mode == "wan_move":
            default_exp_name = DEFAULT_WAN_MOVE_EXP_NAME
            default_ckpt = DEFAULT_WAN_MOVE_CKPT
            mode_exp_env = "TRACK_BUILDER_WAN_MOVE_EXP_NAME"
            mode_ckpt_env = "TRACK_BUILDER_WAN_MOVE_CKPT"
        else:
            default_exp_name = DEFAULT_TRACK_HEAD_EXP_NAME
            default_ckpt = DEFAULT_TRACK_HEAD_CKPT
            mode_exp_env = "TRACK_BUILDER_TRACK_HEAD_EXP_NAME"
            mode_ckpt_env = "TRACK_BUILDER_TRACK_HEAD_CKPT"

        exp_name = (
            os.environ.get("TRACK_BUILDER_EXP_NAME", "").strip()
            or os.environ.get(mode_exp_env, "").strip()
            or default_exp_name
        )
        ckpt = (
            os.environ.get("TRACK_BUILDER_CKPT", "").strip()
            or os.environ.get(mode_ckpt_env, "").strip()
            or default_ckpt
        )
        return str(self.repo_root / "checkpoints" / exp_name / f"checkpoint-{ckpt}")

    def config_snapshot(self) -> dict[str, str]:
        track_condition_mode = self._track_condition_mode()
        is_wan_move = track_condition_mode == "wan_move"
        checkpoint_path = self._checkpoint_path(track_condition_mode)
        checkpoint = Path(checkpoint_path)
        checkpoint_label = checkpoint.name
        if checkpoint.parent.name:
            checkpoint_label = f"{checkpoint.parent.name}/{checkpoint.name}"

        return {
            "runner_mode": self.mode,
            "track_condition_mode": track_condition_mode,
            "transformer_checkpoint_path": checkpoint_path,
            "checkpoint_label": checkpoint_label,
            "cuda_visible_devices": os.environ.get("TRACK_BUILDER_CUDA_VISIBLE_DEVICES", "6"),
            "wan_move_temporal_stride": os.environ.get(
                "TRACK_BUILDER_WAN_MOVE_TEMPORAL_STRIDE",
                "0" if is_wan_move else "",
            ),
            "track_max_points": os.environ.get(
                "TRACK_BUILDER_TRACK_MAX_POINTS",
                "1500" if is_wan_move else "2000",
            ),
            "track_point_sample_mode": os.environ.get(
                "TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE",
                "random",
            ),
            "track_sort_selected_indices": os.environ.get(
                "TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES",
                "true" if is_wan_move else "false",
            ),
            "track_point_id_mode": os.environ.get(
                "TRACK_BUILDER_TRACK_POINT_ID_MODE",
                "original" if is_wan_move else "local",
            ),
        }

    def _build_env(self, job: JobRecord) -> dict[str, str]:
        job_dir = self.store.job_dir(job.job_id)
        track_condition_mode = self._track_condition_mode()
        is_wan_move = track_condition_mode == "wan_move"
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": os.environ.get("TRACK_BUILDER_PYTHON_BIN", env.get("PYTHON_BIN", "python")),
                "CUDA_VISIBLE_DEVICES": os.environ.get("TRACK_BUILDER_CUDA_VISIBLE_DEVICES", "6"),
                "TRANSFORMER_CHECKPOINT_PATH": self._checkpoint_path(track_condition_mode),
                "VALIDATION_IMAGE_START": str(job_dir / job.input.image),
                "TRACK_FILE_PATH": str(job_dir / job.input.tracks),
                "PROMPT": job.prompt.strip() or "a video",
                "SAVE_DIR": str(job_dir / "outputs"),
                "OUTPUT_NAME_SUFFIX": f"{job.mode}_seed{job.seed}_{job.job_id}",
                "GUIDANCE_MODE": job.mode,
                "TEXT_GUIDANCE_WEIGHT": str(job.text_guidance_weight),
                "MOTION_GUIDANCE_WEIGHT": str(job.motion_guidance_weight),
                "TRACK_NORMALIZE": "true",
                "TRACK_CONDITION_MODE": track_condition_mode,
                "WAN_MOVE_TEMPORAL_STRIDE": os.environ.get(
                    "TRACK_BUILDER_WAN_MOVE_TEMPORAL_STRIDE",
                    "0" if is_wan_move else "",
                ),
                "TRACK_HEAD_HIDDEN_DIM": (
                    "" if is_wan_move else os.environ.get("TRACK_BUILDER_TRACK_HEAD_HIDDEN_DIM", "64")
                ),
                "TRACK_LATENT_SCALE": "" if is_wan_move else str(job.track_latent_rest_frame_scale),
                "TRACK_LATENT_FIRST_FRAME_SCALE": (
                    "" if is_wan_move else str(job.track_latent_first_frame_scale)
                ),
                "TRACK_LATENT_REST_FRAME_SCALE": (
                    "" if is_wan_move else str(job.track_latent_rest_frame_scale)
                ),
                "TRACK_MAX_POINTS": os.environ.get(
                    "TRACK_BUILDER_TRACK_MAX_POINTS",
                    "1500" if is_wan_move else "2000",
                ),
                "TRACK_POINT_SAMPLE_MODE": os.environ.get("TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE", "random"),
                "TRACK_SORT_SELECTED_INDICES": os.environ.get(
                    "TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES",
                    "true" if is_wan_move else "false",
                ),
                "TRACK_POINT_ID_MODE": os.environ.get(
                    "TRACK_BUILDER_TRACK_POINT_ID_MODE",
                    "original" if is_wan_move else "local",
                ),
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
                log_file.write(f"[job_runner] track_condition_mode={env.get('TRACK_CONDITION_MODE', '')}\n")
                log_file.write(
                    "[job_runner] track_sampling="
                    f"max_points={env.get('TRACK_MAX_POINTS', '')} "
                    f"mode={env.get('TRACK_POINT_SAMPLE_MODE', '')} "
                    f"sort={env.get('TRACK_SORT_SELECTED_INDICES', '')} "
                    f"point_id={env.get('TRACK_POINT_ID_MODE', '')}\n"
                )
                if env.get("TRACK_CONDITION_MODE") == "wan_move":
                    log_file.write(
                        "[job_runner] wan_move_temporal_stride="
                        f"{env.get('WAN_MOVE_TEMPORAL_STRIDE', '') or '<auto>'}\n"
                    )
                log_file.write(
                    "[job_runner] track_latent_scale="
                    f"first={env.get('TRACK_LATENT_FIRST_FRAME_SCALE', '')} "
                    f"rest={env.get('TRACK_LATENT_REST_FRAME_SCALE', '')}\n"
                )
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
