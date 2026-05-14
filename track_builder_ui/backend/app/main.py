import io
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .job_runner import JobRunner
from .job_store import JobStore
from .schemas import (
    ImageCaptionResponse,
    SaveTrackRequest,
    SaveTrackResponse,
    TrackDocument,
)

BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
TRACK_DIR = DATA_DIR / "tracks"
IMAGE_DIR = DATA_DIR / "images"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
JOBS_ROOT = Path(
    os.environ.get(
        "TRACK_BUILDER_JOBS_ROOT",
        str(REPO_ROOT / "asset" / "track_builder_jobs"),
    )
)

TRACK_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Track Builder UI API", version="0.1.0")
job_store = JobStore(JOBS_ROOT)
job_runner = JobRunner(job_store, REPO_ROOT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

FLORENCE_MODEL_NAME = "microsoft/Florence-2-base"
DEFAULT_FLORENCE_TASK = "<MORE_DETAILED_CAPTION>"

_caption_model: Any | None = None
_caption_processor: Any | None = None
_caption_device: str | None = None
_caption_dtype: Any | None = None


@app.on_event("startup")
def start_job_runner() -> None:
    job_runner.start()


def _normalize_generation_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "joint":
        normalized = "joint_tm"
    if normalized not in {"motion_only", "text_only", "joint_tm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported generation mode: {mode}")
    return normalized


def _default_guidance_weights(mode: str) -> tuple[float, float]:
    if mode == "text_only":
        return 3.0, 0.0
    if mode == "joint_tm":
        return 3.0, 1.5
    return 0.0, 3.0


def _validate_nonnegative_finite(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise HTTPException(status_code=400, detail=f"{name} must be finite and >= 0")
    return parsed


def _get_florence_runtime() -> tuple[Any, Any, str, Any]:
    global _caption_model, _caption_processor, _caption_device, _caption_dtype

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if _caption_model is not None and _caption_processor is not None:
        return _caption_model, _caption_processor, _caption_device or "cpu", _caption_dtype or torch.float32

    _caption_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _caption_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _caption_model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_NAME,
        torch_dtype=_caption_dtype,
        trust_remote_code=True,
    ).to(_caption_device)
    _caption_processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_NAME,
        trust_remote_code=True,
    )
    return _caption_model, _caption_processor, _caption_device, _caption_dtype


def _extract_caption_text(parsed_answer: Any) -> str:
    if isinstance(parsed_answer, str):
        return parsed_answer
    if isinstance(parsed_answer, dict):
        for key in ("caption", "text", "answer"):
            value = parsed_answer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in parsed_answer.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(parsed_answer)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runner/config")
def runner_config() -> dict:
    try:
        return {"runner": job_runner.config_snapshot()}
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "image.png").suffix or ".png"
    image_id = f"img_{uuid4().hex}"
    output_path = IMAGE_DIR / f"{image_id}{suffix}"

    with output_path.open("wb") as f:
        f.write(await file.read())

    return {
        "image_id": image_id,
        "url": f"/static/images/{output_path.name}"
    }


@app.post("/api/tracks", response_model=SaveTrackResponse)
def save_track(payload: SaveTrackRequest) -> SaveTrackResponse:
    track_id = payload.track_id or f"track_{uuid4().hex[:12]}"
    output_path = TRACK_DIR / f"{track_id}.json"
    output_path.write_text(
        payload.document.model_dump_json(indent=2),
        encoding="utf-8"
    )
    return SaveTrackResponse(track_id=track_id, file_path=str(output_path))


@app.put("/api/tracks/{track_id}", response_model=SaveTrackResponse)
def update_track(track_id: str, document: TrackDocument) -> SaveTrackResponse:
    output_path = TRACK_DIR / f"{track_id}.json"
    output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return SaveTrackResponse(track_id=track_id, file_path=str(output_path))


@app.get("/api/tracks/{track_id}", response_model=TrackDocument)
def read_track(track_id: str) -> TrackDocument:
    input_path = TRACK_DIR / f"{track_id}.json"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Track not found")

    return TrackDocument.model_validate_json(input_path.read_text(encoding="utf-8"))


@app.get("/api/tracks")
def list_tracks() -> dict[str, list[str]]:
    ids = [path.stem for path in sorted(TRACK_DIR.glob("*.json"))]
    return {"track_ids": ids}


@app.post("/api/export/package")
async def export_package(
    directory: str = Form(...),
    image: UploadFile = File(...),
    tracks_npz: UploadFile = File(...),
    caption: str = Form(""),
    preview_png: UploadFile | None = File(default=None),
) -> dict:
    target_dir = Path(directory)
    if not target_dir.is_absolute():
        target_dir = DATA_DIR / directory
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    image_path = target_dir / "first_frame.png"
    image_path.write_bytes(await image.read())
    saved["image"] = str(image_path)

    tracks_path = target_dir / "transformed_tracks_grid50_survived.npz"
    tracks_path.write_bytes(await tracks_npz.read())
    saved["tracks"] = str(tracks_path)

    if caption.strip():
        caption_path = target_dir / "image_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")
        saved["caption"] = str(caption_path)

    if preview_png is not None:
        preview_bytes = await preview_png.read()
        if preview_bytes:
            preview_path = target_dir / "track_preview.png"
            preview_path.write_bytes(preview_bytes)
            saved["preview"] = str(preview_path)

    return {"status": "ok", "directory": str(target_dir), "saved": saved}


@app.post("/api/jobs")
async def create_generation_job(
    image: UploadFile = File(...),
    tracks_npz: UploadFile = File(...),
    preview_png: UploadFile | None = File(default=None),
    mode: str = Form("motion_only"),
    prompt: str = Form("a video"),
    seed: int = Form(42),
    text_guidance_weight: float | None = Form(default=None),
    motion_guidance_weight: float | None = Form(default=None),
    track_latent_first_frame_scale: float = Form(0.5),
    track_latent_rest_frame_scale: float = Form(1.8),
) -> dict:
    normalized_mode = _normalize_generation_mode(mode)
    default_text_weight, default_motion_weight = _default_guidance_weights(normalized_mode)
    first_frame_scale = _validate_nonnegative_finite(
        "track_latent_first_frame_scale",
        track_latent_first_frame_scale,
    )
    rest_frame_scale = _validate_nonnegative_finite(
        "track_latent_rest_frame_scale",
        track_latent_rest_frame_scale,
    )
    image_bytes = await image.read()
    tracks_bytes = await tracks_npz.read()
    preview_bytes = await preview_png.read() if preview_png is not None else None
    if preview_bytes == b"":
        preview_bytes = None

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty first-frame image")
    if not tracks_bytes:
        raise HTTPException(status_code=400, detail="Empty track npz")

    job = job_store.create_job(
        image_bytes=image_bytes,
        tracks_bytes=tracks_bytes,
        preview_bytes=preview_bytes,
        mode=normalized_mode,
        prompt=prompt,
        seed=seed,
        text_guidance_weight=(
            default_text_weight if text_guidance_weight is None else text_guidance_weight
        ),
        motion_guidance_weight=(
            default_motion_weight if motion_guidance_weight is None else motion_guidance_weight
        ),
        track_latent_first_frame_scale=first_frame_scale,
        track_latent_rest_frame_scale=rest_frame_scale,
    )
    job_runner.notify()
    return {"job": job}


@app.get("/api/jobs")
def list_generation_jobs() -> dict:
    return {"jobs": job_store.list_jobs()}


@app.get("/api/archive")
def list_archive_jobs() -> dict:
    return {"jobs": job_store.list_archive_jobs()}


@app.get("/api/jobs/{job_id}")
def read_generation_job(job_id: str) -> dict:
    try:
        return {"job": job_store.read_job(job_id)}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error


@app.get("/api/jobs/{job_id}/log")
def read_generation_job_log(job_id: str) -> dict:
    try:
        text = job_store.read_log_tail(job_id)
    except FileNotFoundError:
        text = ""
    return {"job_id": job_id, "text": text}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_generation_job(job_id: str) -> dict:
    try:
        job = job_runner.cancel(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    return {"job": job}


@app.post("/api/jobs/{job_id}/retry")
def retry_generation_job(job_id: str) -> dict:
    try:
        job = job_store.retry_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    job_runner.notify()
    return {"job": job}


@app.delete("/api/jobs/{job_id}")
def delete_generation_job(job_id: str) -> dict:
    try:
        job_store.delete_archived_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"deleted": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}/file/{file_path:path}")
def read_generation_job_file(job_id: str, file_path: str) -> FileResponse:
    try:
        path = job_store.resolve_job_file(job_id, file_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job file not found") from error
    return FileResponse(path)


@app.post("/api/images/caption", response_model=ImageCaptionResponse)
async def caption_image(
    file: UploadFile = File(...),
    task: str = Form(DEFAULT_FLORENCE_TASK),
) -> ImageCaptionResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Invalid image file") from error

    try:
        model, processor, device, runtime_dtype = _get_florence_runtime()
        inputs = processor(
            text=task,
            images=pil_image,
            return_tensors="pt",
        ).to(device, runtime_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height),
        )
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {error}") from error

    return ImageCaptionResponse(
        task=task,
        text=_extract_caption_text(parsed_answer),
        raw_output=parsed_answer,
    )


# Serve frontend SPA — must be mounted last so /api/* routes take priority
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
