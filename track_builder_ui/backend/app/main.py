import io
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoProcessor

from .schemas import (
    ImageCaptionResponse,
    SaveTrackRequest,
    SaveTrackResponse,
    TrackDocument,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TRACK_DIR = DATA_DIR / "tracks"
IMAGE_DIR = DATA_DIR / "images"

TRACK_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Track Builder UI API", version="0.1.0")

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

_caption_model: AutoModelForCausalLM | None = None
_caption_processor: AutoProcessor | None = None
_caption_device: str | None = None
_caption_dtype: torch.dtype | None = None


def _get_florence_runtime() -> tuple[AutoModelForCausalLM, AutoProcessor, str, torch.dtype]:
    global _caption_model, _caption_processor, _caption_device, _caption_dtype

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
