from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class GridConfig(BaseModel):
    type: Literal["square"] = "square"
    spacing: float = Field(default=50.0, ge=1.0)
    offsetX: float = 0.0
    offsetY: float = 0.0
    visible: bool = True


class TrackPath(BaseModel):
    id: str
    name: str
    points: list[Point] = Field(default_factory=list)
    keyframes: list[list[Point]] | None = None
    trackMode: Literal["moving", "static"] = "moving"
    pointMode: str | None = None
    closed: bool = False
    color: str = "#ff5f7a"


class ImageInfo(BaseModel):
    src: str
    width: int
    height: int


class MetaInfo(BaseModel):
    createdAt: datetime
    updatedAt: datetime


class TrackDocument(BaseModel):
    version: Literal["0.1"] = "0.1"
    image: ImageInfo
    grid: GridConfig
    paths: list[TrackPath] = Field(default_factory=list)
    meta: MetaInfo


class SaveTrackRequest(BaseModel):
    track_id: str | None = None
    document: TrackDocument


class SaveTrackResponse(BaseModel):
    track_id: str
    file_path: str


class ImageCaptionResponse(BaseModel):
    task: str
    text: str
    raw_output: dict | list | str | None = None


JobStatus = Literal[
    "queued",
    "running",
    "done",
    "failed",
    "canceled",
    "interrupted",
]

GenerationMode = Literal["motion_only", "text_only", "joint_tm"]


class JobInputPaths(BaseModel):
    image: str
    tracks: str
    preview: str | None = None


class JobOutputPaths(BaseModel):
    video: str | None = None
    overlay_video: str | None = None


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    mode: GenerationMode = "motion_only"
    prompt: str = "a video"
    seed: int = 42
    text_guidance_weight: float = 0.0
    motion_guidance_weight: float = 3.0
    track_latent_first_frame_scale: float = 1.0
    track_latent_rest_frame_scale: float = 4.0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    input: JobInputPaths
    outputs: JobOutputPaths = Field(default_factory=JobOutputPaths)
    log_path: str = "logs/run.log"
    error_message: str | None = None
    return_code: int | None = None
    source_job_id: str | None = None
