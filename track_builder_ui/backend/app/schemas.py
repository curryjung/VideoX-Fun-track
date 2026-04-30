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
