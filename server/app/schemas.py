"""Pydantic schemas for API requests + responses."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    has_video: bool = False
    has_report: bool = False
    input_filename: Optional[str] = None


class PotholeResult(BaseModel):
    track_id: int
    first_frame: int
    last_frame: int
    first_time_s: float
    observations: int
    valid_measurements: int
    confidence: float
    area_cm2: float
    avg_depth_cm: float
    max_depth_cm: float
    volume_cm3: float
    severity_level: str
    severity_score: int
    repair_method: str
    material_name: str
    material_kg: float
    repair_cost: float
    currency: str
    durability_months: int


class VideoInfoDto(BaseModel):
    fps: float
    width: int
    height: int
    frames: int
    duration_s: float


class ProcessingInfoDto(BaseModel):
    wall_s: float
    inference_s: float
    frame_stride: int


class SummaryDto(BaseModel):
    num_potholes: int
    total_area_cm2: float
    total_volume_cm3: float
    total_material_kg: float
    total_cost: float
    currency: str
    total_cracks_detected: int


class AnalysisReport(BaseModel):
    video: VideoInfoDto
    processing: ProcessingInfoDto
    cracks: dict[str, int] = Field(default_factory=dict)
    potholes: list[PotholeResult] = Field(default_factory=list)
    summary: SummaryDto
