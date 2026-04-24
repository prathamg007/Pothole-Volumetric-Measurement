"""HTTP endpoints: upload video, poll job, download results."""
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.utils.logger import get_logger
from app.worker.job_runner import run_job

log = get_logger("routes.analyze")

router = APIRouter(tags=["analyze"])

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_ROOT = SERVER_ROOT / "data" / "uploads"
RESULTS_ROOT = SERVER_ROOT / "data" / "results"


def _state_dict(job: dict) -> dict:
    has_video = bool(job.get("output_video_path")) and Path(job["output_video_path"]).exists()
    has_report = bool(job.get("output_report_path")) and Path(job["output_report_path"]).exists()
    return {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error_message": job.get("error_message"),
        "input_filename": job.get("input_filename"),
        "has_video": has_video,
        "has_report": has_report,
    }


@router.post("/analyze")
async def create_analysis(
    request: Request,
    background: BackgroundTasks,
    video: UploadFile = File(...),
    sensors: Optional[UploadFile] = File(None),
    meta: Optional[UploadFile] = File(None),
):
    app_state = request.app.state

    max_mb = app_state.cfg["server"].get("max_upload_mb", 200)
    # NOTE: UploadFile size isn't reliably populated. Enforce limit while streaming to disk.

    job_id = str(uuid.uuid4())
    upload_dir = UPLOADS_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(video.filename or "video.mp4").name
    suffix = Path(filename).suffix or ".mp4"
    input_path = upload_dir / f"video{suffix}"

    size_written = 0
    max_bytes = max_mb * 1024 * 1024
    with input_path.open("wb") as dst:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size_written += len(chunk)
            if size_written > max_bytes:
                dst.close()
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(413, f"Upload exceeds {max_mb} MB")
            dst.write(chunk)

    if sensors is not None and sensors.filename:
        (upload_dir / "sensors.json").write_bytes(await sensors.read())
    if meta is not None and meta.filename:
        (upload_dir / "meta.json").write_bytes(await meta.read())

    app_state.jobs.create(job_id, input_path, filename)
    results_dir = RESULTS_ROOT / job_id

    background.add_task(
        run_job,
        job_id,
        input_path,
        results_dir,
        app_state.cfg,
        app_state.models,
        app_state.jobs,
        app_state.pipeline_lock,
    )

    log.info(f"queued job {job_id} ({filename}, {size_written/1024/1024:.1f} MB)")
    return {"job_id": job_id, "status": "queued", "bytes": size_written}


@router.get("/jobs")
async def list_jobs(request: Request, limit: int = 20):
    return request.app.state.jobs.list_all(limit=limit)


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return _state_dict(job)


@router.get("/jobs/{job_id}/result")
async def get_result(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']} — no result yet")
    path = Path(job["output_report_path"] or "")
    if not path.exists():
        raise HTTPException(500, "Report file missing on disk")
    return JSONResponse(content=_read_json(path))


@router.get("/jobs/{job_id}/video")
async def get_video(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']} — no video yet")
    path = Path(job["output_video_path"] or "")
    if not path.exists():
        raise HTTPException(500, "Video file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


def _read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
