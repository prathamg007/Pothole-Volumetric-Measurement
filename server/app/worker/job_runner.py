"""Async job runner: acquires the pipeline lock, runs process_video on a worker thread, updates status."""
import asyncio
import json
import traceback
from pathlib import Path

from app.utils.logger import get_logger
from app.worker.job_store import JobStore
from app.worker.models_registry import ModelRegistry
from app.worker.pipeline import process_video

log = get_logger("job_runner")


async def run_job(
    job_id: str,
    input_path: Path,
    output_dir: Path,
    cfg: dict,
    models: ModelRegistry,
    store: JobStore,
    lock: asyncio.Lock,
) -> None:
    async with lock:
        log.info(f"[{job_id}] starting")
        store.mark_processing(job_id)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video = output_dir / "annotated.mp4"
            output_json = output_dir / "report.json"

            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(
                None,
                _run_pipeline,
                input_path,
                output_video,
                cfg,
                models,
            )

            output_json.write_text(json.dumps(report, indent=2))
            store.mark_completed(job_id, output_video, output_json)
            log.info(f"[{job_id}] done")
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            log.error(f"[{job_id}] failed: {err}")
            store.mark_failed(job_id, err)


def _run_pipeline(input_path, output_video, cfg, models):
    return process_video(input_path, output_video, cfg, models=models)
