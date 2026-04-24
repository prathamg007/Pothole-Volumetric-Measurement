import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import analyze, health
from app.utils.config import load_config
from app.utils.logger import get_logger
from app.worker.job_store import JobStore
from app.worker.models_registry import ModelRegistry

SERVER_ROOT = Path(__file__).resolve().parent.parent
log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up...")
    cfg = load_config()

    jobs = JobStore(SERVER_ROOT / "data" / "jobs.db")
    log.info(f"Job store ready at {jobs.db_path}")

    models = ModelRegistry(cfg)
    # Loading is CPU/IO bound; run in executor so we don't block the event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, models.load_all)

    app.state.cfg = cfg
    app.state.jobs = jobs
    app.state.models = models
    app.state.pipeline_lock = asyncio.Lock()
    log.info("Ready to accept requests")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Road Anomaly Analysis Server",
    description=(
        "Upload a road video, get back detected + classified cracks, segmented potholes with "
        "physical measurements, and repair recommendations."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)


@app.get("/")
def root():
    return {"service": "road-anomaly-analysis", "version": app.version, "docs": "/docs"}
