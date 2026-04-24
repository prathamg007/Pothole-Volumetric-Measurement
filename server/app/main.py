from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health

SERVER_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Road Anomaly Analysis Server",
    description="Upload a road video, get back detected + classified cracks, segmented potholes with physical measurements, and repair recommendations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)


@app.get("/")
def root():
    return {
        "service": "road-anomaly-analysis",
        "version": app.version,
        "docs": "/docs",
    }
