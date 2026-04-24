from pathlib import Path
from fastapi import APIRouter

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    weights_dir = SERVER_ROOT / "weights"
    expected = ["pothole_seg.pt", "crack_typology.pt"]
    found = {name: (weights_dir / name).exists() for name in expected}
    return {"status": "ok", "weights": found}
