"""Phase 4 smoke test: assumes the server is already running at http://127.0.0.1:8000.

Steps:
  1. GET /health — confirm reachable
  2. POST /analyze with legacy/InfraSight/road_test.mp4
  3. Poll GET /jobs/{id} until completed
  4. GET /jobs/{id}/result — parse structured report
  5. GET /jobs/{id}/video — download annotated MP4

Start the server separately:
    cd server
    python run_server.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

SERVER = "http://127.0.0.1:8000"
SERVER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = SERVER_ROOT.parent / "legacy" / "InfraSight" / "road_test.mp4"


def wait_for_ready(timeout_s: int = 180) -> bool:
    end = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < end:
        try:
            r = requests.get(f"{SERVER}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except requests.RequestException as e:
            last_err = e
        time.sleep(1)
    if last_err:
        print(f"[wait] last error: {last_err}", file=sys.stderr)
    return False


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    print(f"[1/5] Waiting for server at {SERVER}...")
    if not wait_for_ready():
        print("Server did not become ready. Is `python run_server.py` running?")
        return 1
    print("      Server ready.")

    print(f"[2/5] Uploading {video.name} ({video.stat().st_size/1024/1024:.1f} MB)...")
    with open(video, "rb") as f:
        r = requests.post(f"{SERVER}/analyze", files={"video": (video.name, f, "video/mp4")}, timeout=120)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"      job_id = {job_id}")

    print("[3/5] Polling for completion...")
    last_status = None
    t0 = time.time()
    while True:
        r = requests.get(f"{SERVER}/jobs/{job_id}", timeout=5)
        r.raise_for_status()
        job = r.json()
        status = job["status"]
        if status != last_status:
            elapsed = time.time() - t0
            print(f"      [{elapsed:6.1f}s] status: {status}")
            last_status = status
        if status in ("completed", "failed"):
            break
        time.sleep(2)

    if status == "failed":
        print("\nJob FAILED:")
        print(job.get("error_message", ""))
        return 1

    print("[4/5] Fetching result JSON...")
    r = requests.get(f"{SERVER}/jobs/{job_id}/result", timeout=10)
    r.raise_for_status()
    report = r.json()
    s = report["summary"]
    print(
        f"      {s['num_potholes']} potholes, "
        f"area {s['total_area_cm2']:.0f} cm^2, "
        f"vol {s['total_volume_cm3']:.0f} cm^3, "
        f"cost Rs {s['total_cost']:.0f}"
    )
    print(f"      cracks: {report['cracks']}")
    rs = report.get("road_surface")
    if rs:
        print(
            f"      road: {rs.get('material')} ({rs.get('material_confidence')}) "
            f"+ {rs.get('unevenness')} ({rs.get('unevenness_confidence')}) "
            f"over {rs.get('frames_used')} frames"
        )

    print("[5/5] Downloading annotated video...")
    r = requests.get(f"{SERVER}/jobs/{job_id}/video", timeout=30)
    r.raise_for_status()
    out_path = SERVER_ROOT / "data" / "results" / f"phase4_download_{job_id[:8]}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    print(f"      saved {len(r.content)/1024/1024:.1f} MB to {out_path}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
