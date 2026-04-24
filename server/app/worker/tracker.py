"""IoU-based multi-object tracker for pothole instances across frames."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.physics.volumetric import VolumetricResult


@dataclass
class TrackObservation:
    frame_idx: int
    bbox: tuple[int, int, int, int]
    confidence: float
    measurement: Optional[VolumetricResult]


@dataclass
class Track:
    track_id: int
    first_frame: int
    last_frame: int
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def last_bbox(self) -> tuple[int, int, int, int]:
        return self.observations[-1].bbox


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class PotholeTracker:
    def __init__(self, iou_threshold: float = 0.3, max_gap_frames: int = 15):
        self.tracks: list[Track] = []
        self._next_id = 1
        self.iou_threshold = float(iou_threshold)
        self.max_gap_frames = int(max_gap_frames)

    def update(
        self,
        frame_idx: int,
        detections: list,  # list[(PotholeDetection, Optional[VolumetricResult])]
    ) -> list[int]:
        """Assign each detection to an existing track or start a new one.
        Returns track_ids in the same order as detections.
        """
        active = [t for t in self.tracks if frame_idx - t.last_frame <= self.max_gap_frames]

        pairs: list[tuple[float, int, int]] = []  # (iou, det_idx, active_idx)
        for di, (det, _m) in enumerate(detections):
            for ti, tr in enumerate(active):
                iou = _iou(det.bbox, tr.last_bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)

        det_matched = [False] * len(detections)
        track_matched = [False] * len(active)
        track_ids: list[Optional[int]] = [None] * len(detections)

        for iou, di, ti in pairs:
            if det_matched[di] or track_matched[ti]:
                continue
            det, m = detections[di]
            tr = active[ti]
            tr.observations.append(
                TrackObservation(frame_idx=frame_idx, bbox=det.bbox, confidence=det.confidence, measurement=m)
            )
            tr.last_frame = frame_idx
            track_ids[di] = tr.track_id
            det_matched[di] = True
            track_matched[ti] = True

        for di, (det, m) in enumerate(detections):
            if det_matched[di]:
                continue
            tid = self._next_id
            self._next_id += 1
            new_tr = Track(track_id=tid, first_frame=frame_idx, last_frame=frame_idx)
            new_tr.observations.append(
                TrackObservation(frame_idx=frame_idx, bbox=det.bbox, confidence=det.confidence, measurement=m)
            )
            self.tracks.append(new_tr)
            track_ids[di] = tid

        return [int(t) for t in track_ids]

    def finalize(self, min_observations: int = 2, min_valid_measurements: int = 1) -> list[dict]:
        """Aggregate per-track measurements (median across observations).
        Filters out tracks that appeared for too few frames (likely false positives).
        """
        out: list[dict] = []
        for tr in self.tracks:
            if len(tr.observations) < min_observations:
                continue
            valid = [o.measurement for o in tr.observations if o.measurement is not None]
            if len(valid) < min_valid_measurements:
                continue

            areas = np.array([m.area_cm2 for m in valid])
            davg = np.array([m.avg_depth_cm for m in valid])
            dmax = np.array([m.max_depth_cm for m in valid])
            vols = np.array([m.volume_cm3 for m in valid])

            out.append(
                {
                    "track_id": tr.track_id,
                    "first_frame": tr.first_frame,
                    "last_frame": tr.last_frame,
                    "observations": len(tr.observations),
                    "valid_measurements": len(valid),
                    "area_cm2": float(np.median(areas)),
                    "avg_depth_cm": float(np.median(davg)),
                    "max_depth_cm": float(np.median(dmax)),
                    "volume_cm3": float(np.median(vols)),
                    "confidence": float(np.median([o.confidence for o in tr.observations])),
                }
            )
        return out
