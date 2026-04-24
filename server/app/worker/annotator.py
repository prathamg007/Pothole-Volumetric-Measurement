"""Draw detection masks, bboxes, labels, and frame HUD on video frames."""
from typing import Optional

import cv2
import numpy as np


_POTHOLE_COLOR_BGR = (0, 215, 255)   # amber
_CRACK_COLORS_BGR = {
    "Longitudinal Crack": (255, 180, 60),
    "Transverse Crack": (60, 180, 255),
    "Alligator Crack": (60, 60, 255),
}
_SEVERITY_COLORS_BGR = {
    "LOW": (80, 175, 76),
    "MEDIUM": (0, 152, 255),
    "HIGH": (54, 67, 244),
    "CRITICAL": (176, 39, 156),
}


def annotate_frame(
    frame_bgr: np.ndarray,
    *,
    potholes: list[dict],
    cracks: list[dict],
    frame_idx: int,
    total_frames: int,
    running_totals: Optional[dict] = None,
) -> np.ndarray:
    """Return a new BGR frame with overlays drawn.

    potholes: list of dicts with keys {track_id, bbox, mask, measurement, severity_level, severity_score}
    cracks:   list of dicts with keys {class_name, confidence, bbox}
    """
    out = frame_bgr.copy()

    # Pothole masks (semi-transparent fill)
    for p in potholes:
        mask = p.get("mask")
        if mask is None:
            continue
        color = _SEVERITY_COLORS_BGR.get(p.get("severity_level") or "", _POTHOLE_COLOR_BGR)
        overlay = np.zeros_like(out)
        overlay[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

    # Pothole bboxes + labels
    for p in potholes:
        x1, y1, x2, y2 = p["bbox"]
        color = _SEVERITY_COLORS_BGR.get(p.get("severity_level") or "", _POTHOLE_COLOR_BGR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        tid = p.get("track_id")
        m = p.get("measurement")
        sev = p.get("severity_level") or ""
        if m is not None:
            label = f"#{tid} {sev}  {m.area_cm2:.0f}cm2  d={m.max_depth_cm:.1f}cm"
        else:
            label = f"#{tid} pothole"
        _put_label(out, label, (x1, y1 - 6), color)

    # Crack boxes
    for c in cracks:
        x1, y1, x2, y2 = c["bbox"]
        color = _CRACK_COLORS_BGR.get(c["class_name"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{c['class_name']} {c['confidence']:.2f}"
        _put_label(out, label, (x1, y2 + 18), color)

    # Frame HUD (top-left)
    hud_lines = [f"frame {frame_idx}/{total_frames}"]
    if running_totals:
        hud_lines.append(
            f"potholes {running_totals.get('potholes', 0)}  cracks {running_totals.get('cracks', 0)}"
        )
    _draw_hud(out, hud_lines)

    return out


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color_bgr: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    y = max(y, th + 2)
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + 2), color_bgr, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_hud(img: np.ndarray, lines: list[str]):
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    sizes = [cv2.getTextSize(ln, font, scale, thickness)[0] for ln in lines]
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = sum(s[1] + 4 for s in sizes) + 2 * pad
    cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    y = pad + sizes[0][1]
    for ln in lines:
        cv2.putText(img, ln, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += sizes[0][1] + 4
