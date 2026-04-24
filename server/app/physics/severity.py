"""Pothole severity classification from measurements. Config-driven thresholds."""
import math
from dataclasses import dataclass


@dataclass
class SeverityResult:
    level: str          # LOW / MEDIUM / HIGH / CRITICAL
    score: int          # 1..10
    color: str
    description: str
    priority: str
    risk: str
    estimated_repair_days: int


_LEVEL_META = {
    "LOW": {
        "color": "#4CAF50",
        "priority": "Scheduled (30 days)",
        "risk": "Low risk - minor discomfort",
        "repair_days": 30,
        "score_range": (1, 3),
    },
    "MEDIUM": {
        "color": "#FF9800",
        "priority": "Priority (7 days)",
        "risk": "Medium risk - potential tire damage",
        "repair_days": 7,
        "score_range": (4, 5),
    },
    "HIGH": {
        "color": "#F44336",
        "priority": "Urgent (3 days)",
        "risk": "High risk - vehicle damage & accident potential",
        "repair_days": 3,
        "score_range": (6, 7),
    },
    "CRITICAL": {
        "color": "#9C27B0",
        "priority": "Emergency (24 hours)",
        "risk": "Critical risk - immediate safety hazard",
        "repair_days": 1,
        "score_range": (8, 10),
    },
}


class SeverityClassifier:
    def __init__(self, severity_cfg: dict):
        self.weights = severity_cfg["weights"]

    def classify(self, depth_cm: float, area_cm2: float, volume_cm3: float) -> SeverityResult:
        depth_s = _score_depth(depth_cm)
        area_s = _score_area(area_cm2)
        vol_s = _score_volume(volume_cm3)
        composite = (
            self.weights["depth"] * depth_s
            + self.weights["area"] * area_s
            + self.weights["volume"] * vol_s
        )
        score = max(1, min(10, int(round(composite))))
        level = _score_bracket(score)
        meta = _LEVEL_META[level]

        description = (
            f"Depth {depth_cm:.1f} cm, area {area_cm2:.0f} cm^2, volume {volume_cm3:.0f} cm^3. "
            f"{meta['risk']}."
        )
        return SeverityResult(
            level=level,
            score=int(score),
            color=meta["color"],
            description=description,
            priority=meta["priority"],
            risk=meta["risk"],
            estimated_repair_days=meta["repair_days"],
        )


def _score_depth(depth_cm: float) -> float:
    if depth_cm <= 0:
        return 0.0
    return min(10.0, 2.3 * math.log(depth_cm + 1) + 0.5)


def _score_area(area_cm2: float) -> float:
    if area_cm2 <= 0:
        return 0.0
    return min(10.0, 1.8 * math.sqrt(area_cm2 / 30))


def _score_volume(volume_cm3: float) -> float:
    if volume_cm3 <= 0:
        return 0.0
    return min(10.0, 1.5 * math.log10(volume_cm3 + 1) + 0.5)


def _score_bracket(score: int) -> str:
    for lvl, meta in _LEVEL_META.items():
        lo, hi = meta["score_range"]
        if lo <= score <= hi:
            return lvl
    return "CRITICAL"
