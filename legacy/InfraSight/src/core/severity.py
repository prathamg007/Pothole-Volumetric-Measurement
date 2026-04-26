"""
Pothole Severity Classifier
Classifies pothole damage severity based on volumetric measurements.
Uses engineering standards (IRI/PCI inspired thresholds) adapted for
Indonesian road conditions.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import math
from src.utils.logger import setup_logger

logger = setup_logger("Severity")


@dataclass
class SeverityResult:
    """Container for severity classification results"""
    level: str              # LOW, MEDIUM, HIGH, CRITICAL
    score: int              # 1-10 numeric score
    color: str              # Hex color for visualization
    label_en: str           # English label
    description: str        # Human-readable description
    priority: str           # Repair priority
    risk_level: str         # Safety risk assessment
    estimated_repair_days: int  # Estimated days to repair


class SeverityClassifier:
    """
    Classify pothole severity based on depth, area, and volume.
    
    Classification is based on a weighted scoring system:
    - Depth:  40% weight (most dangerous factor)
    - Area:   30% weight (affects vehicle damage)
    - Volume: 30% weight (indicates material loss)
    
    Thresholds are calibrated for Indonesian urban roads
    (speed limit 40-60 km/h typical context).
    """
    
    # Severity thresholds
    THRESHOLDS = {
        'LOW': {
            'max_depth_cm': 2.5,
            'max_area_cm2': 200,
            'max_volume_cm3': 500,
            'score_range': (1, 3),
            'color': '#4CAF50',        # Green
            'label_en': 'Low',
            'priority': 'Scheduled (30 days)',
            'risk': 'Low risk - minor discomfort',
            'repair_days': 30,
        },
        'MEDIUM': {
            'max_depth_cm': 5.0,
            'max_area_cm2': 700,
            'max_volume_cm3': 3500,
            'score_range': (4, 5),
            'color': '#FF9800',        # Orange
            'label_en': 'Medium',
            'priority': 'Priority (7 days)',
            'risk': 'Medium risk - potential tire damage',
            'repair_days': 7,
        },
        'HIGH': {
            'max_depth_cm': 10.0,
            'max_area_cm2': 2000,
            'max_volume_cm3': 20000,
            'score_range': (6, 7),
            'color': '#F44336',        # Red
            'label_en': 'High',
            'priority': 'Urgent (3 days)',
            'risk': 'High risk - vehicle damage & accident potential',
            'repair_days': 3,
        },
        'CRITICAL': {
            'max_depth_cm': float('inf'),
            'max_area_cm2': float('inf'),
            'max_volume_cm3': float('inf'),
            'score_range': (9, 10),
            'color': '#9C27B0',        # Purple
            'label_en': 'Critical',
            'priority': 'Emergency (24 hours)',
            'risk': 'Critical risk - immediate safety hazard',
            'repair_days': 1,
        },
    }
    
    # Scoring weights
    WEIGHT_DEPTH = 0.40
    WEIGHT_AREA = 0.30
    WEIGHT_VOLUME = 0.30
    
    def classify(
        self,
        depth_cm: float,
        area_cm2: float,
        volume_cm3: float,
    ) -> SeverityResult:
        """
        Classify pothole severity from measurements.
        
        Args:
            depth_cm: Average depth in centimeters
            area_cm2: Surface area in square centimeters
            volume_cm3: Volume in cubic centimeters
            
        Returns:
            SeverityResult with classification details
        """
        # Calculate normalized scores (0-10) for each metric
        depth_score = self._score_depth(depth_cm)
        area_score = self._score_area(area_cm2)
        volume_score = self._score_volume(volume_cm3)
        
        # Weighted composite score
        composite = (
            self.WEIGHT_DEPTH * depth_score +
            self.WEIGHT_AREA * area_score +
            self.WEIGHT_VOLUME * volume_score
        )
        
        # Round to nearest integer (1-10)
        score = max(1, min(10, round(composite)))
        
        # Determine severity level from score
        level = self._score_to_level(score)
        config = self.THRESHOLDS[level]
        
        # Build technical description
        description = (
            f"Depth {depth_cm:.1f} cm, "
            f"area {area_cm2:.0f} cm², "
            f"volume {volume_cm3:.0f} cm³. "
            f"{config['risk']}."
        )
        
        return SeverityResult(
            level=level,
            score=score,
            color=config['color'],
            label_en=config['label_en'],
            description=description,
            priority=config['priority'],
            risk_level=config['risk'],
            estimated_repair_days=config['repair_days'],
        )
    
    def _score_depth(self, depth_cm: float) -> float:
        """Score depth on 0-10 scale using log curve"""
        if depth_cm <= 0:
            return 0.0
        # Log scale: 1cm → ~2.3, 2.5cm → ~4.7, 5cm → ~6.5, 10cm → ~8.3, 15cm+ → 10
        return min(10.0, 2.3 * math.log(depth_cm + 1) + 0.5)
    
    def _score_area(self, area_cm2: float) -> float:
        """Score area on 0-10 scale"""
        if area_cm2 <= 0:
            return 0.0
        # Sqrt scale: 100cm² → ~3.2, 500cm² → ~5.5, 2000cm² → ~8.4
        return min(10.0, 1.8 * math.sqrt(area_cm2 / 30))
    
    def _score_volume(self, volume_cm3: float) -> float:
        """Score volume on 0-10 scale"""
        if volume_cm3 <= 0:
            return 0.0
        # Log scale: 100 → ~2.8, 1000 → ~5.1, 10000 → ~7.5, 50000+ → 10
        return min(10.0, 1.5 * math.log10(volume_cm3 + 1) + 0.5)
    
    def _score_to_level(self, score: int) -> str:
        """Map composite score to severity level"""
        for level, config in self.THRESHOLDS.items():
            low, high = config['score_range']
            if low <= score <= high:
                return level
        return 'CRITICAL'  # Default for scores > 8
    
    def get_color_for_level(self, level: str) -> str:
        """Get display color for severity level"""
        return self.THRESHOLDS.get(level, {}).get('color', '#999999')


if __name__ == "__main__":
    classifier = SeverityClassifier()
    
    # Test cases
    test_cases = [
        {"depth_cm": 1.5, "area_cm2": 100,  "volume_cm3": 150},     # LOW
        {"depth_cm": 4.0, "area_cm2": 400,  "volume_cm3": 1600},    # MEDIUM
        {"depth_cm": 7.5, "area_cm2": 1200, "volume_cm3": 9000},    # HIGH
        {"depth_cm": 15.0, "area_cm2": 3000, "volume_cm3": 45000},  # CRITICAL
    ]
    
    print("=" * 70)
    print("SEVERITY CLASSIFICATION TEST")
    print("=" * 70)
    
    for tc in test_cases:
        result = classifier.classify(**tc)
        logger.info(f"Input: depth={tc['depth_cm']}cm, area={tc['area_cm2']}cm², vol={tc['volume_cm3']}cm³")
        logger.info(f"Level: {result.level} ({result.label_en}) — Score: {result.score}/10")
        logger.info(f"Priority: {result.priority}")
        logger.info(f"Risk: {result.risk_level}")
    
    print(f"\n{'='*70}")
