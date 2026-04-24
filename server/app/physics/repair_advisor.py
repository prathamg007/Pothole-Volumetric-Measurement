"""Repair method, material quantity, and cost — all rates loaded from config.yaml (INR)."""
from dataclasses import dataclass
from typing import List


@dataclass
class RepairStep:
    order: int
    action: str
    description: str
    duration_minutes: int


@dataclass
class RepairRecommendation:
    method: str
    material_key: str
    material_name: str
    material_kg: float
    material_cost: float
    labor_cost: float
    total_cost: float
    currency: str
    tools: List[str]
    steps: List[RepairStep]
    estimated_time_hours: float
    durability_months: int
    notes: str


_METHOD_LABELS = {
    "throw_and_roll": "Throw-and-Roll",
    "semi_permanent": "Semi-Permanent Patch",
    "full_depth": "Full-Depth Repair",
}

_MATERIAL_LABELS = {
    "cold_mix_asphalt": "Cold Mix Asphalt",
    "hot_mix_asphalt": "Hot Mix Asphalt (HMA)",
    "concrete_patch": "Concrete Patch",
}


class RepairAdvisor:
    def __init__(self, repair_cfg: dict):
        self.currency = repair_cfg.get("currency", "INR")
        self.materials = repair_cfg["materials"]
        self.labor = repair_cfg["labor"]
        self.durability = repair_cfg["durability_months"]

    def recommend(
        self,
        volume_cm3: float,
        depth_cm: float,
        area_cm2: float,
        severity_level: str,
        surface_type: str = "asphalt",
    ) -> RepairRecommendation:
        method = self._select_method(severity_level, depth_cm, volume_cm3)
        material_key = self._select_material(method, surface_type)
        mat = self.materials[material_key]

        raw_kg = volume_cm3 * mat["density_kg_per_cm3"]
        total_kg = raw_kg * mat["compaction_factor"]
        material_cost = total_kg * mat["price_per_kg"]
        labor_cost = float(self.labor[method])
        total_cost = material_cost + labor_cost

        steps = _get_steps(method)
        tools = _get_tools(method)
        time_hours = sum(s.duration_minutes for s in steps) / 60.0

        return RepairRecommendation(
            method=_METHOD_LABELS[method],
            material_key=material_key,
            material_name=_MATERIAL_LABELS[material_key],
            material_kg=round(total_kg, 2),
            material_cost=round(material_cost, 2),
            labor_cost=labor_cost,
            total_cost=round(total_cost, 2),
            currency=self.currency,
            tools=tools,
            steps=steps,
            estimated_time_hours=round(time_hours, 1),
            durability_months=int(self.durability[method]),
            notes=_notes(method, total_kg, depth_cm),
        )

    @staticmethod
    def _select_method(severity: str, depth_cm: float, volume_cm3: float) -> str:
        if severity == "LOW" or (depth_cm <= 2.5 and volume_cm3 <= 500):
            return "throw_and_roll"
        if severity in ("MEDIUM", "HIGH") or depth_cm <= 10:
            return "semi_permanent"
        return "full_depth"

    @staticmethod
    def _select_material(method: str, surface_type: str) -> str:
        if surface_type == "concrete":
            return "concrete_patch"
        if method == "throw_and_roll":
            return "cold_mix_asphalt"
        return "hot_mix_asphalt"


def _get_steps(method: str) -> List[RepairStep]:
    if method == "throw_and_roll":
        return [
            RepairStep(1, "Clean", "Remove debris and water from the pothole", 5),
            RepairStep(2, "Fill", "Pour cold mix asphalt into the pothole", 5),
            RepairStep(3, "Level", "Level the surface using a shovel", 3),
            RepairStep(4, "Compact", "Compact with vehicle tires 2-3 times", 5),
        ]
    if method == "semi_permanent":
        return [
            RepairStep(1, "Clean", "Remove all debris, water, and loose material", 10),
            RepairStep(2, "Square Up", "Create vertical/square edges around the pothole", 15),
            RepairStep(3, "Tack Coat", "Apply asphalt emulsion to the bottom and edges", 5),
            RepairStep(4, "Fill", "Pour hot mix asphalt in layers", 10),
            RepairStep(5, "Compact", "Compact with a vibratory plate compactor", 10),
            RepairStep(6, "Level", "Check surface flatness", 5),
        ]
    return [
        RepairStep(1, "Marking", "Mark the repair area larger than the pothole", 10),
        RepairStep(2, "Cutting", "Cut old asphalt using a saw cutter", 30),
        RepairStep(3, "Removal", "Remove and dispose of old material", 20),
        RepairStep(4, "Base Prep", "Prepare and compact the base course", 30),
        RepairStep(5, "Tack Coat", "Apply asphalt emulsion to all surfaces", 10),
        RepairStep(6, "Paving", "Pour hot mix asphalt in multiple layers", 20),
        RepairStep(7, "Compact", "Compact with a roller compactor", 15),
        RepairStep(8, "Finishing", "Seal edges and check surface flatness", 15),
    ]


def _get_tools(method: str) -> List[str]:
    if method == "throw_and_roll":
        return ["Shovel", "Broom", "Cold mix asphalt"]
    if method == "semi_permanent":
        return [
            "Shovel",
            "Broom",
            "Chisel",
            "Asphalt emulsion",
            "Hot mix asphalt",
            "Vibratory plate compactor",
        ]
    return [
        "Saw cutter",
        "Jackhammer",
        "Shovel",
        "Dump truck",
        "Roller compactor",
        "Asphalt emulsion",
        "Hot mix asphalt",
        "Base course material",
        "Leveling tools",
    ]


def _notes(method: str, material_kg: float, depth_cm: float) -> str:
    out = []
    if method == "throw_and_roll":
        out.append("Temporary repair - follow-up required within 3 months.")
    if material_kg > 50:
        out.append(f"Material {material_kg:.1f} kg - transport vehicle recommended.")
    if depth_cm > 10:
        out.append("Depth >10 cm - check base layer and drainage.")
    return " ".join(out) if out else "Standard repair procedure recommended."


def format_currency(amount: float, currency: str = "INR") -> str:
    if currency == "INR":
        return f"Rs {amount:,.0f}"
    return f"{currency} {amount:,.2f}"
