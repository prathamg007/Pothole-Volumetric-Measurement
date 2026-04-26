"""Quick verification test for severity + repair modules"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.severity import SeverityClassifier
from src.core.repair_advisor import RepairAdvisor

sc = SeverityClassifier()
ra = RepairAdvisor()

tests = [
    (1.5, 100, 150, "LOW"),
    (4.0, 400, 1600, "MEDIUM"),
    (7.5, 1200, 9000, "HIGH"),
    (15.0, 3000, 45000, "CRITICAL"),
]

all_pass = True
for d, a, v, expected in tests:
    sev = sc.classify(d, a, v)
    rep = ra.recommend(v, d, a, sev.level)
    ok = sev.level == expected
    if not ok:
        all_pass = False
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] depth={d}, area={a}, vol={v}")
    print(f"       Severity: {sev.level} (score {sev.score}) | Expected: {expected}")
    print(f"       Repair: {rep.method_id} | {rep.material_kg}kg | Rp {rep.total_cost_idr:,.0f}")
    print()

if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
