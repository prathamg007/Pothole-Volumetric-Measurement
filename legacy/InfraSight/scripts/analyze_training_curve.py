"""
Rigorous Regression Analysis with 50 Data Points
Fits multiple models with R² comparison to make a statistically defensible projection.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ── Load data ──────────────────────────────────────────────────────────────────
csv1_path = "runs/detect/models/weights/pothole_det/experiment_b_merged_yolov8m2/results.csv"
csv2_path = "runs/detect/models/weights/pothole_det/experiment_b_merged_yolov8m25/results.csv"
df1 = pd.read_csv(csv1_path)
df1.columns = df1.columns.str.strip()
df1['true_epoch'] = range(1, len(df1)+1)

df2 = pd.read_csv(csv2_path)
df2.columns = df2.columns.str.strip()
df2['true_epoch'] = range(31, 31+len(df2))

df = pd.concat([df1, df2], ignore_index=True)

epochs = df['true_epoch'].values
map50  = df['metrics/mAP50(B)'].values

print(f"Data loaded: {len(epochs)} epochs (ep1-30 + ep31-50)")
for ep in [1, 10, 20, 30, 40, 50]:
    if ep <= len(map50):
        print(f"Epoch {ep:>2} mAP50: {map50[ep-1]:.4f}")

# Growth rates per decade
print()
print("Growth Rates:")
for a, b in [(0,9), (9,19), (19,29), (29,39), (39,49)]:
    if b < len(map50):
        rate = (map50[b] - map50[a]) / (b - a)
        print(f"  Ep {a+1:>2}-{b+1}: {rate:+.5f}/epoch")
print()

# ── Print full per-epoch table ─────────────────────────────────────────────────
print(f"{'Epoch':>6} {'mAP50':>8} {'mAP50-95':>10}")
print("-" * 28)
for i, (e, m) in enumerate(zip(epochs, map50)):
    marker = " ◄" if (i+1) in [10, 20, 30, 40, 50] else ""
    print(f"{e:>6} {m:>8.4f} {df['metrics/mAP50-95(B)'].values[i]:>10.4f}{marker}")

# ── Model definitions ───────────────────────────────────────────────────────────
def log_model(t, a, b):
    return a + b * np.log(t)

def power_model(t, a, b):
    return a * (t ** b)

def sqrt_model(t, a, b):
    return a * np.sqrt(t) + b

def linear_model(t, a, b):
    return a * t + b

models = {
    "Logarithmic  [a + b·ln(t)]": log_model,
    "Power Law    [a · t^b]":      power_model,
    "Square Root  [a·√t + b]":    sqrt_model,
    "Linear       [a·t + b]":     linear_model,
}

# ── Fit each model and compare ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODEL FITTING (all 50 data points)")
print("=" * 65)
print(f"{'Model':<30} {'R²':>6}  {'mAP@100':>9}  {'mAP@150':>9}  {'mAP@200':>9}")
print("-" * 65)

results = {}
for name, fn in models.items():
    try:
        popt, _ = curve_fit(fn, epochs, map50, maxfev=10000)
        pred   = fn(epochs, *popt)
        r2     = r2_score(map50, pred)
        
        p100 = min(fn(100, *popt), 1.0)
        p150 = min(fn(150, *popt), 1.0)
        p200 = min(fn(200, *popt), 1.0)
        
        results[name] = {'r2': r2, 'p100': p100, 'p150': p150, 'p200': p200, 'popt': popt}
        print(f"{name:<30} {r2:>6.4f}  {p100:>9.4f}  {p150:>9.4f}  {p200:>9.4f}")
    except Exception as ex:
        print(f"{name:<30} FAILED: {ex}")

# ── Best model & plateau detection ────────────────────────────────────────────
best = max(results.items(), key=lambda x: x[1]['r2'])
print(f"\n✅ Best fit: {best[0]}  (R²={best[1]['r2']:.4f})")

# Diminishing returns analysis (growth rate over last 10 epochs)
last_10_growth = (map50[-1] - map50[-11]) / 10 if len(map50) >= 11 else None
print(f"\n📈 Growth rate (last 10 epochs): {last_10_growth:.5f} mAP/epoch" if last_10_growth else "")

# ── Conservative projection (mean of top-2 models) ────────────────────────────
sorted_res = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:2]
mean_p100 = np.mean([r['p100'] for _, r in sorted_res])
mean_p150 = np.mean([r['p150'] for _, r in sorted_res])
mean_p200 = np.mean([r['p200'] for _, r in sorted_res])

print("\n" + "=" * 65)
print("FINAL PROJECTION (mean of top-2 models by R²)")
print("=" * 65)
print(f"Epoch 50  (known):   {map50[49]:.4f}" if len(map50) >= 50 else f"Epoch {len(map50)} (latest): {map50[-1]:.4f}")
print(f"Epoch 100 (proj.):   {mean_p100:.4f}")
print(f"Epoch 150 (proj.):   {mean_p150:.4f}")
print(f"Epoch 200 (proj.):   {mean_p200:.4f}")
print(f"\nTarget (mAP ≥ 0.70): {'LIKELY at 200 eps' if mean_p200 >= 0.70 else 'UNCERTAIN at 200 eps'}")

# Estimate epoch to hit 0.70
if best[1]['r2'] > 0.95:
    fn = models[best[0]]
    popt = best[1]['popt']
    for ep in range(50, 300, 5):
        if fn(ep, *popt) >= 0.70:
            print(f"Best model projects 0.70 reached at epoch ~{ep}")
            break
    else:
        print("Best model does NOT project 0.70 within 300 epochs")
