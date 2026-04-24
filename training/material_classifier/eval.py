"""Evaluate a trained material classifier on the held-out test set.

Reports:
  - per-class precision / recall / F1 for material
  - per-class precision / recall / F1 for unevenness (only on samples that have it)
  - confusion matrices
  - overall accuracy
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TestDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root / "test"
        self.transform = transform
        with open(root / "test_labels.csv", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.root / row["filename"]).convert("RGB")
        return self.transform(img), row["material"], row.get("unevenness", "") or ""


class Model(nn.Module):
    def __init__(self, num_materials: int, num_unevenness: int):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = backbone.classifier[0].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(0.2)
        self.material_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_materials),
        )
        self.unevenness_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_unevenness),
        )

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        f = self.dropout(f)
        return self.material_head(f), self.unevenness_head(f)


def per_class_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                  "support": tp + fn}
    return out


def confusion(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    mat = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="curated dir (with test/ + test_labels.csv)")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output JSON path")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    materials = list(ckpt["materials"])
    unevenness = list(ckpt["unevenness"])
    print(f"materials: {materials}")
    print(f"unevenness: {unevenness}")

    model = Model(len(materials), len(unevenness))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = TestDataset(args.data, tx)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    print(f"test size: {len(ds):,}")

    all_mat_true = []
    all_mat_pred = []
    all_u_true = []
    all_u_pred = []

    with torch.inference_mode():
        for img, m_lbl, u_lbl in loader:
            img = img.to(device, non_blocking=True)
            m_logits, u_logits = model(img)
            m_pred = m_logits.argmax(1).cpu().tolist()
            u_pred = u_logits.argmax(1).cpu().tolist()
            for i, (mt, ut) in enumerate(zip(m_lbl, u_lbl)):
                all_mat_true.append(mt)
                all_mat_pred.append(materials[m_pred[i]])
                if ut:
                    all_u_true.append(ut)
                    all_u_pred.append(unevenness[u_pred[i]])

    mat_acc = sum(1 for t, p in zip(all_mat_true, all_mat_pred) if t == p) / len(all_mat_true)
    u_acc = (
        sum(1 for t, p in zip(all_u_true, all_u_pred) if t == p) / len(all_u_true)
        if all_u_true else None
    )

    report = {
        "material": {
            "accuracy": round(mat_acc, 4),
            "n_samples": len(all_mat_true),
            "per_class": per_class_metrics(all_mat_true, all_mat_pred, materials),
            "confusion_matrix": confusion(all_mat_true, all_mat_pred, materials),
        },
        "unevenness": {
            "accuracy": round(u_acc, 4) if u_acc is not None else None,
            "n_samples": len(all_u_true),
            "per_class": per_class_metrics(all_u_true, all_u_pred, unevenness),
            "confusion_matrix": confusion(all_u_true, all_u_pred, unevenness),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Pretty print
    print("\n=== MATERIAL ===")
    print(f"  accuracy: {report['material']['accuracy']:.4f}  (n={report['material']['n_samples']})")
    for c, m in report["material"]["per_class"].items():
        print(f"  {c:9s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    print("\n=== UNEVENNESS ===")
    if u_acc is None:
        print("  (no test samples with unevenness labels)")
    else:
        print(f"  accuracy: {report['unevenness']['accuracy']:.4f}  (n={report['unevenness']['n_samples']})")
        for c, m in report["unevenness"]["per_class"].items():
            print(f"  {c:9s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")

    print(f"\nfull report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
