"""Train a multi-head MobileNetV3-Small for road-surface material + unevenness.

- Backbone:  pretrained MobileNetV3-Small (frozen features early in training,
             unfrozen for later epochs).
- Head 1:    4 materials  (asphalt, concrete, mud, gravel)
- Head 2:    3 unevenness (smooth, slight, severe)

Unevenness is only labeled for asphalt/concrete; we mask the loss for
mud/gravel samples so they only contribute to the material head.

Inputs:
    --data     dir produced by curate.py (contains train/, val/, *_labels.csv)
    --out      dir to write checkpoints + metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MATERIALS = ["asphalt", "concrete", "mud", "gravel"]
UNEVENNESS = ["smooth", "slight", "severe"]


# ------------------------- Dataset -------------------------


class RoadSurfaceDataset(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.root = root / split
        self.transform = transform
        csv_path = root / f"{split}_labels.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.root / row["filename"]).convert("RGB")
        img = self.transform(img)
        material_label = MATERIALS.index(row["material"])
        u_str = row.get("unevenness", "") or ""
        if u_str:
            unevenness_label = UNEVENNESS.index(u_str)
            unevenness_mask = 1
        else:
            unevenness_label = 0  # placeholder; mask will zero its loss
            unevenness_mask = 0
        return img, material_label, unevenness_label, unevenness_mask


# ------------------------- Model -------------------------


class RoadSurfaceModel(nn.Module):
    def __init__(self, num_materials: int = 4, num_unevenness: int = 3):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = backbone.classifier[0].in_features  # 576
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

    def freeze_backbone(self, frozen: bool):
        for p in self.features.parameters():
            p.requires_grad = not frozen


# ------------------------- Loss -------------------------


def multi_head_loss(
    material_logits: torch.Tensor,
    unevenness_logits: torch.Tensor,
    material_labels: torch.Tensor,
    unevenness_labels: torch.Tensor,
    unevenness_mask: torch.Tensor,
    unevenness_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    m_loss = F.cross_entropy(material_logits, material_labels)
    mask = unevenness_mask.bool()
    if mask.any():
        u_loss = F.cross_entropy(unevenness_logits[mask], unevenness_labels[mask])
    else:
        u_loss = torch.zeros((), device=material_logits.device)
    total = m_loss + unevenness_weight * u_loss
    return total, {"m_loss": float(m_loss.item()), "u_loss": float(u_loss.item())}


# ------------------------- Training -------------------------


@dataclass
class EpochStats:
    epoch: int
    train_m_loss: float
    train_u_loss: float
    val_m_loss: float
    val_u_loss: float
    val_m_acc: float
    val_u_acc: float
    seconds: float


def evaluate(model, loader, device) -> dict:
    model.eval()
    n = 0
    n_u = 0
    correct_m = 0
    correct_u = 0
    sum_m = 0.0
    sum_u = 0.0
    with torch.inference_mode():
        for img, m, u, mask in loader:
            img = img.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            m_logits, u_logits = model(img)
            sum_m += float(F.cross_entropy(m_logits, m).item()) * img.size(0)
            n += img.size(0)
            correct_m += int((m_logits.argmax(1) == m).sum().item())
            mb = mask.bool()
            if mb.any():
                sum_u += float(F.cross_entropy(u_logits[mb], u[mb]).item()) * mb.sum().item()
                correct_u += int((u_logits[mb].argmax(1) == u[mb]).sum().item())
                n_u += int(mb.sum().item())
    return {
        "m_loss": sum_m / max(1, n),
        "u_loss": sum_u / max(1, n_u),
        "m_acc": correct_m / max(1, n),
        "u_acc": correct_u / max(1, n_u),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--unfreeze-after", type=int, default=5,
                    help="epoch at which to unfreeze the backbone")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_tx = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = RoadSurfaceDataset(args.data, "train", train_tx)
    val_ds = RoadSurfaceDataset(args.data, "val", eval_tx)
    print(f"train: {len(train_ds):,}, val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = RoadSurfaceModel().to(device)
    model.freeze_backbone(True)

    head_params = list(model.material_head.parameters()) + list(model.unevenness_head.parameters())
    optim = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val_m_acc = 0.0
    epochs_since_improve = 0
    metrics_log: list[EpochStats] = []
    best_path = args.out / "material_classifier_best.pt"

    for epoch in range(args.epochs):
        if epoch == args.unfreeze_after:
            print(f"epoch {epoch}: unfreezing backbone")
            model.freeze_backbone(False)
            optim = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": args.lr_backbone},
                    {"params": head_params, "lr": args.lr_head},
                ],
                weight_decay=1e-4,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs - epoch)

        model.train()
        t0 = time.time()
        running_m = 0.0
        running_u = 0.0
        n = 0
        n_u = 0
        for img, m, u, mask in train_loader:
            img = img.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            m_logits, u_logits = model(img)
            loss, parts = multi_head_loss(m_logits, u_logits, m, u, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_m += parts["m_loss"] * img.size(0)
            running_u += parts["u_loss"] * mask.sum().item()
            n += img.size(0)
            n_u += int(mask.sum().item())
        sched.step()

        train_stats = {
            "m_loss": running_m / max(1, n),
            "u_loss": running_u / max(1, n_u),
        }
        val_stats = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        es = EpochStats(
            epoch=epoch,
            train_m_loss=train_stats["m_loss"],
            train_u_loss=train_stats["u_loss"],
            val_m_loss=val_stats["m_loss"],
            val_u_loss=val_stats["u_loss"],
            val_m_acc=val_stats["m_acc"],
            val_u_acc=val_stats["u_acc"],
            seconds=elapsed,
        )
        metrics_log.append(es)
        print(
            f"ep {epoch:02d}  "
            f"train m={train_stats['m_loss']:.3f} u={train_stats['u_loss']:.3f}  "
            f"val m={val_stats['m_loss']:.3f}/{val_stats['m_acc']:.3f}  "
            f"u={val_stats['u_loss']:.3f}/{val_stats['u_acc']:.3f}  "
            f"({elapsed:.0f}s)"
        )

        if val_stats["m_acc"] > best_val_m_acc:
            best_val_m_acc = val_stats["m_acc"]
            epochs_since_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "materials": MATERIALS,
                    "unevenness": UNEVENNESS,
                    "epoch": epoch,
                    "val_m_acc": best_val_m_acc,
                    "val_u_acc": val_stats["u_acc"],
                },
                best_path,
            )
            print(f"  saved best -> {best_path}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    # Write metrics
    with open(args.out / "metrics.json", "w") as f:
        json.dump(
            [vars(es) for es in metrics_log],
            f,
            indent=2,
        )
    print(f"best val material acc: {best_val_m_acc:.3f}  weights: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
