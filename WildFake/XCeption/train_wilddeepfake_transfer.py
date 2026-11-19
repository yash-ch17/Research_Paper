#!/usr/bin/env python
"""Fine-tune the existing FF++ Xception-compression model on WildDeepfake.

This script READS the FF++ checkpoint and writes all new artifacts under a
separate WildDeepfake output directory. It never writes to the FF++ project.

Expected dataset:
  D:\\newdataset\\Dataset\\train\\fake|real
  D:\\newdataset\\Dataset\\valid\\fake|real
  D:\\newdataset\\Dataset\\test\\fake|real

The known FF++ checkpoint architecture is:
  timm Xception backbone (num_classes=0, global_pool='avg')
  Linear(2048, 256) -> ReLU -> Dropout(0.35) -> Linear(256, 1)

FF++ training labels used by the source model were real=0, fake=1. The
WildDeepfake ImageFolder maps fake=0, real=1, so this script explicitly maps
its labels to fake=1, real=0 to preserve the checkpoint's output meaning.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_CLASSES = {"fake": 0, "real": 1}


def parse_args():
    p = argparse.ArgumentParser(description="FF++ Xception checkpoint -> WildDeepfake transfer learning")
    p.add_argument("--data-dir", default=r"D:\newdataset\Dataset")
    p.add_argument("--ffpp-checkpoint", default=r"D:\faceforensics\checkpoints_xception_compression\best_compressed.pth",
                   help="Read-only source checkpoint from the FF++ compression Xception experiment")
    p.add_argument("--output-dir", default=r"D:\newdataset\wilddeepfake_experiment\transfer_xception")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-train-batches", type=int, default=0, help="Pilot only; 0 = all selected training batches")
    p.add_argument("--max-valid-batches", type=int, default=0, help="Pilot only; 0 = all validation batches")
    p.add_argument("--max-train-images", type=int, default=30000,
                   help="Deterministic balanced subset size for manageable runtime; 0 = use all training images")
    p.add_argument("--skip-test", action="store_true", help="Do not evaluate the held-out test split")
    p.add_argument("--resume", default="", help="Resume from this run's last_checkpoint.pth")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def atomic_torch_save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


class WildLabelAdapter(Dataset):
    """Map ImageFolder fake=0/real=1 to checkpoint semantics fake=1/real=0."""
    def __init__(self, imagefolder):
        self.base = imagefolder
        if imagefolder.class_to_idx != EXPECTED_CLASSES:
            raise ValueError(f"Unexpected classes {imagefolder.class_to_idx}; expected {EXPECTED_CLASSES}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, folder_label = self.base[index]
        fake_target = 1.0 if folder_label == EXPECTED_CLASSES["fake"] else 0.0
        return image, torch.tensor(fake_target, dtype=torch.float32)


def make_balanced_indices(imagefolder, max_images: int, seed: int):
    if max_images <= 0 or max_images >= len(imagefolder):
        return list(range(len(imagefolder)))
    # ImageFolder targets keep original class indices: fake=0 and real=1.
    labels = np.asarray(imagefolder.targets)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == cls) for cls in (0, 1)]
    per_class = max(1, max_images // 2)
    chosen = []
    for indices in class_indices:
        n = min(per_class, len(indices))
        chosen.extend(rng.choice(indices, size=n, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def make_transforms(image_size: int):
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_t, eval_t


def make_model():
    backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
    if getattr(backbone, "num_features", None) != 2048:
        raise RuntimeError(f"Expected Xception backbone with 2048 features, got {getattr(backbone, 'num_features', None)}")
    return XceptionFFPPStyle(backbone)


class XceptionFFPPStyle(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)


def load_ffpp_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FF++ source checkpoint not found: {checkpoint_path}\n"
            "Check the path or pass --ffpp-checkpoint with the correct file. "
            "The FF++ checkpoint is only read; this script will not modify it."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state is None and isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError(f"Could not find a model state dict inside {checkpoint_path}. Top-level keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
    # Normalize DataParallel prefixes and common names used for the same sequential head.
    normalized = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        normalized[clean] = value
    target_state = model.state_dict()
    remapped = dict(normalized)
    # The FF++ scripts may call this sequential module "classifier" or "head".
    # Match only non-backbone head tensors by shape when their names differ.
    used_source_keys = set()
    for target_key, target_tensor in target_state.items():
        if target_key in normalized:
            used_source_keys.add(target_key)
            continue
        if not target_key.startswith("head."):
            continue
        shape = tuple(target_tensor.shape)
        candidates = [
            k for k, v in normalized.items()
            if not k.startswith("backbone.") and tuple(v.shape) == shape
            and any(tag in k.lower() for tag in ("head", "classifier", "fc", "mlp"))
            and k not in used_source_keys
        ]
        if len(candidates) == 1:
            remapped[target_key] = normalized[candidates[0]]
            # Remove the old alias key so strict loading won't reject it as unexpected.
            remapped.pop(candidates[0], None)
            used_source_keys.add(candidates[0])
    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as exc:
        state_keys = list(normalized.keys())[:20] if isinstance(normalized, dict) else []
        raise RuntimeError(
            "FF++ checkpoint architecture does not exactly match this FF++-style Xception model. "
            "No training was started. If this occurs, send me the error and the FF++ model class/training script. "
            f"Example checkpoint keys: {state_keys}. Original error: {exc}"
        ) from exc
    print(f"Loaded FF++ Xception checkpoint (strict match): {checkpoint_path}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint metadata: epoch={checkpoint.get('epoch', 'n/a')}; keys={list(checkpoint.keys())}")
    return checkpoint


def auc_from_scores(y_true_fake, score_fake):
    y = np.asarray(y_true_fake, dtype=bool)
    scores = np.asarray(score_fake, dtype=np.float64)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics_at_threshold(y_fake, p_fake, threshold):
    y = np.asarray(y_fake, dtype=np.int64)  # 1=fake, 0=real
    pred = (np.asarray(p_fake) >= threshold).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(len(y), 1)
    return {
        "count": int(len(y)), "threshold_fake": float(threshold), "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2), "fake_precision": float(precision),
        "fake_recall": float(recall), "fake_f1": float(f1), "specificity_real": float(specificity),
        "roc_auc_fake": auc_from_scores(y == 1, p_fake),
        "confusion_matrix_labels_true_fake_real": [[tp, fn], [fp, tn]],
    }


def best_balanced_accuracy_threshold(y, p):
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(p, dtype=np.float64))))
    best_t, best_ba = 0.5, -1.0
    for t in candidates:
        m = metrics_at_threshold(y, p, float(t))
        if m["balanced_accuracy"] > best_ba:
            best_ba, best_t = m["balanced_accuracy"], float(t)
    return best_t


def make_loader(folder, transform, args, device, shuffle, limit_images=0, shuffle_capped_valid=False):
    raw = datasets.ImageFolder(str(folder), transform=transform)
    if raw.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping in {folder}: {raw.class_to_idx}")
    if limit_images > 0 and len(raw) > limit_images:
        indices = make_balanced_indices(raw, limit_images, args.seed)
        raw = Subset(raw, indices)
    wrapped = WildLabelAdapter(raw) if isinstance(raw, datasets.ImageFolder) else Subset(WildLabelAdapter(raw.dataset), raw.indices)
    do_shuffle = shuffle or shuffle_capped_valid
    kwargs = dict(dataset=wrapped, batch_size=args.batch_size, shuffle=do_shuffle,
                  num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=False)
    if args.workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs), len(wrapped)


def evaluate(model, loader, criterion, device, amp_enabled, max_batches=0, label="Validation"):
    model.eval()
    labels, logits = [], []
    total_loss, total = 0.0, 0
    started = time.time()
    with torch.inference_mode():
        for batch_i, (images, targets) in enumerate(loader):
            if max_batches and batch_i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)
            n = targets.size(0)
            total_loss += float(loss.item()) * n
            total += n
            labels.append(targets.cpu().numpy().astype(np.int64))
            logits.append(outputs.float().cpu().numpy())
    if labels:
        y = np.concatenate(labels)
        z = np.concatenate(logits)
        p_fake = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    else:
        y, p_fake = np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    threshold = best_balanced_accuracy_threshold(y, p_fake) if label.lower().startswith("validation") and len(y) else 0.5
    metrics = metrics_at_threshold(y, p_fake, threshold) if len(y) else {"count": 0, "roc_auc_fake": None, "accuracy": None}
    metrics["loss"] = total_loss / total if total else None
    metrics["elapsed_seconds"] = round(time.time() - started, 2)
    print(f"{label}: n={total}, loss={metrics.get('loss')}, acc={metrics.get('accuracy')}, "
          f"balanced_acc={metrics.get('balanced_accuracy')}, fake_F1={metrics.get('fake_f1')}, "
          f"fake_AUC={metrics.get('roc_auc_fake')}, threshold={metrics.get('threshold_fake')}, "
          f"elapsed={metrics['elapsed_seconds']}s", flush=True)
    return metrics, y, p_fake


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive, workers cannot be negative")
    if args.max_train_images < 0 or args.max_train_batches < 0 or args.max_valid_batches < 0:
        raise ValueError("Image/batch limits cannot be negative")
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    for split in ("train", "valid", "test"):
        if not (data_root / split).is_dir():
            raise FileNotFoundError(f"Required split not found: {data_root / split}")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_t, eval_t = make_transforms(args.image_size)
    train_loader, train_count = make_loader(data_root / "train", train_t, args, device, True, args.max_train_images)
    valid_loader, valid_count = make_loader(data_root / "valid", eval_t, args, device, False,
                                            0, shuffle_capped_valid=args.max_valid_batches > 0)
    test_loader = None
    if not args.skip_test:
        test_loader, test_count = make_loader(data_root / "test", eval_t, args, device, False)
    else:
        test_count = len(datasets.ImageFolder(str(data_root / "test")))

    print("=" * 72)
    print("WILDDEEPFAKE TRANSFER RUN — FF++ XCEPTION CHECKPOINT (READ ONLY)")
    print(f"Device={device}; GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a'}; AMP={amp_enabled}")
    print(f"Dataset={data_root}; split image counts train={train_count}, valid={valid_count}, test={test_count}")
    print(f"WildDeepfake ImageFolder labels fake=0, real=1; converted for FF++ model semantics fake=1, real=0")
    print(f"Source FF++ checkpoint={args.ffpp_checkpoint}")
    print(f"Output (separate)={output_dir}; batch={args.batch_size}; image_size={args.image_size}; epochs={args.epochs}; lr={args.lr}")
    print("=" * 72, flush=True)

    model = make_model().to(device)
    source_checkpoint = load_ffpp_checkpoint(model, Path(args.ffpp_checkpoint), device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_auc, best_epoch, no_improve, history = 0, -float("inf"), 0, 0, []
    best_path = output_dir / "checkpoints" / "best_model.pth"
    last_path = output_dir / "checkpoints" / "last_checkpoint.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_auc = float(ckpt.get("best_auc", best_auc))
        best_epoch = int(ckpt.get("best_epoch", 0))
        no_improve = int(ckpt.get("epochs_without_improvement", 0))
        history = ckpt.get("history", [])
        print(f"Resumed own WildDeepfake run: {resume_path}; next epoch={start_epoch + 1}")

    json_write(output_dir / "config.json", {**vars(args), "source_checkpoint_epoch": source_checkpoint.get("epoch") if isinstance(source_checkpoint, dict) else None,
                                             "label_semantics": "model logit positive = fake; source FF++ labels preserved"})
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, total_loss, seen = time.time(), 0.0, 0
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        for batch_i, (images, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_i >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = targets.size(0)
            total_loss += float(loss.item()) * n
            seen += n
            if (batch_i + 1) % 50 == 0:
                print(f"  batch {batch_i + 1}: mean_train_loss={total_loss / max(seen, 1):.5f}; images={seen}", flush=True)
        train_loss = total_loss / seen if seen else None
        val_metrics, _, _ = evaluate(model, valid_loader, criterion, device, amp_enabled, args.max_valid_batches, "Validation")
        val_auc = val_metrics.get("roc_auc_fake")
        score = float(val_auc) if val_auc is not None else float(val_metrics.get("balanced_accuracy") or -1.0)
        scheduler.step(score)
        improved = score > best_auc
        if improved:
            best_auc, best_epoch, no_improve = score, epoch + 1, 0
            atomic_torch_save(best_path, {
                "epoch": epoch, "model_state_dict": model.state_dict(), "best_auc": best_auc,
                "best_epoch": best_epoch, "validation_metrics": val_metrics,
                "model_name": "timm xception + FF++ compression head", "image_size": args.image_size,
                "class_semantics": {"fake": 1, "real": 0}, "source_checkpoint": str(args.ffpp_checkpoint),
            })
            print(f"  Saved new best WildDeepfake checkpoint: {best_path} (validation AUC={best_auc:.5f})", flush=True)
        else:
            no_improve += 1

        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_metrics.get("loss"),
               "validation_accuracy_at_selected_threshold": val_metrics.get("accuracy"),
               "validation_balanced_accuracy": val_metrics.get("balanced_accuracy"),
               "validation_fake_precision": val_metrics.get("fake_precision"),
               "validation_fake_recall": val_metrics.get("fake_recall"), "validation_fake_f1": val_metrics.get("fake_f1"),
               "validation_roc_auc_fake": val_metrics.get("roc_auc_fake"), "validation_threshold_fake": val_metrics.get("threshold_fake"),
               "epoch_seconds": round(time.time() - t0, 2), "lr_next_epoch": optimizer.param_groups[0]["lr"]}
        history.append(row)
        with (output_dir / "reports" / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader(); writer.writerows(history)
        json_write(output_dir / "reports" / "validation_history.json", history)
        atomic_torch_save(last_path, {
            "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc, "best_epoch": best_epoch, "epochs_without_improvement": no_improve, "history": history,
            "source_checkpoint": str(args.ffpp_checkpoint), "image_size": args.image_size,
        })
        print(f"Epoch complete: train_loss={train_loss}; best_epoch={best_epoch}; best_val_AUC={best_auc:.5f}; elapsed={time.time()-t0:.1f}s", flush=True)
        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epoch(s) without validation improvement.", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was produced; check logs and dataset.")
    report = {"experiment": "WildDeepfake transfer from FF++ compression Xception", "source_checkpoint": str(args.ffpp_checkpoint),
              "dataset_root": str(data_root), "output_dir": str(output_dir), "class_semantics": {"fake": 1, "real": 0},
              "train_images_used": train_count, "validation_images": valid_count, "test_images": test_count,
              "best_epoch": best_epoch, "best_validation_auc": best_auc, "best_checkpoint": str(best_path),
              "test_evaluated": False, "history": history}
    if test_loader is not None:
        print("\nLoading best validation checkpoint for one final test evaluation...", flush=True)
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        measured_test_metrics, test_y, test_p = evaluate(model, test_loader, criterion, device, amp_enabled, 0, "FINAL TEST")
        # Threshold selected from the validation set only, then used unchanged on test.
        val_threshold = float(best_ckpt["validation_metrics"].get("threshold_fake", 0.5))
        test_metrics = metrics_at_threshold(test_y, test_p, val_threshold)
        test_metrics["loss"] = measured_test_metrics.get("loss")
        test_metrics["elapsed_seconds"] = measured_test_metrics.get("elapsed_seconds")
        test_metrics["threshold_source"] = "validation set; fixed before test evaluation"
        report["test_evaluated"] = True
        report["test_metrics"] = test_metrics
        json_write(output_dir / "reports" / "test_results.json", test_metrics)
    json_write(output_dir / "reports" / "final_report.json", report)
    print("\nRUN COMPLETE")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Final report: {output_dir / 'reports' / 'final_report.json'}")
    if args.skip_test:
        print("Test set was not evaluated (pilot mode).")


if __name__ == "__main__":
    main()
