#!/usr/bin/env python
"""Train an isolated Xception baseline on a train/valid/test ImageFolder dataset.

Expected layout:
  Dataset/train/fake, Dataset/train/real
  Dataset/valid/fake, Dataset/valid/real
  Dataset/test/fake,  Dataset/test/real

Fake is class index 0 and real is class index 1 (ImageFolder alphabetical order).
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_CLASSES = {"fake": 0, "real": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WildDeepfake Xception baseline")
    parser.add_argument("--data-dir", default=r"D:\newdataset\Dataset", help="Dataset root containing train/valid/test")
    parser.add_argument("--output-dir", default=r"D:\newdataset\wilddeepfake_experiment\run1", help="This run's output folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience on validation AUC")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Pilot only: 0 means all batches")
    parser.add_argument("--max-valid-batches", type=int, default=0, help="Pilot only: 0 means all batches")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate test set (use for pilot)")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize Xception randomly; not recommended")
    parser.add_argument("--resume", default="", help="Path to a last_checkpoint.pth from the same run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def make_loader(path: Path, transform, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    ds = datasets.ImageFolder(str(path), transform=transform)
    if ds.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class mapping in {path}: {ds.class_to_idx}; expected {EXPECTED_CLASSES}. "
            "The code assumes fake=0 and real=1."
        )
    kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def auc_from_scores(y_fake: np.ndarray, score_fake: np.ndarray):
    """ROC-AUC via average ranks, avoiding a scikit-learn dependency."""
    y_fake = np.asarray(y_fake, dtype=np.bool_)
    score_fake = np.asarray(score_fake, dtype=np.float64)
    positives = int(y_fake.sum())
    negatives = int((~y_fake).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score_fake, kind="mergesort")
    sorted_scores = score_fake[order]
    ranks = np.empty(len(score_fake), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # 1-based average rank for ties
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    rank_sum_positive = ranks[y_fake].sum()
    return float((rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives))


def metrics_from_outputs(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {"count": 0, "accuracy": None, "balanced_accuracy": None, "fake_precision": None,
                "fake_recall": None, "fake_f1": None, "real_recall": None, "specificity": None, "roc_auc_fake": None,
                "confusion_matrix_labels_fake_real": None}
    # By design, class 0 = fake and class 1 = real.
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    p_fake = probs[:, 0]
    predicted = (p_fake >= 0.5).astype(np.int64)  # predictions in ImageFolder class indices
    labels = labels.astype(np.int64)

    tp_fake = int(np.sum((labels == 0) & (predicted == 0)))
    fn_fake = int(np.sum((labels == 0) & (predicted == 1)))
    fp_fake = int(np.sum((labels == 1) & (predicted == 0)))
    tn_real = int(np.sum((labels == 1) & (predicted == 1)))
    accuracy = float(np.mean(predicted == labels))
    fake_precision = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) else 0.0
    fake_recall = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) else 0.0
    real_recall = tn_real / (tn_real + fp_fake) if (tn_real + fp_fake) else 0.0
    specificity = real_recall
    fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall) if (fake_precision + fake_recall) else 0.0
    return {
        "count": int(len(labels)),
        "accuracy": accuracy,
        "balanced_accuracy": float((fake_recall + real_recall) / 2.0),
        "fake_precision": float(fake_precision),
        "fake_recall": float(fake_recall),
        "fake_f1": float(fake_f1),
        "real_recall": float(real_recall),
        "specificity": float(specificity),
        "roc_auc_fake": auc_from_scores(labels == 0, p_fake),
        "confusion_matrix_labels_fake_real": [[tp_fake, fn_fake], [fp_fake, tn_real]],
        "decision_rule": "predict fake when softmax probability for fake >= 0.5",
    }


def evaluate(model, loader, criterion, device, amp_enabled: bool, max_batches: int = 0, label: str = "Valid"):
    model.eval()
    all_labels, all_logits = [], []
    total_loss, total_count = 0.0, 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            batch_count = labels.size(0)
            total_loss += float(loss.item()) * batch_count
            total_count += batch_count
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())
    labels_np = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    logits_np = np.concatenate(all_logits) if all_logits else np.empty((0, 2), dtype=np.float32)
    metrics = metrics_from_outputs(labels_np, logits_np)
    metrics["loss"] = total_loss / total_count if total_count else None
    print(f"{label}: n={total_count}, loss={metrics['loss']}, accuracy={metrics['accuracy']}, "
          f"balanced_acc={metrics['balanced_accuracy']}, fake_F1={metrics['fake_f1']}, "
          f"fake_AUC={metrics['roc_auc_fake']}, elapsed={time.time()-started:.1f}s", flush=True)
    return metrics


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    if args.max_train_batches < 0 or args.max_valid_batches < 0:
        raise ValueError("batch limits cannot be negative")

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    needed = [data_root / split for split in ("train", "valid", "test")]
    for path in needed:
        if not path.is_dir():
            raise FileNotFoundError(f"Required split folder not found: {path}")

    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = make_loader(data_root / "train", train_transform, args.batch_size, args.workers, True, device)
    # Shuffle a capped validation pilot so a small slice is not just one alphabetically ordered class.
    valid_shuffle = args.max_valid_batches > 0
    valid_loader = make_loader(data_root / "valid", eval_transform, args.batch_size, args.workers, valid_shuffle, device)
    test_loader = None
    if not args.skip_test:
        test_loader = make_loader(data_root / "test", eval_transform, args.batch_size, args.workers, False, device)

    print("=" * 68)
    print("WILDDEEPFAKE XCEPTION BASELINE — SEPARATE EXPERIMENT")
    print(f"Device: {device}; mixed precision: {amp_enabled}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a'}")
    print(f"Dataset root: {data_root}")
    print(f"Class mapping: {train_loader.dataset.class_to_idx}")
    print(f"Split sizes: train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, test={len(datasets.ImageFolder(str(data_root / 'test')))}")
    print(f"Batch size: {args.batch_size}; workers: {args.workers}; epochs: {args.epochs}")
    if args.max_train_batches or args.max_valid_batches:
        print(f"PILOT MODE: max_train_batches={args.max_train_batches or 'all'}, max_valid_batches={args.max_valid_batches or 'all'}")
    print("=" * 68, flush=True)

    try:
        model = timm.create_model("xception", pretrained=not args.no_pretrained, num_classes=2)
    except Exception as exc:
        if not args.no_pretrained:
            raise RuntimeError(
                "Could not initialize pretrained Xception weights. Check the internet connection and retry. "
                "For a quick pipeline-only check, add --no-pretrained (not recommended for final results)."
            ) from exc
        raise
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = checkpoint.get("history", [])
        print(f"Resumed from {resume_path}; next epoch={start_epoch + 1}, current best score={best_score}")

    write_json(output_dir / "config.json", vars(args))
    best_path = output_dir / "checkpoints" / "best_model.pth"
    last_path = output_dir / "checkpoints" / "last_checkpoint.pth"
    best_path.parent.mkdir(exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        running_loss, seen = 0.0, 0
        max_steps = args.max_train_batches
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        for batch_index, (images, labels) in enumerate(train_loader):
            if max_steps and batch_index >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = labels.size(0)
            running_loss += float(loss.item()) * count
            seen += count
            if (batch_index + 1) % 50 == 0:
                print(f"  batch {batch_index + 1}: mean_train_loss={running_loss / max(seen, 1):.4f}; images={seen}", flush=True)

        train_loss = running_loss / seen if seen else None
        val_metrics = evaluate(model, valid_loader, criterion, device, amp_enabled, args.max_valid_batches, "Validation")
        val_score = val_metrics.get("roc_auc_fake")
        if val_score is None:
            val_score = val_metrics.get("accuracy")
        if val_score is None:
            val_score = -float("inf")
        scheduler.step(val_score)
        improved = val_score > best_score
        if improved:
            best_score = float(val_score)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(best_path, {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_to_idx": EXPECTED_CLASSES,
                "model_name": "xception",
                "image_size": args.image_size,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "validation_metrics": val_metrics,
            })
            print(f"  Saved new best model: {best_path} (validation score={best_score:.5f})", flush=True)
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_metrics.get("loss"),
            "validation_accuracy": val_metrics.get("accuracy"),
            "validation_balanced_accuracy": val_metrics.get("balanced_accuracy"),
            "validation_fake_precision": val_metrics.get("fake_precision"),
            "validation_fake_recall": val_metrics.get("fake_recall"),
            "validation_fake_f1": val_metrics.get("fake_f1"),
            "validation_roc_auc_fake": val_metrics.get("roc_auc_fake"),
            "learning_rate_next_epoch": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - started, 2),
        }
        history.append(row)
        with (output_dir / "reports" / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        write_json(output_dir / "reports" / "validation_history.json", history)
        save_checkpoint(last_path, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "class_to_idx": EXPECTED_CLASSES,
            "model_name": "xception",
            "image_size": args.image_size,
        })
        train_loss_text = f"{train_loss:.5f}" if train_loss is not None else "n/a"
        print(f"Epoch complete: train_loss={train_loss_text}; best_epoch={best_epoch}; "
              f"best_val_score={best_score:.5f}; elapsed={time.time()-started:.1f}s", flush=True)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no validation improvement for {args.patience} epoch(s).", flush=True)
            break

    final_report = {
        "experiment": "WildDeepfake Xception baseline",
        "data_dir": str(data_root),
        "output_dir": str(output_dir),
        "class_to_idx": EXPECTED_CLASSES,
        "best_epoch": best_epoch,
        "best_validation_score_auc_or_accuracy_fallback": best_score,
        "best_checkpoint": str(best_path),
        "test_evaluated": False,
        "history": history,
    }
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was written; check whether training batches were processed.")

    if test_loader is not None:
        print("\nLoading best validation checkpoint for the one final test evaluation...", flush=True)
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, amp_enabled, 0, "FINAL TEST")
        final_report["test_evaluated"] = True
        final_report["test_metrics"] = test_metrics
        write_json(output_dir / "reports" / "test_results.json", test_metrics)
    write_json(output_dir / "reports" / "final_report.json", final_report)

    print("\nRUN COMPLETE", flush=True)
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Final report: {output_dir / 'reports' / 'final_report.json'}")
    if args.skip_test:
        print("Test set was not evaluated (as requested for pilot).", flush=True)


if __name__ == "__main__":
    # Required on Windows when DataLoader uses worker subprocesses.
    main()
