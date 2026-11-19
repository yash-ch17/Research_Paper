"""
Cross-dataset evaluation: WildDeepfake-trained Xception -> FaceForensics++ test set.
Evaluation only: no training/fine-tuning on FF++ and no threshold tuning on its test set.

Default checkpoint:
  D:\\newdataset\\wilddeepfake_experiment\\transfer_xception\\checkpoints\\best_model.pth
Default FF++ face-crop dataset:
  D:\\faceforensics\\ffpp_faces_final
  Expected layout: test\\real and test\\fake (or test\\0 and test\\1 if class mapping is explicitly changed).

IMPORTANT:
The default dataset loader expects ImageFolder-style test folders:
  D:\\faceforensics\\ffpp_faces_final\\test\\fake
  D:\\faceforensics\\ffpp_faces_final\\test\\real

If your FF++ dataset is organised by video/manifests instead, set --data-root to an
ImageFolder-style directory containing held-out FF++ test images. Do not use training
or validation images as the target test set.

Run:
  python D:\\newdataset\\cross_eval_wilddeepfake_to_ffpp.py

Optional:
  python cross_eval_wilddeepfake_to_ffpp.py --threshold 0.50
  python cross_eval_wilddeepfake_to_ffpp.py --data-root "D:\\path\\to\\ffpp_imagefolder"
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)

DEFAULT_CHECKPOINT = Path(
    r"D:\newdataset\wilddeepfake_experiment\transfer_xception\checkpoints\best_model.pth"
)
DEFAULT_DATA_ROOT = Path(r"D:\faceforensics\ffpp_faces_final")
DEFAULT_TEST_MANIFEST = Path(r"D:\faceforensics\manifests\test.csv")
DEFAULT_OUTPUT = Path(
    r"D:\newdataset\wilddeepfake_experiment\cross_eval_wilddeepfake_to_ffpp"
)
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class XceptionFFPPStyle(nn.Module):
    """Matches the WildDeepfake transfer Xception model: backbone + 2048->256->1 head."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "xception", pretrained=False, num_classes=0, global_pool="avg"
        )
        if getattr(self.backbone, "num_features", None) != 2048:
            raise RuntimeError(f"Expected 2048 Xception features, got {self.backbone.num_features}")
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def get_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary.")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(f"No model state dict found. Top-level keys: {list(checkpoint.keys())[:40]}")


def load_checkpoint_strict(model, path):
    if not path.is_file():
        raise FileNotFoundError(
            f"WildDeepfake Xception checkpoint not found: {path}\n"
            "Pass the correct checkpoint using --wild-checkpoint. No training was started."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = get_state(checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    target = model.state_dict()
    mapped = dict(state)
    used = set()
    for target_key, target_tensor in target.items():
        if target_key in mapped:
            used.add(target_key)
            continue
        if not target_key.startswith("head."):
            continue
        shape = tuple(target_tensor.shape)
        candidates = [
            k for k, value in state.items()
            if not k.startswith("backbone.")
            and tuple(value.shape) == shape
            and any(s in k.lower() for s in ("head", "classifier", "fc", "mlp"))
            and k not in used
        ]
        if len(candidates) == 1:
            mapped[target_key] = state[candidates[0]]
            mapped.pop(candidates[0], None)
            used.add(candidates[0])
    try:
        model.load_state_dict(mapped, strict=True)
    except RuntimeError as exc:
        keys = list(state.keys())[:25]
        raise RuntimeError(
            "The checkpoint does not exactly match this Xception + 2048->256->1 architecture. "
            "This script stopped safely; no weights were silently ignored. "
            f"Sample checkpoint keys: {keys}. Original error: {exc}"
        ) from exc
    print(f"Loaded WildDeepfake Xception checkpoint: {path}")
    if isinstance(checkpoint, dict):
        for key in ("epoch", "best_epoch", "val_auc", "best_auc", "best_score", "image_size"):
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")
    return checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wild-checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="ImageFolder root with test/fake and test/real subfolders.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Fixed fake-probability threshold; not tuned on FF++ test data.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--test-manifest", default=str(DEFAULT_TEST_MANIFEST),
                   help="Existing held-out FF++ test.csv with columns video,label.")
    p.add_argument("--frames-per-video", type=int, default=8,
                   help="Number of evenly spaced frames sampled per test video.")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.wild_checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    manifest_path = Path(args.test_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FF++ held-out test manifest not found: {manifest_path}")
    if args.frames_per_video < 1:
        raise ValueError("--frames-per-video must be >= 1")

    # The existing FF++ pipeline stores frames as original/<video>/*.jpg for label 0
    # and fake/<video>/*.jpg for label 1. The test manifest is video-level.
    video_rows = []
    skipped = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"video", "label"}.issubset(set(reader.fieldnames)):
            raise ValueError(f"Expected columns 'video' and 'label' in {manifest_path}; found {reader.fieldnames}")
        for row in reader:
            video = row["video"]
            label = int(row["label"])  # FF++ convention: 0=real, 1=fake
            folder = data_root / ("original" if label == 0 else "fake") / video
            frames = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
            if not frames:
                skipped.append({"video": video, "reason": f"no JPG frames in {folder}"})
                continue
            # Evenly sample frames across the video; use all if fewer than requested.
            n = min(args.frames_per_video, len(frames))
            indices = np.linspace(0, len(frames) - 1, num=n, dtype=int)
            selected = [frames[i] for i in sorted(set(indices.tolist()))]
            video_rows.append({"video": video, "label": label, "frames": selected})
    if not video_rows:
        raise RuntimeError("No usable test videos found. Check --test-manifest and --data-root.")
    frame_records = [(row["video"], row["label"], path) for row in video_rows for path in row["frames"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    print("=" * 76)
    print("CROSS-DATASET EVALUATION: WILDDEEPFAKE XCEPTION -> FF++ TEST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Source checkpoint: {checkpoint_path}")
    print(f"Target test manifest: {manifest_path}")
    print(f"FF++ face root: {data_root}")
    print(f"Usable test videos: {len(video_rows):,} | sampled frames: {len(frame_records):,}")
    print(f"Skipped manifest rows with missing frames: {len(skipped)}")
    print(f"Fixed fake-probability threshold: {args.threshold:.4f}")
    print("No fine-tuning. No threshold selection on FF++ test set.")
    print("Video score = mean probability across sampled frames.")
    print("=" * 76, flush=True)

    model = XceptionFFPPStyle()
    checkpoint = load_checkpoint_strict(model, checkpoint_path)
    model.to(device).eval()

    video_probs = {}
    video_targets = {}
    started = time.time()
    batch_size = args.batch_size
    for base in range(0, len(frame_records), batch_size):
        records = frame_records[base:base + batch_size]
        tensors = []
        for _, _, img_path in records:
            with Image.open(img_path) as im:
                tensors.append(transform(im.convert("RGB")))
        images = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            probs_batch = torch.sigmoid(logits.float()).cpu().numpy().tolist()
        for (video, label, _), prob in zip(records, probs_batch):
            video_probs.setdefault(video, []).append(float(prob))
            video_targets[video] = int(label)
        processed = min(base + len(records), len(frame_records))
        if processed % (batch_size * 50) == 0 or processed == len(frame_records):
            print(f"Processed {processed:,}/{len(frame_records):,} frames ({time.time()-started:.1f}s)", flush=True)

    # Evaluate one averaged score per held-out video, not each correlated frame as an independent sample.
    ordered_videos = [row["video"] for row in video_rows]
    all_targets = [video_targets[v] for v in ordered_videos]
    all_probs = [float(np.mean(video_probs[v])) for v in ordered_videos]
    all_paths = [";".join(str(pth) for pth in row["frames"]) for row in video_rows]
    y_true = np.asarray(all_targets, dtype=np.int64)
    probs = np.asarray(all_probs, dtype=np.float64)
    y_pred = (probs >= args.threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "real_specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision_fake": float(average_precision_score(y_true, probs)),
        "threshold": float(args.threshold),
        "confusion_matrix_labels_0_real_1_fake": cm.tolist(),
        "n_test_videos": int(len(y_true)),
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
    }
    report = {
        "experiment": "Cross-dataset: WildDeepfake Xception checkpoint evaluated on FF++",
        "source_dataset": "WildDeepfake",
        "target_dataset": "FaceForensics++",
        "source_checkpoint": str(checkpoint_path),
        "target_test_manifest": str(manifest_path),
        "target_face_root": str(data_root),
        "skipped_manifest_rows": skipped,
        "frames_per_video_requested": int(args.frames_per_video),
        "evaluation_unit": "video-level; mean probability across sampled frames per held-out test video",
        "training_or_finetuning_on_target": False,
        "threshold_selection": "fixed 0.50 by default; not tuned on FF++ test set",
        "threshold": float(args.threshold),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": MEAN, "std": STD},
        "model_checkpoint_metadata": {
            key: checkpoint[key] for key in
            ("epoch", "best_epoch", "val_auc", "best_auc", "best_score", "image_size")
            if isinstance(checkpoint, dict) and key in checkpoint
        },
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 2),
        "note": (
            "Video-level cross-dataset result from the existing FF++ held-out test manifest. "
            "For publication-grade thresholded metrics, use a threshold selected on held-out WildDeepfake "
            "validation data if available; never tune threshold on the FF++ test set."
        ),
    }
    report_path = output_dir / "wilddeepfake_to_ffpp_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "wilddeepfake_to_ffpp_predictions.npz",
                        y_true=y_true, probabilities=probs, y_pred=y_pred,
                        paths=np.asarray(all_paths, dtype=str))

    print("\n" + "=" * 76)
    print("FINAL CROSS-DATASET RESULTS")
    print("=" * 76)
    for key, value in metrics.items():
        print(f"{key:42s}: {value}")
    print(f"Report: {report_path}")
    print(f"Predictions: {output_dir / 'wilddeepfake_to_ffpp_predictions.npz'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
