"""
Cross-dataset evaluation: WildDeepfake Fusion V2 -> FaceForensics++.

Uses the already-trained WildDeepfake Xception, Frequency V2, and Fusion V2
checkpoints. It does NOT train or fine-tune on FF++ and does NOT select a
threshold using the FF++ test labels.

Expected paths (edit CONFIG below if yours differ):
  D:\newdataset\wilddeepfake_experiment\transfer_xception\checkpoints\best_model.pth
  D:\newdataset\wilddeepfake_experiment\frequency_v2\checkpoints\best_frequency_v2.pth
  D:\newdataset\wilddeepfake_experiment\fusion_v2\checkpoints\best_fusion_v2.pth
  D:\faceforensics\manifests\test.csv
  D:\faceforensics\ffpp_faces_final
  D:\faceforensics\ffpp_faces_v2

Run:
  python eval_wilddeepfake_fusion_v2_to_ffpp.py

Output:
  D:\newdataset\wilddeepfake_experiment\cross_eval_wilddeepfake_fusion_v2_to_ffpp\
      wilddeepfake_fusion_v2_to_ffpp_report.json
      wilddeepfake_fusion_v2_to_ffpp_predictions.csv

Label convention in reports: real=0, fake=1.
"""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

# ------------------------------ CONFIG ---------------------------------------
FFPP_TEST_MANIFEST = Path(r"D:\faceforensics\manifests\test.csv")
FFPP_FINAL_FACE_ROOT = Path(r"D:\faceforensics\ffpp_faces_final")
FFPP_V2_FACE_ROOT = Path(r"D:\faceforensics\ffpp_faces_v2")

EXPERIMENT_ROOT = Path(r"D:\newdataset\wilddeepfake_experiment")
XCEPTION_CKPT = EXPERIMENT_ROOT / "transfer_xception" / "checkpoints" / "best_model.pth"
FREQUENCY_CKPT = EXPERIMENT_ROOT / "frequency_v2" / "checkpoints" / "best_frequency_v2.pth"
FUSION_CKPT = EXPERIMENT_ROOT / "fusion_v2" / "checkpoints" / "best_fusion_v2.pth"

OUT_DIR = EXPERIMENT_ROOT / "cross_eval_wilddeepfake_fusion_v2_to_ffpp"
REPORT_PATH = OUT_DIR / "wilddeepfake_fusion_v2_to_ffpp_report.json"
PREDICTIONS_PATH = OUT_DIR / "wilddeepfake_fusion_v2_to_ffpp_predictions.csv"

IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 8
# Keep this fixed from the WildDeepfake validation experiment. Do not tune on FF++ test.
THRESHOLD = 0.28
BATCH_SIZE = 8
NUM_WORKERS = 0  # Windows-safe
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if AMP:
    torch.cuda.manual_seed_all(SEED)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ------------------------------ MODELS ---------------------------------------
class XceptionFFPPStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "xception", pretrained=False, num_classes=0, global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class FrequencyCNNV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.50), nn.Linear(128, 1)
        )

    def forward_features(self, x):
        return torch.flatten(self.features(x), 1)

    def forward(self, x):
        return self.classifier(self.forward_features(x)).squeeze(1)


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ------------------------------ HELPERS --------------------------------------
def load_state(path, model, label, state_keys=("model_state_dict", "state_dict", "model")):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} checkpoint not found:\n{path}\n"
            "Check the configured checkpoint path before running."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = None
    if isinstance(ckpt, dict):
        for key in state_keys:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    if state is None:
        raise ValueError(f"Could not locate state_dict in {path}. Top-level keys: {list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded {label}: {path}")
    return ckpt


def build_frequency_input(rgb_normalized):
    """Match the WildDeepfake Frequency V2 training preprocessing."""
    mean = torch.tensor(MEAN, dtype=rgb_normalized.dtype, device=rgb_normalized.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=rgb_normalized.dtype, device=rgb_normalized.device).view(1, 3, 1, 1)
    rgb = (rgb_normalized * std + mean).clamp(0, 1)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    spectrum = torch.log1p(torch.abs(torch.fft.fftshift(
        torch.fft.fft2(gray), dim=(-2, -1)
    )))
    mu = spectrum.mean(dim=(-2, -1), keepdim=True)
    sd = spectrum.std(dim=(-2, -1), keepdim=True, unbiased=False)
    spectrum = (spectrum - mu) / (sd + 1e-6)
    return spectrum.clamp(-5, 5)


def frame_dir_candidates(video_name, label):
    """
    Return preferred per-video crop directories. FF++ convention:
    final/original/<video>.mp4/ for real and final/fake/<video>.mp4/ for fake.
    V2 contains fake crops only, so it is a fallback for fake videos only.
    """
    video_name = str(video_name).strip()
    if not video_name:
        return []
    if not video_name.lower().endswith(".mp4"):
        video_name += ".mp4"

    if label == 0:
        return [FFPP_FINAL_FACE_ROOT / "original" / video_name]
    candidates = [
        FFPP_FINAL_FACE_ROOT / "fake" / video_name,
        FFPP_V2_FACE_ROOT / "fake" / video_name,
    ]
    return candidates


def list_frames(video_name, label):
    for folder in frame_dir_candidates(video_name, label):
        if folder.is_dir():
            frames = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            )
            if len(frames) >= 1:
                # Use one crop source per video; never mix final and V2 frames.
                return frames, str(folder)
    return [], ""


def choose_frames(frames, count):
    """Deterministic, evenly spaced frame selection."""
    if len(frames) <= count:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=count, dtype=int)
    return [frames[i] for i in idx]


def load_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f"FF++ test manifest not found: {path}")
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Manifest must have columns 'video,label'; found {reader.fieldnames}")
        for row in reader:
            name = str(row["video"]).strip()
            try:
                label = int(row["label"])
            except Exception:
                continue
            if label not in (0, 1):
                continue
            rows.append((name, label))
    return rows


@torch.inference_mode()
def predict_frame_batch(paths, xception, frequency, fusion):
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(TRANSFORM(im.convert("RGB")))
    rgb = torch.stack(images).to(DEVICE, non_blocking=True)
    freq = build_frequency_input(rgb)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
        rgb_features = xception.backbone(rgb)
        freq_features = frequency.forward_features(freq)
        combined = torch.cat([rgb_features.float(), freq_features.float()], dim=1)
        logits = fusion(combined)
    return torch.sigmoid(logits.float()).cpu().numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("CROSS-DATASET: WildDeepfake Fusion V2 -> FaceForensics++")
    print("Device:", DEVICE, "| CUDA available:", torch.cuda.is_available())
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Threshold fixed from WildDeepfake validation:", THRESHOLD)
    print("No FF++ fine-tuning and no threshold tuning on FF++ test.")
    print("=" * 80)

    xception = XceptionFFPPStyle()
    load_state(XCEPTION_CKPT, xception, "WildDeepfake Xception")
    frequency = FrequencyCNNV2()
    load_state(FREQUENCY_CKPT, frequency, "WildDeepfake Frequency V2")
    fusion = FusionHead()
    fusion_ckpt = load_state(FUSION_CKPT, fusion, "WildDeepfake Fusion V2")

    xception.to(DEVICE).eval()
    frequency.to(DEVICE).eval()
    fusion.to(DEVICE).eval()
    for model in (xception, frequency, fusion):
        for p in model.parameters():
            p.requires_grad = False

    # Validate feature sizes before reading the full test set.
    with torch.inference_mode():
        rgb_dummy = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        freq_dummy = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        dims = (xception.backbone(rgb_dummy).shape[1], frequency.forward_features(freq_dummy).shape[1])
    print(f"Feature dimensions: RGB={dims[0]}, frequency={dims[1]}, fused={sum(dims)}")
    if dims != (2048, 256):
        raise RuntimeError(f"Checkpoint/model feature dimensions unexpected: {dims}; expected (2048, 256)")

    manifest_rows = load_manifest(FFPP_TEST_MANIFEST)
    y_true, y_probs, records = [], [], []
    skipped = []
    started = time.time()
    processed_frames = 0
    source_counts = {"final_original": 0, "final_fake": 0, "v2_fake_fallback": 0}

    for i, (video, label) in enumerate(manifest_rows, start=1):
        frames, source = list_frames(video, label)
        if len(frames) < 1:
            skipped.append({"video": video, "label": label, "reason": "no usable crop frames"})
            continue

        selected = choose_frames(frames, FRAMES_PER_VIDEO)
        frame_probs = []
        # Small batches keep the script safe on 4-GB laptop GPUs.
        for start in range(0, len(selected), BATCH_SIZE):
            batch_paths = selected[start:start + BATCH_SIZE]
            frame_probs.extend(predict_frame_batch(batch_paths, xception, frequency, fusion).tolist())
        processed_frames += len(selected)

        # Model was trained on individual WildDeepfake images; average frame probabilities
        # to produce one video-level FF++ score.
        video_prob = float(np.mean(frame_probs))
        pred = int(video_prob >= THRESHOLD)
        y_true.append(label)
        y_probs.append(video_prob)
        if "ffpp_faces_v2" in source.lower():
            source_key = "v2_fake_fallback"
        elif source.endswith("original\\" + (video if video.lower().endswith(".mp4") else video + ".mp4")) or "\\original\\" in source.lower():
            source_key = "final_original"
        else:
            source_key = "final_fake"
        source_counts[source_key] += 1
        records.append({
            "video": video,
            "true_label": label,
            "true_class": "fake" if label == 1 else "real",
            "fake_probability_mean_frames": video_prob,
            "predicted_label": pred,
            "predicted_class": "fake" if pred == 1 else "real",
            "frames_used": len(selected),
            "available_frames_in_source": len(frames),
            "crop_source": source,
        })

        if i % 100 == 0 or i == len(manifest_rows):
            print(f"Manifest rows {i}/{len(manifest_rows)} | usable videos {len(y_true)} | "
                  f"frames processed {processed_frames} | elapsed {(time.time()-started)/60:.1f} min",
                  flush=True)

    if not y_true:
        raise RuntimeError("No FF++ test videos had usable face crops. Check crop-root paths and manifest naming.")

    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_probs, dtype=np.float64)
    pred = (probs >= THRESHOLD).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fake_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "fake_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "real_specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        "average_precision_fake": float(average_precision_score(y, probs)) if len(np.unique(y)) == 2 else None,
        "threshold": THRESHOLD,
        "confusion_matrix_labels_0_real_1_fake": cm.tolist(),
        "n_test_videos": int(len(y)),
        "n_real": int(np.sum(y == 0)),
        "n_fake": int(np.sum(y == 1)),
    }

    report = {
        "experiment": "Cross-dataset: WildDeepfake Fusion V2 evaluated on FaceForensics++",
        "source_dataset": "WildDeepfake",
        "target_dataset": "FaceForensics++",
        "evaluation_unit": "video-level; mean fusion probability across up to 8 sampled frames per video",
        "source_checkpoints": {
            "xception": str(XCEPTION_CKPT),
            "frequency_v2": str(FREQUENCY_CKPT),
            "fusion_v2": str(FUSION_CKPT),
        },
        "target_manifest": str(FFPP_TEST_MANIFEST),
        "target_final_face_root": str(FFPP_FINAL_FACE_ROOT),
        "target_v2_face_root": str(FFPP_V2_FACE_ROOT),
        "training_or_finetuning_on_target": False,
        "threshold_selection": "fixed at 0.28 from WildDeepfake validation; not tuned on FF++ test",
        "frames_per_video_max": FRAMES_PER_VIDEO,
        "frame_sampling": "deterministic evenly spaced; mean probability over sampled frames",
        "feature_dimensions": {"xception_rgb": 2048, "frequency_v2": 256, "concatenated": 2304},
        "crop_source_counts": source_counts,
        "skipped_manifest_rows": skipped,
        "n_manifest_rows": len(manifest_rows),
        "n_processed_frames": processed_frames,
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 2),
        "fusion_checkpoint_epoch": fusion_ckpt.get("epoch") if isinstance(fusion_ckpt, dict) else None,
        "notes": [
            "No FF++ target training or threshold tuning was performed.",
            "WildDeepfake Fusion V2 was trained on individual images; frame probabilities are averaged to obtain a video-level score.",
            "FF++ V2 crop fallback is used only for fake videos because the V2 crop set contains no real videos.",
            "Only one crop source is used per video; frames from different crop versions are never mixed.",
            "The number of evaluated videos may be lower than the manifest count due to missing or insufficient face crops.",
            "The fixed threshold is the validation-selected threshold recorded for WildDeepfake Fusion V2; thresholded metrics reflect this operating point."
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 80)
    print("FINAL RESULTS — WILDDEEPFAKE FUSION V2 -> FF++")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key:42s}: {value}")
    print("Crop sources:", source_counts)
    print("Skipped manifest rows:", len(skipped))
    print("Report:", REPORT_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
