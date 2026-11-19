
import os
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

FACE_ROOT = Path(r"D:\faceforensics\ffpp_faces_final")
TRAIN_CSV = Path(r"D:\faceforensics\manifests\train.csv")
VAL_CSV = Path(r"D:\faceforensics\manifests\val.csv")
TEST_CSV = Path(r"D:\faceforensics\manifests\test.csv")

XCEPTION_CKPT = Path(
    r"D:\faceforensics\checkpoints_xception_compression\best_compressed.pth"
)
FREQUENCY_CKPT = Path(
    r"D:\faceforensics\checkpoints_frequency\best_frequency.pth"
)

OUTPUT_DIR = Path(r"D:\faceforensics\checkpoints_fusion_improved")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUTPUT_DIR / "best_fusion.pth"
REPORT_JSON = OUTPUT_DIR / "final_report_fusion.json"

IMG_SIZE = 224
NUM_FRAMES = 8

BATCH_SIZE = 1
GRAD_ACCUM = 8
NUM_WORKERS = 0

EPOCHS = 16
PATIENCE = 5

# Tiny learning rates for already-trained branches.
XCEPTION_LR = 5e-7
FREQUENCY_LR = 1e-6
FUSION_LR = 1e-4
WEIGHT_DECAY = 5e-4

DROPOUT = 0.40

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

print("=" * 72)
print("DEEPFAKE RGB + FREQUENCY FEATURE FUSION")
print("=" * 72)
print("Device:", DEVICE)
print("Output directory:", OUTPUT_DIR)
print("Face root:", FACE_ROOT)
print("Train:", TRAIN_CSV)
print("Val  :", VAL_CSV)
print("Test :", TEST_CSV)
print("Xception checkpoint:", XCEPTION_CKPT)
print("Frequency checkpoint:", FREQUENCY_CKPT)
print("=" * 72)


# ============================================================
# CSV / DATASET
# ============================================================

def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header[:2] != ["video", "label"]:
            raise RuntimeError(
                f"Unexpected manifest header in {path}: {header}"
            )

        for line in f:
            line = line.strip()
            if not line:
                continue
            video, label = line.rsplit(",", 1)
            rows.append((video, int(label)))

    return rows


def video_folder(face_root, video_name, label):
    # Your final dataset uses:
    #   fake/      for label 1
    #   original/  for label 0
    folder_name = "fake" if label == 1 else "original"
    return face_root / folder_name / video_name


def available_frames(folder):
    if not folder.exists():
        return []

    frames = sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    return frames


class FusionDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.augment = augment
        raw = read_manifest(csv_path)

        self.samples = []
        missing = 0
        short = 0

        for video, label in raw:
            folder = video_folder(FACE_ROOT, video, label)
            frames = available_frames(folder)

            if len(frames) == 0:
                missing += 1
                continue

            if len(frames) < 4:
                short += 1
                continue

            self.samples.append(
                {
                    "video": video,
                    "label": label,
                    "folder": folder,
                    "frames": frames,
                }
            )

        print(
            f"{csv_path.name}: usable={len(self.samples)}, "
            f"missing={missing}, <4 frames={short}"
        )

    def __len__(self):
        return len(self.samples)

    def choose_frames(self, frames):
        if len(frames) >= NUM_FRAMES:
            # Evenly sample the available frames so all videos use the
            # same temporal length without requiring preprocessing.
            idx = np.linspace(
                0, len(frames) - 1, NUM_FRAMES
            ).round().astype(int)
            return [frames[i] for i in idx]

        # Repeat frames if a rare video has 4-7 valid crops.
        result = list(frames)
        while len(result) < NUM_FRAMES:
            result.append(result[len(result) % len(frames)])
        return result[:NUM_FRAMES]

    def load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        if self.augment:
            # Apply the same image-level transformations before BOTH RGB
            # normalization and FFT generation, so the two branches see
            # corresponding versions of each frame.
            if random.random() < 0.5:
                img = ImageOps.mirror(img)

            # Mild camera / color variation.
            if random.random() < 0.35:
                img = ImageEnhance.Brightness(img).enhance(
                    random.uniform(0.85, 1.15)
                )
            if random.random() < 0.35:
                img = ImageEnhance.Contrast(img).enhance(
                    random.uniform(0.85, 1.15)
                )
            if random.random() < 0.20:
                img = ImageEnhance.Color(img).enhance(
                    random.uniform(0.85, 1.15)
                )

            # Simulate compression artifacts.
            if random.random() < 0.55:
                quality = random.randint(35, 90)
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                with Image.open(buf) as compressed:
                    img = compressed.convert("RGB")

            # Simulate low-resolution uploads followed by upscaling.
            if random.random() < 0.25:
                side = random.choice([112, 144, 168, 192])
                img = img.resize((side, side), Image.Resampling.BILINEAR)
                img = img.resize(
                    (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
                )

            # A small amount of blur, not applied to every frame.
            if random.random() < 0.12:
                img = img.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8))
                )

        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)

        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=x.dtype
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=x.dtype
        ).view(3, 1, 1)

        x = (x - mean) / std
        return x

    def rgb_to_frequency(self, rgb):
        # Reverse ImageNet normalization first.
        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=rgb.dtype
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=rgb.dtype
        ).view(3, 1, 1)

        img = (rgb * std + mean).clamp(0.0, 1.0)

        gray = (
            0.299 * img[0]
            + 0.587 * img[1]
            + 0.114 * img[2]
        )

        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft)
        mag = torch.abs(fft)

        freq = torch.log1p(mag)

        # Same per-image normalization concept used by Frequency V1.
        freq = (freq - freq.mean()) / (freq.std() + 1e-6)
        freq = freq.unsqueeze(0)

        return freq

    def __getitem__(self, idx):
        item = self.samples[idx]
        selected = self.choose_frames(item["frames"])

        rgb_frames = []
        freq_frames = []

        for frame_path in selected:
            rgb = self.load_rgb(frame_path)
            freq = self.rgb_to_frequency(rgb)

            rgb_frames.append(rgb)
            freq_frames.append(freq)

        rgb = torch.stack(rgb_frames, dim=0)
        freq = torch.stack(freq_frames, dim=0)

        label = torch.tensor(
            float(item["label"]), dtype=torch.float32
        )

        return rgb, freq, label, item["video"]


def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# EXACT XCEPTION ARCHITECTURE USED BY COMPRESSION CHECKPOINT
# ============================================================

class XceptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logit = self.head(feat)
        return feat, logit


# ============================================================
# EXACT FREQUENCY V1 ARCHITECTURE
#
# This is copied from the user's actual train_frequency.py.
# IMPORTANT:
# classifier indices are:
#   classifier.1 -> Linear(256,128)
#   classifier.4 -> Linear(128,1)
# ============================================================

class FrequencyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1, 32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # [B,T,1,H,W]
        batch_size = x.shape[0]
        time_steps = x.shape[1]

        channels = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]

        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        features = self.features(x)
        features = features.flatten(1)

        features = features.reshape(
            batch_size,
            time_steps,
            -1,
        )

        frame_features = features.reshape(
            batch_size * time_steps,
            -1,
        )

        frame_logits = self.classifier(frame_features)

        frame_logits = frame_logits.reshape(
            batch_size,
            time_steps,
        )

        return features, frame_logits


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):
    """
    Supports:
      checkpoint["model"]
      checkpoint["model_state_dict"]
      checkpoint["state_dict"]
      direct state_dict
    Also supports an nn.Module under one of those keys.
    """
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    for key in ("model", "model_state_dict", "state_dict"):
        if key in checkpoint:
            value = checkpoint[key]

            if isinstance(value, nn.Module):
                return value.state_dict()

            if isinstance(value, dict):
                return value

    # Direct state dict fallback.
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    raise RuntimeError(
        "Could not find a model state_dict in checkpoint."
    )


def compatibility_report(model, state_dict, name):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    print()
    print("=" * 72)
    print(f"CHECKPOINT CHECK: {name}")
    print("=" * 72)
    print("Checkpoint tensor keys:", len(ckpt_keys))
    print("Expected model keys  :", len(model_keys))

    if missing:
        print("\nMissing keys:")
        for k in missing[:20]:
            print("  ", k)

    if unexpected:
        print("\nUnexpected keys:")
        for k in unexpected[:20]:
            print("  ", k)

    if not missing and not unexpected:
        print("STATUS: EXACT MATCH")
        return True

    return False


def load_branch_checkpoint(model, path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} checkpoint does not exist:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    ok = compatibility_report(
        model,
        state_dict,
        name,
    )

    if not ok:
        raise RuntimeError(
            f"\n{name} checkpoint is NOT compatible.\n"
            "The training will not continue with partially loaded weights."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(f"{name}: checkpoint loaded successfully.")
    return checkpoint


# ============================================================
# FUSION MODEL
# ============================================================

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb = XceptionModel()
        self.frequency = FrequencyCNN()

        # Xception feature = 2048
        # Frequency feature = 256
        fusion_dim = 2048 + 256

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 1),
        )

        self.freeze_branches = True

    def set_branch_trainable(self, enabled):
        for p in self.rgb.parameters():
            p.requires_grad = enabled

        for p in self.frequency.parameters():
            p.requires_grad = enabled

        self.freeze_branches = not enabled

    def train_fusion_only(self):
        self.train()

        # Frozen branch BatchNorm layers should NOT update running
        # statistics during the fusion-head-only epoch.
        self.rgb.eval()
        self.frequency.eval()
        self.fusion_head.train()

    def forward(self, rgb, freq):
        b, t, c, h, w = rgb.shape

        rgb_flat = rgb.reshape(
            b * t, c, h, w
        )

        bf, tf, fc, fh, fw = freq.shape

        freq_flat = freq.reshape(
            bf * tf, fc, fh, fw
        )

        # -------------------------------
        # RGB branch
        # -------------------------------
        rgb_chunks = []
        chunk_size = 1  # conservative for RTX 3050 4 GB

        if self.freeze_branches:
            with torch.no_grad():
                for start in range(0, rgb_flat.shape[0], chunk_size):
                    chunk = rgb_flat[start:start + chunk_size]
                    rgb_chunks.append(
                        self.rgb.backbone(chunk)
                    )
        else:
            for start in range(0, rgb_flat.shape[0], chunk_size):
                chunk = rgb_flat[start:start + chunk_size]
                rgb_chunks.append(
                    self.rgb.backbone(chunk)
                )

        rgb_features = torch.cat(
            rgb_chunks,
            dim=0,
        )

        # -------------------------------
        # Frequency branch
        # -------------------------------
        if self.freeze_branches:
            with torch.no_grad():
                freq_features = self.frequency.features(freq_flat)
        else:
            freq_features = self.frequency.features(freq_flat)

        freq_features = freq_features.flatten(1)

        # -------------------------------
        # Concatenate
        # -------------------------------
        fused = torch.cat(
            [rgb_features, freq_features],
            dim=1,
        )

        fused = self.fusion_head(fused)

        # [B,T,1] -> [B,T]
        fused = fused.reshape(b, t)

        # Video score = mean frame logit
        video_logit = fused.mean(dim=1)

        return video_logit, fused


# ============================================================
# SCORE FUSION BASELINE
# ============================================================

@torch.no_grad()
def collect_branch_scores(model, loader):
    model.eval()

    rgb_scores = []
    freq_scores = []
    labels = []

    for rgb, freq, y, _ in loader:
        rgb = rgb.to(DEVICE, non_blocking=True)
        freq = freq.to(DEVICE, non_blocking=True)

        b, t, c, h, w = rgb.shape

        rgb_flat = rgb.reshape(
            b * t, c, h, w
        )

        rgb_logits = []

        for start in range(0, rgb_flat.shape[0], 1):
            feat = model.rgb.backbone(
                rgb_flat[start:start + 1]
            )
            logit = model.rgb.head(feat)
            rgb_logits.append(logit)

        rgb_logits = torch.cat(rgb_logits, dim=0)
        rgb_logits = rgb_logits.reshape(b, t)
        rgb_video = rgb_logits.mean(dim=1)

        freq_feat, freq_frame_logits = model.frequency(freq)
        freq_video = freq_frame_logits.mean(dim=1)

        rgb_scores.extend(
            torch.sigmoid(rgb_video).cpu().numpy().tolist()
        )
        freq_scores.extend(
            torch.sigmoid(freq_video).cpu().numpy().tolist()
        )
        labels.extend(y.numpy().tolist())

    return (
        np.asarray(rgb_scores),
        np.asarray(freq_scores),
        np.asarray(labels).astype(int),
    )


def best_score_fusion_alpha(rgb, freq, y):
    best = None

    for alpha in np.linspace(0.0, 1.0, 21):
        score = alpha * rgb + (1.0 - alpha) * freq

        try:
            auc = roc_auc_score(y, score)
        except ValueError:
            auc = 0.0

        if best is None or auc > best["auc"]:
            best = {
                "alpha": float(alpha),
                "auc": float(auc),
            }

    return best


def best_threshold_balanced_accuracy(scores, y):
    best = None

    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (scores >= threshold).astype(int)
        bal = balanced_accuracy_score(y, pred)

        if best is None or bal > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(bal),
            }

    return best


# ============================================================
# TRAINING METRICS
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(auc),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    scores = []
    labels = []

    for rgb, freq, y, _ in loader:
        rgb = rgb.to(DEVICE, non_blocking=True)
        freq = freq.to(DEVICE, non_blocking=True)

        logits, _ = model(rgb, freq)

        scores.extend(
            torch.sigmoid(logits).cpu().numpy().tolist()
        )
        labels.extend(y.numpy().tolist())

    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    # Threshold MUST be selected on validation only.
    threshold_result = best_threshold_balanced_accuracy(
        scores,
        labels,
    )

    metrics = calculate_metrics(
        labels,
        scores,
        threshold_result["threshold"],
    )

    return metrics, labels, scores


@torch.no_grad()
def evaluate_fixed_threshold(model, loader, threshold):
    model.eval()

    scores = []
    labels = []

    for rgb, freq, y, _ in loader:
        rgb = rgb.to(DEVICE, non_blocking=True)
        freq = freq.to(DEVICE, non_blocking=True)

        logits, _ = model(rgb, freq)

        scores.extend(
            torch.sigmoid(logits).cpu().numpy().tolist()
        )
        labels.extend(y.numpy().tolist())

    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    metrics = calculate_metrics(
        labels,
        scores,
        threshold,
    )

    return metrics, labels, scores


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------
    required = [
        FACE_ROOT,
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        XCEPTION_CKPT,
        FREQUENCY_CKPT,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{p}"
            )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    train_ds = FusionDataset(
        TRAIN_CSV,
        augment=True,
    )

    val_ds = FusionDataset(
        VAL_CSV,
        augment=False,
    )

    test_ds = FusionDataset(
        TEST_CSV,
        augment=False,
    )

    train_labels = np.asarray(
        [s["label"] for s in train_ds.samples]
    )

    class_counts = np.bincount(
        train_labels,
        minlength=2,
    )

    print()
    print("Training class counts:")
    print("REAL:", int(class_counts[0]))
    print("FAKE:", int(class_counts[1]))

    weights = np.zeros_like(
        train_labels,
        dtype=np.float64,
    )

    for cls in [0, 1]:
        if class_counts[cls] > 0:
            weights[train_labels == cls] = (
                1.0 / class_counts[cls]
            )

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = make_loader(
        train_ds,
        sampler=sampler,
    )

    val_loader = make_loader(
        val_ds,
    )

    test_loader = make_loader(
        test_ds,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    print()
    print("Building exact branch architectures...")

    model = FusionModel()

    print()
    print("Loading Xception compression checkpoint...")
    load_branch_checkpoint(
        model.rgb,
        XCEPTION_CKPT,
        "Xception compression",
    )

    print()
    print("Loading Frequency V1 checkpoint...")
    load_branch_checkpoint(
        model.frequency,
        FREQUENCY_CKPT,
        "Frequency V1",
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Score fusion baseline
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("VALIDATION SCORE-FUSION BASELINE")
    print("=" * 72)

    rgb_val, freq_val, y_val = collect_branch_scores(
        model,
        val_loader,
    )

    alpha_info = best_score_fusion_alpha(
        rgb_val,
        freq_val,
        y_val,
    )

    alpha = alpha_info["alpha"]
    val_score_fused = (
        alpha * rgb_val
        + (1.0 - alpha) * freq_val
    )

    threshold_info = best_threshold_balanced_accuracy(
        val_score_fused,
        y_val,
    )

    print(
        f"Best alpha (RGB): {alpha:.3f}"
    )
    print(
        f"Frequency weight: {1.0 - alpha:.3f}"
    )
    print(
        f"Validation AUC: {alpha_info['auc']:.4f}"
    )
    print(
        f"Validation threshold: "
        f"{threshold_info['threshold']:.3f}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    model.set_branch_trainable(False)

    fusion_params = list(
        model.fusion_head.parameters()
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": fusion_params,
                "lr": FUSION_LR,
            },
            {
                "params": model.rgb.backbone.parameters(),
                "lr": XCEPTION_LR,
            },
            {
                "params": model.frequency.features.parameters(),
                "lr": FREQUENCY_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):

        print()
        print("=" * 72)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 72)

        # Epochs 1-2: fusion-head warmup with both pretrained
        # branches frozen. This gives the new fusion head a stable
        # starting point before changing the pretrained features.
        #
        # Epochs 3+: jointly fine-tune the branches with tiny LRs.
        if epoch <= 2:
            model.set_branch_trainable(False)
            model.train_fusion_only()
        else:
            model.set_branch_trainable(True)
            model.train()

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        seen = 0

        for step, (rgb, freq, y, _) in enumerate(
            train_loader,
            start=1,
        ):
            rgb = rgb.to(
                DEVICE,
                non_blocking=True,
            )
            freq = freq.to(
                DEVICE,
                non_blocking=True,
            )
            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            logits, _ = model(rgb, freq)

            loss = criterion(
                logits,
                y,
            )

            loss_for_backward = loss / GRAD_ACCUM
            loss_for_backward.backward()

            if (
                step % GRAD_ACCUM == 0
                or step == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            seen += 1

            if step % 100 == 0:
                print(
                    f"  step {step}/{len(train_loader)} "
                    f"loss={running_loss / seen:.5f}"
                )

        train_loss = (
            running_loss / max(1, seen)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
        )

        print()
        print("TRAIN LOSS:", f"{train_loss:.6f}")
        print(
            "VAL AUC:",
            f"{val_metrics['roc_auc']:.6f}",
        )
        print(
            "VAL Accuracy:",
            f"{val_metrics['accuracy']:.6f}",
        )
        print(
            "VAL Balanced Accuracy:",
            f"{val_metrics['balanced_accuracy']:.6f}",
        )
        print(
            "VAL F1:",
            f"{val_metrics['f1']:.6f}",
        )
        print(
            "VAL threshold:",
            f"{val_metrics['threshold']:.3f}",
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": val_metrics,
            }
        )

        # ----------------------------------------------------
        # Save best by validation AUC
        # ----------------------------------------------------
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "score_fusion_alpha": alpha,
                    "score_fusion_validation_auc": alpha_info["auc"],
                },
                BEST_CKPT,
            )

            print(
                f"*** NEW BEST MODEL SAVED: "
                f"epoch {epoch}, "
                f"val AUC={best_val_auc:.6f}"
            )
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):
                print(
                    "Early stopping."
                )
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Reload best
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("LOADING BEST FUSION CHECKPOINT")
    print("=" * 72)

    best = torch.load(
        BEST_CKPT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        best["model"],
        strict=True,
    )

    # Threshold is selected from validation ONLY.
    final_val_metrics, _, _ = evaluate_model(
        model,
        val_loader,
    )

    final_threshold = final_val_metrics["threshold"]

    print(
        "Final validation threshold:",
        f"{final_threshold:.3f}",
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------
    test_metrics, test_y, test_scores = (
        evaluate_fixed_threshold(
            model,
            test_loader,
            final_threshold,
        )
    )

    # --------------------------------------------------------
    # Score-fusion test baseline
    # --------------------------------------------------------
    rgb_test, freq_test, y_test = collect_branch_scores(
        model,
        test_loader,
    )

    score_fused_test = (
        alpha * rgb_test
        + (1.0 - alpha) * freq_test
    )

    score_fusion_test_metrics = calculate_metrics(
        y_test,
        score_fused_test,
        threshold_info["threshold"],
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    report = {
        "device": str(DEVICE),
        "face_root": str(FACE_ROOT),
        "train_manifest": str(TRAIN_CSV),
        "val_manifest": str(VAL_CSV),
        "test_manifest": str(TEST_CSV),

        "xception_checkpoint": str(XCEPTION_CKPT),
        "frequency_checkpoint": str(FREQUENCY_CKPT),

        "model": (
            "Xception compression features + "
            "Frequency V1 features + fusion MLP"
        ),

        "num_frames": NUM_FRAMES,
        "image_size": IMG_SIZE,

        "train_videos": len(train_ds),
        "val_videos": len(val_ds),
        "test_videos": len(test_ds),

        "best_epoch": best_epoch,
        "best_validation_auc": best_val_auc,
        "warmup_epochs": 2,
        "total_max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,

        "final_validation_metrics": final_val_metrics,

        "test_metrics": test_metrics,

        "score_fusion_baseline": {
            "rgb_weight": alpha,
            "frequency_weight": 1.0 - alpha,
            "validation_auc": alpha_info["auc"],
            "validation_threshold": threshold_info["threshold"],
            "test_metrics": score_fusion_test_metrics,
        },

        "history": history,
    }

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("FINAL FUSION TEST RESULTS")
    print("=" * 72)

    for key, value in test_metrics.items():
        print(f"{key:22s}: {value}")

    print()
    print("=" * 72)
    print("SCORE-FUSION BASELINE TEST RESULTS")
    print("=" * 72)

    for key, value in score_fusion_test_metrics.items():
        print(f"{key:22s}: {value}")

    print()
    print("Best checkpoint:")
    print(BEST_CKPT)

    print()
    print("Report:")
    print(REPORT_JSON)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
