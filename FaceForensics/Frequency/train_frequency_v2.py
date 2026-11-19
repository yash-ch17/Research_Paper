# train_frequency_v2.py
# Frequency-Domain Deepfake Detection — V2
#
# V2 changes vs V1:
# 1. Stronger dropout (0.50)
# 2. Lower learning rate (5e-5)
# 3. Label smoothing (0.05)
# 4. Frequency-domain augmentation during training
# 5. Spatial augmentation before FFT
# 6. Validation-AUC based checkpointing + early stopping
# 7. Validation threshold is used for final TEST evaluation
#
# V1 baseline:
# Accuracy ≈ 70.95%
# Balanced Accuracy ≈ 65.04%
# ROC-AUC ≈ 69.13%

import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"D:\faceforensics")

FACE_ROOT = ROOT / "ffpp_faces_final"
MANIFEST_DIR = ROOT / "manifests"

TRAIN_CSV = MANIFEST_DIR / "train.csv"
VAL_CSV = MANIFEST_DIR / "val.csv"
TEST_CSV = MANIFEST_DIR / "test.csv"

OUT_DIR = ROOT / "checkpoints_frequency_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUT_DIR / "best_frequency_v2.pth"
HISTORY_FILE = OUT_DIR / "history_frequency_v2.json"
REPORT_FILE = OUT_DIR / "final_report_frequency_v2.json"

IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 8

BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_WORKERS = 0

EPOCHS = 15
PATIENCE = 3

LR = 5e-5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05

DROPOUT = 0.50

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

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

print("=" * 70)
print("FREQUENCY V2 — OVERFITTING-REDUCED TRAINING")
print("=" * 70)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Face root:", FACE_ROOT)
print("Output:", OUT_DIR)
print()


# ============================================================
# DATA HELPERS
# ============================================================

def read_manifest(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video = row["video"]
            label = int(row["label"])

            # Existing face dataset uses:
            # original -> real
            # fake -> fake
            if label == 0:
                folder = FACE_ROOT / "original" / video
            else:
                folder = FACE_ROOT / "fake" / video

            if not folder.exists():
                continue

            frames = sorted(folder.glob("*.jpg"))

            if len(frames) < FRAMES_PER_VIDEO:
                continue

            rows.append({
                "video": video,
                "label": label,
                "frames": frames,
            })

    return rows


train_rows = read_manifest(TRAIN_CSV)
val_rows = read_manifest(VAL_CSV)
test_rows = read_manifest(TEST_CSV)

print("Usable videos:")
print("Train:", len(train_rows))
print("Val  :", len(val_rows))
print("Test :", len(test_rows))
print()


# ============================================================
# IMAGE / FFT PROCESSING
# ============================================================

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def jpeg_compress(image, quality):
    """
    Optional JPEG recompression.
    This is intentionally mild and is only used in training.
    """
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    import io

    buffer = io.BytesIO()

    pil.save(buffer, format="JPEG", quality=int(quality))

    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")
    compressed = np.array(compressed)

    return cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)


def random_spatial_augmentation(image):
    """
    Mild spatial augmentation before FFT.
    Keeps the augmentation realistic for face frames.
    """

    # Horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # Very small brightness/contrast change
    if random.random() < 0.30:
        alpha = random.uniform(0.90, 1.10)
        beta = random.uniform(-8.0, 8.0)

        image = cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=beta
        )

    return image


def build_fft(image, augment=False):
    """
    Convert RGB/BGR face image into normalized FFT magnitude map.

    Pipeline:
        grayscale
        -> 2D FFT
        -> fftshift
        -> magnitude
        -> log1p
        -> normalization
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    gray = gray.astype(np.float32) / 255.0

    # FFT
    fft = np.fft.fft2(gray)

    # Move zero frequency to center
    fft_shift = np.fft.fftshift(fft)

    # Magnitude
    magnitude = np.abs(fft_shift)

    # Log compression
    spectrum = np.log1p(magnitude)

    # --------------------------------------------------------
    # Frequency-domain augmentation
    # --------------------------------------------------------
    if augment:

        # 1. Mild multiplicative frequency noise
        if random.random() < 0.35:
            noise = np.random.normal(
                loc=1.0,
                scale=0.035,
                size=spectrum.shape
            ).astype(np.float32)

            spectrum = spectrum * noise

        # 2. Random high-frequency attenuation/emphasis
        #
        # Do NOT destroy the central low-frequency region.
        # This creates mild domain variation while preserving
        # the basic frequency structure.
        if random.random() < 0.30:

            h, w = spectrum.shape

            cy = h // 2
            cx = w // 2

            yy, xx = np.ogrid[:h, :w]

            radius = np.sqrt(
                (yy - cy) ** 2 +
                (xx - cx) ** 2
            )

            max_radius = min(h, w) / 2.0

            high_freq = radius > (0.35 * max_radius)

            factor = random.uniform(0.85, 1.15)

            spectrum[high_freq] *= factor

        # 3. Tiny additive noise
        if random.random() < 0.20:
            spectrum += np.random.normal(
                0.0,
                0.002,
                spectrum.shape
            ).astype(np.float32)

    # Remove invalid values
    spectrum = np.nan_to_num(
        spectrum,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Per-image normalization
    mean = spectrum.mean()
    std = spectrum.std()

    spectrum = (spectrum - mean) / (std + 1e-6)

    spectrum = np.clip(spectrum, -5.0, 5.0)

    return spectrum.astype(np.float32)


def build_rgb(image):
    """
    RGB image used only for the spatial augmentation step
    before converting to grayscale FFT.
    """

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return image


# ============================================================
# DATASET
# ============================================================

class FrequencyDataset(Dataset):

    def __init__(self, rows, train=False):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        item = self.rows[index]

        frames = item["frames"]

        # Exactly 8 frames.
        # Spread them across the available frames so we don't
        # always use only the first 8.
        if len(frames) > FRAMES_PER_VIDEO:

            positions = np.linspace(
                0,
                len(frames) - 1,
                FRAMES_PER_VIDEO
            ).astype(int)

            selected = [frames[i] for i in positions]

        else:
            selected = frames[:FRAMES_PER_VIDEO]

        fft_frames = []

        for frame_path in selected:

            image = cv2.imread(str(frame_path))

            if image is None:
                image = np.zeros(
                    (IMAGE_SIZE, IMAGE_SIZE, 3),
                    dtype=np.uint8
                )

            if self.train:

                image = random_spatial_augmentation(image)

                # Mild JPEG degradation augmentation.
                # Not applied every time.
                if random.random() < 0.40:
                    quality = random.randint(45, 90)
                    image = jpeg_compress(image, quality)

            spectrum = build_fft(
                image,
                augment=self.train
            )

            spectrum = torch.from_numpy(spectrum).unsqueeze(0)

            fft_frames.append(spectrum)

        # [T, 1, H, W]
        fft_frames = torch.stack(fft_frames, dim=0)

        label = torch.tensor(
            item["label"],
            dtype=torch.float32
        )

        return fft_frames, label


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = FrequencyDataset(train_rows, train=True)
val_dataset = FrequencyDataset(val_rows, train=False)
test_dataset = FrequencyDataset(test_rows, train=False)


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

train_labels = np.array(
    [row["label"] for row in train_rows],
    dtype=np.int64
)

class_counts = np.bincount(train_labels, minlength=2)

print("Training class counts:")
print("REAL:", class_counts[0])
print("FAKE:", class_counts[1])
print()

class_weights = 1.0 / np.maximum(class_counts, 1)

sample_weights = np.array(
    [class_weights[label] for label in train_labels],
    dtype=np.float64
)

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)


# ============================================================
# FREQUENCY CNN V2
# ============================================================

class FrequencyCNNV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        # Stronger regularization than V1
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = torch.flatten(x, 1)

        return self.classifier(x).squeeze(1)


model = FrequencyCNNV2().to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

# Label smoothing for BCE:
# positive target becomes 0.95
# negative target becomes 0.05
criterion = nn.BCEWithLogitsLoss(
    pos_weight=None
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-6
)


# ============================================================
# METRICS
# ============================================================

def tune_threshold(y_true, probs):

    best_threshold = 0.50
    best_balanced = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):

        preds = (probs >= threshold).astype(int)

        score = balanced_accuracy_score(
            y_true,
            preds
        )

        if score > best_balanced:

            best_balanced = score
            best_threshold = float(threshold)

    return best_threshold


def calculate_metrics(y_true, probs, threshold):

    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        preds,
        labels=[0, 1]
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, preds)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, preds)
        ),

        "precision": float(
            precision_score(
                y_true,
                preds,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                preds,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                preds,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probs)
        ),

        "threshold": float(threshold),

        "confusion_matrix": cm.tolist()
    }


# ============================================================
# TRAINING
# ============================================================

history = []

best_val_auc = -float("inf")
best_epoch = -1
epochs_without_improvement = 0

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


def smooth_targets(labels):

    # 0 -> 0.05
    # 1 -> 0.95
    return (
        labels * (1.0 - LABEL_SMOOTHING)
        + LABEL_SMOOTHING / 2.0
    )


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (fft_frames, labels) in enumerate(train_loader):

        # [B, T, 1, H, W]
        fft_frames = fft_frames.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        B, T, C, H, W = fft_frames.shape

        # Flatten frames
        x = fft_frames.view(
            B * T,
            C,
            H,
            W
        )

        # Forward in smaller chunks to protect 4GB VRAM.
        frame_logits = []

        for start in range(0, B * T, 4):

            chunk = x[start:start + 4]

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):

                logits = model(chunk)

            frame_logits.append(logits)

        frame_logits = torch.cat(
            frame_logits,
            dim=0
        )

        # Restore [B, T]
        frame_logits = frame_logits.view(B, T)

        # Mean temporal aggregation
        video_logits = frame_logits.mean(dim=1)

        # Label smoothing
        smooth = smooth_targets(labels)

        loss = criterion(
            video_logits.float(),
            smooth.float()
        )

        loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (
            (step + 1) % GRAD_ACCUM == 0
            or (step + 1) == len(train_loader)
        ):

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * GRAD_ACCUM

    train_loss = running_loss / max(len(train_loader), 1)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_labels = []
    val_probs = []

    with torch.no_grad():

        for fft_frames, labels in val_loader:

            fft_frames = fft_frames.to(
                DEVICE,
                non_blocking=True
            )

            B, T, C, H, W = fft_frames.shape

            x = fft_frames.view(
                B * T,
                C,
                H,
                W
            )

            frame_logits = []

            for start in range(0, B * T, 8):

                chunk = x[start:start + 8]

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available()
                ):

                    logits = model(chunk)

                frame_logits.append(logits)

            frame_logits = torch.cat(
                frame_logits,
                dim=0
            )

            frame_logits = frame_logits.view(B, T)

            video_probs = torch.sigmoid(
                frame_logits.mean(dim=1)
            )

            val_probs.extend(
                video_probs.float().cpu().numpy().tolist()
            )

            val_labels.extend(
                labels.numpy().tolist()
            )

    val_labels_np = np.asarray(
        val_labels,
        dtype=np.int64
    )

    val_probs_np = np.asarray(
        val_probs,
        dtype=np.float64
    )

    # NaN protection
    val_probs_np = np.nan_to_num(
        val_probs_np,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )

    val_threshold = tune_threshold(
        val_labels_np,
        val_probs_np
    )

    val_metrics = calculate_metrics(
        val_labels_np,
        val_probs_np,
        val_threshold
    )

    val_auc = val_metrics["roc_auc"]

    scheduler.step(val_auc)

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "learning_rate": float(current_lr),
        "validation": val_metrics
    }

    history.append(epoch_record)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss {train_loss:.5f} | "
        f"Val Acc {val_metrics['accuracy']:.4f} | "
        f"Val BalAcc {val_metrics['balanced_accuracy']:.4f} | "
        f"Val F1 {val_metrics['f1']:.4f} | "
        f"Val AUC {val_auc:.4f} | "
        f"Thr {val_threshold:.2f} | "
        f"LR {current_lr:.2e}"
    )

    # ========================================================
    # SAVE BEST MODEL BY VALIDATION AUC
    # ========================================================

    if val_auc > best_val_auc + 0.002:

        best_val_auc = val_auc
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": float(val_auc),
                "val_threshold": float(val_threshold),
                "config": {
                    "image_size": IMAGE_SIZE,
                    "frames": FRAMES_PER_VIDEO,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "dropout": DROPOUT,
                    "label_smoothing": LABEL_SMOOTHING,
                    "epochs": EPOCHS,
                    "model": "FrequencyCNNV2"
                }
            },
            BEST_CKPT
        )

        print(
            f"  >>> NEW BEST MODEL "
            f"(Val AUC = {val_auc:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  No sufficient improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

        if epochs_without_improvement >= PATIENCE:

            print("  >>> Early stopping.")
            break


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

if not BEST_CKPT.exists():

    raise RuntimeError(
        "Best checkpoint was not created."
    )

checkpoint = torch.load(
    BEST_CKPT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_epoch = checkpoint["epoch"]
best_val_auc = checkpoint["val_auc"]
best_val_threshold = checkpoint["val_threshold"]

print()
print("=" * 70)
print("BEST FREQUENCY V2 CHECKPOINT")
print("=" * 70)
print("Best epoch:", best_epoch)
print("Best validation AUC:", best_val_auc)
print("Validation threshold:", best_val_threshold)


# ============================================================
# FINAL VALIDATION + TEST
# ============================================================

def predict_dataset(loader):

    model.eval()

    labels_all = []
    probs_all = []

    with torch.no_grad():

        for fft_frames, labels in loader:

            fft_frames = fft_frames.to(
                DEVICE,
                non_blocking=True
            )

            B, T, C, H, W = fft_frames.shape

            x = fft_frames.view(
                B * T,
                C,
                H,
                W
            )

            frame_logits = []

            for start in range(0, B * T, 8):

                chunk = x[start:start + 8]

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available()
                ):

                    logits = model(chunk)

                frame_logits.append(logits)

            frame_logits = torch.cat(
                frame_logits,
                dim=0
            )

            frame_logits = frame_logits.view(B, T)

            probs = torch.sigmoid(
                frame_logits.mean(dim=1)
            )

            probs_all.extend(
                probs.float().cpu().numpy().tolist()
            )

            labels_all.extend(
                labels.numpy().tolist()
            )

    labels_all = np.asarray(
        labels_all,
        dtype=np.int64
    )

    probs_all = np.asarray(
        probs_all,
        dtype=np.float64
    )

    probs_all = np.nan_to_num(
        probs_all,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )

    return labels_all, probs_all


# Recalculate validation using best checkpoint
val_y, val_p = predict_dataset(val_loader)

final_val_threshold = tune_threshold(
    val_y,
    val_p
)

final_val_metrics = calculate_metrics(
    val_y,
    val_p,
    final_val_threshold
)

# IMPORTANT:
# This validation threshold is carried directly to TEST.
test_y, test_p = predict_dataset(test_loader)

test_metrics = calculate_metrics(
    test_y,
    test_p,
    final_val_threshold
)


# ============================================================
# FINAL REPORT
# ============================================================

report = {
    "model": "FrequencyCNNV2 - FFT Frequency Domain",

    "best_epoch": int(best_epoch),

    "best_validation_auc": float(best_val_auc),

    "validation": final_val_metrics,

    "test": test_metrics,

    "config": {
        "face_root": str(FACE_ROOT),
        "image_size": IMAGE_SIZE,
        "frames_per_video": FRAMES_PER_VIDEO,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "epochs_requested": EPOCHS,
        "patience": PATIENCE,
        "frequency_augmentation": True,
        "spatial_augmentation": True,
        "jpeg_augmentation": True,
        "threshold_selection": "validation only"
    },

    "v1_baseline": {
        "accuracy": 0.709480122324159,
        "balanced_accuracy": 0.6504034065441506,
        "precision": 0.7922077922077922,
        "recall": 0.7956521739130434,
        "f1": 0.7939262472885033,
        "roc_auc": 0.6912595248767368,
        "threshold": 0.41,
        "confusion_matrix": [
            [49, 48],
            [47, 183]
        ]
    }
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL FREQUENCY V2 TEST")
print("=" * 70)

print(
    f"Accuracy            : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy   : "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Precision           : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall              : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1                  : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC             : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"Threshold (from VAL): "
    f"{final_val_threshold:.2f}"
)

print(
    "Confusion Matrix    :",
    test_metrics["confusion_matrix"]
)

print()
print("Best Epoch          :", best_epoch)
print("Best Validation AUC :", f"{best_val_auc:.4f}")

print()
print("V2 files:")
print("Checkpoint:", BEST_CKPT)
print("History   :", HISTORY_FILE)
print("Report    :", REPORT_FILE)

print()
print("=" * 70)
print("V1 BASELINE")
print("=" * 70)
print("Accuracy          : 0.7095")
print("Balanced Accuracy : 0.6504")
print("F1                : 0.7939")
print("ROC-AUC           : 0.6913")
print()
print("Compare V2 against these numbers before moving to fusion.")
print("=" * 70)
