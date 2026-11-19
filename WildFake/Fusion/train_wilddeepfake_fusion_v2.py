"""
WildDeepfake feature-level fusion (Xception RGB + FrequencyCNN V2).

Uses the already-trained WildDeepfake checkpoints; it DOES NOT retrain either branch.
1) Extracts and caches deterministic features for train/valid/test.
2) Trains a small fusion head on cached 2048-D RGB + 256-D frequency features.
3) Selects threshold on validation only, then evaluates test once.
4) Saves everything under wilddeepfake_experiment\fusion_v2 (does not overwrite FF++).

Expected checkpoints:
  D:\newdataset\wilddeepfake_experiment\transfer_xception\checkpoints\best_model.pth
  D:\newdataset\wilddeepfake_experiment\frequency_v2\checkpoints\best_frequency_v2.pth

Run:
  python D:\newdataset\train_wilddeepfake_fusion_v2.py
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ----------------------------- PATHS -----------------------------------------
DATA_ROOT = Path(r"D:\newdataset\Dataset")
EXPERIMENT_ROOT = Path(r"D:\newdataset\wilddeepfake_experiment")
XCEPTION_CKPT = EXPERIMENT_ROOT / "transfer_xception" / "checkpoints" / "best_model.pth"
FREQUENCY_CKPT = EXPERIMENT_ROOT / "frequency_v2" / "checkpoints" / "best_frequency_v2.pth"
OUT_DIR = EXPERIMENT_ROOT / "fusion_v2"
CACHE_DIR = OUT_DIR / "feature_cache"
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_DIR = OUT_DIR / "reports"

for folder in (CACHE_DIR, CKPT_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------- CONFIG ----------------------------------------
IMAGE_SIZE = 224
EXTRACT_BATCH_SIZE = 8       # safe starting point for 4-GB RTX 3050
NUM_WORKERS = 0              # Windows-safe
HEAD_BATCH_SIZE = 2048        # cached features are small
EPOCHS = 40
PATIENCE = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if AMP:
    torch.cuda.manual_seed_all(SEED)

# ----------------------------- DATA ------------------------------------------
class WildImageDataset(Dataset):
    """Stable sorted sample order; label convention is real=0, fake=1."""
    def __init__(self, split):
        self.root = DATA_ROOT / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")
        self.samples = []
        for class_name, label in (("real", 0), ("fake", 1)):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class folder: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VALID_EXTS:
                    self.samples.append((str(path), label))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")
        # Keep class order deterministic and report mapping.
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        n_fake = sum(y == 1 for _, y in self.samples)
        n_real = len(self.samples) - n_fake
        print(f"{split}: {len(self.samples):,} images | real={n_real:,} fake={n_fake:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            im = im.convert("RGB")
            rgb_tensor = self.transform(im)
        return rgb_tensor, path, label


def build_fft_from_rgb_tensor(rgb_tensor):
    """
    Input: normalized ImageNet RGB tensor [B,3,224,224].
    Reverse ImageNet normalization, convert to grayscale, then use the same
    FFT/log-magnitude/per-image z-score/clip pipeline as frequency V2 training.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=rgb_tensor.dtype, device=rgb_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=rgb_tensor.dtype, device=rgb_tensor.device).view(1, 3, 1, 1)
    rgb = (rgb_tensor * std + mean).clamp(0, 1)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    spectrum = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))))
    mu = spectrum.mean(dim=(-2, -1), keepdim=True)
    sd = spectrum.std(dim=(-2, -1), keepdim=True, unbiased=False)
    spectrum = (spectrum - mu) / (sd + 1e-6)
    return spectrum.clamp(-5, 5)


# ----------------------------- MODELS ----------------------------------------
class XceptionFFPPStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
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
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
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


def load_state(path, model, label):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} checkpoint missing:\n{path}\n"
            "Check the existing checkpoint path. No model training was started."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model")))
    if state is None:
        raise ValueError(f"Could not find model state dict in {path}; keys={list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded {label}: {path}")
    if isinstance(ckpt, dict):
        print(f"  checkpoint metadata: epoch={ckpt.get('epoch', 'n/a')}")
    return ckpt


# ----------------------------- FEATURE CACHE ---------------------------------
@torch.inference_mode()
def extract_split(split, xception, frequency):
    dataset = WildImageDataset(split)
    cache_x = CACHE_DIR / f"{split}_features.npy"
    cache_y = CACHE_DIR / f"{split}_labels.npy"
    cache_meta = CACHE_DIR / f"{split}_cache_meta.json"

    # Reuse cache only when it matches current sample list/order and dimensions.
    sample_signature = {
        "count": len(dataset),
        "first_path": dataset.samples[0][0],
        "last_path": dataset.samples[-1][0],
        "xception_checkpoint": str(XCEPTION_CKPT),
        "frequency_checkpoint": str(FREQUENCY_CKPT),
        "feature_shape": [2304],
    }
    if cache_x.is_file() and cache_y.is_file() and cache_meta.is_file():
        try:
            old_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if old_meta == sample_signature:
                feats = np.load(cache_x, mmap_mode="r")
                labels = np.load(cache_y, mmap_mode="r")
                if feats.shape == (len(dataset), 2304) and labels.shape == (len(dataset),):
                    print(f"Using existing feature cache for {split}: {cache_x}")
                    return cache_x, cache_y
        except Exception:
            pass
        print(f"Cache metadata differs for {split}; recomputing this split only.")

    loader = DataLoader(dataset, batch_size=EXTRACT_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=AMP)
    feats_out = np.lib.format.open_memmap(
        cache_x, mode="w+", dtype=np.float16, shape=(len(dataset), 2304)
    )
    labels_out = np.lib.format.open_memmap(
        cache_y, mode="w+", dtype=np.int64, shape=(len(dataset),)
    )

    start = time.time()
    cursor = 0
    for batch_idx, (rgb, _paths, labels) in enumerate(loader, start=1):
        rgb = rgb.to(DEVICE, non_blocking=True)
        freq = build_fft_from_rgb_tensor(rgb)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            rgb_features = xception.backbone(rgb)
            freq_features = frequency.forward_features(freq)
        combined = torch.cat([rgb_features.float(), freq_features.float()], dim=1)
        b = combined.shape[0]
        feats_out[cursor:cursor+b] = combined.cpu().numpy().astype(np.float16)
        labels_out[cursor:cursor+b] = labels.numpy().astype(np.int64)
        cursor += b
        if batch_idx % 500 == 0 or cursor == len(dataset):
            print(f"  {split}: {cursor:,}/{len(dataset):,} | elapsed {(time.time()-start)/60:.1f} min")
    feats_out.flush()
    labels_out.flush()
    del feats_out, labels_out
    cache_meta.write_text(json.dumps(sample_signature, indent=2), encoding="utf-8")
    print(f"Saved {split} cache: {cache_x}")
    return cache_x, cache_y


def load_cached_tensors(feature_path, label_path):
    # Copies mmap-backed arrays into regular tensors for reliable Windows handling.
    x = torch.from_numpy(np.array(np.load(feature_path, mmap_mode="r"), dtype=np.float32))
    y = torch.from_numpy(np.array(np.load(label_path, mmap_mode="r"), dtype=np.float32))
    return x, y


# ----------------------------- METRICS ---------------------------------------
def metrics(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = 0.5
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "fake_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "fake_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "real_specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix_labels_0_real_1_fake": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def probabilities_for(model, x, batch_size=8192):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = x[start:start+batch_size].to(DEVICE)
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(chunks)


def main():
    print("=" * 78)
    print("WildDeepfake feature-level fusion: frozen Xception + frozen FrequencyCNN V2")
    print("Device:", DEVICE, "| CUDA:", torch.cuda.is_available())
    print("Data:", DATA_ROOT)
    print("Output:", OUT_DIR)
    print("Label convention: real=0, fake=1")
    print("=" * 78)

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    xception = XceptionFFPPStyle()
    load_state(XCEPTION_CKPT, xception, "WildDeepfake Xception transfer model")
    frequency = FrequencyCNNV2()
    load_state(FREQUENCY_CKPT, frequency, "WildDeepfake FrequencyCNN V2")

    xception.to(DEVICE).eval()
    frequency.to(DEVICE).eval()
    for p in xception.parameters():
        p.requires_grad = False
    for p in frequency.parameters():
        p.requires_grad = False

    # Sanity check architecture and checkpoint dimensions before processing the dataset.
    with torch.inference_mode():
        dummy_rgb = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        dummy_freq = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        rgb_dim = xception.backbone(dummy_rgb).shape[1]
        freq_dim = frequency.forward_features(dummy_freq).shape[1]
    print(f"Feature dimensions: Xception={rgb_dim}, Frequency V2={freq_dim}, fused={rgb_dim+freq_dim}")
    if (rgb_dim, freq_dim) != (2048, 256):
        raise RuntimeError(f"Unexpected feature dimensions: {rgb_dim} + {freq_dim}; expected 2048 + 256")

    cache_paths = {}
    for split in ("train", "valid", "test"):
        cache_paths[split] = extract_split(split, xception, frequency)

    # The cached train/valid/test labels are real=0, fake=1.
    x_train, y_train = load_cached_tensors(*cache_paths["train"])
    x_valid, y_valid = load_cached_tensors(*cache_paths["valid"])
    x_test, y_test = load_cached_tensors(*cache_paths["test"])
    print(f"Cached tensors: train={tuple(x_train.shape)}, valid={tuple(x_valid.shape)}, test={tuple(x_test.shape)}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=HEAD_BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=AMP
    )
    fusion_head = FusionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_epoch = 0
    stale = 0
    history = []
    best_path = CKPT_DIR / "best_fusion_v2.pth"
    latest_path = CKPT_DIR / "latest_fusion_v2.pth"
    history_path = REPORT_DIR / "history_fusion_v2.json"

    for epoch in range(1, EPOCHS + 1):
        fusion_head.train()
        total_loss = 0.0
        total_n = 0
        start = time.time()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = fusion_head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)

        val_probs = probabilities_for(fusion_head, x_valid)
        yv = y_valid.numpy().astype(np.int64)
        val_auc = float(roc_auc_score(yv, val_probs))
        # Choose threshold from validation only by highest F1 for fake class.
        thresholds = np.arange(0.10, 0.901, 0.01)
        val_f1s = [f1_score(yv, (val_probs >= t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(val_f1s))
        val_threshold = float(thresholds[best_idx])
        val_metrics = metrics(yv, val_probs, val_threshold)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_n),
            "val_auc": val_auc,
            "val_threshold": val_threshold,
            "val_metrics": val_metrics,
            "epoch_seconds": time.time() - start,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={record['train_loss']:.4f} "
            f"| val_auc={val_auc:.4f} | val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"| val_fake_f1={val_metrics['fake_f1']:.4f} | threshold={val_threshold:.2f} "
            f"| {record['epoch_seconds']:.1f}s"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": fusion_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
            "val_threshold": val_threshold,
            "config": {
                "data_root": str(DATA_ROOT), "xception_checkpoint": str(XCEPTION_CKPT),
                "frequency_checkpoint": str(FREQUENCY_CKPT), "feature_dims": [2048, 256],
                "label_mapping": {"real": 0, "fake": 1}, "seed": SEED,
            },
        }, latest_path)

        # Best model selected by validation AUC; no test data is used for selection.
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": fusion_head.state_dict(),
                "val_auc": val_auc,
                "val_threshold": val_threshold,
                "val_metrics": val_metrics,
                "config": {
                    "data_root": str(DATA_ROOT), "xception_checkpoint": str(XCEPTION_CKPT),
                    "frequency_checkpoint": str(FREQUENCY_CKPT), "feature_dims": [2048, 256],
                    "label_mapping": {"real": 0, "fake": 1}, "seed": SEED,
                },
            }, best_path)
            print(f"  Saved best fusion head: {best_path}")
        else:
            stale += 1
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= PATIENCE:
            print(f"Early stopping: validation AUC did not improve for {PATIENCE} epochs.")
            break

    # Restore best head, choose the threshold stored with that checkpoint, then test once.
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    fusion_head.load_state_dict(best["model_state_dict"], strict=True)
    threshold = float(best["val_threshold"])
    valid_probs = probabilities_for(fusion_head, x_valid)
    test_probs = probabilities_for(fusion_head, x_test)
    valid_metrics = metrics(y_valid.numpy().astype(np.int64), valid_probs, threshold)
    test_metrics = metrics(y_test.numpy().astype(np.int64), test_probs, threshold)

    report = {
        "experiment": "WildDeepfake frozen-feature fusion V2",
        "data_root": str(DATA_ROOT),
        "label_mapping": {"real": 0, "fake": 1},
        "feature_dimensions": {"xception_rgb": 2048, "frequency_v2": 256, "concatenated": 2304},
        "method": "Precompute deterministic frozen branch features, train only fusion head",
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "threshold_selected_on_validation": threshold,
        "validation": valid_metrics,
        "test": test_metrics,
        "paths": {
            "xception_checkpoint": str(XCEPTION_CKPT),
            "frequency_checkpoint": str(FREQUENCY_CKPT),
            "best_fusion_checkpoint": str(best_path),
            "feature_cache": str(CACHE_DIR),
        },
        "notes": [
            "Both feature extractors remained frozen; only FusionHead was trained.",
            "Test set was not used for checkpoint or threshold selection.",
            "Feature caches are float16 on disk and can be reused when metadata/checkpoint paths match.",
            "The validation split has known exact cross-split duplicates with train; validation metrics may be optimistic.",
        ],
    }
    (REPORT_DIR / "final_report_fusion_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("FINAL TEST RESULTS — WILDDEEPFAKE FUSION V2")
    print("=" * 78)
    for key, value in test_metrics.items():
        print(f"{key:42s}: {value}")
    print(f"Best epoch: {best_epoch} | best validation AUC: {best_auc:.4f}")
    print(f"Report: {REPORT_DIR / 'final_report_fusion_v2.json'}")
    print(f"Checkpoint: {best_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
