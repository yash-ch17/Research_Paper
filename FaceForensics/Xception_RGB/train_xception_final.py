
import os, csv, json, math, random, argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(r"D:\faceforensics")
FACE_ROOT = ROOT / "ffpp_faces_final"
MANIFEST_ROOT = ROOT / "manifests"   # ORIGINAL fixed video-level split
OUT = ROOT / "checkpoints_xception_final"

IMAGE_SIZE = 224
FRAMES = 8
BATCH_SIZE = 2
ACCUM = 4
WORKERS = 2
EPOCHS = 12
PATIENCE = 3
LR = 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class VideoDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.transform = transform
        self.items = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        dropped = 0
        for r in rows:
            video = r["video"].strip()
            label = int(r["label"])
            cls = "fake" if label == 1 else "original"
            folder = FACE_ROOT / cls / video
            frames = sorted(folder.glob("*.jpg")) if folder.exists() else []
            if len(frames) < 4:
                dropped += 1
                continue

            # Use all 8 when available; if fewer than 8, repeat last frame.
            if len(frames) >= FRAMES:
                idx = torch.linspace(0, len(frames)-1, FRAMES).long().tolist()
                chosen = [frames[i] for i in idx]
            else:
                chosen = frames + [frames[-1]] * (FRAMES-len(frames))

            self.items.append((video, label, chosen))

        self.dropped = dropped
        print(Path(csv_path).name, "usable:", len(self.items), "dropped:", dropped,
              "REAL:", sum(x[1] == 0 for x in self.items),
              "FAKE:", sum(x[1] == 1 for x in self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        video, label, paths = self.items[i]
        imgs = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
            except:
                im = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
            imgs.append(self.transform(im))
        return torch.stack(imgs), torch.tensor(label, dtype=torch.long)

class XceptionVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: B,T,C,H,W
        b,t,c,h,w = x.shape
        x = x.reshape(b*t,c,h,w)
        feats = []
        # Small chunks protect 4GB VRAM.
        for s in range(0, b*t, 2):
            feats.append(self.backbone(x[s:s+2]))
        f = torch.cat(feats, dim=0).reshape(b,t,-1)
        logits = self.head(f).squeeze(-1)
        # Video score = mean of frame logits.
        return logits.mean(dim=1)

def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "recall": float(recall_score(y,pred,zero_division=0)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y,pred).tolist()
    }
    try: out["roc_auc"] = float(roc_auc_score(y,p))
    except: out["roc_auc"] = 0.5
    return out

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x,y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            z = model(x)
            prob = torch.sigmoid(z)
        ys.extend(y.numpy().tolist())
        ps.extend(prob.float().cpu().numpy().tolist())

    ys = __import__("numpy").array(ys)
    ps = __import__("numpy").array(ps)

    # Choose threshold on validation only, optimizing balanced accuracy.
    best_t, best_b = 0.5, -1
    for t in [i/100 for i in range(10,91)]:
        b = balanced_accuracy_score(ys, (ps >= t).astype(int))
        if b > best_b:
            best_b, best_t = b, t
    return metrics(ys, ps, best_t), ys, ps

def run_epoch(model, loader, optimizer, scaler, train=True):
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step,(x,y) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.float().to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            z = model(x)
            loss = loss_fn(z,y) / ACCUM

        if train:
            scaler.scale(loss).backward()
            if (step+1) % ACCUM == 0 or step == len(loader)-1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total += loss.item() * ACCUM

    return total / max(1,len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if not FACE_ROOT.exists():
        raise FileNotFoundError(FACE_ROOT)

    OUT.mkdir(parents=True, exist_ok=True)

    train = VideoDataset(MANIFEST_ROOT/"train.csv", train_tf)
    val = VideoDataset(MANIFEST_ROOT/"val.csv", eval_tf)
    test = VideoDataset(MANIFEST_ROOT/"test.csv", eval_tf)

    # Balanced video-level sampler.
    labels = [x[1] for x in train.items]
    counts = Counter(labels)
    weights = [1.0/counts[y] for y in labels]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=WORKERS, pin_memory=True)

    model = XceptionVideo().to(DEVICE)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR},
        {"params": head_params, "lr": HEAD_LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

    best_auc = -1
    best_epoch = 0
    bad = 0
    history = []

    print("="*70)
    print("CLEAN PRETRAINED XCEPTION VIDEO TRAINING")
    print("="*70)
    print("Device:", DEVICE)
    print("Face root:", FACE_ROOT)
    print("Epochs:", EPOCHS)
    print("No FFT / GRU / attention / contrastive / GRL")
    print("="*70)

    for epoch in range(1,EPOCHS+1):
        loss = run_epoch(model, train_loader, optimizer, scaler, True)
        vm,_,_ = evaluate(model, val_loader)
        scheduler.step()

        rec = {"epoch":epoch,"loss":loss,**{f"val_{k}":v for k,v in vm.items()}}
        history.append(rec)

        print(f"\nEPOCH {epoch}/{EPOCHS}")
        print(f"Loss       : {loss:.4f}")
        print(f"Val Acc    : {vm['accuracy']:.4f}")
        print(f"Val BalAcc : {vm['balanced_accuracy']:.4f}")
        print(f"Val F1     : {vm['f1']:.4f}")
        print(f"Val AUC    : {vm['roc_auc']:.4f}")
        print(f"Val Thresh : {vm['threshold']:.2f}")

        score = vm["roc_auc"]
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "epoch":epoch,
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "threshold":vm["threshold"],
                "val_metrics":vm
            }, OUT/"best.pth")
            print(">>> SAVED BEST")
        else:
            bad += 1

        torch.save({"epoch":epoch,"model":model.state_dict()}, OUT/"latest.pth")
        (OUT/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

        if bad >= PATIENCE:
            print("Early stopping.")
            break

    ckpt = torch.load(OUT/"best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    test_m,_,_ = evaluate(model, test_loader)
    report = {
        "best_epoch": best_epoch,
        "validation_auc": best_auc,
        "test": test_m,
        "data": {
            "face_root": str(FACE_ROOT),
            "train_videos": len(train),
            "val_videos": len(val),
            "test_videos": len(test)
        },
        "model": "pretrained Xception frame scoring + mean temporal aggregation"
    }
    (OUT/"final_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("\n"+"="*70)
    print("FINAL TEST")
    print("="*70)
    for k,v in test_m.items():
        print(f"{k:20s}: {v}")
    print("Report:", OUT/"final_report.json")

if __name__ == "__main__":
    main()
