"""
FF++ -> WildDeepfake generalization experiment.
Fine-tunes the FF++ Xception checkpoint with video-level class balancing and
compression/resize augmentation, selects the checkpoint and threshold on FF++
validation only, then evaluates WildDeepfake test once.

This is an experiment, not a guarantee of higher target accuracy.
"""
import argparse, io, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

DEFAULT_FFPP_ROOT = Path(r"D:\faceforensics")
DEFAULT_CKPT = Path(r"D:\faceforensics\checkpoints_xception_compression\best_compressed.pth")
DEFAULT_OUT = Path(r"D:\faceforensics\checkpoints_xception_balanced_generalization")
DEFAULT_WILD_ROOT = Path(r"D:\newdataset\Dataset")
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class XceptionFFPPStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
        if getattr(self.backbone, "num_features", None) != 2048:
            raise RuntimeError(f"Expected 2048 Xception features; got {self.backbone.num_features}")
        self.head = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

def get_state(ckpt):
    if not isinstance(ckpt, dict): raise RuntimeError("Checkpoint is not a dictionary")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt.get(key), dict): return ckpt[key]
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()): return ckpt
    raise RuntimeError(f"No model state found. Keys: {list(ckpt)[:25]}")

def load_initial(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in get_state(ckpt).items()}
    target = model.state_dict()
    mapped = dict(state)
    used = set()
    for target_key, tensor in target.items():
        if target_key in mapped:
            used.add(target_key); continue
        if not target_key.startswith("head."): continue
        candidates = [k for k,v in state.items()
                      if not k.startswith("backbone.") and tuple(v.shape)==tuple(tensor.shape)
                      and any(s in k.lower() for s in ("head","classifier","fc","mlp")) and k not in used]
        if len(candidates)==1:
            mapped[target_key] = state[candidates[0]]
            mapped.pop(candidates[0], None)
            used.add(candidates[0])
    model.load_state_dict(mapped, strict=True)
    print(f"Loaded pretrained FF++ checkpoint: {path}")
    return ckpt

def resolve_video_dir(root, video, label):
    # Manifest labels: 0=real, 1=fake. In ffpp_faces_final folders: original/fake.
    sub = "fake" if int(label) == 1 else "original"
    base = root / "ffpp_faces_final" / sub
    v = str(video).strip().replace("/", "\\")
    candidates = [base / v]
    if not v.lower().endswith(".mp4"): candidates.append(base / (v + ".mp4"))
    for c in candidates:
        if c.is_dir(): return c
    # Some manifests contain a relative path or an ID with extension differences.
    name = Path(v).name
    if not name.lower().endswith(".mp4"): name += ".mp4"
    direct = base / name
    if direct.is_dir(): return direct
    return None

def collect_videos(root, manifest):
    df = pd.read_csv(manifest)
    if not {"video","label"}.issubset(df.columns):
        raise ValueError(f"{manifest} must contain columns 'video' and 'label'")
    items, missing = [], 0
    for row in df.itertuples(index=False):
        video, label = str(getattr(row, "video")), int(getattr(row, "label"))
        folder = resolve_video_dir(root, video, label)
        frames = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT]) if folder else []
        if not frames:
            missing += 1; continue
        items.append({"video": video, "label": label, "frames": frames})
    print(f"{manifest.name}: usable videos={len(items)}, missing/empty={missing}, "
          f"real={sum(x['label']==0 for x in items)}, fake={sum(x['label']==1 for x in items)}")
    if not items: raise RuntimeError(f"No frames resolved for {manifest}. Check paths and manifest video names.")
    return items

class VideoFrameDataset(Dataset):
    def __init__(self, items, training):
        self.items, self.training = items, training
        if training:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD)
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(), transforms.Normalize(MEAN, STD)
            ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        # One random frame per video per epoch: balances the unit of sampling by video.
        frame = random.choice(item["frames"]) if self.training else item["frames"][len(item["frames"])//2]
        try:
            with Image.open(frame) as im: im = im.convert("RGB")
            if self.training:
                # Random JPEG re-encoding models compression shift without changing labels.
                if random.random() < 0.65:
                    quality = random.randint(35, 90)
                    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality); buf.seek(0)
                    with Image.open(buf) as comp: im = comp.convert("RGB")
                if random.random() < 0.25:
                    small = random.choice([112, 160, 192])
                    im = im.resize((small, small), Image.Resampling.BILINEAR).resize(
                        (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            x = self.tf(im)
        except Exception as e:
            raise RuntimeError(f"Failed reading {frame}: {e}") from e
        return x, int(item["label"]), str(frame)

class WildImageDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for cls, label in (("real",0), ("fake",1)):
            folder = root / cls
            if not folder.is_dir(): raise FileNotFoundError(f"Missing class folder: {folder}")
            self.samples.extend((p,label) for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXT)
        if not self.samples: raise RuntimeError(f"No images found under {root}")
        self.tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,y = self.samples[i]
        with Image.open(p) as im: x=self.tf(im.convert("RGB"))
        return x,y,str(p)

def metric_pack(y, p, threshold):
    y=np.asarray(y,dtype=np.int64); p=np.asarray(p,dtype=np.float64)
    pred=(p>=threshold).astype(np.int64)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {
        "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "fake_precision":float(precision_score(y,pred,pos_label=1,zero_division=0)),
        "fake_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
        "fake_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
        "real_specificity":float(recall_score(y,pred,pos_label=0,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,p)),
        "average_precision_fake":float(average_precision_score(y,p)),
        "threshold":float(threshold),
        "confusion_matrix_labels_0_real_1_fake":cm.tolist(),
        "n":int(len(y)), "n_real":int((y==0).sum()), "n_fake":int((y==1).sum())
    }

def predict(model, loader, device):
    model.eval(); probs=[]; ys=[]; paths=[]
    with torch.inference_mode():
        for x,y,p in loader:
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                logits=model(x)
            probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist()); paths.extend(list(p))
    return np.asarray(ys), np.asarray(probs), paths

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ffpp-root", default=str(DEFAULT_FFPP_ROOT))
    ap.add_argument("--initial-checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--wild-root", default=str(DEFAULT_WILD_ROOT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-backbone", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=3)
    args=ap.parse_args()
    seed_everything(args.seed)
    root=Path(args.ffpp_root); ckpt_path=Path(args.initial_checkpoint); wild_root=Path(args.wild_root)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not ckpt_path.is_file(): raise FileNotFoundError(f"Initial checkpoint not found: {ckpt_path}")
    for name in ("train.csv","val.csv"):
        if not (root/"manifests"/name).is_file(): raise FileNotFoundError(f"Missing manifest: {root/'manifests'/name}")
    train_items=collect_videos(root,root/"manifests"/"train.csv")
    val_items=collect_videos(root,root/"manifests"/"val.csv")
    train_ds=VideoFrameDataset(train_items,True); val_ds=VideoFrameDataset(val_items,False)
    labels=np.array([x["label"] for x in train_items])
    counts=np.bincount(labels,minlength=2)
    weights=np.array([1.0/max(counts[y],1) for y in labels],dtype=np.float64)
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=len(train_items),replacement=True)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=XceptionFFPPStyle(); load_initial(model,ckpt_path); model.to(device)
    # Fine-tune only the last Xception stage and classification head to reduce catastrophic forgetting.
    for p in model.backbone.parameters(): p.requires_grad=False
    trainable=[]
    for name,p in model.backbone.named_parameters():
        if name.startswith(("block12.","block13.","conv4.","bn4.")):
            p.requires_grad=True; trainable.append(name)
    for p in model.head.parameters(): p.requires_grad=True
    if not trainable:
        # If this timm version uses different names, keep the backbone frozen rather than accidentally unfreezing all.
        print("WARNING: no late-stage names matched; training classifier head only.")
    print(f"Device: {device}; train videos={len(train_items)}; val videos={len(val_items)}")
    print(f"Trainable late-backbone tensors: {len(trainable)}")
    optimizer=torch.optim.AdamW([
        {"params":[p for p in model.backbone.parameters() if p.requires_grad],"lr":args.lr_backbone},
        {"params":list(model.head.parameters()),"lr":args.lr_head}
    ],weight_decay=1e-4)
    criterion=nn.BCEWithLogitsLoss()
    best_auc=-1; best_epoch=0; no_improve=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        # Freeze BatchNorm running stats for stability on small/imbalanced batches.
        for m in model.backbone.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.eval()
        total=0; loss_sum=0
        for x,y,_ in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
                logits=model(x); loss=criterion(logits,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step(); loss_sum += loss.item()*len(y); total += len(y)
        vy,vp,_=predict(model,val_loader,device)
        # AUC is the checkpoint-selection criterion; threshold is selected on FF++ validation only.
        val_auc=float(roc_auc_score(vy,vp))
        candidates=np.unique(np.quantile(vp,np.linspace(0.05,0.95,91)))
        best_thr=0.5; best_bacc=-1
        for th in candidates:
            bacc=balanced_accuracy_score(vy,(vp>=th).astype(int))
            if bacc>best_bacc: best_bacc=float(bacc); best_thr=float(th)
        vm=metric_pack(vy,vp,best_thr)
        history.append({"epoch":epoch,"train_loss":loss_sum/max(total,1),"val_auc":val_auc,"val_balanced_accuracy":vm["balanced_accuracy"],"val_threshold":best_thr})
        print(f"Epoch {epoch}/{args.epochs} | train_loss={loss_sum/max(total,1):.4f} | val_AUC={val_auc:.4f} | val_bal_acc={vm['balanced_accuracy']:.4f} | val_threshold={best_thr:.4f}",flush=True)
        if val_auc>best_auc:
            best_auc=val_auc; best_epoch=epoch; no_improve=0
            torch.save({"epoch":epoch,"model":model.state_dict(),"val_auc":best_auc,"val_threshold":best_thr,"val_metrics":vm,"config":vars(args)},out/"best_balanced_xception.pth")
        else:
            no_improve+=1
            if no_improve>=args.patience:
                print("Early stopping: validation AUC did not improve."); break
    # Reload best source-validation checkpoint and select threshold only on FF++ validation.
    saved=torch.load(out/"best_balanced_xception.pth",map_location="cpu",weights_only=False)
    model.load_state_dict(saved["model"],strict=True); model.to(device).eval()
    vy,vp,_=predict(model,val_loader,device)
    candidates=np.unique(np.quantile(vp,np.linspace(0.05,0.95,91)))
    best_thr=0.5; best_bacc=-1
    for th in candidates:
        bacc=balanced_accuracy_score(vy,(vp>=th).astype(int))
        if bacc>best_bacc: best_bacc=float(bacc); best_thr=float(th)
    val_metrics=metric_pack(vy,vp,best_thr)
    # Evaluate target test once. No target labels are used for training/model selection/threshold choice.
    wild_test=wild_root/"test"
    wild_ds=WildImageDataset(wild_test)
    wild_loader=DataLoader(wild_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=(device.type=="cuda"))
    wy,wp,wpaths=predict(model,wild_loader,device)
    wild_metrics=metric_pack(wy,wp,best_thr)
    report={
        "experiment":"FF++ balanced fine-tuning -> WildDeepfake cross-dataset test",
        "initial_checkpoint":str(ckpt_path),"best_checkpoint":str(out/"best_balanced_xception.pth"),
        "source_train_manifest":str(root/"manifests"/"train.csv"),
        "source_validation_manifest":str(root/"manifests"/"val.csv"),
        "target_test_directory":str(wild_test),
        "training_strategy":"video-level weighted sampling (real/fake balanced in expectation), one random frame per video per epoch, JPEG compression and low-resolution augmentation, late Xception fine-tuning",
        "threshold_source":"selected by balanced accuracy on FF++ validation only",
        "target_training_or_threshold_tuning":False,
        "best_epoch":best_epoch,"best_ffpp_val_auc":best_auc,"ffpp_val_threshold":best_thr,
        "ffpp_validation_metrics":val_metrics,"wilddeepfake_test_metrics":wild_metrics,
        "history":history,"elapsed_seconds":None
    }
    report["elapsed_seconds"]=None
    (out/"ffpp_to_wilddeepfake_balanced_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(out/"ffpp_to_wilddeepfake_balanced_predictions.npz",y_true=wy,probabilities=wp,y_pred=(wp>=best_thr).astype(int),paths=np.asarray(wpaths,dtype=str))
    print("\\nFINAL RESULTS")
    print("FF++ validation:",json.dumps(val_metrics,indent=2))
    print("WildDeepfake test:",json.dumps(wild_metrics,indent=2))
    print("Report:",out/"ffpp_to_wilddeepfake_balanced_report.json")

if __name__=="__main__": main()
