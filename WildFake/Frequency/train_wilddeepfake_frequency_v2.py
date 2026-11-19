import json, random, time, io
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# WildDeepfake image-level adaptation of train_frequency_v2.py.
# Labels match train_fusion.py: real=0, fake=1.
DATA_ROOT = Path(r"D:\newdataset\Dataset")
OUT_DIR = Path(r"D:\newdataset\wilddeepfake_experiment\frequency_v2")
CKPT_DIR, REPORT_DIR = OUT_DIR / "checkpoints", OUT_DIR / "reports"
BEST_CKPT = CKPT_DIR / "best_frequency_v2.pth"
LATEST_CKPT = CKPT_DIR / "latest_frequency_v2.pth"
HISTORY_FILE = REPORT_DIR / "history_frequency_v2.json"
REPORT_FILE = REPORT_DIR / "final_report_frequency_v2.json"
IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS = 224, 16, 0
EPOCHS, PATIENCE = 15, 3
LR, WEIGHT_DECAY, DROPOUT, LABEL_SMOOTHING, SEED = 5e-5, 5e-4, 0.50, 0.05, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
for d in (CKPT_DIR, REPORT_DIR): d.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if AMP: torch.cuda.manual_seed_all(SEED)

def build_fft(image, augment=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    if augment:
        if random.random() < .35:
            spectrum *= np.random.normal(1.0, .035, spectrum.shape).astype(np.float32)
        if random.random() < .30:
            h, w = spectrum.shape
            yy, xx = np.ogrid[:h, :w]
            radius = np.sqrt((yy-h//2)**2 + (xx-w//2)**2)
            spectrum[radius > (.35 * min(h,w)/2)] *= random.uniform(.85, 1.15)
        if random.random() < .20:
            spectrum += np.random.normal(0, .002, spectrum.shape).astype(np.float32)
    spectrum = np.nan_to_num(spectrum, nan=0, posinf=0, neginf=0)
    spectrum = (spectrum-spectrum.mean())/(spectrum.std()+1e-6)
    return np.clip(spectrum, -5, 5).astype(np.float32)

class WildDeepfakeFrequencyDataset(Dataset):
    def __init__(self, split, train=False):
        self.root, self.train, self.samples = DATA_ROOT/split, train, []
        if not self.root.is_dir(): raise FileNotFoundError(self.root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.counts = {"fake": 0, "real": 0}
        for cls in ("fake", "real"):
            folder = self.root/cls
            if not folder.is_dir(): raise FileNotFoundError(folder)
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    self.samples.append((p, 1 if cls == "fake" else 0))
                    self.counts[cls] += 1
        if not self.samples: raise RuntimeError(f"No images in {self.root}")
        print(f"{split}: {len(self.samples):,} images | fake={self.counts['fake']:,} real={self.counts['real']:,}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, label = self.samples[i]
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None: raise RuntimeError(f"Unreadable image: {p}")
        if self.train:
            if random.random() < .5: im = cv2.flip(im, 1)
            if random.random() < .3:
                im = cv2.convertScaleAbs(im, alpha=random.uniform(.9,1.1), beta=random.uniform(-8,8))
            if random.random() < .4:
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=random.randint(45,90))
                buf.seek(0)
                im = cv2.cvtColor(np.asarray(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)
        return torch.from_numpy(build_fft(im, self.train)).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

class FrequencyCNNV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(256,128), nn.ReLU(inplace=True), nn.Dropout(DROPOUT), nn.Linear(128,1))
    def forward_features(self, x): return torch.flatten(self.features(x), 1)
    def forward(self, x): return self.classifier(self.forward_features(x)).squeeze(1)

def calc_metrics(y, p, t):
    pred = (p >= t).astype(np.int64)
    try: auc = float(roc_auc_score(y,p))
    except ValueError: auc = .5
    return {"accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "fake_precision":float(precision_score(y,pred,pos_label=1,zero_division=0)),
            "fake_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
            "fake_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
            "real_specificity":float(recall_score(y,pred,pos_label=0,zero_division=0)),
            "roc_auc":auc,"threshold":float(t),
            "confusion_matrix_labels_0_real_1_fake":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def choose_threshold(y,p):
    return max((float(t) for t in np.arange(.10,.911,.01)),
               key=lambda t: balanced_accuracy_score(y,(p>=t).astype(np.int64)))

@torch.no_grad()
def predict(model, loader):
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        x=x.to(DEVICE,non_blocking=True)
        with torch.amp.autocast("cuda",enabled=AMP): z=model(x)
        ys.extend(y.numpy().astype(int).tolist())
        ps.extend(torch.sigmoid(z.float()).cpu().numpy().tolist())
    return np.asarray(ys,dtype=np.int64), np.nan_to_num(np.asarray(ps,dtype=np.float64),nan=.5,posinf=1,neginf=0)

def main():
    if not DATA_ROOT.is_dir(): raise FileNotFoundError(DATA_ROOT)
    print("="*68, "\nWILDDEEPFAKE FREQUENCY BRANCH V2\n", "="*68, sep="")
    print("Device:",DEVICE, "| GPU:",torch.cuda.get_device_name(0) if AMP else "CPU")
    print("Dataset:",DATA_ROOT,"| Output:",OUT_DIR)
    print("Labels: real=0, fake=1; FFT: grayscale -> fft2 -> fftshift -> magnitude -> log1p -> normalize")
    train=WildDeepfakeFrequencyDataset("train",True)
    val=WildDeepfakeFrequencyDataset("valid",False)
    test=WildDeepfakeFrequencyDataset("test",False)
    labels=np.asarray([y for _,y in train.samples],dtype=np.int64)
    counts=np.bincount(labels,minlength=2)
    weights=np.asarray([1/max(counts[y],1) for y in labels],dtype=np.float64)
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),replacement=True)
    train_loader=DataLoader(train,batch_size=BATCH_SIZE,sampler=sampler,num_workers=NUM_WORKERS,pin_memory=AMP)
    val_loader=DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=AMP)
    test_loader=DataLoader(test,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=AMP)
    model=FrequencyCNNV2().to(DEVICE)
    criterion=nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1,min_lr=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=AMP)
    best_auc=-1.; best_epoch=0; bad=0; history=[]
    for epoch in range(1,EPOCHS+1):
        start=time.time(); model.train(); total=0.
        for step,(x,y) in enumerate(train_loader):
            x=x.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True)
            y_smooth=y*(1-LABEL_SMOOTHING)+LABEL_SMOOTHING/2
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=AMP): logits=model(x); loss=criterion(logits.float(),y_smooth.float())
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update(); total+=loss.item()
            if (step+1)%1000==0: print(f"  batch {step+1:,}/{len(train_loader):,} loss={loss.item():.4f}")
        train_loss=total/max(1,len(train_loader))
        vy,vp=predict(model,val_loader); threshold=choose_threshold(vy,vp); vm=calc_metrics(vy,vp,threshold)
        sched.step(vm["roc_auc"]); elapsed=round(time.time()-start,2)
        history.append({"epoch":epoch,"train_loss":train_loss,"validation":vm,"seconds":elapsed})
        print(f"Epoch {epoch:02d}/{EPOCHS} loss={train_loss:.4f} val_acc={vm['accuracy']:.4f} "
              f"bal_acc={vm['balanced_accuracy']:.4f} fake_f1={vm['fake_f1']:.4f} "
              f"AUC={vm['roc_auc']:.4f} threshold={threshold:.2f} time={elapsed:.1f}s")
        if vm["roc_auc"]>best_auc+.002:
            best_auc=vm["roc_auc"]; best_epoch=epoch; bad=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"val_auc":best_auc,
                        "val_threshold":threshold,"class_to_label":{"real":0,"fake":1},
                        "config":{"image_size":IMAGE_SIZE,"model":"FrequencyCNNV2","dropout":DROPOUT,
                                  "lr":LR,"weight_decay":WEIGHT_DECAY,"label_smoothing":LABEL_SMOOTHING,
                                  "fft":"grayscale/fft2/fftshift/abs/log1p/per-image-normalize"}},BEST_CKPT)
            print("  >>> Saved best checkpoint")
        else:
            bad+=1; print(f"  No sufficient improvement ({bad}/{PATIENCE})")
        torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"optimizer_state_dict":opt.state_dict()},LATEST_CKPT)
        HISTORY_FILE.write_text(json.dumps(history,indent=2),encoding="utf-8")
        if bad>=PATIENCE: print("Early stopping."); break
    if not BEST_CKPT.exists(): raise RuntimeError("Best checkpoint was not created.")
    ckpt=torch.load(BEST_CKPT,map_location=DEVICE,weights_only=False); model.load_state_dict(ckpt["model_state_dict"])
    vy,vp=predict(model,val_loader); val_threshold=choose_threshold(vy,vp); val_m=calc_metrics(vy,vp,val_threshold)
    ty,tp=predict(model,test_loader); test_m=calc_metrics(ty,tp,val_threshold)
    report={"model":"WildDeepfake FrequencyCNNV2","best_epoch":int(ckpt["epoch"]),
            "best_validation_auc_checkpoint":float(ckpt["val_auc"]),"validation":val_m,"test":test_m,
            "threshold_source":"validation; reused unchanged for test",
            "data":{"root":str(DATA_ROOT),"train_images":len(train),"valid_images":len(val),"test_images":len(test),
                    "train_class_counts":train.counts,"valid_class_counts":val.counts,"test_class_counts":test.counts},
            "checkpoint":str(BEST_CKPT),"architecture":"4 conv blocks; 256-d pooled features; binary classifier"}
    REPORT_FILE.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nFINAL TEST (threshold chosen on validation)")
    for k,v in test_m.items(): print(f"{k:42s}: {v}")
    print("Best checkpoint:",BEST_CKPT,"\nReport:",REPORT_FILE)

if __name__=="__main__": main()
