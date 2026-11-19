
import csv, io, json, random
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import timm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT=Path(r"D:\faceforensics")
FACE_ROOT=ROOT/"ffpp_faces_final"
MANIFEST=ROOT/"manifests"
CKPT=ROOT/"checkpoints_xception_final"/"best.pth"
OUT=ROOT/"checkpoints_xception_compression"
IMG=224; FRAMES=8; BATCH=2; ACCUM=4; WORKERS=2; EPOCHS=5
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MEAN=[.485,.456,.406]; STD=[.229,.224,.225]
random.seed(42); torch.manual_seed(42)

base=T.Compose([T.Resize((IMG,IMG)),T.RandomHorizontalFlip(.5)])
norm=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])

def jpeg(im,q):
    b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0)
    return Image.open(b).convert("RGB")

class DS(Dataset):
    def __init__(self,csvp,train=False):
        self.train=train; self.items=[]
        with open(csvp,encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
        for r in rows:
            v=r["video"].strip(); y=int(r["label"]); cls="fake" if y else "original"
            fs=sorted((FACE_ROOT/cls/v).glob("*.jpg"))
            if len(fs)<4: continue
            if len(fs)>=FRAMES:
                ids=torch.linspace(0,len(fs)-1,FRAMES).long().tolist(); fs=[fs[i] for i in ids]
            else: fs += [fs[-1]]*(FRAMES-len(fs))
            self.items.append((v,y,fs))
        print(Path(csvp).name,len(self.items),"REAL",sum(x[1]==0 for x in self.items),"FAKE",sum(x[1]==1 for x in self.items))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        v,y,fs=self.items[i]; out=[]
        for p in fs:
            try: im=Image.open(p).convert("RGB")
            except: im=Image.new("RGB",(IMG,IMG))
            im=base(im)
            if self.train and random.random()<.70: im=jpeg(im,random.randint(30,90))
            out.append(norm(im))
        return torch.stack(out),torch.tensor(y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=timm.create_model("xception",pretrained=False,num_classes=0,global_pool="avg")
        d=self.backbone.num_features
        self.head=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,1))
    def forward(self,x):
        b,t,c,h,w=x.shape; x=x.reshape(b*t,c,h,w); fs=[]
        for s in range(0,b*t,2): fs.append(self.backbone(x[s:s+2]))
        f=torch.cat(fs).reshape(b,t,-1)
        return self.head(f).squeeze(-1).mean(1)

@torch.no_grad()
def evaluate(model,loader):
    import numpy as np
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        x=x.to(DEVICE)
        p=torch.sigmoid(model(x))
        ys+=y.numpy().tolist(); ps+=p.float().cpu().numpy().tolist()
    y=np.asarray(ys); p=np.asarray(ps); p=np.nan_to_num(p,nan=.5,posinf=1.0,neginf=0.0); best_t=.5; best_b=-1
    for i in range(10,91):
        t=i/100; b=balanced_accuracy_score(y,(p>=t).astype(int))
        if b>best_b: best_b=b; best_t=t
    pred=(p>=best_t).astype(int)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(best_b),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "f1":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)),"threshold":float(best_t),
            "confusion_matrix":confusion_matrix(y,pred).tolist()}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not CKPT.exists(): raise FileNotFoundError(CKPT)
    tr=DS(MANIFEST/"train.csv",True); va=DS(MANIFEST/"val.csv"); te=DS(MANIFEST/"test.csv")
    cnt=Counter(x[1] for x in tr.items); w=[1/cnt[x[1]] for x in tr.items]
    tl=DataLoader(tr,batch_size=BATCH,sampler=WeightedRandomSampler(torch.DoubleTensor(w),len(w),replacement=True),num_workers=WORKERS,pin_memory=True)
    vl=DataLoader(va,batch_size=BATCH,shuffle=False,num_workers=WORKERS,pin_memory=True)
    tel=DataLoader(te,batch_size=BATCH,shuffle=False,num_workers=WORKERS,pin_memory=True)
    m=Model().to(DEVICE); ck=torch.load(CKPT,map_location=DEVICE); m.load_state_dict(ck["model"])
    print("Loaded:",CKPT,"epoch:",ck.get("epoch"),"threshold:",ck.get("threshold"))
    opt=torch.optim.AdamW([{"params":m.backbone.parameters(),"lr":3e-6},{"params":m.head.parameters(),"lr":5e-5}],weight_decay=1e-4)
    scaler=None; best=-1
    for ep in range(1,EPOCHS+1):
        m.train(); opt.zero_grad(set_to_none=True); total=0
        for step,(x,y) in enumerate(tl):
            x=x.to(DEVICE); y=y.float().to(DEVICE)
            z=torch.nan_to_num(m(x),nan=0.0,posinf=20.0,neginf=-20.0)
            loss=nn.functional.binary_cross_entropy_with_logits(z,y)/ACCUM
            loss.backward()
            if (step+1)%ACCUM==0 or step==len(tl)-1:
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
            total+=loss.item()*ACCUM
        vm=evaluate(m,vl)
        print(f"\nEPOCH {ep}/{EPOCHS} | Loss {total/len(tl):.4f} | Val Acc {vm['accuracy']:.4f} | BalAcc {vm['balanced_accuracy']:.4f} | F1 {vm['f1']:.4f} | AUC {vm['roc_auc']:.4f}")
        if vm["roc_auc"]>best:
            best=vm["roc_auc"]; torch.save({"epoch":ep,"model":m.state_dict(),"val_metrics":vm},OUT/"best_compressed.pth"); print(">>> SAVED BEST")
    ck=torch.load(OUT/"best_compressed.pth",map_location=DEVICE); m.load_state_dict(ck["model"]); tm=evaluate(m,tel)
    report={"best_epoch":ck["epoch"],"validation_auc":best,"test":tm,"training":{"compression_probability":.70,"jpeg_quality":[30,90],"epochs":EPOCHS},"model":"pretrained Xception fine-tuned with JPEG compression + mean temporal aggregation"}
    (OUT/"final_report_compression.json").write_text(json.dumps(report,indent=2))
    print("\n"+"="*70+"\nFINAL COMPRESSED-MODEL TEST\n"+"="*70)
    for k,v in tm.items(): print(f"{k:20s}: {v}")
    print("Report:",OUT/"final_report_compression.json")
if __name__=="__main__": main()
