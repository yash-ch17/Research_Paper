"""Image-level proxy evaluation of FF++ RGB+FFT Fusion checkpoint on WildDeepfake.
Each target image is repeated 8 times because WildDeepfake test is ImageFolder (not videos).
This is NOT a true video-level evaluation. No target training or threshold tuning is performed.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

DEFAULT_CHECKPOINT = Path(r"D:\faceforensics\checkpoints_fusion_epoch2_ft\best_fusion_epoch2_ft.pth")
DEFAULT_DATA_ROOT = Path(r"D:\newdataset\Dataset")
DEFAULT_OUTPUT = Path(r"D:\newdataset\wilddeepfake_experiment\cross_eval_fusion_epoch2_to_wilddeepfake")
IMG = 224
T = 8
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class XceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(nn.Linear(2048,256), nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256,1))
    def forward(self, x):
        return self.backbone(x), self.head(self.backbone(x))

class FrequencyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256,128), nn.ReLU(inplace=True), nn.Dropout(0.40), nn.Linear(128,1))
    def forward(self, x):
        b,t,c,h,w=x.shape
        f=self.features(x.reshape(b*t,c,h,w)).flatten(1)
        logits=self.classifier(f).reshape(b,t)
        return f.reshape(b,t,-1), logits

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb=XceptionModel()
        self.frequency=FrequencyCNN()
        self.fusion_head=nn.Sequential(nn.Linear(2304,512),nn.ReLU(inplace=True),nn.Dropout(0.40),nn.Linear(512,128),nn.ReLU(inplace=True),nn.Dropout(0.40),nn.Linear(128,1))
    def forward(self,rgb,freq):
        b,t,c,h,w=rgb.shape
        rf=self.rgb.backbone(rgb.reshape(b*t,c,h,w))
        ff=self.frequency.features(freq.reshape(b*t,1,h,w)).flatten(1)
        logits=self.fusion_head(torch.cat([rf,ff],dim=1)).reshape(b,t)
        return logits.mean(dim=1)

def state_dict_from(ckpt):
    if not isinstance(ckpt,dict): raise RuntimeError("Checkpoint is not a dictionary.")
    for k in ("model","model_state_dict","state_dict"):
        if isinstance(ckpt.get(k),dict): return ckpt[k]
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()): return ckpt
    raise RuntimeError(f"No state_dict found; keys={list(ckpt)[:20]}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--data-root",default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--output-dir",default=str(DEFAULT_OUTPUT))
    ap.add_argument("--threshold",type=float,default=0.45)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--workers",type=int,default=0)
    a=ap.parse_args()
    if not 0<a.threshold<1: raise ValueError("threshold must be between 0 and 1")
    ckpt_path=Path(a.checkpoint); test_dir=Path(a.data_root)/"test"; out=Path(a.output_dir)
    if not ckpt_path.is_file(): raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}")
    if not test_dir.is_dir(): raise FileNotFoundError(f"WildDeepfake test directory not found: {test_dir}")
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform=transforms.Compose([transforms.Resize((IMG,IMG)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=datasets.ImageFolder(str(test_dir),transform=transform)
    expected={"fake":0,"real":1}
    if ds.class_to_idx != expected: raise RuntimeError(f"Unexpected class mapping {ds.class_to_idx}; expected {expected}")
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=(device.type=="cuda"))
    model=FusionModel()
    ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    state={k.removeprefix("module."):v for k,v in state_dict_from(ckpt).items()}
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    probs_all=[]; y_all=[]; paths=[]; start=time.time()
    mean=torch.tensor(MEAN,device=device).view(1,1,3,1,1)
    std=torch.tensor(STD,device=device).view(1,1,3,1,1)
    with torch.inference_mode():
        for bi,(images,folder_labels) in enumerate(loader,1):
            images=images.to(device,non_blocking=True)
            y=(folder_labels==expected["fake"]).long()
            rgb=images.unsqueeze(1).repeat(1,T,1,1,1)
            unnorm=(rgb*std+mean).clamp(0,1)
            gray=0.299*unnorm[:,:,0]+0.587*unnorm[:,:,1]+0.114*unnorm[:,:,2]
            fft=torch.fft.fftshift(torch.fft.fft2(gray),dim=(-2,-1))
            freq=torch.log1p(torch.abs(fft))
            freq=(freq-freq.mean(dim=(-2,-1),keepdim=True))/(freq.std(dim=(-2,-1),keepdim=True)+1e-6)
            freq=freq.unsqueeze(2)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
                logits=model(rgb,freq)
            probs=torch.sigmoid(logits.float()).cpu().numpy()
            probs_all.extend(probs.tolist()); y_all.extend(y.cpu().numpy().tolist())
            begin=(bi-1)*a.batch_size
            paths.extend([x[0] for x in ds.samples[begin:begin+len(images)]])
            if bi%50==0 or begin+len(images)>=len(ds): print(f"Processed {min(begin+len(images),len(ds))}/{len(ds)} images",flush=True)
    y=np.asarray(y_all,dtype=np.int64); p=np.asarray(probs_all,dtype=np.float64); pred=(p>=a.threshold).astype(np.int64)
    cm=confusion_matrix(y,pred,labels=[0,1])
    metrics={"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "fake_precision":float(precision_score(y,pred,pos_label=1,zero_division=0)),"fake_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
      "fake_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),"real_specificity":float(recall_score(y,pred,pos_label=0,zero_division=0)),
      "roc_auc":float(roc_auc_score(y,p)),"average_precision_fake":float(average_precision_score(y,p)),"threshold":float(a.threshold),
      "confusion_matrix_labels_0_real_1_fake":cm.tolist(),"n_test_images":int(len(y)),"n_real":int((y==0).sum()),"n_fake":int((y==1).sum())}
    report={"experiment":"FF++ Fusion checkpoint -> WildDeepfake test images",
      "source_checkpoint":str(ckpt_path),"target_test_directory":str(test_dir),
      "evaluation_unit":"image-level proxy; each image repeated 8 times to fit 8-frame fusion input; not true video-level evaluation",
      "target_training_or_threshold_tuning":False,"threshold_source":"fixed 0.45 from FF++ validation; not tuned on WildDeepfake",
      "image_size":IMG,"normalization":{"mean":MEAN,"std":STD},"metrics":metrics,"elapsed_seconds":round(time.time()-start,2)}
    rp=out/"fusion_epoch2_to_wilddeepfake_report.json"
    rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(out/"fusion_epoch2_to_wilddeepfake_predictions.npz",y_true=y,probabilities=p,y_pred=pred,paths=np.asarray(paths,dtype=str))
    print("\\nFINAL CROSS-DATASET METRICS")
    for k,v in metrics.items(): print(f"{k:38s}: {v}")
    print("Report:",rp); print("Output:",out)

if __name__=="__main__": main()
