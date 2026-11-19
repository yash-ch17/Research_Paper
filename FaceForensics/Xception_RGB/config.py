from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(r"D:\faceforensics\FaceForensics_C23")
FRAME_ROOT = Path(r"D:\faceforensics\ffpp_frames")
FACE_ROOT = Path(r"D:\faceforensics\ffpp_faces")

MANIFEST_DIR = PROJECT_DIR / "manifests"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
REPORT_DIR = PROJECT_DIR / "reports"

for p in (MANIFEST_DIR, CHECKPOINT_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# FaceForensics++ domains.
DOMAIN_NAMES = [
    "original",
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]
DOMAIN_TO_ID = {x.lower(): i for i, x in enumerate(DOMAIN_NAMES)}

REAL_NAMES = {"original", "real", "reals"}
FAKE_NAMES = {
    "deepfakedetection",
    "deepfakes",
    "face2face",
    "faceshifter",
    "faceswap",
    "neuraltextures",
    "fake",
}

# Model/data.
IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 4

# RTX 3050 4 GB-safe starting point.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 2

# Training.
SEED = 2026
EPOCHS = 12
PATIENCE = 4

LR_BACKBONE = 1e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Multi-objective loss:
# total = CE + contrastive_weight * SupCon + domain_weight * domain_CE
LAMBDA_CONTRAST = 0.05
LAMBDA_DOMAIN = 0.02
CONTRAST_TEMPERATURE = 0.07
GRL_MAX = 0.10

# Temporal branch.
GRU_HIDDEN = 192
GRU_LAYERS = 1
DROPOUT = 0.30

# Backbone.
BACKBONE_NAME = "xception"
PRETRAINED = True

# Face crop.
FACE_MARGIN = 0.20
MIN_FACE_SIZE = 40
