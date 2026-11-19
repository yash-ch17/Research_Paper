from pathlib import Path
import csv
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Domain IDs used by the domain-adversarial branch.
# Keep this mapping fixed across train/val/test.
DOMAIN_MAP = {
    "original": 0,
    "DeepFakeDetection": 1,
    "Deepfakes": 2,
    "Face2Face": 3,
    "FaceShifter": 4,
    "FaceSwap": 5,
    "NeuralTextures": 6,
}


class VideoFaceDataset(Dataset):
    """
    One sample = one video.

    Each sample returns:
        frames -> [T, 3, H, W]
        label  -> 0 = REAL, 1 = FAKE
        domain -> manipulation/source domain ID
        video_id -> video filename without extension
    """

    def __init__(
        self,
        csv_path,
        face_root,
        frames_per_video=4,
        image_size=224,
        train=False,
        seed=2026,
    ):
        self.csv_path = Path(csv_path)
        self.face_root = Path(face_root)
        self.frames_per_video = frames_per_video
        self.train = train
        self.seed = seed

        # ---------------------------------------------------------
        # Read CSV
        # ---------------------------------------------------------
        with self.csv_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise RuntimeError(
                f"No videos found in {self.csv_path}"
            )

        # Verify required columns
        required_columns = {"video", "label"}

        actual_columns = set(self.rows[0].keys())

        missing = required_columns - actual_columns

        if missing:
            raise RuntimeError(
                f"CSV {self.csv_path} is missing columns: {missing}\n"
                f"Found columns: {actual_columns}"
            )

        # ---------------------------------------------------------
        # Transformations
        # ---------------------------------------------------------
        if train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),

                T.RandomHorizontalFlip(p=0.5),

                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=5,
                            translate=(0.03, 0.03),
                            scale=(0.95, 1.05),
                        )
                    ],
                    p=0.25,
                ),

                T.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),

                T.RandomApply(
                    [
                        T.GaussianBlur(
                            kernel_size=3,
                            sigma=(0.1, 1.2),
                        )
                    ],
                    p=0.12,
                ),

                T.ToTensor(),

                T.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ])

        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),

                T.ToTensor(),

                T.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ])

    # =========================================================
    # Basic dataset functions
    # =========================================================

    def __len__(self):
        return len(self.rows)

    # =========================================================
    # Extract video ID
    # =========================================================

    def _video_id(self, row):
        """
        Example:

        FaceForensics_C23__NeuralTextures__421_423.mp4

        becomes:

        FaceForensics_C23__NeuralTextures__421_423
        """

        video = str(row["video"]).strip()

        return Path(video).stem

    # =========================================================
    # Extract domain
    # =========================================================

    def _domain_name(self, row):
        """
        Extract manipulation/source domain from filename.

        Example:

        FaceForensics_C23__NeuralTextures__421_423.mp4

        -> NeuralTextures

        FaceForensics_C23__original__917.mp4

        -> original
        """

        video_id = self._video_id(row)

        parts = video_id.split("__")

        if len(parts) >= 3:
            domain_name = parts[1]
        else:
            domain_name = "unknown"

        return domain_name

    # =========================================================
    # Domain ID
    # =========================================================

    def _domain_id(self, row):
        domain_name = self._domain_name(row)

        if domain_name in DOMAIN_MAP:
            return DOMAIN_MAP[domain_name]

        # Unknown domains get an extra ID.
        # This should normally never happen with your FF++ data.
        return len(DOMAIN_MAP)

    # =========================================================
    # Find face crop paths
    # =========================================================

    def _paths(self, row):
   
        video = str(row["video"]).strip()

        label = int(str(row["label"]).strip())

        # Your folders are simply:
        # label 0 -> original
        # label 1 -> fake
        class_folder = "fake" if label == 1 else "original"

        # IMPORTANT:
        # The folder name includes .mp4
        video_folder = self.face_root / class_folder / video

        if video_folder.exists() and video_folder.is_dir():

            paths = sorted(
                [
                    p
                    for p in video_folder.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in VALID_EXTS
                ]
            )

            if paths:
                return paths

        # ---------------------------------------------------------
        # Fallback: search recursively for the exact video folder
        # ---------------------------------------------------------

        matches = list(
            self.face_root.rglob(video)
        )

        for match in matches:

            if match.is_dir():

                paths = sorted(
                    [
                        p
                        for p in match.iterdir()
                        if p.is_file()
                        and p.suffix.lower() in VALID_EXTS
                    ]
                )

                if paths:
                    return paths

        raise FileNotFoundError(
            "\n"
            "=============================================\n"
            "FACE CROPS NOT FOUND\n"
            "=============================================\n"
            f"Video      : {video}\n"
            f"Label      : {label}\n"
            f"Expected   : {video_folder}\n"
            f"Face root  : {self.face_root}\n"
            "=============================================\n"
        )
        # =========================================================
    # Temporal sampling
    # =========================================================

    def _sample(self, paths, index):

        if not paths:
            raise RuntimeError(
                f"No frames found for dataset index {index}"
            )

        # -----------------------------------------------------
        # If video has fewer frames than requested,
        # repeat the last frame.
        # -----------------------------------------------------

        if len(paths) <= self.frames_per_video:

            selected = list(paths)

            while len(selected) < self.frames_per_video:
                selected.append(paths[-1])

            return selected

        # -----------------------------------------------------
        # Training:
        # Random temporal sampling
        # -----------------------------------------------------

        if self.train:

            # Use deterministic but different seed per sample.
            rng = random.Random(
                self.seed + index
            )

            indices = sorted(
                rng.sample(
                    range(len(paths)),
                    self.frames_per_video,
                )
            )

        # -----------------------------------------------------
        # Validation / Test:
        # Uniform temporal sampling
        # -----------------------------------------------------

        else:

            indices = (
                torch.linspace(
                    0,
                    len(paths) - 1,
                    self.frames_per_video,
                )
                .long()
                .tolist()
            )

        return [
            paths[i]
            for i in indices
        ]

    # =========================================================
    # Load one video
    # =========================================================

    def __getitem__(self, index):

        row = self.rows[index]

        video_id = self._video_id(row)

        paths = self._paths(row)

        paths = self._sample(
            paths,
            index,
        )

        frames = []

        # -----------------------------------------------------
        # Load frames
        # -----------------------------------------------------

        for path in paths:

            try:

                with Image.open(path) as img:

                    img = img.convert("RGB")

                    frame = self.transform(img)

                    frames.append(frame)

            except Exception as e:

                print(
                    f"[WARNING] Could not read frame: "
                    f"{path} | {e}"
                )

        # -----------------------------------------------------
        # If every frame failed
        # -----------------------------------------------------

        if not frames:

            raise RuntimeError(
                f"No readable frames for video:\n"
                f"{row['video']}"
            )

        # -----------------------------------------------------
        # Pad if some frames failed
        # -----------------------------------------------------

        while len(frames) < self.frames_per_video:

            frames.append(
                frames[-1].clone()
            )

        frames = frames[
            :self.frames_per_video
        ]

        # -----------------------------------------------------
        # Label
        # -----------------------------------------------------

        label = int(
            str(row["label"]).strip()
        )

        # -----------------------------------------------------
        # Domain
        # -----------------------------------------------------

        domain = self._domain_id(row)

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {
            "frames": torch.stack(frames),

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "domain": torch.tensor(
                domain,
                dtype=torch.long,
            ),

            "video_id": video_id,
        }