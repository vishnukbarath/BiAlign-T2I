import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = r"C:\Users\vishn\Documents\pw2\data"
IMG_DIR = os.path.join(DATA_ROOT, "train2017")
ANN_FILE = os.path.join(DATA_ROOT, "annotations", "captions_train2017.json")

BATCH_SIZE = 8          # safe start
EPOCHS = 1              # start with 1 (increase later)
LR = 5e-5
MAX_SAMPLES = 5000      # IMPORTANT: start small

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD ANNOTATIONS
# -----------------------------
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

annotations = coco["annotations"][:MAX_SAMPLES]
id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

# -----------------------------
# DATASET
# -----------------------------
class CocoCaptionDataset(Dataset):
    def __init__(self, annotations, processor):
        self.annotations = annotations
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(IMG_DIR, id_to_file[ann["image_id"]])
        image = Image.open(image_path).convert("RGB")
        caption = ann["caption"]

        inputs = self.processor(
            image,
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40
        )


