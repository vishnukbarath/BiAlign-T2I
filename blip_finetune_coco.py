import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = r"C:\Users\vishn\Documents\pw2\data"
IMG_DIR = os.path.join(DATA_ROOT, "train2017")
ANN_FILE = os.path.join(DATA_ROOT, "annotations", "captions_train2017.json")


BATCH_SIZE = 8
EPOCHS = 1
LR = 5e-5
MAX_SAMPLES = 5000

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

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=40
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # ✅ CORRECT: keep input_ids AND create labels
        labels = encoding["input_ids"].clone()

        # Ignore padding tokens in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

# -----------------------------
# LOAD MODEL
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

dataset = CocoCaptionDataset(annotations, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

# -----------------------------
# TRAINING LOOP
# -----------------------------
model.train()

for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=f"{loss.item():.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_pretrained("blip_finetuned")
processor.save_pretrained("blip_finetuned")

print("✅ BLIP fine-tuning completed and model saved.")
