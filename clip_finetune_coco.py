import torch
import random
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_ROOT = r"C:\Users\vishn\Documents\pw2\data"
IMG_DIR = os.path.join(DATA_ROOT, "train2017")
ANN_FILE = os.path.join(DATA_ROOT, "annotations", "captions_train2017.json")

BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-5
MAX_SAMPLES = 10000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD COCO ----------------
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

images = {img["id"]: img["file_name"] for img in coco["images"]}
annotations = coco["annotations"][:MAX_SAMPLES]
all_captions = [ann["caption"] for ann in annotations]

# ---------------- DATASET ----------------
class ClipDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(IMG_DIR, images[ann["image_id"]])
        image = Image.open(image_path).convert("RGB")

        # Positive or negative pair
        if random.random() > 0.5:
            text = ann["caption"]
            label = 1
        else:
            text = random.choice(all_captions)
            label = 0

        return image, text, label

# ---------------- COLLATE FUNCTION ----------------
def clip_collate(batch):
    images, texts, labels = zip(*batch)
    inputs = processor(
        text=list(texts),
        images=list(images),
        padding=True,
        return_tensors="pt"
    )
    labels = torch.tensor(labels)
    return inputs, labels

# ---------------- MODEL ----------------
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

dataset = ClipDataset(annotations)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=clip_collate
)

optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch, labels in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)

        outputs = model(**batch)
        logits = outputs.logits_per_image

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        acc = correct / total
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    print(f"\nEpoch {epoch+1} Summary")
    print(f"Avg Loss: {total_loss/len(loader):.4f}")
    print(f"Accuracy: {acc:.4f}")

# ---------------- SAVE ----------------
model.save_pretrained("clip_finetuned")
processor.save_pretrained("clip_finetuned")

print("✅ CLIP fine‑tuning completed successfully")
