import torch
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

# ---------------- DATASET ----------------
class CocoClipDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(IMG_DIR, images[ann["image_id"]])
        image = Image.open(image_path).convert("RGB")
        caption = ann["caption"]
        return image, caption

# ---------------- COLLATE ----------------
def clip_collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True
    )
    return inputs

# ---------------- MODEL ----------------
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

dataset = CocoClipDataset(annotations)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=clip_collate_fn
)

optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_acc = 0
    steps = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)

        logits_img = outputs.logits_per_image   # [B, B]
        logits_txt = outputs.logits_per_text    # [B, B]

        labels = torch.arange(logits_img.size(0)).to(DEVICE)

        loss_i = loss_fn(logits_img, labels)
        loss_t = loss_fn(logits_txt, labels)
        loss = (loss_i + loss_t) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accuracy
        preds = logits_img.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        steps += 1

        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    print(f"\nEpoch {epoch+1} Summary")
    print(f"Avg Loss: {total_loss/steps:.4f}")
    print(f"Avg Accuracy: {total_acc/steps:.4f}")

# ---------------- SAVE ----------------
model.save_pretrained("clip_finetuned_correct")
processor.save_pretrained("clip_finetuned_correct")

print("✅ CLIP fine‑tuning finished successfully")
