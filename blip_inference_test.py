import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------
# CHECK DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# LOAD BLIP MODEL
# -----------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("BLIP model loaded successfully")

# -----------------------------
# LOAD TEST IMAGE
# -----------------------------
image_path = r"C:\Users\vishn\Documents\pw2\data\train2017\000000000009.jpg"
image = Image.open(image_path).convert("RGB")

print("Image loaded successfully")

# -----------------------------
# RUN INFERENCE
# -----------------------------
inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)

# -----------------------------
# OUTPUT RESULT
# -----------------------------
print("\nGenerated Caption:")
print(caption)
