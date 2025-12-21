import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD FINETUNED MODEL
processor = BlipProcessor.from_pretrained("blip_finetuned")
model = BlipForConditionalGeneration.from_pretrained("blip_finetuned").to(device)

# TEST IMAGE
image_path = r"C:\Users\vishn\Documents\pw2\data\val2017\000000000139.jpg"
image = Image.open(image_path).convert("RGB")

inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:")
print(caption)
