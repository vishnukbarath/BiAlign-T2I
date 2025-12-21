import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

# INPUTS
image_path = r"C:\Users\vishn\Downloads\img-hero-slide-4.jpg.webp"
user_prompt = "a large pool with a view of the ocean"

# Load image
image = Image.open(image_path).convert("RGB")

# ---- BLIP CAPTION ----
inputs = blip_processor(images=image, return_tensors="pt").to(device)
out = blip_model.generate(**inputs, max_new_tokens=30)
caption = blip_processor.decode(out[0], skip_special_tokens=True)

# ---- CLIP SIMILARITY ----
clip_inputs = clip_processor(
    text=[user_prompt, caption],
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

outputs = clip_model(**clip_inputs)
logits = outputs.logits_per_image.softmax(dim=1)

sim_user = logits[0][0].item()
sim_caption = logits[0][1].item()

# ---- RESULTS ----
print("\n===== TEST RESULT =====")
print("Image:", image_path)
print("User Prompt:", user_prompt)
print("BLIP Caption:", caption)
print(f"Similarity (User Prompt): {sim_user:.3f}")
print(f"Similarity (BLIP Caption): {sim_caption:.3f}")
