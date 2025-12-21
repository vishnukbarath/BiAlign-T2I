import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- CONFIG ----------------
image_path = r"C:\Users\vishn\Downloads\img-hero-slide-4.jpg.webp"
user_prompt = "a large pool with a view of the ocean"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
print("Using device:", device)

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# ---------------- LOAD IMAGE ----------------
image = Image.open(image_path).convert("RGB")

# ---------------- BLIP CAPTION ----------------
inputs = blip_processor(image, return_tensors="pt").to(device)
caption_ids = blip_model.generate(**inputs, max_new_tokens=30)
blip_caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

# ---------------- CLIP SIMILARITY FUNCTION ----------------
def clip_similarity(image, text):
    with torch.no_grad():
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([text]).to(device)

        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

        # Normalize (IMPORTANT)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        return similarity

# ---------------- COMPUTE SIMILARITIES ----------------
sim_user = clip_similarity(image, user_prompt)
sim_blip = clip_similarity(image, blip_caption)

# ---------------- PRINT RESULTS ----------------
print("\n===== TEXT–IMAGE CONSISTENCY CHECK =====")
print("Image:", image_path)
print("User Prompt:", user_prompt)
print("BLIP Caption:", blip_caption)
print(f"Similarity (User Prompt): {sim_user:.3f}")
print(f"Similarity (BLIP Caption): {sim_blip:.3f}")
