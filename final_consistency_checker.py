import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load BLIP
# -------------------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# -------------------------
# Load CLIP (fine-tuned or base)
# -------------------------
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# -------------------------
# Functions
# -------------------------
def generate_blip_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def clip_similarity(image, text):
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds

    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    similarity = (image_emb @ text_emb.T).item()
    return similarity

# -------------------------
# MAIN PIPELINE
# -------------------------
def consistency_check(image_path, user_text):
    image = Image.open(image_path).convert("RGB")

    # BLIP caption
    blip_caption = generate_blip_caption(image)

    # CLIP similarities
    sim_user = clip_similarity(image, user_text)
    sim_blip = clip_similarity(image, blip_caption)

    # Final score
    final_score = 0.6 * sim_user + 0.4 * sim_blip

    # Decision
    if final_score >= 0.75:
        verdict = "Highly Consistent"
    elif final_score >= 0.50:
        verdict = "Partially Consistent"
    else:
        verdict = "Mismatch"

    return {
        "User Prompt": user_text,
        "BLIP Caption": blip_caption,
        "CLIP(Image, User Text)": round(sim_user, 4),
        "CLIP(Image, BLIP Caption)": round(sim_blip, 4),
        "Final Consistency Score": round(final_score, 4),
        "Decision": verdict
    }

# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    image_path = "test.jpg"
    user_prompt = "A yellow container filled with food"

    result = consistency_check(image_path, user_prompt)

    for k, v in result.items():
        print(f"{k}: {v}")
