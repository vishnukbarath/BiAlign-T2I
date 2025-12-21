import torch
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)


def consistency_check(image_path, user_prompt):
    image = Image.open(image_path).convert("RGB")

    # -------- BLIP CAPTION --------
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # -------- CLIP SIMILARITY --------
    clip_inputs = clip_processor(
        text=[user_prompt, caption],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**clip_inputs)

    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds

    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sim_user = (image_emb @ text_emb[0].unsqueeze(1)).item()
    sim_caption = (image_emb @ text_emb[1].unsqueeze(1)).item()

    consistency_score = (sim_user + sim_caption) / 2

    if consistency_score > 0.75:
        verdict = "MATCH"
    elif consistency_score > 0.45:
        verdict = "PARTIAL MATCH"
    else:
        verdict = "MISMATCH"

    return {
        "caption": caption,                     # ✅ ALWAYS PRESENT
        "similarity_user": sim_user,
        "similarity_caption": sim_caption,
        "consistency_score": consistency_score,
        "verdict": verdict
    }
