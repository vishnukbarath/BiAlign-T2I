import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

# ------------------ DEVICE ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------ LOAD MODELS ------------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)


# ------------------ CORE FUNCTION ------------------
def consistency_check(image_path, user_prompt):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # -------- BLIP CAPTION --------
    blip_inputs = blip_processor(
        images=image, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        blip_out = blip_model.generate(
            **blip_inputs, max_new_tokens=30
        )

    caption = blip_processor.decode(
        blip_out[0], skip_special_tokens=True
    )

    # -------- CLIP SIMILARITY --------
    clip_inputs = clip_processor(
        text=[user_prompt, caption],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**clip_inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    sim_user = torch.matmul(
        image_features, text_features[0].unsqueeze(1)
    ).item()

    sim_caption = torch.matmul(
        image_features, text_features[1].unsqueeze(1)
    ).item()

    # -------- CONSISTENCY SCORE --------
    score = (sim_user + sim_caption) / 2

    # -------- VERDICT --------
    if score >= 0.75:
        verdict = "MATCH"
    elif score >= 0.45:
        verdict = "PARTIAL MATCH"
    else:
        verdict = "MISMATCH"

    # 🔒 LOCKED RETURN FORMAT
    return {
        "caption": caption,
        "sim_user": round(sim_user, 3),
        "sim_caption": round(sim_caption, 3),
        "score": round(score, 3),
        "verdict": verdict
    }
