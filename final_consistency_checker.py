import os
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

# ===============================
# DEVICE
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ===============================
# LOAD BLIP
# ===============================
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()

# ===============================
# LOAD CLIP
# ===============================
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
clip_model.eval()

# ===============================
# CONSISTENCY CHECK FUNCTION
# ===============================
def consistency_check(image_path, user_text):
    image = Image.open(image_path).convert("RGB")

    # ---------- BLIP: Caption ----------
    with torch.no_grad():
        blip_inputs = blip_processor(
            image, return_tensors="pt"
        ).to(device)

        caption_ids = blip_model.generate(
            **blip_inputs,
            max_new_tokens=30
        )

        caption = blip_processor.decode(
            caption_ids[0],
            skip_special_tokens=True
        )

    # ---------- CLIP: Relative Similarity ----------
    with torch.no_grad():
        # Image embedding
        img_inputs = clip_processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        image_embed = clip_model.get_image_features(**img_inputs)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # User text embedding
        user_inputs = clip_processor(
            text=[user_text],
            return_tensors="pt",
            padding=True
        ).to(device)

        user_embed = clip_model.get_text_features(**user_inputs)
        user_embed = user_embed / user_embed.norm(dim=-1, keepdim=True)

        # BLIP caption embedding
        caption_inputs = clip_processor(
            text=[caption],
            return_tensors="pt",
            padding=True
        ).to(device)

        caption_embed = clip_model.get_text_features(**caption_inputs)
        caption_embed = caption_embed / caption_embed.norm(dim=-1, keepdim=True)

        # Cosine similarities
        sim_user = (image_embed @ user_embed.T).item()
        sim_caption = (image_embed @ caption_embed.T).item()

        # Relative consistency score
        consistency_score = sim_user / sim_caption

    # ---------- DECISION ----------
    if consistency_score >= 0.75:
        verdict = "MATCH"
    elif consistency_score >= 0.40:
        verdict = "PARTIAL MATCH"
    else:
        verdict = "MISMATCH"

    return {
        "Image": os.path.basename(image_path),
        "User Prompt": user_text,
        "BLIP Caption": caption,
        "Similarity (User)": round(sim_user, 3),
        "Similarity (Caption)": round(sim_caption, 3),
        "Consistency Score": round(consistency_score, 3),
        "Verdict": verdict
    }

# ===============================
# MAIN (COCO val2017 TEST)
# ===============================
if __name__ == "__main__":

    image_dir = r"C:\Users\vishn\Documents\pw2\data\val2017"
    user_prompt = "bedroom bed window nightstand lamp cozy"
    num_images = 5

    images = os.listdir(image_dir)[:num_images]

    print("\n===== TEXT–IMAGE CONSISTENCY CHECK =====")

    for img in images:
        img_path = os.path.join(image_dir, img)

        result = consistency_check(img_path, user_prompt)

        print("\n----------------------------------")
        for k, v in result.items():
            print(f"{k}: {v}")
