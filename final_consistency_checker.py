import os
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# LOAD BLIP (Caption Model)
# =========================
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# =========================
# LOAD CLIP (Similarity Model)
# =========================
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

# =========================
# CONSISTENCY CHECK FUNCTION
# =========================
def consistency_check(image_path, user_text):
    image = Image.open(image_path).convert("RGB")

    # ---- BLIP: Image → Caption ----
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

    # ---- CLIP: Image ↔ Text Similarity ----
    clip_inputs = clip_processor(
        text=[user_text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = clip_model(**clip_inputs)

    similarity_score = outputs.logits_per_image.softmax(dim=1)[0][0].item()
    similarity_percentage = similarity_score * 100

    # ---- Decision ----
    if similarity_percentage >= 75:
        verdict = "MATCH"
    elif similarity_percentage >= 40:
        verdict = "PARTIAL MATCH"
    else:
        verdict = "MISMATCH"

    return {
        "Image": os.path.basename(image_path),
        "User Prompt": user_text,
        "BLIP Caption": caption,
        "CLIP Similarity (%)": round(similarity_percentage, 2),
        "Verdict": verdict
    }

# =========================
# MAIN (BATCH TEST)
# =========================
if __name__ == "__main__":

    # 🔹 COCO val2017 path
    image_dir = r"C:\Users\vishn\Documents\pw2\data\val2017"

    # 🔹 User text prompt
    user_prompt = "A yellow food container"

    # 🔹 Number of images to test
    num_images = 5

    images = os.listdir(image_dir)[:num_images]

    print("\n===== TEXT–IMAGE CONSISTENCY CHECK =====")

    for img in images:
        img_path = os.path.join(image_dir, img)

        result = consistency_check(img_path, user_prompt)

        print("\n----------------------------------")
        for key, value in result.items():
            print(f"{key}: {value}")
