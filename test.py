from PIL import Image
from final_consistency_checker import consistency_check

image_path = r"C:\Users\vishn\Downloads\img-hero-slide-4.jpg.webp"
user_prompt = "a modern hero banner image with people and text"

result = consistency_check(image_path, user_prompt)

print("==== TEXT–IMAGE CONSISTENCY CHECK ====")
print("Image:", image_path)
print("User Prompt:", user_prompt)
print("BLIP Caption:", result["blip_caption"])
print("Similarity (User):", result["sim_user"])
print("Similarity (Caption):", result["sim_caption"])
print("Consistency Score:", result["score"])
print("Verdict:", result["verdict"])
