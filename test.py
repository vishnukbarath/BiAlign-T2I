from final_consistency_checker import consistency_check

print("TEST SCRIPT STARTED")

image_path = r"C:\Users\vishn\Downloads\generated-image.png"
user_prompt = "a modern hero banner image with people and text"

result = consistency_check(image_path, user_prompt)

print("\n==== TEXT–IMAGE CONSISTENCY CHECK ====")
print("Image:", image_path)
print("User Prompt:", user_prompt)
print("BLIP Caption:", result["caption"])
print("Similarity (User):", result["sim_user"])
print("Similarity (Caption):", result["sim_caption"])
print("Consistency Score:", result["score"])
print("Verdict:", result["verdict"])
print("TEST SCRIPT FINISHED")
