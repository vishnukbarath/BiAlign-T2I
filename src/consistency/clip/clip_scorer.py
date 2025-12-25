from typing import List, Optional
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class CLIPScorer:
    """Compute global CLIP similarity scores between images and prompts."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def compute_score(self, images: List, prompts: List[str], batch_size: int = 8) -> np.ndarray:
        assert len(images) == len(prompts), "images and prompts must be same length"
        sims = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]
                batch_prompts = prompts[i : i + batch_size]
                inputs = self.processor(text=batch_prompts, images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
                image_embeds = self.model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
                text_embeds = self.model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                batch_sims = (image_embeds * text_embeds).sum(dim=-1).cpu().numpy()
                sims.append(batch_sims)
        return np.concatenate(sims, axis=0)
