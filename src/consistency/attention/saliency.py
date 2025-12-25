from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..clip.clip_scorer import CLIPScorer


class AttentionMapper:
    """Compute a saliency map for an (image, prompt) pair using input gradients.

    This is a simple, model-agnostic saliency approach:
      1. Run CLIP to get image & text embeddings
      2. Compute similarity and backprop to the input pixel tensor
      3. Use absolute gradient magnitude averaged over channels as the saliency map

    Note: This is not a full Grad-CAM for ViT, but provides a fast, interpretable map.
    """

    def __init__(self, clip_scorer: Optional[CLIPScorer] = None, model_name: str = "openai/clip-vit-base-patch32"):
        self.scorer = clip_scorer or CLIPScorer(model_name=model_name)
        self.model = self.scorer.model
        self.processor = self.scorer.processor
        self.device = self.scorer.device

    def saliency_map(self, image: Image.Image, prompt: str, upsample_to_image: bool = True) -> np.ndarray:
        """Return a normalized saliency map (H x W) as float32 in [0, 1].

        Args:
            image: PIL.Image
            prompt: text prompt
            upsample_to_image: if True, resize map to the original image size
        """
        self.model.eval()
        # prepare inputs
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # ensure pixel_values requires grad
        inputs["pixel_values"].requires_grad_(True)

        with torch.enable_grad():
            img_feats = self.model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
            txt_feats = self.model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
            img_feats = img_feats / (img_feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            txt_feats = txt_feats / (txt_feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = (img_feats * txt_feats).sum()
            # backprop to pixel values
            self.model.zero_grad()
            sim.backward()
            grads = inputs["pixel_values"].grad.detach().cpu()[0]  # C x H x W
        # aggregate across channels
        sal = grads.abs().mean(axis=0)  # H x W
        sal = sal.numpy()
        # normalize
        sal = sal - sal.min()
        if sal.max() > 0:
            sal = sal / sal.max()
        # upsample to original image size if requested
        if upsample_to_image:
            sal_t = torch.from_numpy(sal[None, None, ...]).float()
            sal_up = F.interpolate(sal_t, size=(image.height, image.width), mode="bilinear", align_corners=False)
            sal = sal_up[0, 0].numpy()
            sal = sal - sal.min()
            if sal.max() > 0:
                sal = sal / sal.max()
        return sal.astype(np.float32)
