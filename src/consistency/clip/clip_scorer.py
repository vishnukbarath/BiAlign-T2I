from typing import List, Optional, Callable, Dict
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class CLIPScorer:
    """Compute global CLIP similarity scores between images and prompts.

    Improvements:
    - Batching and device handling
    - In-memory caching for repeated prompts or image objects
    - Public methods to get image/text embeddings separately
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Simple in-memory caches
        self._image_cache: Dict[int, torch.Tensor] = {}
        self._text_cache: Dict[str, torch.Tensor] = {}

    def embed_images(self, images: List, batch_size: int = 8, use_cache: bool = False) -> torch.Tensor:
        """Return L2-normalized image embeddings (torch.Tensor, shape [N, D]).

        Caching uses the object's id(image) when `use_cache=True` which is suitable when the
        same PIL.Image objects are reused in a single process.
        """
        embeds = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                to_compute = []
                indices = []
                # check cache
                for j, img in enumerate(batch):
                    key = id(img)
                    if use_cache and key in self._image_cache:
                        embeds.append(self._image_cache[key])
                    else:
                        to_compute.append(img)
                        indices.append(j)
                if to_compute:
                    inputs = self.processor(images=to_compute, return_tensors="pt", padding=True).to(self.device)
                    im_feats = self.model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
                    im_feats = im_feats / im_feats.norm(p=2, dim=-1, keepdim=True)
                    # insert computed embeddings in order
                    ci = 0
                    for j in range(len(batch)):
                        key = id(batch[j])
                        if use_cache and key in self._image_cache:
                            continue
                        emb = im_feats[ci].detach().cpu()
                        if use_cache:
                            self._image_cache[key] = emb
                        embeds.append(emb)
                        ci += 1
        return torch.stack(embeds, dim=0)

    def embed_texts(self, prompts: List[str], batch_size: int = 8, use_cache: bool = False) -> torch.Tensor:
        """Return L2-normalized text embeddings (torch.Tensor, shape [N, D]).

        Caching uses prompt string as the cache key when `use_cache=True`.
        """
        embeds = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                to_compute = []
                indices = []
                for j, p in enumerate(batch):
                    if use_cache and p in self._text_cache:
                        embeds.append(self._text_cache[p])
                    else:
                        to_compute.append(p)
                        indices.append(j)
                if to_compute:
                    inputs = self.processor(text=to_compute, return_tensors="pt", padding=True).to(self.device)
                    txt_feats = self.model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
                    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
                    ci = 0
                    for j in range(len(batch)):
                        p = batch[j]
                        if use_cache and p in self._text_cache:
                            continue
                        emb = txt_feats[ci].detach().cpu()
                        if use_cache:
                            self._text_cache[p] = emb
                        embeds.append(emb)
                        ci += 1
        return torch.stack(embeds, dim=0)

    def compute_score(self, images: List, prompts: List[str], batch_size: int = 8, use_cache: bool = False, image_embeds: Optional[torch.Tensor] = None, text_embeds: Optional[torch.Tensor] = None) -> np.ndarray:
        """Compute cosine similarity per pair (image_i, prompt_i).

        Optionally accepts precomputed `image_embeds` and `text_embeds` (torch tensors on CPU)
        which should be L2-normalized.
        """
        assert len(images) == len(prompts), "images and prompts must be same length"
        if image_embeds is None:
            image_embeds = self.embed_images(images, batch_size=batch_size, use_cache=use_cache)
        if text_embeds is None:
            text_embeds = self.embed_texts(prompts, batch_size=batch_size, use_cache=use_cache)
        # ensure tensors are on CPU and same dtype
        image_embeds = image_embeds.cpu()
        text_embeds = text_embeds.cpu()
        # cosine similarity per row
        sims = (image_embeds * text_embeds).sum(dim=-1).numpy()
        return sims

    def clear_cache(self):
        """Clear internal caches for images and texts."""
        self._image_cache.clear()
        self._text_cache.clear()
