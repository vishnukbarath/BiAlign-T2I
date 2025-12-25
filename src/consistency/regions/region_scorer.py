from typing import List, Sequence, Optional
from PIL import Image
import torch
from ..clip.clip_scorer import CLIPScorer


class RegionScorer:
    """Score image regions (bboxes) against prompts using CLIP embeddings.

    Usage:
        rs = RegionScorer()
        scores = rs.score_regions(image, boxes, prompts)

    `boxes` is a sequence of [x1, y1, x2, y2] in pixel coordinates.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.clip = CLIPScorer(model_name=clip_model_name, device=device)

    def _crop(self, image: Image.Image, box: Sequence[int]) -> Image.Image:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        # ensure within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width - 1, max(x1, x2))
        y2 = min(image.height - 1, max(y1, y2))
        return image.crop((x1, y1, x2 + 1, y2 + 1))

    def score_regions(self, image: Image.Image, boxes: List[Sequence[int]], prompts: List[str], batch_size: int = 8, use_cache: bool = False):
        """Return array of cosine similarities, one per box/prompt pair."""
        assert len(boxes) == len(prompts), "boxes and prompts must be same length"
        crops = [self._crop(image, b) for b in boxes]
        return self.clip.compute_score(crops, prompts, batch_size=batch_size, use_cache=use_cache)

    def score_regions_against_prompt(self, image: Image.Image, boxes: List[Sequence[int]], prompt: str, batch_size: int = 8, use_cache: bool = False):
        """Score each region against the same prompt; returns array of similarities."""
        prompts = [prompt] * len(boxes)
        return self.score_regions(image, boxes, prompts, batch_size=batch_size, use_cache=use_cache)
