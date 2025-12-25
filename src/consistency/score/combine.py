from typing import Optional, List, Dict
import numpy as np
from PIL import Image

from ..clip import CLIPScorer
from ..regions import RegionScorer
from ..attention import AttentionMapper
from ..detector import DetectorWrapper


class CompositeScorer:
    """Combine multiple signals into a single semantic alignment score.

    Components:
      - global: CLIP image-prompt similarity
      - region: average CLIP score across detected regions
      - attention: average saliency inside detected regions
      - detection: weighted agreement between detection confidence and region CLIP scores

    Weights are configurable and should sum to 1.0 (they will be normalized).
    """

    def __init__(self,
                 clip: Optional[CLIPScorer] = None,
                 region: Optional[RegionScorer] = None,
                 attention: Optional[AttentionMapper] = None,
                 detector: Optional[DetectorWrapper] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.clip = clip or CLIPScorer()
        self.region = region or RegionScorer()
        self.attention = attention or AttentionMapper(self.clip)
        self.detector = detector or DetectorWrapper()
        self.weights = weights or {"global": 0.4, "region": 0.3, "attention": 0.2, "detection": 0.1}
        # normalize weights
        s = sum(self.weights.values())
        if s <= 0:
            raise ValueError("weights must sum to positive value")
        self.weights = {k: float(v) / s for k, v in self.weights.items()}

    def score(self, image: Image.Image, prompt: str, detections: Optional[List[Dict]] = None, boxes: Optional[List[List[int]]] = None) -> Dict[str, float]:
        # Global score
        global_score = float(self.clip.compute_score([image], [prompt], batch_size=1, use_cache=False)[0])

        # Get boxes (from detections or provided boxes). Detections are dicts with 'bbox' and 'score'
        if boxes is None:
            if detections is None:
                detections = self.detector.detect(image)
            boxes = [d["bbox"] for d in detections]
            det_scores = [d.get("score", 1.0) for d in detections]
        else:
            det_scores = [1.0 for _ in boxes]

        # Region scores (CLIP per bbox)
        region_score = 0.0
        region_scores = []
        if len(boxes) > 0:
            prompts = [prompt] * len(boxes)
            region_scores = self.region.score_regions(image, boxes, prompts, batch_size=8, use_cache=False)
            region_score = float(np.mean(region_scores))

        # Attention: mean saliency within boxes (weighted by detection scores)
        attention_score = 0.0
        if len(boxes) > 0:
            sal = self.attention.saliency_map(image, prompt, upsample_to_image=True)
            # compute mean saliency inside each box
            box_means = []
            for b in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width - 1, max(x1, x2))
                y2 = min(image.height - 1, max(y1, y2))
                patch = sal[y1 : y2 + 1, x1 : x2 + 1]
                if patch.size == 0:
                    box_means.append(0.0)
                else:
                    box_means.append(float(patch.mean()))
            # weighted by detection scores
            ds = np.array(det_scores, dtype=float)
            if ds.sum() > 0:
                attention_score = float((np.array(box_means) * ds).sum() / ds.sum())
            else:
                attention_score = float(np.mean(box_means))

        # Detection agreement: measure of how well detector confidences align with region CLIP relevances
        detection_score = 0.0
        if len(boxes) > 0 and len(region_scores) > 0:
            rs = np.array(region_scores)
            ds = np.array(det_scores)
            # compute weighted correlation-like measure: normalized dot product
            if ds.sum() > 0 and np.linalg.norm(rs) > 0:
                detection_score = float((rs * ds).sum() / (np.linalg.norm(rs) * np.linalg.norm(ds)))
            else:
                detection_score = float((rs * ds).sum())

        # Combine
        comps = {
            "global": global_score,
            "region": region_score,
            "attention": attention_score,
            "detection": detection_score,
        }
        composite = sum(self.weights[k] * comps[k] for k in comps)
        comps["composite"] = float(composite)
        return comps