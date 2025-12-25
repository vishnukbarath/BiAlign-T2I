from typing import Iterable, Dict, Any, Optional
import csv
from pathlib import Path

from ..score import CompositeScorer


class EvaluationRunner:
    """Run evaluation on a dataset (iterable of dicts with keys: image, prompt, optional 'objects' or ground truth)."""

    def __init__(self, scorer: Optional[CompositeScorer] = None):
        self.scorer = scorer or CompositeScorer()

    def run(self, items: Iterable[Dict[str, Any]], output_csv: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        """Run evaluation and optionally save a CSV of results.

        Each item should be a dict with at least 'image' (PIL.Image) and 'prompt' (str). If items provide 'objects' (list of labels), they'll be preserved in the output row.
        """
        results = []
        for it in items:
            img = it["image"]
            prompt = it["prompt"]
            detections = it.get("detections")
            boxes = it.get("boxes")
            res = self.scorer.score(img, prompt, detections=detections, boxes=boxes)
            row = {"prompt": prompt, **res}
            # copy through some metadata
            for k in ("image_path", "id", "objects"):
                if k in it:
                    row[k] = it[k]
            results.append(row)
        if output_csv is not None:
            p = Path(output_csv)
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
        return results