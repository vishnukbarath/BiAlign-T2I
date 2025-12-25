from typing import Iterable, Dict, Any, Optional
import csv
from pathlib import Path

from ..score import CompositeScorer
from ..visualize import overlay_saliency_and_boxes, HTMLReport


class EvaluationRunner:
    """Run evaluation on a dataset (iterable of dicts with keys: image, prompt, optional 'objects' or ground truth)."""

    def __init__(self, scorer: Optional[CompositeScorer] = None):
        self.scorer = scorer or CompositeScorer()

    def run(self, items: Iterable[Dict[str, Any]], output_csv: Optional[str] = None, save_visuals: bool = False, report_dir: Optional[str] = None, heatmap_alpha: float = 0.5) -> Iterable[Dict[str, Any]]:
        """Run evaluation and optionally save a CSV of results and visualizations.

        Each item should be a dict with at least 'image' (PIL.Image) and 'prompt' (str). If items provide 'objects' (list of labels), they'll be preserved in the output row.

        Args:
            save_visuals: if True, generate overlay images per item and add to HTML report
            report_dir: where to save the HTML report and overlay images (required if save_visuals=True)
            heatmap_alpha: alpha blending for the heatmap overlay
        """
        report = None
        if save_visuals:
            if report_dir is None:
                raise ValueError("report_dir must be specified when save_visuals=True")
            report = HTMLReport(report_dir)

        results = []
        for idx, it in enumerate(items):
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

            if save_visuals and report is not None:
                # compute saliency and overlay
                sal = self.scorer.attention.saliency_map(img, prompt, upsample_to_image=True)
                # choose boxes from scorer output (prefer given boxes/detections)
                used_boxes = boxes if boxes is not None else ([d["bbox"] for d in (detections or [])])
                labels = None
                scores = None
                if detections is not None:
                    labels = [d.get("label") for d in detections]
                    scores = [d.get("score") for d in detections]
                overlay = overlay_saliency_and_boxes(img, sal, boxes=used_boxes, box_labels=labels, box_scores=scores, alpha=heatmap_alpha)
                # name for report image
                name = f"item_{idx}"
                metadata = {"scores": res}
                report.add_row(overlay, name, metadata=metadata)

        if output_csv is not None and len(results) > 0:
            p = Path(output_csv)
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                for r in results:
                    writer.writerow(r)

        if report is not None:
            report.write()

        return results