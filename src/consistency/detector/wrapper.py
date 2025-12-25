from typing import List, Dict


class DetectorWrapper:
    """Simple wrapper for object detection. Default backend: ultralytics YOLOv8 if available."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("ultralytics is required for the DetectorWrapper. Install via `pip install ultralytics`") from e
        self.model = YOLO(model_name)

    def detect(self, image) -> List[Dict]:
        # accepts PIL.Image or numpy array
        results = self.model(image)
        out = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") else []
            scores = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") else []
            labels = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, "boxes") else []
            for b, s, l in zip(boxes, scores, labels):
                out.append({"bbox": b.tolist(), "score": float(s), "label": int(l)})
        return out
