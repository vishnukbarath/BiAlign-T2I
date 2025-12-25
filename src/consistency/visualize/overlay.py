from typing import List, Optional, Sequence, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io


def _apply_colormap(sal: np.ndarray, cmap: str = "jet") -> Image.Image:
    # sal expected H x W float in [0,1]
    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(sal)[:, :, :3]  # H x W x 3
    arr = (colored * 255).astype(np.uint8)
    return Image.fromarray(arr)


def overlay_saliency_and_boxes(image: Image.Image, saliency: np.ndarray, boxes: Optional[List[Sequence[int]]] = None, box_labels: Optional[List[str]] = None, box_scores: Optional[List[float]] = None, alpha: float = 0.5) -> Image.Image:
    """Return a PIL.Image with the saliency heatmap overlayed and optional boxes drawn.

    - image: PIL.Image (RGB)
    - saliency: numpy array H x W (float 0..1). If different size, it's resized to image size.
    - boxes: list of [x1,y1,x2,y2]
    - box_labels: optional list of labels to draw next to boxes
    - box_scores: optional list of float scores to show
    - alpha: overlay alpha for heatmap
    """
    img = image.convert("RGBA")
    h, w = saliency.shape
    if (h, w) != (image.height, image.width):
        # resize saliency to image size
        sal = Image.fromarray((saliency * 255).astype('uint8'))
        sal = sal.resize((image.width, image.height), resample=Image.BILINEAR)
        sal = np.asarray(sal).astype(float) / 255.0
    else:
        sal = saliency

    heat = _apply_colormap(sal)
    heat = heat.convert("RGBA")
    # blend
    blended = Image.blend(img, heat, alpha=alpha).convert("RGBA")

    draw = ImageDraw.Draw(blended)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=2)
            label = None
            if box_labels is not None and i < len(box_labels):
                label = box_labels[i]
            if box_scores is not None and i < len(box_scores):
                score = box_scores[i]
                s = f"{score:.2f}"
                label = f"{label} {s}" if label is not None else s
            if label:
                text_pos = (x1 + 4, max(0, y1 - 12))
                draw.rectangle([text_pos, (text_pos[0] + 120, text_pos[1] + 12)], fill=(0, 0, 0, 160))
                draw.text(text_pos, label, fill=(255, 255, 255, 255), font=font)

    return blended.convert("RGB")
