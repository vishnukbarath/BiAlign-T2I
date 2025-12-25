from PIL import Image
import numpy as np
from consistency.visualize import overlay_saliency_and_boxes, HTMLReport
import tempfile
import os


def test_overlay_and_report(tmp_path):
    img = Image.new("RGB", (64, 32), color=(128, 128, 128))
    sal = np.zeros((32, 64), dtype=float)
    # draw a left-hot spot
    sal[:, :32] = 1.0

    boxes = [[0, 0, 31, 31], [32, 0, 63, 31]]
    labels = ["left", "right"]
    scores = [0.9, 0.8]

    out = overlay_saliency_and_boxes(img, sal, boxes=boxes, box_labels=labels, box_scores=scores)
    assert isinstance(out, Image.Image)

    # test HTML report
    d = tmp_path / "rep"
    rep = HTMLReport(str(d))
    rep.add_row(out, "sample", {"score": 0.5})
    rep.write()
    assert (d / "report.html").exists()
    assert (d / "sample.png").exists()
