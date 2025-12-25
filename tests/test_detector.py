from PIL import Image
from consistency.detector import DetectorWrapper


def make_blank_image():
    return Image.new("RGB", (64, 64), color=(255, 255, 255))


def test_detector_instantiation_and_detect():
    dw = DetectorWrapper()
    img = make_blank_image()
    res = dw.detect(img)
    assert isinstance(res, list)
    # If any detections exist, they should have bbox, score, label keys
    if len(res) > 0:
        assert all(k in res[0] for k in ("bbox", "score", "label"))
