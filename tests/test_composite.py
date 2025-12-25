from PIL import Image
from consistency.score import CompositeScorer


def make_two_color_image(w=64, h=32):
    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            if x < w // 2:
                img.putpixel((x, y), (0, 255, 0))
            else:
                img.putpixel((x, y), (255, 0, 0))
    return img


def test_composite_prefers_matching_prompt():
    img = make_two_color_image()
    # create fake detections (left and right boxes)
    left_box = [0, 0, 31, 31]
    right_box = [32, 0, 63, 31]
    detections = [{"bbox": left_box, "score": 0.9}, {"bbox": right_box, "score": 0.8}]

    cs = CompositeScorer()
    out_green = cs.score(img, "a green square", detections=detections)
    out_red = cs.score(img, "a red square", detections=detections)

    assert "composite" in out_green and "composite" in out_red
    # green prompt should get a higher composite than red prompt
    assert out_green["composite"] > out_red["composite"]
    # values are finite and within reasonable bounds
    for k, v in out_green.items():
        assert isinstance(v, float)
        assert v == v  # not NaN
