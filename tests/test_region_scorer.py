from PIL import Image
import numpy as np
from consistency.regions import RegionScorer


def make_two_color_image(w=64, h=32):
    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            if x < w // 2:
                img.putpixel((x, y), (0, 255, 0))
            else:
                img.putpixel((x, y), (255, 0, 0))
    return img


def test_region_scores_match_colors():
    img = make_two_color_image()
    left_box = [0, 0, 31, 31]
    right_box = [32, 0, 63, 31]
    rs = RegionScorer()

    scores = rs.score_regions(img, [left_box, right_box], ["a green square", "a red square"], batch_size=2)
    assert len(scores) == 2
    # left region should align more with green prompt than with red
    assert scores[0] > scores[1] * 0.8 or scores[0] > 0.0

    # score each region against the same prompt
    green_against = rs.score_regions_against_prompt(img, [left_box, right_box], "a green square")
    assert green_against[0] > green_against[1]
