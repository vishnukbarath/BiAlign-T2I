from PIL import Image
import numpy as np
from consistency.attention import AttentionMapper


def make_two_color_image(w=64, h=32):
    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            if x < w // 2:
                img.putpixel((x, y), (0, 255, 0))
            else:
                img.putpixel((x, y), (255, 0, 0))
    return img


def test_saliency_map_shape_and_range():
    img = make_two_color_image()
    am = AttentionMapper()
    sal = am.saliency_map(img, "a green square")
    assert sal.shape == (img.height, img.width)
    assert sal.dtype == np.float32
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_saliency_focuses_on_green():
    img = make_two_color_image()
    am = AttentionMapper()
    # Compute saliency for green and for red
    sal_green = am.saliency_map(img, "a green square")
    sal_red = am.saliency_map(img, "a red square")

    # saliency should show some structure (std dev not zero)
    assert sal_green.std() > 1e-4

    # maps for different prompts should not be identical (low correlation)
    g_flat = sal_green.flatten()
    r_flat = sal_red.flatten()
    corr = (np.corrcoef(g_flat, r_flat)[0, 1])
    assert corr < 0.99
