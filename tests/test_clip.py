from PIL import Image
import numpy as np
from consistency.clip import CLIPScorer


def make_img(color=(128, 128, 128)):
    return Image.new("RGB", (64, 64), color=color)


def test_compute_score_shape_and_range():
    imgs = [make_img((0, 255, 0)), make_img((255, 0, 0))]
    prompts = ["a green square", "a red square"]
    scorer = CLIPScorer()
    sims = scorer.compute_score(imgs, prompts, batch_size=2)
    assert sims.shape[0] == 2
    assert np.all(sims <= 1.0) and np.all(sims >= -1.0)


def test_cache_consistency():
    imgs = [make_img((0, 255, 0)), make_img((255, 0, 0))]
    prompts = ["a green square", "a red square"]
    scorer = CLIPScorer()
    # compute without cache
    s1 = scorer.compute_score(imgs, prompts, batch_size=2, use_cache=False)
    # compute with cache enabled
    scorer.clear_cache()
    s2 = scorer.compute_score(imgs, prompts, batch_size=1, use_cache=True)
    assert np.allclose(s1, s2, atol=1e-5)


def test_precomputed_embeddings():
    imgs = [make_img((0, 255, 0)), make_img((255, 0, 0))]
    prompts = ["a green square", "a red square"]
    scorer = CLIPScorer()
    img_emb = scorer.embed_images(imgs, batch_size=2)
    txt_emb = scorer.embed_texts(prompts, batch_size=2)
    s_embed = scorer.compute_score(imgs, prompts, image_embeds=img_emb, text_embeds=txt_emb)
    s_direct = scorer.compute_score(imgs, prompts)
    assert np.allclose(s_embed, s_direct, atol=1e-5)
