from PIL import Image
import json
import os
from consistency.data import PromptImageIterableDataset, PromptImageDataset


def create_sample_image(path):
    img = Image.new("RGB", (16, 16), color=(123, 222, 64))
    img.save(path)


def test_streaming_jsonl(tmp_path):
    img_path = tmp_path / "img1.jpg"
    create_sample_image(img_path)

    # JSONL with relative path
    jl = tmp_path / "ann.jsonl"
    jl.write_text(json.dumps({"image": str(img_path.name), "prompt": "a green square"}) + "\n")

    ds = PromptImageIterableDataset(str(jl), images_root=str(tmp_path))
    items = list(ds)
    assert len(items) == 1
    assert items[0]["prompt"] == "a green square"
    assert os.path.exists(items[0]["image_path"])


def test_inmemory_list_and_coco(tmp_path):
    # create two sample images
    for i in range(2):
        create_sample_image(tmp_path / f"img{i}.jpg")

    # list JSON
    list_ann = [
        {"image": "img0.jpg", "prompt": "first"},
        {"image": "img1.jpg", "prompt": "second"}
    ]
    list_file = tmp_path / "list.json"
    list_file.write_text(json.dumps(list_ann))

    ds = PromptImageDataset(str(list_file), images_root=str(tmp_path))
    assert len(ds) == 2
    first = ds[0]
    assert first["prompt"] == "first"
    assert os.path.exists(first["image_path"])

    # COCO style
    coco = {
        "images": [{"id": 1, "file_name": "img0.jpg"}, {"id": 2, "file_name": "img1.jpg"}],
        "annotations": [{"image_id": 1, "caption": "first_coco"}, {"image_id": 2, "caption": "second_coco"}]
    }
    coco_file = tmp_path / "coco.json"
    coco_file.write_text(json.dumps(coco))

    ds2 = PromptImageDataset(str(coco_file), images_root=str(tmp_path))
    assert len(ds2) == 2
    assert ds2[0]["prompt"] == "first_coco"


def test_validate_images_skips_missing(tmp_path):
    # JSON with one missing image and one present
    create_sample_image(tmp_path / "exists.jpg")
    ann = [{"image": "exists.jpg", "prompt": "ok"}, {"image": "missing.jpg", "prompt": "nope"}]
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(ann))

    ds = PromptImageDataset(str(ann_file), images_root=str(tmp_path), validate_images=True)
    assert len(ds) == 1
    assert ds[0]["prompt"] == "ok"
