from consistency.data import PromptImageDataset
import pytest


def test_dataset_loads():
    # If COCO formatted file exists in workspace, try to load it. Otherwise, ensure constructor doesn't crash on missing file.
    try:
        ds = PromptImageDataset("data/annotations/formatted.json", images_root="data/val2017")
        assert len(ds) >= 0
        item = ds[0]
        assert "image" in item and "prompt" in item
    except FileNotFoundError:
        pytest.skip("No sample annotations available in workspace")
