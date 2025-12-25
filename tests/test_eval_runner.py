from PIL import Image
from consistency.eval import EvaluationRunner


def make_item(prompt, color):
    img = Image.new("RGB", (32, 32), color=color)
    return {"image": img, "prompt": prompt, "image_path": None}


def test_eval_runner_csv_and_results(tmp_path):
    items = [make_item("a green square", (0, 255, 0)), make_item("a red square", (255, 0, 0))]
    er = EvaluationRunner()
    out = er.run(items)
    assert isinstance(out, list)
    assert "composite" in out[0]

    csv_path = tmp_path / "res.csv"
    er.run(items, output_csv=str(csv_path))
    assert csv_path.exists()
