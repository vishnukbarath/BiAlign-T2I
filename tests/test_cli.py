import os
from pathlib import Path
from PIL import Image
from consistency.cli import main as cli_main


def test_cli_run_demo(tmp_path):
    out = tmp_path / "demo"
    cli_main(["run-demo", "--out-dir", str(out)])
    assert (out / "report.html").exists()
    assert (out / "results.csv").exists()


def test_cli_evaluate_single(tmp_path):
    img = tmp_path / "img.png"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(img)
    out = tmp_path / "eval"
    cli_main(["evaluate", "--image", str(img), "--prompt", "a green square", "--out-dir", str(out)])
    assert (out / "report.html").exists()
    assert (out / "results.csv").exists()