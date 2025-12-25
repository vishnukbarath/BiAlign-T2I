"""Non-interactive demo: create sample images, run evaluation, and produce CSV + HTML report."""
from pathlib import Path
from PIL import Image
import argparse

from consistency.eval import EvaluationRunner


def main(out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = out_dir / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)

    img_green = demo_dir / "green.png"
    img_red = demo_dir / "red.png"
    Image.new("RGB", (128, 128), color=(0, 255, 0)).save(img_green)
    Image.new("RGB", (128, 128), color=(255, 0, 0)).save(img_red)

    items = [
        {"image": Image.open(img_green), "prompt": "a green square", "id": "g1", "image_path": str(img_green)},
        {"image": Image.open(img_red), "prompt": "a red square", "id": "r1", "image_path": str(img_red)},
    ]

    runner = EvaluationRunner()
    csv_path = out_dir / "results.csv"
    results = runner.run(items, output_csv=str(csv_path), save_visuals=True, report_dir=str(out_dir))

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out_dir / 'report.html'}")
    for r in results:
        print(r)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="examples/demo_report", help="Directory to write demo artifacts")
    args = p.parse_args()
    main(args.out_dir)
