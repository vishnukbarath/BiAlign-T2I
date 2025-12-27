import argparse
from pathlib import Path
from typing import List, Optional
from PIL import Image

from .eval import EvaluationRunner


def _run_demo(args: argparse.Namespace):
    """Efficient demo runner using internal APIs.

    This avoids importing/running the script module and directly uses the
    project's `EvaluationRunner`, creating images in-memory and saving only
    the artifacts (results CSV + report) to the requested output directory.
    """
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # create a small demo data folder (optional, helpful for reproducibility)
    demo_dir = out / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from .eval import EvaluationRunner





    # create two example images in-memory and save copies for audit
    img_green = Image.new("RGB", (128, 128), color=(0, 255, 0))
    img_red = Image.new("RGB", (128, 128), color=(255, 0, 0))
    img_green_path = demo_dir / "green.png"
    img_red_path = demo_dir / "red.png"
    img_green.save(img_green_path)
    img_red.save(img_red_path)

    items = [
        {"image": img_green, "prompt": "a green square", "id": "g1", "image_path": str(img_green_path)},
        {"image": img_red, "prompt": "a red square", "id": "r1", "image_path": str(img_red_path)},
    ]

    runner = EvaluationRunner()




    # Run evaluation: produces results.csv and a report with overlays
    csv_path = str(out / "results.csv")
    runner.run(items, output_csv=csv_path, save_visuals=True, report_dir=str(out))

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out / 'report.html'}")


def _evaluate(args: argparse.Namespace):
    image = Image.open(args.image).convert("RGB")
    runner = EvaluationRunner()
    items = [{"image": image, "prompt": args.prompt, "image_path": args.image}]
    out_dir = args.out_dir
    csv_path = None
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(out_dir) / "results.csv")
    results = runner.run(items, output_csv=csv_path, save_visuals=bool(out_dir), report_dir=out_dir if out_dir else None)
    print(results)




def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="consistency")
    sub = p.add_subparsers(dest="cmd")

    d_demo = sub.add_parser("run-demo", help="Run the non-interactive demo")
    d_demo.add_argument("--out-dir", default="examples/demo_report")

    d_eval = sub.add_parser("evaluate", help="Evaluate single image against a prompt")
    d_eval.add_argument("--image", required=True)
    d_eval.add_argument("--prompt", required=True)
    d_eval.add_argument("--out-dir", default=None, help="directory to write visuals and CSV (optional)")

    args = p.parse_args(argv)
    if args.cmd == "run-demo":
        _run_demo(args)
    elif args.cmd == "evaluate":
        _evaluate(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()