import argparse
from pathlib import Path
from typing import List, Optional
from PIL import Image

from .eval import EvaluationRunner


def _run_demo(args: argparse.Namespace):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # re-use examples/run_demo.py logic
    from examples.run_demo import main as run_demo_main
    run_demo_main(str(out))


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