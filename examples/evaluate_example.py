from pathlib import Path
from PIL import Image
from consistency.eval import EvaluationRunner


def main():
    items = [
        {"image": Image.new("RGB", (128, 128), color=(0, 255, 0)), "prompt": "a green square", "id": "g1"},
        {"image": Image.new("RGB", (128, 128), color=(255, 0, 0)), "prompt": "a red square", "id": "r1"},
    ]
    er = EvaluationRunner()
    results = er.run(items, output_csv=str(Path.cwd() / "eval_results.csv"))
    print("Wrote eval_results.csv")

if __name__ == '__main__':
    main()
