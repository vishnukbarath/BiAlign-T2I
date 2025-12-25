import argparse
from PIL import Image
from consistency.clip import CLIPScorer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    args = p.parse_args()

    img = Image.open(args.image).convert("RGB")
    scorer = CLIPScorer()
    score = scorer.compute_score([img], [args.prompt])[0]
    print(f"CLIP similarity: {score:.4f}")


if __name__ == "__main__":
    main()
