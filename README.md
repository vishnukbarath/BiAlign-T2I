# Semantic Alignment (Text–Image Consistency)

Lightweight toolkit to evaluate whether generated images match a text prompt using CLIP, object detection, and attention-based signals.

Features
- Global CLIP scoring
- Pluggable object detector wrapper (YOLOv8 by default)
- Dataset loaders for COCO / JSONL of (prompt, image_path)
- Scoring and visualization utilities (planned)

Quick start
1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the example (replace paths as needed):

```bash
python examples\run_example.py --image data\test2017\000000000139.jpg --prompt "A dog running in a park"
```

See `docs/usage.md` for more details.
