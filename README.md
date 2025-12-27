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

Demo script

```bash
python examples\run_demo.py --out-dir examples\demo_report
```

This will create two demo images, run the evaluation pipeline, produce `results.csv` and `report.html` under `examples/demo_report`.

See `docs/usage.md` for more details.

---

## Project overview

A lightweight toolkit to evaluate whether generated images match a text prompt using a combination of CLIP, object detection, and attention-based signals.

This repository implements a reproducible pipeline to compute a semantic alignment score between a text prompt and an image. The score combines multiple signals:

- Global image-text similarity via CLIP
- Region-level CLIP scores derived from object detections
- Attention/saliency maps that indicate which image regions the model focuses on for a given prompt
- Agreement between detection confidence and region relevance

The goal is to provide an extensible, testable, and reproducible framework for evaluating text–image consistency for generated images (e.g., for model evaluation or dataset curation).

**Commit 1/5:** Add project overview and high-level goals.

---

## Installation & Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
pip install -e .            # optional - install package for local dev
```

2. Run the interactive example:

```bash
python examples/run_example.py --image data/test2017/000000000139.jpg --prompt "A dog running in a park"
```

3. Run the quick demo (creates a demo report):

```bash
python examples/run_demo.py --out-dir examples/demo_report
```

4. Run tests:

```bash
PYTHONPATH=src pytest -q
```

**Commit 2/5:** Add installation and quick start instructions.

---

## Project architecture & components

The codebase is organized as a small Python package under `src/consistency` with the following main components:

- `data` — dataset loaders (COCO, JSONL stream) and collate utilities.
- `clip` — CLIP-based global scorer and embedding caching utilities.
- `detector` — object detector wrapper (YOLOv8 by default) returning bboxes, labels, and scores.
- `attention` — saliency map utilities (input-gradient saliency) to localize the prompt's influence.
- `regions` — region cropping and region-level CLIP scoring.
- `score` — composite scorer combining global, region, attention, and detector agreement into one score.
- `eval` — evaluation runner, metrics, and CSV reporting.
- `visualize` — heatmap overlays and HTML report generator.
- `cli` — simple command line interface for demo and single-image evaluation.

### Typical data flow

1. Load an (image, prompt) pair from the dataset.
2. Compute a global CLIP similarity.
3. Detect objects to produce bboxes and scores.
4. For each bbox, crop and compute region CLIP similarity.
5. Compute saliency map for prompt; measure mean saliency inside bboxes.
6. Combine signals with configurable weights to produce a single `composite` score.

**Commit 3/5:** Add architecture and components documentation.

