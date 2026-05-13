import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_retinanet_model(model_path: str, convert: bool):
    try:
        from keras_retinanet import models
    except ImportError as exc:
        raise ImportError(
            "keras_retinanet is required to run the provided .h5 detector. "
            "Install it with: pip install git+https://github.com/fizyr/keras-retinanet.git"
        ) from exc

    model = models.load_model(model_path, backbone_name="resnet50")
    if convert:
        model = models.convert_model(model)
    return model


def run_one_image(model, image_path: Path, score_threshold: float, max_detections: int):
    from keras_retinanet.utils.image import preprocess_image, resize_image

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image_rgb.astype(np.float32))
    processed, scale = resize_image(processed)

    boxes, scores, labels, *_ = model.predict_on_batch(np.expand_dims(processed, axis=0))
    boxes = boxes[0] / scale
    scores = scores[0]
    labels = labels[0]

    detections = []
    for box, score, label in zip(boxes[:max_detections], scores[:max_detections], labels[:max_detections]):
        score = float(score)
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "image_name": image_path.name,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "score": round(score, 6),
            "class_id": int(label),
            "category_id": int(label) + 1,
            "source": "keras_retinanet_h5",
        })

    return detections


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Return set of image names already processed in the checkpoint file."""
    if not checkpoint_path.exists():
        return set()
    seen = set()
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Each line is a list of detections for one image (may be empty)
                if record:
                    seen.add(record[0]["image_name"])
                else:
                    # Empty detection list — image was processed but had no hits.
                    # We stored the name separately; see write path below.
                    pass
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def load_checkpoint_v2(checkpoint_path: Path) -> set[str]:
    """
    Each JSONL line: {"image_name": "...", "detections": [...]}
    Returns set of already-processed image names.
    """
    if not checkpoint_path.exists():
        return set()
    seen = set()
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                seen.add(record["image_name"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def merge_checkpoint_to_json(checkpoint_path: Path, output_path: Path):
    """Flatten all detections from JSONL checkpoint into final JSON file."""
    all_detections = []
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_detections.extend(record["detections"])
            except (json.JSONDecodeError, KeyError):
                continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    return len(all_detections)


def main():
    parser = argparse.ArgumentParser(
        description="Run the provided Keras RetinaNet .h5 product detector and export detections.json."
    )
    parser.add_argument("--model", required=True, help="Path to .h5 Keras RetinaNet model")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_convert", action="store_true", help="Skip keras-retinanet convert_model step")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".jsonl")  # e.g. s3_detections.jsonl

    image_paths = list_images(Path(args.images_dir))
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # Resume: skip images already in checkpoint
    already_done = load_checkpoint_v2(checkpoint_path)
    if already_done:
        print(f"Resuming: {len(already_done)} images already processed, skipping.")
    remaining = [p for p in image_paths if p.name not in already_done]

    print(f"Loading model: {args.model}")
    model = load_retinanet_model(args.model, convert=not args.no_convert)
    print(f"Images total: {len(image_paths)} | Remaining: {len(remaining)}")

    with checkpoint_path.open("a", encoding="utf-8") as ckpt_f:
        for image_path in tqdm(remaining, desc="Detecting products"):
            try:
                detections = run_one_image(
                    model=model,
                    image_path=image_path,
                    score_threshold=args.score_threshold,
                    max_detections=args.max_detections,
                )
            except FileNotFoundError as e:
                print(f"\nWarning: {e} — skipping.")
                detections = []

            # Write one JSONL record per image immediately
            record = {"image_name": image_path.name, "detections": detections}
            ckpt_f.write(json.dumps(record) + "\n")
            ckpt_f.flush()

    # Merge checkpoint → final JSON
    total = merge_checkpoint_to_json(checkpoint_path, output_path)
    print(f"Wrote {total} detections to: {output_path}")
    print(f"Checkpoint preserved at: {checkpoint_path}")


if __name__ == "__main__":
    main()