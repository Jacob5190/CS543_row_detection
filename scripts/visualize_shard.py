import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def visualize_one(processed_dir: Path, image_name: str):
    image_path = processed_dir / "images" / image_name
    label_path = processed_dir / "labels_json" / f"{Path(image_name).stem}.json"
    output_path = processed_dir / "visualizations" / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")

    with label_path.open("r", encoding="utf-8") as f:
        label = json.load(f)

    draw = ImageDraw.Draw(img)

    for x1, y1, x2, y2 in label["lines"]:
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed/SHARD")
    parser.add_argument("--image_name", required=True)

    args = parser.parse_args()

    visualize_one(
        processed_dir=Path(args.processed_dir),
        image_name=args.image_name,
    )


if __name__ == "__main__":
    main()