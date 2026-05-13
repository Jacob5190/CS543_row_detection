import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_detections(path: Path) -> dict[str, list[dict]]:
    with path.open("r", encoding="utf-8") as f:
        detections = json.load(f)
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["image_name"]].append(det)
    return dict(grouped)


def draw_label(draw, xy, text, color):
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(255, 255, 255), outline=color)
    draw.text((x, y), text, fill=color, font=font)


def main():
    parser = argparse.ArgumentParser(description="Visualize product detections before Stage 3 localization.")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    grouped = load_detections(Path(args.detections))
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(grouped)
    if args.limit is not None:
        names = names[:args.limit]

    written = 0
    for image_name in names:
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"SKIP missing image: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for det in grouped[image_name]:
            x1, y1, x2, y2 = [int(round(float(det[k]))) for k in ("x1", "y1", "x2", "y2")]
            color = (255, 64, 64)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            label = f"{det.get('score', 1.0):.2f}" if isinstance(det.get("score", 1.0), float) else str(det.get("score", ""))
            if det.get("ean"):
                label = f"{label} {det['ean']}"
            draw_label(draw, (x1 + 4, max(0, y1 - 14)), label, color)

        img.save(output_dir / image_name)
        written += 1

    print(f"Wrote {written} detection visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
