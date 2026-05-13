import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_localized(json_path: Path) -> dict[str, list[dict]]:
    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_name"]].append(row)
    return dict(grouped)


def load_row_predictions(csv_path: Path) -> dict[str, list[float]]:
    row_preds = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parts = [part.strip() for part in raw_line.split(";") if part.strip()]
            if not parts:
                continue
            row_preds[parts[0]] = [float(part) for part in parts[1:]]
    return row_preds


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    bg = (255, 255, 255)
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=bg,
        outline=fill,
    )
    draw.text((x, y), text, fill=fill, font=font)


def visualize_image(
    image_path: Path,
    boxes: list[dict],
    row_fracs: list[float],
    output_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for frac in row_fracs:
        y = int(round(frac * height))
        draw.line((0, y, width - 1, y), fill=(30, 144, 255), width=4)

    for box in boxes:
        discarded = bool(box.get("discarded"))
        color = (180, 180, 180) if discarded else (255, 64, 64)
        x1, y1, x2, y2 = [int(round(box[key])) for key in ("x1", "y1", "x2", "y2")]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

        if discarded:
            label = "discarded"
        else:
            label = f"r{box.get('shelf_row')}-c{box.get('column')}-s{box.get('subrow')}"
            if box.get("ean") is not None:
                label = f"{label} {box['ean']}"

        draw_label(draw, (x1 + 4, max(0, y1 - 14)), label, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Draw shelf rows and localized product boxes."
    )
    parser.add_argument("--localized", required=True, help="localized_products.json")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--row_preds", default=None, help="Optional row_predictions.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    grouped = load_localized(Path(args.localized))
    row_preds = load_row_predictions(Path(args.row_preds)) if args.row_preds else {}
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    image_names = sorted(grouped)
    if args.limit is not None:
        image_names = image_names[:args.limit]

    written = 0
    skipped = 0
    for image_name in image_names:
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"SKIP missing image: {image_path}")
            skipped += 1
            continue

        visualize_image(
            image_path=image_path,
            boxes=grouped[image_name],
            row_fracs=row_preds.get(image_name, []),
            output_path=output_dir / image_name,
        )
        written += 1

    print(f"Wrote {written} visualizations to: {output_dir}")
    if skipped:
        print(f"Skipped {skipped} missing images")


if __name__ == "__main__":
    main()
