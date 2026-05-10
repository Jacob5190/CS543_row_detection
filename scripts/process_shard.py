import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(images_dir: Path, image_name: str) -> Path | None:
    candidate = images_dir / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def parse_annotation_row(row):
    image_name = row[0].strip()

    coords = []

    if len(row) > 1:
        coord_string = row[1].strip()
        if coord_string:
            coords = [float(x.strip()) for x in coord_string.split(",") if x.strip()]

    return image_name, coords


def process_shard(
    images_dir: Path,
    annotation_file: Path,
    output_dir: Path,
    line_thickness: int,
    copy_images: bool,
):
    out_images = output_dir / "images"
    out_labels = output_dir / "labels_json"
    out_masks = output_dir / "masks"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    index_rows = []
    missing_images = []

    with annotation_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    for row in tqdm(rows, desc="Processing SHARD"):
        if not row or len(row) < 2:
            continue

        try:
            image_name, rows_normalized = parse_annotation_row(row)
        except ValueError:
            continue

        if image_name.lower() in {"filename", "image", "image_name", "file"}:
            continue

        image_path = find_image(images_dir, image_name)
        if image_path is None:
            missing_images.append(image_name)
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

        rows_y = [
            int(round(coord * height))
            for coord in rows_normalized
        ]

        rows_y = [
            max(0, min(height - 1, y))
            for y in rows_y
        ]

        lines = [
            [0, y, width - 1, y]
            for y in rows_y
        ]

        label = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "rows_normalized": rows_normalized,
            "rows_y": rows_y,
            "lines": lines,
        }

        label_path = out_labels / f"{image_path.stem}.json"
        with label_path.open("w", encoding="utf-8") as jf:
            json.dump(label, jf, indent=2)

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for x1, y1, x2, y2 in lines:
            draw.line((x1, y1, x2, y2), fill=255, width=line_thickness)

        mask_path = out_masks / f"{image_path.stem}.png"
        mask.save(mask_path)

        if copy_images:
            shutil.copy2(image_path, out_images / image_path.name)

        index_rows.append({
            "image": image_path.name,
            "width": width,
            "height": height,
            "num_rows": len(rows_y),
            "rows_normalized": ";".join(map(str, rows_normalized)),
            "rows_y": ";".join(map(str, rows_y)),
            "label_json": str(label_path.relative_to(output_dir)),
            "mask": str(mask_path.relative_to(output_dir)),
        })

    index_path = output_dir / "shard_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "width",
            "height",
            "num_rows",
            "rows_normalized",
            "rows_y",
            "label_json",
            "mask",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    if missing_images:
        missing_path = output_dir / "missing_images.txt"
        missing_path.write_text("\n".join(missing_images), encoding="utf-8")
        print(f"Missing images: {len(missing_images)}")
        print(f"Missing list saved to: {missing_path}")

    print(f"Processed images: {len(index_rows)}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--line_thickness", type=int, default=5)
    parser.add_argument("--copy_images", action="store_true")

    args = parser.parse_args()

    process_shard(
        images_dir=Path(args.images_dir),
        annotation_file=Path(args.annotation_file),
        output_dir=Path(args.output_dir),
        line_thickness=args.line_thickness,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()