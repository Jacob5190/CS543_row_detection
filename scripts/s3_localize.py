"""
localize_products.py
--------------------
Stage 3 – Product Localization (Section 3.3, Pietrini et al. 2024)

Takes the outputs of the two preceding stages:
  • Stage 1 – shelf row detection  (predict_row_dht_1d.py)
  • Stage 2 – product detection    (your product detector)

and assigns every surviving bounding box a (row, column, subrow) triple
using the three deterministic procedures described in the paper:
  AssignRow / AssignColumn / AssignSubRow

plus the false-positive filter that discards boxes whose area overlaps a
shelf-row line by more than `shelf_overlap_threshold`.

Expected input formats
----------------------
Row predictions CSV  (one row per image, same format as SHARD annotations):
    image_name, p1, p2, ..., pN
    where pi is the Y-coordinate of shelf row i as a FRACTION of image height,
    ordered top-to-bottom (0 = top, 1 = bottom).

Product detection JSON  (list of dicts, one per detected box):
    [
      {
        "image_name": "abc.jpg",
        "x1": 120, "y1": 45, "x2": 200, "y2": 130,
        "score": 0.87,          # optional confidence score
        "ean": "8000500179864"  # optional – filled in by Stage 4 recognition
      },
      ...
    ]
    Coordinates are PIXEL values.

Image size JSON  (needed to convert shelf-row fractions → pixel coords):
    { "abc.jpg": {"width": 800, "height": 600}, ... }
    If this file is not supplied the script will attempt to read image files
    directly from --image_dir.

Output
------
A JSON file where each entry has the original box fields plus:
    "shelf_row" : int  (1-based, counting from top)
    "column"    : int  (1-based, left → right within the row)
    "subrow"    : int  (1 = bottom layer, 2 = stacked on top, …)
    "discarded" : bool (True when the box overlapped a shelf-row line)

Usage
-----
python localize_products.py \
    --row_preds   data/processed/SHARD/row_predictions.csv \
    --detections  data/processed/s3_detections.json \
    --image_sizes data/processed/image_sizes.json \
    --output      data/processed/s3_localized_products.json \
    [--image_dir  data/raw/SHARD/images] \
    [--shelf_overlap_threshold 0.6] \
    [--shelf_line_thickness 0.015]
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional: use PIL/OpenCV to derive image sizes when no size JSON is given
# ---------------------------------------------------------------------------
try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_row_predictions(csv_path: str) -> dict[str, list[float]]:
    """
    Returns  { image_name: [y0, y1, ..., yN] }
    where yi are shelf-row Y-coordinates as fractions of image height,
    sorted top → bottom (ascending).
    """
    rows = {}
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Row prediction file not found: {csv_path}. "
            "Create it with scripts/s3_export_row_predictions.py first."
        )
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for line in reader:
            line = [tok.strip() for tok in line if tok.strip()]
            if not line:
                continue
            image_name = line[0]
            coords = sorted(float(v) for v in line[1:])
            rows[image_name] = coords
    return rows


def load_detections(json_path: str) -> dict[str, list[dict]]:
    """
    Returns  { image_name: [box_dict, ...] }
    Each box_dict has at minimum x1, y1, x2, y2 in pixels.
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(
            f"Detection file not found: {json_path}. "
            "Stage 3 needs product bounding boxes. For a quick pipeline test, "
            "create synthetic boxes with scripts/create_mock_detections.py; "
            "for real results, provide detector output in detections.json format."
        )
    with open(json_path) as f:
        raw = json.load(f)
    grouped = defaultdict(list)
    for det in raw:
        grouped[det["image_name"]].append(det)
    return dict(grouped)


def load_image_sizes(json_path: str) -> dict[str, dict]:
    with open(json_path) as f:
        return json.load(f)


def get_image_size(image_name: str, sizes: dict, image_dir: str | None):
    if image_name in sizes:
        s = sizes[image_name]
        return s["width"], s["height"]
    if image_dir:
        path = os.path.join(image_dir, image_name)
        if not os.path.exists(path):
            raise ValueError(f"Image file not found: {path}")
        if _HAS_PIL:
            with _PIL_Image.open(path) as img:
                return img.size
        if _HAS_CV2:
            img = _cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    raise ValueError(
        f"Cannot determine size for '{image_name}'. "
        "Supply --image_sizes or --image_dir with PIL/OpenCV installed."
    )


# ===========================================================================
# Core localization logic  (Algorithms from the paper)
# ===========================================================================

def _box_center(box: dict) -> tuple[float, float]:
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return cx, cy


def _box_area(box: dict) -> float:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def _horizontally_contains(base_box: dict, upper_box: dict) -> bool:
    """
    Returns True when upper_box is horizontally supported by base_box.
    The paper's subrow step uses horizontal containment to detect stacks.
    """
    return base_box["x1"] <= upper_box["x1"] and base_box["x2"] >= upper_box["x2"]


def filter_shelf_overlaps(
    boxes: list[dict],
    row_ys_px: list[float],
    thickness_px: float,
    threshold: float = 0.6,
) -> tuple[list[dict], list[dict]]:
    """
    Discard bounding boxes that overlap a shelf-row line band by more than
    `threshold` of the box's own area.

    A shelf-row line at y_px is modelled as a horizontal band
    [y_px - thickness/2, y_px + thickness/2].

    Returns (kept_boxes, discarded_boxes).
    """
    kept, discarded = [], []
    for box in boxes:
        box_area = _box_area(box)
        if box_area == 0:
            discarded.append({**box, "discarded": True})
            continue

        is_fp = False
        for y_px in row_ys_px:
            band_top = y_px - thickness_px / 2
            band_bot = y_px + thickness_px / 2

            # Intersection with the band
            inter_top = max(box["y1"], band_top)
            inter_bot = min(box["y2"], band_bot)
            inter_h   = max(0, inter_bot - inter_top)
            inter_w   = box["x2"] - box["x1"]
            inter_area = inter_h * inter_w

            if inter_area / box_area > threshold:
                is_fp = True
                break

        if is_fp:
            discarded.append({**box, "discarded": True})
        else:
            kept.append(box)

    return kept, discarded


def assign_rows(boxes: list[dict], row_ys_px: list[float]) -> None:
    """
    Procedure AssignRow (paper Algorithm 1 / Procedure AssignRow).

    shelf_rows are sorted top → bottom (ascending Y in image coords).
    Row numbering is 1-based counting from the TOP of the shelf.

    Logic: scan shelf-row boundaries top-to-bottom; the product lives
    in the first row whose Y boundary is BELOW (greater than) the
    product's center Y.  If no boundary is found the product is in the
    last row.
    """
    # row_ys_px is already sorted ascending (top→bottom)
    for box in boxes:
        _, cy = _box_center(box)
        assigned_row = len(row_ys_px) + 1   # default: below all lines → last row
        for idx, y_px in enumerate(row_ys_px):
            if cy < y_px:                   # center is above this line
                assigned_row = idx + 1      # rows are 1-based
                break
        box["shelf_row"] = assigned_row


def assign_columns(boxes: list[dict]) -> None:
    """
    Procedure AssignColumn.
    Within each row sort boxes by center-X left→right and assign 1, 2, 3 …
    """
    row_groups: dict[int, list[dict]] = defaultdict(list)
    for box in boxes:
        row_groups[box["shelf_row"]].append(box)

    for row_id, row_boxes in row_groups.items():
        sorted_boxes = sorted(row_boxes, key=lambda b: _box_center(b)[0])
        for col_idx, box in enumerate(sorted_boxes, start=1):
            box["column"] = col_idx


def assign_subrows(boxes: list[dict]) -> None:
    """
    Procedure AssignSubRow.

    A box A is considered stacked ON TOP of box B when:
      B horizontally contains A
      B starts below A's vertical center

    Each box starts at subrow = 1; for every such B found, A's subrow
    is incremented.  Higher subrow = higher up in the stack.
    """
    row_groups: dict[int, list[dict]] = defaultdict(list)
    for box in boxes:
        row_groups[box["shelf_row"]].append(box)

    for row_boxes in row_groups.values():
        # Initialise
        for box in row_boxes:
            box["subrow"] = 1

        for box_a in row_boxes:
            cx_a, cy_a = _box_center(box_a)
            for box_b in row_boxes:
                if box_a is box_b:
                    continue
                if _horizontally_contains(box_b, box_a) and box_b["y1"] > cy_a:
                    box_a["subrow"] += 1


def write_csv(results: list[dict], csv_path: str) -> None:
    if not results:
        fieldnames = [
            "image_name", "x1", "y1", "x2", "y2", "score", "ean",
            "shelf_row", "column", "subrow", "discarded",
        ]
    else:
        priority = [
            "image_name", "x1", "y1", "x2", "y2", "score", "ean",
            "shelf_row", "column", "subrow", "discarded",
        ]
        extras = sorted({key for row in results for key in row} - set(priority))
        fieldnames = [key for key in priority if any(key in row for row in results)] + extras

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def localize(
    boxes: list[dict],
    row_ys_px: list[float],
    shelf_line_thickness_px: float,
    shelf_overlap_threshold: float,
) -> list[dict]:
    """
    Full pipeline for one image.  Returns all boxes (kept + discarded).
    """
    kept, discarded = filter_shelf_overlaps(
        boxes, row_ys_px, shelf_line_thickness_px, shelf_overlap_threshold
    )

    if kept:
        assign_rows(kept, row_ys_px)
        assign_columns(kept)
        assign_subrows(kept)
        for box in kept:
            box["discarded"] = False

    return kept + discarded


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 – Product Localization")
    p.add_argument("--row_preds",  required=True,
                   help="CSV of shelf-row predictions (SHARD format)")
    p.add_argument("--detections", required=True,
                   help="JSON list of product detections (pixel coords)")
    p.add_argument("--output",     required=True,
                   help="Output JSON path")
    p.add_argument("--output_csv", default=None,
                   help="Optional CSV copy of the localized product output")
    p.add_argument("--image_sizes", default=None,
                   help="JSON mapping image_name → {width, height}")
    p.add_argument("--image_dir",   default=None,
                   help="Directory of raw images (used to derive sizes)")
    p.add_argument("--shelf_overlap_threshold", type=float, default=0.6,
                   help="IoU-style threshold to discard boxes over shelf lines")
    p.add_argument("--shelf_line_thickness", type=float, default=0.015,
                   help="Shelf-line thickness as fraction of image height")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading row predictions …")
    row_preds = load_row_predictions(args.row_preds)

    print("Loading product detections …")
    detections = load_detections(args.detections)

    sizes: dict = {}
    if args.image_sizes:
        print("Loading image sizes …")
        sizes = load_image_sizes(args.image_sizes)

    all_results = []
    images_processed = 0
    images_skipped   = 0

    all_image_names = set(row_preds) | set(detections)
    for image_name in sorted(all_image_names):

        # ── image dimensions ──────────────────────────────────────────────
        try:
            img_w, img_h = get_image_size(image_name, sizes, args.image_dir)
        except ValueError as e:
            print(f"  SKIP {image_name}: {e}")
            images_skipped += 1
            continue

        # ── shelf rows in pixel coords ────────────────────────────────────
        row_fracs  = row_preds.get(image_name, [])
        row_ys_px  = [f * img_h for f in row_fracs]   # ascending (top→bottom)
        thickness_px = args.shelf_line_thickness * img_h

        # ── product boxes for this image ──────────────────────────────────
        boxes = detections.get(image_name, [])
        if not boxes:
            images_processed += 1
            continue

        # Deep-copy so we don't mutate the source list
        boxes = [dict(b) for b in boxes]

        # ── run localization ──────────────────────────────────────────────
        results = localize(
            boxes,
            row_ys_px,
            thickness_px,
            args.shelf_overlap_threshold,
        )

        for r in results:
            r["image_name"] = image_name   # ensure field is present

        all_results.extend(results)
        images_processed += 1

    # ── write output ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    if args.output_csv:
        write_csv(all_results, args.output_csv)

    kept      = sum(1 for r in all_results if not r.get("discarded"))
    discarded = sum(1 for r in all_results if     r.get("discarded"))
    print(f"\nDone.  Processed {images_processed} images "
          f"({images_skipped} skipped).")
    print(f"  Boxes kept:      {kept}")
    print(f"  Boxes discarded: {discarded}")
    print(f"  JSON written:    {args.output}")
    if args.output_csv:
        print(f"  CSV written:     {args.output_csv}")


if __name__ == "__main__":
    main()
