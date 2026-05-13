import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.s3_row_dht_1d import RowDHT1D
from scripts.s3_predict_row_dht_1d import extract_peaks


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


@torch.no_grad()
def predict_row_fractions(
    model,
    image_path: Path,
    device: str,
    image_size: tuple[int, int],
    threshold: float,
    min_distance: int,
) -> list[float]:
    out_h, out_w = image_size

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (out_w, out_h))

    x = resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)[0].cpu().numpy()
    peaks = extract_peaks(prob, threshold=threshold, min_distance=min_distance)

    return [round(y / (out_h - 1), 6) for y, _ in peaks]


def main():
    parser = argparse.ArgumentParser(
        description="Export RowDHT1D shelf-row predictions for a directory of images."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min_distance", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_paths = list_images(images_dir)
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")

    model = RowDHT1D(out_height=args.image_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        for image_path in tqdm(image_paths, desc="Exporting rows"):
            row_fracs = predict_row_fractions(
                model=model,
                image_path=image_path,
                device=device,
                image_size=(args.image_size, args.image_size),
                threshold=args.threshold,
                min_distance=args.min_distance,
            )
            writer.writerow([image_path.name, *row_fracs])

    print(f"Wrote row predictions to: {output_path}")


if __name__ == "__main__":
    main()
