import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from models.s3_row_dht_1d import RowDHT1D


def extract_peaks(prob, threshold=0.35, min_distance=20):
    peaks = []

    for i in range(1, len(prob) - 1):
        if prob[i] < threshold:
            continue

        if prob[i] >= prob[i - 1] and prob[i] >= prob[i + 1]:
            peaks.append((i, float(prob[i])))

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    selected = []
    for y, score in peaks:
        if all(abs(y - prev_y) >= min_distance for prev_y, _ in selected):
            selected.append((y, score))

    selected = sorted(selected, key=lambda x: x[0])
    return selected


@torch.no_grad()
def predict_one(model, image_path, device, image_size=(512, 512), threshold=0.35):
    out_h, out_w = image_size

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (out_w, out_h))

    x = resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)[0].cpu().numpy()

    peaks = extract_peaks(
        prob=prob,
        threshold=threshold,
        min_distance=20,
    )

    rows_y_original = [
        int(round((y / (out_h - 1)) * (orig_h - 1)))
        for y, score in peaks
    ]

    return rows_y_original, peaks


def draw_rows(image_path, rows_y, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for y in rows_y:
        draw.line((0, y, width - 1, y), fill=(255, 0, 0), width=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="prediction.jpg")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = RowDHT1D(out_height=512).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows_y, peaks = predict_one(
        model=model,
        image_path=Path(args.image),
        device=device,
        image_size=(512, 512),
        threshold=args.threshold,
    )

    print("Predicted y rows:", rows_y)
    print("Peak positions/scores:", peaks)

    draw_rows(
        image_path=Path(args.image),
        rows_y=rows_y,
        output_path=Path(args.output),
    )

    print(f"Saved prediction visualization to: {args.output}")


if __name__ == "__main__":
    main()