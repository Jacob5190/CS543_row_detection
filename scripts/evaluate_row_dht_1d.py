import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

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
def predict_rows(model, image_path, device, image_size=(512, 512), threshold=0.35, min_distance=20):
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
        min_distance=min_distance,
    )

    pred_y = [
        int(round((y / (out_h - 1)) * (orig_h - 1)))
        for y, score in peaks
    ]

    return pred_y, peaks


def get_ground_truth_rows(label, image_height):
    if "rows_normalized" in label:
        coords = label["rows_normalized"]
        return [
            max(0, min(image_height - 1, int(round(coord * image_height))))
            for coord in coords
        ]

    if "rows_y" in label:
        return [int(y) for y in label["rows_y"]]

    raise KeyError("Label JSON must contain either 'rows_normalized' or 'rows_y'.")


def match_rows(gt_y, pred_y, max_match_distance):
    """
    Greedy top-to-bottom one-to-one matching.
    """
    gt_y = sorted(gt_y)
    pred_y = sorted(pred_y)

    i = 0
    j = 0
    matches = []

    while i < len(gt_y) and j < len(pred_y):
        gt = gt_y[i]
        pred = pred_y[j]
        error = abs(gt - pred)

        if error <= max_match_distance:
            matches.append((gt, pred, error))
            i += 1
            j += 1
        elif pred < gt:
            j += 1
        else:
            i += 1

    matched_gt = len(matches)
    total_gt = len(gt_y)
    total_pred = len(pred_y)

    false_negatives = total_gt - matched_gt
    false_positives = total_pred - matched_gt

    return matches, false_negatives, false_positives


def draw_comparison(image_path, gt_y, pred_y, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # GT: green
    for y in gt_y:
        draw.line((0, y, width - 1, y), fill=(0, 255, 0), width=4)

    # Prediction: red
    for y in pred_y:
        draw.line((0, y, width - 1, y), fill=(255, 0, 0), width=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed/SHARD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min_distance", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_match_distance_ratio", type=float, default=0.05)
    parser.add_argument("--out_dir", default="runs/row_dht_1d/eval_100")
    parser.add_argument("--save_visualizations", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels_json"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = RowDHT1D(out_height=args.image_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    label_paths = sorted(labels_dir.glob("*.json"))

    if len(label_paths) == 0:
        raise RuntimeError(f"No label JSON files found in {labels_dir}")

    random.seed(args.seed)
    selected_labels = random.sample(label_paths, min(args.num_images, len(label_paths)))

    results = []

    all_errors = []
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_fn = 0
    total_fp = 0

    for label_path in tqdm(selected_labels, desc="Evaluating"):
        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f)

        image_path = images_dir / label["image"]

        with Image.open(image_path) as img:
            width, height = img.size

        gt_y = get_ground_truth_rows(label, image_height=height)

        pred_y, peaks = predict_rows(
            model=model,
            image_path=image_path,
            device=device,
            image_size=(args.image_size, args.image_size),
            threshold=args.threshold,
            min_distance=args.min_distance,
        )

        max_match_distance = int(round(args.max_match_distance_ratio * height))

        matches, fn, fp = match_rows(
            gt_y=gt_y,
            pred_y=pred_y,
            max_match_distance=max_match_distance,
        )

        errors = [error for _, _, error in matches]

        image_mae = float(np.mean(errors)) if errors else None
        image_max_error = int(max(errors)) if errors else None

        all_errors.extend(errors)
        total_gt += len(gt_y)
        total_pred += len(pred_y)
        total_matched += len(matches)
        total_fn += fn
        total_fp += fp

        results.append({
            "image": label["image"],
            "height": height,
            "width": width,
            "num_gt": len(gt_y),
            "num_pred": len(pred_y),
            "num_matched": len(matches),
            "false_negatives": fn,
            "false_positives": fp,
            "mae_pixels": image_mae,
            "max_error_pixels": image_max_error,
            "gt_y": ";".join(map(str, gt_y)),
            "pred_y": ";".join(map(str, pred_y)),
            "matched_errors": ";".join(map(str, errors)),
        })

        if args.save_visualizations:
            vis_path = out_dir / "visualizations" / label["image"]
            draw_comparison(
                image_path=image_path,
                gt_y=gt_y,
                pred_y=pred_y,
                output_path=vis_path,
            )

    csv_path = out_dir / "eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "height",
            "width",
            "num_gt",
            "num_pred",
            "num_matched",
            "false_negatives",
            "false_positives",
            "mae_pixels",
            "max_error_pixels",
            "gt_y",
            "pred_y",
            "matched_errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    overall_mae = float(np.mean(all_errors)) if all_errors else None
    overall_median = float(np.median(all_errors)) if all_errors else None
    overall_max = int(max(all_errors)) if all_errors else None

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    summary = {
        "num_images": len(selected_labels),
        "threshold": args.threshold,
        "min_distance": args.min_distance,
        "max_match_distance_ratio": args.max_match_distance_ratio,
        "total_gt_rows": total_gt,
        "total_pred_rows": total_pred,
        "total_matched_rows": total_matched,
        "total_false_negatives": total_fn,
        "total_false_positives": total_fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_error_pixels": overall_mae,
        "median_error_pixels": overall_median,
        "max_error_pixels": overall_max,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation summary")
    print("------------------")
    print(f"Images evaluated:       {len(selected_labels)}")
    print(f"GT rows:                {total_gt}")
    print(f"Predicted rows:         {total_pred}")
    print(f"Matched rows:           {total_matched}")
    print(f"False negatives:        {total_fn}")
    print(f"False positives:        {total_fp}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1:                     {f1:.4f}")

    if overall_mae is not None:
        print(f"Average error:          {overall_mae:.2f} px")
        print(f"Median error:           {overall_median:.2f} px")
        print(f"Max matched error:      {overall_max} px")
    else:
        print("Average error:          N/A, no matched rows")

    print(f"\nSaved per-image results to: {csv_path}")
    print(f"Saved summary to:           {summary_path}")

    if args.save_visualizations:
        print(f"Saved visualizations to:    {out_dir / 'visualizations'}")
        print("Visualization colors: green = ground truth, red = prediction")


if __name__ == "__main__":
    main()