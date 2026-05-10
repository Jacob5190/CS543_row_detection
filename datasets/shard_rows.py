import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SHARDRowDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        image_size=(512, 512),
        sigma=3.0,
    ):
        self.processed_dir = Path(processed_dir)
        self.images_dir = self.processed_dir / "images"
        self.labels_dir = self.processed_dir / "labels_json"

        self.image_size = image_size
        self.out_h, self.out_w = image_size
        self.sigma = sigma

        self.label_paths = sorted(self.labels_dir.glob("*.json"))

    def __len__(self):
        return len(self.label_paths)

    def _make_target(self, rows_normalized):
        target = np.zeros((self.out_h,), dtype=np.float32)

        xs = np.arange(self.out_h)

        for coord in rows_normalized:
            y = int(round(coord * (self.out_h - 1)))
            gaussian = np.exp(-((xs - y) ** 2) / (2 * self.sigma ** 2))
            target = np.maximum(target, gaussian)

        return target

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]

        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f)

        image_path = self.images_dir / label["image"]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.out_w, self.out_h))

        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        rows_normalized = label["rows_normalized"]
        target = self._make_target(rows_normalized)

        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "image_name": label["image"],
        }