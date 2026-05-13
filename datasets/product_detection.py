import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class CocoProductDetectionDataset(Dataset):
    """
    Minimal COCO-format product detection dataset.

    Expected annotation format:
      {
        "images": [{"id": 1, "file_name": "abc.jpg", "width": 800, "height": 600}],
        "annotations": [{"image_id": 1, "bbox": [x, y, w, h], "category_id": 1}],
        "categories": [{"id": 1, "name": "product"}]
      }

    For the paper's Stage 2 product detector, product detection is usually a
    single foreground class: "product". If the annotation file contains
    multiple category IDs, they are mapped to contiguous zero-based labels.
    """

    def __init__(self, images_dir, annotation_file):
        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)

        with self.annotation_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.image_id_to_index = {
            image_info["id"]: idx for idx, image_info in enumerate(self.images)
        }

        category_ids = sorted({cat["id"] for cat in coco.get("categories", [])})
        if not category_ids:
            category_ids = sorted({ann["category_id"] for ann in coco["annotations"]})
        self.category_id_to_label = {
            category_id: idx for idx, category_id in enumerate(category_ids)
        }
        self.label_to_category_id = {
            label: category_id for category_id, label in self.category_id_to_label.items()
        }

        self.annotations_by_image_id = {image_info["id"]: [] for image_info in self.images}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            self.annotations_by_image_id.setdefault(ann["image_id"], []).append(ann)

    @property
    def num_classes(self):
        return len(self.category_id_to_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        for ann in self.annotations_by_image_id.get(image_info["id"], []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "file_name": image_info["file_name"],
        }

        return TF.to_tensor(image), target


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
