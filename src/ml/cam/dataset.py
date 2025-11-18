"""PyTorch dataset for camera image object detection.

This dataset loads images and their corresponding YOLO‑formatted label
files.  Each label file should contain one line per object with the
format `class_id x_center y_center width height`, where the
coordinates are normalised to the range [0,1].  The dataset returns
images as torch tensors and a target dictionary containing bounding
boxes in pixel coordinates and class labels.  The target format is
compatible with PyTorch object detection models.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class CamDataset(Dataset):
    """Dataset for object detection in camera images."""

    def __init__(self, image_dir: str, label_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.files: List[Tuple[Path, Path]] = []
        for img_file in self.image_dir.glob("*.png"):
            stem = img_file.stem
            label_file = self.label_dir / f"{stem}.txt"
            if label_file.exists():
                self.files.append((img_file, label_file))
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path, lbl_path = self.files[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        # PIL does not return a NumPy array by default; convert explicitly
        import numpy as np
        image_np = np.asarray(image, dtype=np.float32)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1) / 255.0)
        # Parse labels
        boxes: List[List[float]] = []
        labels: List[int] = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, bw, bh = map(float, parts)
                # Convert to pixel coordinates: YOLO format uses centre and width/height relative to image size
                x_center = cx * w
                y_center = cy * h
                box_width = bw * w
                box_height = bh * h
                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2
                x_max = x_center + box_width / 2
                y_max = y_center + box_height / 2
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }
        if self.transform:
            image_tensor, target = self.transform(image_tensor, target)
        return image_tensor, target