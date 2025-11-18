"""PyTorch dataset for BEV segmentation.

This dataset expects a directory containing `.npz` files for inputs
and corresponding `.png` mask files for labels.  Each `.npz` file
should store an array named `image` of shape `(C, H, W)` containing the
BEV channels (e.g. max height, mean intensity, density).  The mask
image should be a single‑channel image where pixel values encode the
class index (0 for background, 1..N for objects).  A simple QA flag
can be provided via an optional mapping file, but this is not
implemented here.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BEVDataset(Dataset):
    """Basic dataset for BEV images and segmentation masks."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        # Gather file names by matching stems
        self.files: List[Tuple[Path, Path]] = []
        for npz_file in self.image_dir.glob("*.npz"):
            stem = npz_file.stem
            mask_file = self.mask_dir / f"{stem}.png"
            if mask_file.exists():
                self.files.append((npz_file, mask_file))
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npz_path, mask_path = self.files[idx]
        data = np.load(npz_path)
        image = data.get('image')
        if image is None:
            raise KeyError(f"'image' key missing in {npz_path}")
        # Convert to float32 and normalise to [0,1] range
        image = image.astype(np.float32)
        # Normalise density channel separately to avoid huge values
        if image.shape[0] >= 3:
            density = image[2]
            max_density = density.max() if density.max() > 0 else 1.0
            image[2] = density / max_density
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image)
        # Load mask
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        mask_tensor = torch.from_numpy(mask)
        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor