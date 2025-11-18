"""Inference utilities for BEV segmentation.

This module provides a function to run a trained U‑Net model on a
single BEV image.  The model weights are loaded from a saved
checkpoint.  The function returns a 2D array of class indices.  For
batch inference you can adapt this function or write a new script.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .model import UNet


def infer_bev(model_path: str, image_npz: str, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a single BEV tile.

    Parameters
    ----------
    model_path : str
        Path to a saved `.pt` file containing the trained model
        parameters.
    image_npz : str
        Path to an `.npz` file containing an `image` array of shape
        `(C, H, W)`.
    device : str or None
        Device on which to run inference ('cpu' or 'cuda').  If
        `None`, defaults to CUDA if available.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple of (predicted mask, probability scores).  The mask is a
        2D array of class indices and the scores array has shape
        `(num_classes, H, W)` containing per‑class probabilities.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Load image
    data = np.load(image_npz)
    image = data.get('image')
    if image is None:
        raise KeyError("'image' key missing from NPZ file")
    image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    # Instantiate model and load weights
    n_channels = image.shape[0]
    # We don't know the number of classes, so infer from checkpoint size
    state_dict = torch.load(model_path, map_location=device)
    # Derive n_classes from the final conv weight shape
    sample_key = next(k for k in state_dict.keys() if k.endswith('outc.conv.weight'))
    n_classes = state_dict[sample_key].shape[0]
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)[0]  # Remove batch dim
        probs = torch.softmax(logits, dim=0)
        mask = torch.argmax(probs, dim=0)
    return mask.cpu().numpy(), probs.cpu().numpy()