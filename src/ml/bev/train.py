"""Training script for BEV semantic segmentation models.

This module defines a function to train a U‑Net model on BEV data.
It loads training and validation datasets from directories and runs a
basic training loop using cross‑entropy loss.  For the sake of
simplicity the optimiser and hyperparameters are hard‑coded; modify
them or load from a YAML config as needed.  The trained model is
saved to the specified output file at the end of training.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import BEVDataset
from .model import UNet


def train_bev_model(
    train_image_dir: str,
    train_mask_dir: str,
    val_image_dir: Optional[str],
    val_mask_dir: Optional[str],
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_classes: int = 2,
    model_out: str = "bev_unet.pt",
    device: Optional[str] = None,
) -> None:
    """Train a UNet model on BEV data.

    Parameters
    ----------
    train_image_dir : str
        Directory containing `.npz` files with BEV images.
    train_mask_dir : str
        Directory containing `.png` mask files corresponding to the images.
    val_image_dir : str or None
        Directory for validation images.  If `None`, no validation is performed.
    val_mask_dir : str or None
        Directory for validation masks.
    epochs : int
        Number of training epochs (default 10).
    batch_size : int
        Batch size for training (default 4).
    learning_rate : float
        Initial learning rate for the optimiser (default 1e-3).
    num_classes : int
        Number of output classes (default 2).
    model_out : str
        Path to write the trained model (default 'bev_unet.pt').
    device : str or None
        Device on which to train ('cpu' or 'cuda').  If `None`, will
        use CUDA if available.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Create datasets
    train_ds = BEVDataset(train_image_dir, train_mask_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    if val_image_dir and val_mask_dir:
        val_ds = BEVDataset(val_image_dir, val_mask_dir)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        val_loader = None
    # Instantiate model
    sample_image, _ = train_ds[0]
    model = UNet(n_channels=sample_image.shape[0], n_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        # Simple validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            print(f"  Val loss={avg_val:.4f}")
    # Save model
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")