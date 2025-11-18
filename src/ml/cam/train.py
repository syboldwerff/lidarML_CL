"""Training script for camera object detection.

This module provides a wrapper around the Ultralytics YOLO training
function.  Given a dataset directory and model configuration, it
invokes YOLO training and saves the trained model to disk.  See the
Ultralytics documentation for details on available hyperparameters
and options.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def train_cam_model(
    model_cfg: str,
    data_yaml: str,
    epochs: int = 20,
    batch_size: int = 16,
    imgsz: int = 640,
    output_dir: str = "cam_model",
) -> None:
    """Train a YOLO model on the given dataset.

    Parameters
    ----------
    model_cfg : str
        Path to a YOLO model configuration or model name (e.g. 'yolov8s.yaml').
    data_yaml : str
        Path to a dataset YAML file specifying train/val splits in YOLO format.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    imgsz : int
        Input image resolution.
    output_dir : str
        Directory to save the trained model and results.
    """
    model = YOLO(model_cfg)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=output_dir,
    )