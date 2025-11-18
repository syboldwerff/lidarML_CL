"""Camera object detection models and datasets."""

from .dataset import CamDataset
from .train import train_cam_model
from .infer import infer_cam

__all__ = ["CamDataset", "train_cam_model", "infer_cam"]