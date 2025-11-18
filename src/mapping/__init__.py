"""Mapping between 2D pixels and 3D points.

This package provides utilities to project between image pixels and
3D coordinates.  It also includes functions to apply sensor
alignment (boresight) corrections and fuse detections from multiple
modalities.
"""

from .pixel_to_3d import backproject_points
from .boresight import apply_boresight
from .fusion import fuse_point_sets

__all__ = [
    "backproject_points",
    "apply_boresight",
    "fuse_point_sets",
]