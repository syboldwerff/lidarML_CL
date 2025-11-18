"""Sensor alignment and boresight corrections.

This module contains functions to apply fixed boresight rotations and
translations between the LiDAR and other sensors (e.g. cameras).
Boresight calibration corrects for small misalignments between the
coordinate frames of different sensors.  In this simplified example
we apply a constant 3D offset and an optional 3×3 rotation matrix.
In practice you would estimate these parameters from calibration
targets or by solving for alignment using overlapping scans.
"""

from typing import Iterable, Tuple

import numpy as np


def apply_boresight(points: np.ndarray, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0), rotation: np.ndarray | None = None) -> np.ndarray:
    """Apply boresight transformation to a set of points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) or (N, M) containing XYZ coordinates in
        the first three columns.
    offset : tuple of float, optional
        Translation (dx, dy, dz) to apply.  Default is no translation.
    rotation : numpy.ndarray, optional
        3×3 rotation matrix to apply.  If `None`, no rotation is
        applied.

    Returns
    -------
    numpy.ndarray
        Transformed points.
    """
    coords = points.copy()
    if rotation is not None:
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")
        coords[:, :3] = coords[:, :3].dot(rotation.T)
    coords[:, :3] += np.asarray(offset)
    return coords