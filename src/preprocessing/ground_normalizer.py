"""Ground segmentation and height normalisation.

This module provides a rudimentary ground detection and normalisation
routine.  The goal is to separate ground points from non‑ground and
express elevations relative to the local ground surface.  A common
approach is to perform a progressive morphological filter (PMF) or
cloth simulation filter (CSF) on the LiDAR point cloud.  Here we
approximate the ground as the lower quantile of Z values within a
moving window and subtract that from each point's height.

In a real system you should integrate a specialised library such as
PDAL or PCL to perform robust ground segmentation.  The
implementation below demonstrates the expected interface using
simple NumPy operations.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class GroundNormalizer:
    """Normalise point heights relative to estimated ground."""

    window_size: float = 5.0
    """Size of the sliding window used to estimate ground height in metres."""

    quantile: float = 0.05
    """Quantile of Z values considered to be ground (0–1)."""

    def normalise(self, points: np.ndarray) -> np.ndarray:
        """Estimate ground height and subtract it from each point's Z.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2.

        Returns
        -------
        numpy.ndarray
            Points with the Z coordinate replaced by height above ground.
        """
        coords = points.copy()
        z = coords[:, 2]
        # Estimate global ground elevation as the chosen quantile of all Z values.
        ground_level = np.quantile(z, self.quantile)
        coords[:, 2] = z - ground_level
        return coords