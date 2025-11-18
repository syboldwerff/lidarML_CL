"""Ground segmentation and height normalisation.

This module provides ground detection and normalisation routines.  The
goal is to separate ground points from non‑ground and express
elevations relative to the local ground surface.  Common approaches
include progressive morphological filter (PMF) or cloth simulation
filter (CSF).

This implementation provides a simple quantile-based approach and a
more sophisticated grid-based local ground estimation.  For production
use, consider integrating specialised libraries such as PDAL or PCL
for robust ground segmentation.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GroundNormalizer:
    """Normalise point heights relative to estimated ground."""

    window_size: float = 5.0
    """Size of the sliding window used to estimate ground height in metres."""

    quantile: float = 0.05
    """Quantile of Z values considered to be ground (0–1)."""

    grid_resolution: float = 2.0
    """Resolution of the ground estimation grid in metres."""

    ground_threshold: float = 0.2
    """Maximum height above local ground to be classified as ground (metres)."""

    def normalise_simple(self, points: np.ndarray) -> np.ndarray:
        """Estimate ground height and subtract it from each point's Z.

        This is a simple global approach that estimates the ground as
        the chosen quantile of all Z values.

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

    def estimate_local_ground(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate local ground height using a grid-based approach.

        The point cloud is divided into a 2D grid, and for each cell
        the ground height is estimated as the minimum or low quantile
        of Z values.  Each point's height is then computed relative
        to its local ground estimate.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple of (normalised points, is_ground mask).
        """
        coords = points.copy()
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # Compute grid bounds
        x_min, y_min = x.min(), y.min()
        x_max, y_max = x.max(), y.max()

        # Create grid indices
        grid_i = np.floor((x - x_min) / self.grid_resolution).astype(int)
        grid_j = np.floor((y - y_min) / self.grid_resolution).astype(int)

        # Build a dictionary mapping grid cells to point indices
        grid_cells = {}
        for idx, (gi, gj) in enumerate(zip(grid_i, grid_j)):
            key = (gi, gj)
            grid_cells.setdefault(key, []).append(idx)

        # Estimate ground height for each cell
        ground_heights = {}
        for key, indices in grid_cells.items():
            cell_z = z[indices]
            # Use low quantile as ground estimate
            ground_heights[key] = np.quantile(cell_z, self.quantile)

        # Assign local ground height to each point
        z_ground = np.zeros_like(z)
        for idx, (gi, gj) in enumerate(zip(grid_i, grid_j)):
            key = (gi, gj)
            z_ground[idx] = ground_heights.get(key, z[idx])

        # Compute height relative to ground
        z_rel = z - z_ground

        # Classify ground points
        is_ground = np.abs(z_rel) <= self.ground_threshold

        # Update Z coordinate
        coords[:, 2] = z_rel

        return coords, is_ground

    def normalise(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise point heights and label ground points.

        This method uses the grid-based local ground estimation to
        compute relative heights and identify ground points.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple of (points with normalised Z, is_ground boolean mask).
        """
        return self.estimate_local_ground(points)
