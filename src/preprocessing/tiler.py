"""Tile point clouds into spatially indexed chunks.

This module partitions a point cloud into a regular grid of square
tiles.  For each tile we record its bounding box, basic QA metrics
such as point count and density, and optionally compute roll/pitch
stability using trajectory data.  Tiles are saved as separate
structures to facilitate parallel processing in downstream steps
(e.g. BEV generation and machine learning).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Tile:
    """Represents a square tile of a point cloud."""
    idx: Tuple[int, int]
    bbox: Tuple[float, float, float, float]
    points: np.ndarray
    qa_flags: Dict[str, bool]


@dataclass
class Tiler:
    """Partition a point cloud into square tiles and compute QA flags."""

    size: float = 20.0
    """Tile size in metres."""

    density_threshold: float = 50.0
    """Minimum acceptable point density (points per square metre)."""

    def tile(self, points: np.ndarray) -> List[Tile]:
        """Split the point cloud into a grid of tiles.

        The algorithm computes the bounding box of the point cloud in the
        XY plane and then divides it into a grid of square cells of
        side length `size`.  All points falling within each cell are
        grouped into a `Tile` object.  A simple QA flag is computed
        based on point density.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2.

        Returns
        -------
        list of Tile
            All non‑empty tiles.
        """
        x = points[:, 0]
        y = points[:, 1]
        x_min, y_min = x.min(), y.min()
        x_max, y_max = x.max(), y.max()
        # Compute tile indices for each point
        i = np.floor((x - x_min) / self.size).astype(int)
        j = np.floor((y - y_min) / self.size).astype(int)
        tiles: Dict[Tuple[int, int], List[int]] = {}
        for idx, (ti, tj) in enumerate(zip(i, j)):
            key = (ti, tj)
            tiles.setdefault(key, []).append(idx)
        result: List[Tile] = []
        for key, indices in tiles.items():
            subset = points[indices]
            # Bounding box of this tile
            bbox = (
                x_min + key[0] * self.size,
                y_min + key[1] * self.size,
                x_min + (key[0] + 1) * self.size,
                y_min + (key[1] + 1) * self.size,
            )
            # Compute density
            area = self.size * self.size
            density = len(subset) / area
            qa_flags = {
                "density_ok": density >= self.density_threshold
            }
            result.append(Tile(key, bbox, subset, qa_flags))
        return result