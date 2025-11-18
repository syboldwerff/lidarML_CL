"""Bird's‑eye view (BEV) rasterisation of LiDAR point clouds.

This module contains the `BEVGenerator` class which converts a tile of
points into a raster image.  Each pixel corresponds to a fixed area
on the ground, and multiple channels can be encoded (e.g. maximum
height, mean height, intensity and point density).  A mapping from
pixel indices to original point indices is returned to enable
back‑projection of segmentation results to 3D space.

Numba is used to accelerate the rasterisation loop.  If Numba is not
available the pure NumPy fallback will be used, though it may be
slower on large point clouds.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class BEVGenerator:
    """Generate BEV images and point mappings from point clouds."""

    resolution: float = 0.1
    """Pixel size in metres.  A resolution of 0.1 m corresponds to a
    10×10 cm pixel."""

    def rasterize(self, tile: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], list]]:
        """Rasterise a tile of points into a multi‑channel image.

        The output image has shape `(C, H, W)` where `C` is the number
        of channels (currently 3: max height, mean intensity and
        density), and `H` and `W` are determined by the tile's extent
        and the chosen `resolution`.  A mapping is also returned that
        maps pixel coordinates `(row, col)` to the indices of points in
        the input tile that contributed to that pixel.

        Parameters
        ----------
        tile : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2 and intensity
            in column 4.

        Returns
        -------
        (numpy.ndarray, dict)
            Tuple of (image, mapping).  The image is a float array of
            shape `(3, H, W)`, and the mapping is a dict keyed by
            `(row, col)` with values being lists of point indices.
        """
        pts = tile
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        intensity = pts[:, 4] if pts.shape[1] > 4 else np.zeros_like(z)
        # Determine the grid extents
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        width = int(np.ceil((x_max - x_min) / self.resolution)) + 1
        height = int(np.ceil((y_max - y_min) / self.resolution)) + 1
        # Shift points to grid coordinates
        col = ((x - x_min) / self.resolution).astype(int)
        row = ((y - y_min) / self.resolution).astype(int)
        # Initialise channels
        max_z = np.full((height, width), -np.inf, dtype=float)
        sum_intensity = np.zeros((height, width), dtype=float)
        count = np.zeros((height, width), dtype=int)
        mapping: Dict[Tuple[int, int], list] = {}
        if NUMBA_AVAILABLE:
            # Use a Numba accelerated loop
            max_z, sum_intensity, count, mapping = _rasterize_numba(
                row, col, z, intensity, height, width
            )
        else:
            # Pure Python fallback
            for idx, (r, c) in enumerate(zip(row, col)):
                if r < 0 or r >= height or c < 0 or c >= width:
                    continue
                if z[idx] > max_z[r, c]:
                    max_z[r, c] = z[idx]
                sum_intensity[r, c] += intensity[idx]
                count[r, c] += 1
                mapping.setdefault((r, c), []).append(idx)
        # Compute mean intensity
        mean_intensity = np.divide(
            sum_intensity, count, out=np.zeros_like(sum_intensity), where=count > 0
        )
        density = count.astype(float)
        # Stack channels into (C, H, W)
        image = np.stack([
            np.where(np.isfinite(max_z), max_z, 0.0),
            mean_intensity,
            density,
        ], axis=0)
        return image, mapping


if NUMBA_AVAILABLE:
    @njit
    def _rasterize_numba(row, col, z, intensity, height, width):
        max_z = np.full((height, width), -1e9, dtype=np.float64)
        sum_intensity = np.zeros((height, width), dtype=np.float64)
        count = np.zeros((height, width), dtype=np.int64)
        # We cannot use Python dict in Numba nopython mode, so we
        # construct a mapping of pixel indices to lists after the fact.
        mapping_rows = []
        mapping_cols = []
        mapping_indices = []
        for idx in range(row.shape[0]):
            r = row[idx]
            c = col[idx]
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            # Track max height
            if z[idx] > max_z[r, c]:
                max_z[r, c] = z[idx]
            sum_intensity[r, c] += intensity[idx]
            count[r, c] += 1
            mapping_rows.append(r)
            mapping_cols.append(c)
            mapping_indices.append(idx)
        # Build a Python dict outside of Numba context
        mapping: Dict[Tuple[int, int], list] = {}
        for r, c, idx in zip(mapping_rows, mapping_cols, mapping_indices):
            key = (int(r), int(c))
            if key in mapping:
                mapping[key].append(int(idx))
            else:
                mapping[key] = [int(idx)]
        return max_z, sum_intensity, count, mapping