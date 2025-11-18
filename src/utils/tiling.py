"""General tiling utilities for spatial indexing.

This module provides a helper function to generate a grid of square
tiles over a bounding box.  Each tile is defined by its lower-left
coordinate and upper-right coordinate.  This is independent from the
point cloud tiler, which partitions point clouds into tiles.
"""

from typing import Iterable, List, Tuple


def generate_tiles(x_min: float, y_min: float, x_max: float, y_max: float, size: float) -> List[Tuple[float, float, float, float]]:
    """Generate a list of tile bounding boxes covering a region.

    Parameters
    ----------
    x_min, y_min : float
        Lower left corner of the region.
    x_max, y_max : float
        Upper right corner of the region.
    size : float
        Side length of each square tile.

    Returns
    -------
    list of (float, float, float, float)
        A list of bounding boxes `(xmin, ymin, xmax, ymax)` for each tile.
    """
    tiles: List[Tuple[float, float, float, float]] = []
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            tiles.append((x, y, x + size, y + size))
            y += size
        x += size
    return tiles