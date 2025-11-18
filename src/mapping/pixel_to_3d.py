"""Back‑projection of pixel labels to 3D points.

This module implements a simple utility for back‑projecting labels
assigned to pixels in a BEV or camera image back to the original
points.  Given a mapping from pixel indices to point indices (as
produced by the BEV and camera generators), it returns the set of
point indices corresponding to a labelled region.
"""

from typing import Dict, Iterable, List, Tuple


def backproject_points(mapping: Dict[Tuple[int, int], List[int]], pixels: Iterable[Tuple[int, int]]) -> List[int]:
    """Collect point indices from a set of pixels.

    Parameters
    ----------
    mapping : dict
        Mapping from `(row, col)` pixel indices to lists of point indices
        as returned by the BEV or camera generators.
    pixels : iterable of (row, col)
        Pixel indices that have been assigned to a particular label.

    Returns
    -------
    list of int
        The union of point indices that project to the given pixels.
    """
    indices: List[int] = []
    for pix in pixels:
        pts = mapping.get(tuple(pix))
        if pts:
            indices.extend(pts)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique