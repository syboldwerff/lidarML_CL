"""Fusion of 3D point sets from different modalities.

When objects are detected independently in BEV and camera modalities,
their corresponding point sets can be combined to increase confidence
and to ensure all relevant points are collected.  This module
provides a simple union operation.  More sophisticated approaches
might perform intersection or incorporate temporal continuity.
"""

from typing import Iterable, List, Set


def fuse_point_sets(*point_sets: Iterable[int]) -> List[int]:
    """Compute the union of multiple sets of point indices.

    Parameters
    ----------
    *point_sets : iterable of iterables of int
        The point index collections to union.

    Returns
    -------
    list of int
        Unique point indices from the union, sorted in ascending order.
    """
    union: Set[int] = set()
    for pts in point_sets:
        union.update(int(p) for p in pts)
    return sorted(union)