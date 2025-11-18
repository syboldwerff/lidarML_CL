"""Extraction of lighting poles or masts from clustered points.

A mast is approximated as a vertical cylinder.  We estimate the XY
centre of the cylinder as the mean of the points' XY coordinates,
compute the average radial distance as the cylinder radius, and take
the height as the range of Z values.  In practice you should use a
RANSAC cylinder model to handle outliers and partial observations.
"""

from typing import Optional, Tuple

import numpy as np


def extract_masts(points: np.ndarray) -> Optional[Tuple[np.ndarray, float, float]]:
    """Extract a mast (vertical cylinder) from a cluster of points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) with XYZ coordinates of mast points.

    Returns
    -------
    (centre_xy, radius, height) or None
        Returns the XY centre, cylinder radius and height if the
        cluster meets simple criteria; otherwise None.
    """
    if len(points) < 10:
        return None
    xy = points[:, :2]
    z = points[:, 2]
    centre = xy.mean(axis=0)
    # Estimate radius as RMS distance from centre
    r = np.sqrt(((xy - centre)**2).sum(axis=1)).mean()
    height = z.max() - z.min()
    # Simple heuristics: radius between 0.02 and 0.5 m, height > 2 m
    if 0.02 < r < 0.5 and height > 2.0:
        return centre, r, height
    return None