"""Extraction of manhole covers from clustered points.

A manhole cover is approximated as a circular disk on the road
surface.  This module fits a circle to the XY positions of a cluster
of points and returns the estimated centre and radius.  A simple
least squares circle fit is used.  In practice you should verify the
fit error and radius range to ensure the cluster truly represents a
manhole.
"""

from typing import Optional, Tuple

import numpy as np


def fit_circle(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a circle to 2D points using linear least squares.

    Returns the centre and radius of the best fitting circle.  See
    Kåsa's method for details.  If fewer than 3 points are provided,
    returns the centroid and zero radius.
    """
    if len(points) < 3:
        centre = points[:, :2].mean(axis=0)
        return centre, 0.0
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(points))]
    b = x**2 + y**2
    c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    centre = np.array([cx, cy])
    radius = np.sqrt(c0 + cx**2 + cy**2)
    return centre, radius


def extract_manholes(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Extract a manhole cover from a cluster of points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) containing XYZ coordinates of a cluster
        suspected to be a manhole.

    Returns
    -------
    (centre, radius) or None
        Tuple of centre (2D) and radius if a reasonable fit was found;
        otherwise `None`.
    """
    centre, radius = fit_circle(points[:, :2])
    # Simple sanity check on radius (manhole covers are roughly 0.3–1 m)
    if 0.1 < radius < 1.5:
        return centre, radius
    return None