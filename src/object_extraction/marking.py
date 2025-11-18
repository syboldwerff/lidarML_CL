"""Extraction of road markings from clustered points.

Road markings such as lane lines and arrows are typically thin,
elongated objects on the ground.  This module provides a simple
function to extract polylines from a cluster of points representing a
marking.  A principal component analysis (PCA) is used to estimate
the predominant orientation of the points.  Points are then
projected onto this axis and sorted to form a polyline.  In practice
you may need to split the cluster into multiple segments if sharp
turns or branches are present.
"""

from typing import List, Tuple

import numpy as np


def extract_markings(points: np.ndarray) -> List[np.ndarray]:
    """Fit a polyline through a cluster of marking points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) containing XYZ coordinates of marking
        points.

    Returns
    -------
    list of numpy.ndarray
        A list containing a single 2D polyline (in XY space) sorted
        along its length.  If the cluster is degenerate or too small,
        an empty list is returned.
    """
    if len(points) < 2:
        return []
    # Use PCA to find dominant direction in XY plane
    xy = points[:, :2]
    mean_xy = xy.mean(axis=0)
    centred = xy - mean_xy
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    # Choose eigenvector with largest eigenvalue
    idx = np.argmax(eigvals)
    direction = eigvecs[:, idx]
    # Project points onto direction
    t = centred.dot(direction)
    order = np.argsort(t)
    polyline = xy[order]
    return [polyline]