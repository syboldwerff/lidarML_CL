"""Extraction of traffic signs from clustered points.

Traffic signs are approximated as flat planar surfaces mounted on
vertical posts.  We estimate the sign plane by computing the normal
vector using PCA, then derive the bounding rectangle in the plane.
Height above ground is taken from the minimum Z value in the cluster.
This is a rudimentary implementation; more robust plane fitting and
rectangle extraction should be used for production systems.
"""

from typing import Optional, Tuple

import numpy as np


def extract_signs(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Extract a traffic sign from a cluster of points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) of sign points.

    Returns
    -------
    (centre, normal, height) or None
        Returns the centre of the sign plane, the normal vector and
        height above ground.  Returns None if the cluster is too
        small.
    """
    if len(points) < 5:
        return None
    xyz = points[:, :3]
    centre = xyz.mean(axis=0)
    # Compute covariance matrix and eigenvectors
    cov = np.cov((xyz - centre).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    # The normal vector corresponds to the smallest eigenvalue
    idx = np.argmin(eigvals)
    normal = eigvecs[:, idx]
    # Height above ground: min Z coordinate
    height = xyz[:, 2].min()
    return centre, normal / np.linalg.norm(normal), height