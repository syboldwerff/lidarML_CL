"""Extraction of guardrails from clustered points.

A guardrail is approximated as a sequence of posts connected by a
continuous line.  This module fits a principal axis through the
cluster (using PCA in the XY plane) and orders the points along that
axis to produce a polyline.  The Z coordinate is preserved to
represent the rail height variation.  A simple heuristic filters
clusters that are too short or too small.
"""

from typing import List

import numpy as np


def extract_guardrails(points: np.ndarray) -> List[np.ndarray]:
    """Extract a guardrail polyline from a cluster of points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) containing XYZ coordinates.

    Returns
    -------
    list of numpy.ndarray
        A list containing one 3D polyline if the cluster meets
        heuristics; otherwise an empty list.
    """
    if len(points) < 10:
        return []
    xyz = points[:, :3]
    mean_xy = xyz[:, :2].mean(axis=0)
    centred = xyz[:, :2] - mean_xy
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    direction = eigvecs[:, idx]
    # Project onto principal axis for ordering
    t = centred.dot(direction)
    order = np.argsort(t)
    polyline = xyz[order]
    # Heuristic: guardrails longer than 2 m
    length = np.linalg.norm(polyline[-1, :2] - polyline[0, :2])
    if length < 2.0:
        return []
    return [polyline]