"""Compute elevation profiles across the road surface.

Elevation profiles describe the height of the road surface as a
function of distance along the road.  This module provides a simple
function to compute the median elevation of points within cross
sections orthogonal to a reference alignment at regular intervals.
In a full implementation you would account for alignment curvature
and road crossfall.
"""

from typing import Dict, Tuple

import numpy as np


def compute_profiles(points: np.ndarray, alignment: np.ndarray, step: float = 0.5) -> Dict[float, float]:
    """Compute road elevation profiles.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) containing XYZ coordinates of ground
        points.
    alignment : numpy.ndarray
        Array of shape (M, 2) containing XY coordinates of the road
        alignment sampled at regular distance along its length.
    step : float
        Sampling interval along the alignment in metres.

    Returns
    -------
    dict
        Mapping from distance along the alignment (m) to median
        elevation above an arbitrary datum.
    """
    profile: Dict[float, float] = {}
    # Precompute cumulative distances along alignment
    diffs = np.diff(alignment, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s_cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = s_cumulative[-1]
    distances = np.arange(0, total_length + step, step)
    # For each sample distance find nearest alignment point
    for s in distances:
        idx = np.searchsorted(s_cumulative, s, side='right') - 1
        if idx >= len(alignment) - 1:
            idx = len(alignment) - 2
        # Position along the segment
        seg_start = alignment[idx]
        seg_end = alignment[idx + 1]
        t = (s - s_cumulative[idx]) / max(seg_lengths[idx], 1e-6)
        point_on_alignment = seg_start + t * (seg_end - seg_start)
        # Compute cross section plane: use segment normal in XY plane
        seg_dir = seg_end - seg_start
        normal = np.array([-seg_dir[1], seg_dir[0]])
        normal /= np.linalg.norm(normal) + 1e-12
        # Select ground points within a corridor of width 5 m around alignment
        xy = points[:, :2]
        rel = xy - point_on_alignment
        # Distance along normal axis
        dist_normal = rel.dot(normal)
        mask = np.abs(dist_normal) < 2.5
        z = points[mask, 2]
        if len(z) > 0:
            profile[s] = float(np.median(z))
        else:
            profile[s] = float('nan')
    return profile