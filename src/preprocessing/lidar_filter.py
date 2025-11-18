"""LiDAR filtering utilities.

This module defines a set of functions to clean raw LiDAR point
clouds.  These filters remove points that fall outside a desired
range, have low return intensity, or exhibit spurious height spikes.
They also optionally resample the point cloud to achieve a more
uniform density.

The point cloud is assumed to be stored in an `(N, M)` array, where
the first three columns hold the XYZ coordinates and a fourth column
contains return intensity or range.  Additional columns (e.g. point
index) are preserved by the filters.
"""

from typing import Tuple

import numpy as np

def filter_range(points: np.ndarray, max_range: float) -> np.ndarray:
    """Remove points beyond a maximum range.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, M) with XYZ and possibly intensity in the
        first columns.  A fourth column is assumed to be range if
        present.
    max_range : float
        Points with a range greater than this value will be removed.

    Returns
    -------
    numpy.ndarray
        Filtered array of points.
    """
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("points must have at least four columns (x,y,z,range)")
    mask = points[:, 3] <= max_range
    return points[mask]


def filter_intensity_low(points: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Remove points with return intensity below a threshold.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, M) containing at least five columns; the
        fifth column (index 4) is assumed to be intensity.
    threshold : float, optional
        Minimum intensity value to keep a point.  Default is 0.0
        (remove only negative intensity values).

    Returns
    -------
    numpy.ndarray
        Filtered array of points.
    """
    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError("points must have at least five columns (x,y,z,range,intensity)")
    mask = points[:, 4] >= threshold
    return points[mask]


def spike_filter_z(points: np.ndarray, kernel: int = 5, threshold: float = 1.0) -> np.ndarray:
    """Remove isolated height spikes.

    A simple median filter is applied to the Z coordinate.  Points whose
    Z deviates from the median of their neighbourhood by more than
    `threshold` metres are discarded.  This is a naive implementation
    which assumes the point cloud is evenly sampled; for a real
    pipeline you should use a 3D k‑nearest neighbour filter.

    Parameters
    ----------
    points : numpy.ndarray
        Array with XYZ in columns 0–2.
    kernel : int, optional
        Size of the sliding window used to compute the median.  Must
        be odd.  Default is 5.
    threshold : float, optional
        Height difference (metres) above which a point is considered a
        spike.  Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Point cloud with spikes removed.
    """
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    z = points[:, 2]
    pad = kernel // 2
    padded = np.pad(z, (pad, pad), mode="edge")
    medians = np.array([
        np.median(padded[i:i + kernel]) for i in range(len(z))
    ])
    mask = np.abs(z - medians) <= threshold
    return points[mask]


def uniformize_density(points: np.ndarray, target_ppm: float = 100.0) -> np.ndarray:
    """Downsample the point cloud to approximate a uniform density.

    The algorithm estimates a sampling probability such that the
    resulting point density (points per square metre) approaches
    `target_ppm`.  A very rough estimation of the bounding box area is
    used to compute the initial density.  In practice, density should
    be computed locally and adjusted based on scanning geometry.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, M) with XYZ in columns 0–2.
    target_ppm : float, optional
        Desired average number of points per square metre.  Default is
        100.

    Returns
    -------
    numpy.ndarray
        Downsampled point cloud.
    """
    coords = points[:, :3]
    # Estimate current density using bounding box area
    x_min, y_min = coords[:, 0:2].min(axis=0)
    x_max, y_max = coords[:, 0:2].max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    if area <= 0:
        return points
    current_density = len(points) / area
    keep_prob = min(1.0, target_ppm / max(current_density, 1e-6))
    mask = np.random.rand(len(points)) < keep_prob
    return points[mask]