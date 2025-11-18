"""Lane reconstruction from road edge and centre line geometry.

This module defines helper functions to construct lane polygons from
detected road boundaries and centre lines.  A lane polygon is
represented as a sequence of vertices in XY space.  The simplest
lane reconstruction is to offset the centre line by half the lane
width on each side.  More sophisticated approaches use detected
markings as inputs.
"""

from typing import List, Tuple

import numpy as np


def construct_lanes(center_line: np.ndarray, lane_width: float, num_lanes: int = 2) -> List[np.ndarray]:
    """Construct lane polygons from a centre line.

    Parameters
    ----------
    center_line : numpy.ndarray
        Array of shape (N, 2) containing XY coordinates of the road
        centre line sampled at regular intervals.
    lane_width : float
        Width of a single lane in metres.
    num_lanes : int
        Number of lanes to construct (assumes symmetrical lanes).

    Returns
    -------
    list of numpy.ndarray
        List of lane polygons, each an array of shape (N, 2) giving
        the lane centre line; polygons are left as simple line strips
        rather than closed polygons.  Use this as a starting point
        before constructing actual polygons.
    """
    lanes: List[np.ndarray] = []
    # Compute tangent vector along the centre line using finite differences
    diffs = np.gradient(center_line, axis=0)
    # Normal vectors (rotated 90 degrees)
    normals = np.column_stack([-diffs[:, 1], diffs[:, 0]])
    # Normalise
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)
    # Construct lane centre lines by offsetting the road centre
    for i in range(num_lanes):
        offset = (i - (num_lanes - 1) / 2) * lane_width
        lane = center_line + normals * offset
        lanes.append(lane)
    return lanes