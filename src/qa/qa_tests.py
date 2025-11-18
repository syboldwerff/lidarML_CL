"""Quality assurance tests for processed data and road models.

This module defines classes encapsulating QA routines for different
parts of the pipeline.  `PointCloudQA` operates on point cloud tiles
to verify densities and intensities.  `RoadModelQA` validates the
final road model to ensure lane continuity and reasonable geometry.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PointCloudQA:
    """Perform QA tests on a point cloud tile."""
    density_threshold: float = 50.0
    intensity_threshold: float = 1.0

    def run(self, points: np.ndarray, tile_size: float) -> Dict[str, bool]:
        area = tile_size * tile_size
        density = len(points) / area
        density_ok = density >= self.density_threshold
        if points.shape[1] > 4:
            mean_intensity = np.mean(points[:, 4])
        else:
            mean_intensity = 1.0
        intensity_ok = mean_intensity >= self.intensity_threshold
        return {
            "density_ok": density_ok,
            "intensity_ok": intensity_ok,
        }


@dataclass
class RoadModelQA:
    """Validate a reconstructed road model."""
    max_lane_overlap: float = 0.1
    """Maximum allowed overlap between lanes (fraction of length)."""

    def check_lane_overlap(self, lanes: List[np.ndarray]) -> bool:
        """Check that lanes do not overlap excessively.

        Lanes are represented as polylines in XY space.  This simplistic
        check computes pairwise distances between lane centre lines and
        ensures they are at least half a lane width apart on average.
        """
        if not lanes:
            return True
        n = len(lanes)
        min_sep_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Compute mean distance between lane i and j
                li = lanes[i]
                lj = lanes[j]
                # Interpolate shorter lane to same number of points
                m = min(len(li), len(lj))
                li_sample = li[np.linspace(0, len(li) - 1, m).astype(int)]
                lj_sample = lj[np.linspace(0, len(lj) - 1, m).astype(int)]
                dists = np.linalg.norm(li_sample - lj_sample, axis=1)
                min_sep_sum += np.mean(dists)
                count += 1
        if count == 0:
            return True
        avg_sep = min_sep_sum / count
        # Assume typical lane width of 3.5 m; lanes should be at least
        # ~3 m apart to avoid overlap.
        return avg_sep >= 3.0

    def run(self, lanes: List[np.ndarray]) -> Dict[str, bool]:
        return {
            "lane_separation_ok": self.check_lane_overlap(lanes),
        }