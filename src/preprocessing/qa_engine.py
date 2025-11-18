"""Quality assurance engine for preprocessing.

The `QAEngine` collects a set of tests that verify whether a tile or
view meets certain quality standards.  Each test returns a boolean
value indicating pass or fail.  The engine then aggregates these
results into a dictionary of QA flags.  In a real system you might
record additional metadata or use more sophisticated scoring.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class QAEngine:
    """Run quality tests on tiles or views."""

    density_threshold: float = 50.0
    intensity_threshold: float = 1.0

    def test_density(self, tile_points: np.ndarray, tile_size: float) -> bool:
        """Check whether point density exceeds threshold."""
        area = tile_size * tile_size
        density = len(tile_points) / area
        return density >= self.density_threshold

    def test_intensity(self, intensities: np.ndarray) -> bool:
        """Check if mean intensity is above threshold."""
        return np.mean(intensities) >= self.intensity_threshold

    def run(self, tile_points: np.ndarray, tile_size: float) -> Dict[str, bool]:
        """Run all QA tests on a tile.

        Parameters
        ----------
        tile_points : numpy.ndarray
            Array of points within a tile.
        tile_size : float
            Side length of the tile in metres.

        Returns
        -------
        dict
            Mapping from test names to boolean pass/fail values.
        """
        z = tile_points[:, 2]
        intensities = tile_points[:, 4] if tile_points.shape[1] > 4 else np.array([])
        return {
            "density_ok": self.test_density(tile_points, tile_size),
            "intensity_ok": self.test_intensity(intensities) if intensities.size else True,
        }