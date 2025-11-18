"""Quality assurance engine for preprocessing.

The `QAEngine` collects a set of tests that verify whether a tile or
view meets certain quality standards.  Each test returns a boolean
value indicating pass or fail.  The engine then aggregates these
results into a dictionary of QA flags.  In a real system you might
record additional metadata or use more sophisticated scoring.

This module now supports comprehensive QA checks including CRS
validation, ground segmentation quality, RGB quality, intensity
quality and density checks.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class QAEngine:
    """Run quality tests on tiles or views."""

    density_threshold: float = 50.0
    """Minimum acceptable point density (points per square metre)."""

    intensity_threshold: float = 1.0
    """Minimum mean intensity value."""

    ground_fraction_min: float = 0.05
    """Minimum fraction of ground points."""

    ground_fraction_max: float = 0.95
    """Maximum fraction of ground points."""

    rgb_min: int = 0
    """Expected minimum RGB value."""

    rgb_max: int = 255
    """Expected maximum RGB value."""

    # RD New + NAP coordinate ranges
    rd_x_range: Tuple[float, float] = (-7000.0, 300000.0)
    rd_y_range: Tuple[float, float] = (289000.0, 629000.0)
    nap_z_range: Tuple[float, float] = (-20.0, 400.0)

    def test_density(self, tile_points: np.ndarray, tile_size: float) -> bool:
        """Check whether point density exceeds threshold.

        Parameters
        ----------
        tile_points : numpy.ndarray
            Points within a tile.
        tile_size : float
            Side length of the tile in metres.

        Returns
        -------
        bool
            True if density is acceptable.
        """
        area = tile_size * tile_size
        density = len(tile_points) / area
        return density >= self.density_threshold

    def test_intensity(self, intensities: np.ndarray) -> bool:
        """Check if mean intensity is above threshold.

        Parameters
        ----------
        intensities : numpy.ndarray
            Intensity values.

        Returns
        -------
        bool
            True if mean intensity is acceptable.
        """
        if len(intensities) == 0:
            return False
        return np.mean(intensities) >= self.intensity_threshold

    def test_ground_fraction(self, is_ground: np.ndarray) -> bool:
        """Check if ground point fraction is within acceptable range.

        Parameters
        ----------
        is_ground : numpy.ndarray
            Boolean array indicating ground points.

        Returns
        -------
        bool
            True if ground fraction is acceptable.
        """
        if len(is_ground) == 0:
            return False

        fraction = np.sum(is_ground) / len(is_ground)
        return self.ground_fraction_min <= fraction <= self.ground_fraction_max

    def test_rgb(self, rgb: np.ndarray) -> bool:
        """Check if RGB values are within expected range [0, 255].

        Parameters
        ----------
        rgb : numpy.ndarray
            RGB array of shape (N, 3).

        Returns
        -------
        bool
            True if RGB values are valid.
        """
        if rgb.size == 0:
            return False

        rgb_min_actual = rgb.min()
        rgb_max_actual = rgb.max()

        return (rgb_min_actual >= self.rgb_min and
                rgb_max_actual <= self.rgb_max)

    def test_crs(self, tile_points: np.ndarray) -> bool:
        """Check if coordinates fall within expected RD New + NAP ranges.

        Parameters
        ----------
        tile_points : numpy.ndarray
            Points with XYZ in columns 0-2.

        Returns
        -------
        bool
            True if coordinates appear to be in RD New + NAP.
        """
        if len(tile_points) == 0:
            return False

        x = tile_points[:, 0]
        y = tile_points[:, 1]
        z = tile_points[:, 2]

        x_ok = (x.min() >= self.rd_x_range[0] and
                x.max() <= self.rd_x_range[1])
        y_ok = (y.min() >= self.rd_y_range[0] and
                y.max() <= self.rd_y_range[1])
        z_ok = (z.min() >= self.nap_z_range[0] and
                z.max() <= self.nap_z_range[1])

        return x_ok and y_ok and z_ok

    def run(
        self,
        tile_points: np.ndarray,
        tile_size: float,
        is_ground: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """Run all QA tests on a tile.

        Parameters
        ----------
        tile_points : numpy.ndarray
            Array of points within a tile.  Expected columns:
            0-2: XYZ
            3: range (optional)
            4: intensity (optional)
            5-7: RGB (optional)
            8: is_ground (optional, or pass via is_ground parameter)
        tile_size : float
            Side length of the tile in metres.
        is_ground : numpy.ndarray, optional
            Boolean array indicating ground points.  If not provided,
            the method will look for an is_ground column in tile_points.

        Returns
        -------
        dict
            Mapping from test names to boolean pass/fail values.
        """
        results = {}

        # CRS check
        results["crs_ok"] = self.test_crs(tile_points)

        # Density check
        results["density_ok"] = self.test_density(tile_points, tile_size)

        # Intensity check
        if tile_points.shape[1] > 4:
            intensities = tile_points[:, 4]
            results["intensity_ok"] = self.test_intensity(intensities)
        else:
            results["intensity_ok"] = True  # No intensity data to check

        # RGB check
        if tile_points.shape[1] >= 8:
            rgb = tile_points[:, 5:8]
            results["rgb_ok"] = self.test_rgb(rgb)
        else:
            results["rgb_ok"] = True  # No RGB data to check

        # Ground check
        if is_ground is not None:
            results["ground_ok"] = self.test_ground_fraction(is_ground)
        elif tile_points.shape[1] > 8:
            # Try to extract is_ground from column 8
            is_ground_col = tile_points[:, 8].astype(bool)
            results["ground_ok"] = self.test_ground_fraction(is_ground_col)
        else:
            results["ground_ok"] = True  # No ground data to check

        return results
