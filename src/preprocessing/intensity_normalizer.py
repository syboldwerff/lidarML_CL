"""Intensity normalisation with material preclassification.

Raw LiDAR intensity values depend on both surface reflectivity and
sensor geometry (e.g. range, incidence angle).  To compare
intensities across scenes and trajectories, we first compensate for
range falloff and then perform a simple material preclassification to
identify materials with similar reflectance.  A more robust approach
could employ sensor calibration tables and supervised material
classification.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class IntensityNormalizer:
    """Normalise LiDAR intensities and preclassify materials."""

    reference_range: float = 50.0
    """Reference range (metres) used for range compensation.  Intensities
    are scaled by `(range / reference_range) ** 2`.
    """

    n_clusters: int = 3
    """Number of material classes for k‑means preclassification."""

    def compensate_range(self, distances: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Apply inverse square law range compensation.

        Parameters
        ----------
        distances : numpy.ndarray
            Range values (metres) for each LiDAR return.
        intensities : numpy.ndarray
            Raw intensity values for each return.

        Returns
        -------
        numpy.ndarray
            Range compensated intensities.
        """
        ratio = np.maximum(distances, 1e-3) / self.reference_range
        compensated = intensities * ratio**2
        return compensated

    def preclassify(self, intensities: np.ndarray, roughness: np.ndarray) -> np.ndarray:
        """Cluster points into material classes using k‑means.

        Parameters
        ----------
        intensities : numpy.ndarray
            Compensated intensities.
        roughness : numpy.ndarray
            A measure of local surface roughness.  In this simplified
            implementation we use the absolute deviation of Z from its
            local mean as a proxy for roughness.

        Returns
        -------
        numpy.ndarray
            Array of cluster labels of the same length as inputs.
        """
        features = np.column_stack((intensities, roughness))
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=0)
        labels = kmeans.fit_predict(features)
        return labels

    def normalise(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise intensities and return material labels.

        Assumes that column 3 of `points` contains range values and
        column 4 contains raw intensity values.  Additional columns are
        preserved in the output array.  The returned tuple contains
        the updated points array and an array of cluster labels.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with range in col 3 and intensity in
            col 4.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple of (updated points, material labels).
        """
        pts = points.copy()
        distances = pts[:, 3]
        raw_int = pts[:, 4]
        comp_int = self.compensate_range(distances, raw_int)
        # Roughness proxy: local absolute deviation of Z from mean.
        z = pts[:, 2]
        z_mean = np.mean(z)
        roughness = np.abs(z - z_mean)
        labels = self.preclassify(comp_int, roughness)
        pts[:, 4] = comp_int
        return pts, labels