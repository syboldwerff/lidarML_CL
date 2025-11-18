"""Intensity and RGB normalisation with material preclassification.

Raw LiDAR intensity values depend on both surface reflectivity and
sensor geometry (e.g. range, incidence angle).  RGB values may vary
due to lighting conditions and sensor calibration.  This module
provides normalisation for both intensity and RGB channels.

For intensity, we compensate for range falloff and perform material
preclassification.  For RGB, we clamp values to the valid range [0, 255]
and optionally perform white balance correction.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class IntensityNormalizer:
    """Normalise LiDAR intensities, RGB values and preclassify materials."""

    reference_range: float = 50.0
    """Reference range (metres) used for range compensation.  Intensities
    are scaled by `(range / reference_range) ** 2`.
    """

    n_clusters: int = 3
    """Number of material classes for k‑means preclassification."""

    intensity_min: float = 1.0
    """Minimum valid intensity value after compensation."""

    intensity_max: float = 255.0
    """Maximum valid intensity value after compensation."""

    rgb_min: int = 0
    """Minimum valid RGB value."""

    rgb_max: int = 255
    """Maximum valid RGB value."""

    apply_white_balance: bool = False
    """Whether to apply simple white balance correction to RGB."""

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

    def clamp_intensity(self, intensities: np.ndarray) -> np.ndarray:
        """Clamp intensity values to valid range.

        Parameters
        ----------
        intensities : numpy.ndarray
            Intensity values to clamp.

        Returns
        -------
        numpy.ndarray
            Clamped intensities in range [intensity_min, intensity_max].
        """
        return np.clip(intensities, self.intensity_min, self.intensity_max)

    def normalise_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Normalise RGB values to valid range [0, 255].

        Parameters
        ----------
        rgb : numpy.ndarray
            Array of shape (N, 3) with RGB values.

        Returns
        -------
        numpy.ndarray
            Normalised RGB values.
        """
        rgb_norm = np.clip(rgb, self.rgb_min, self.rgb_max)

        if self.apply_white_balance:
            # Simple grey world white balance
            r_mean = np.mean(rgb_norm[:, 0])
            g_mean = np.mean(rgb_norm[:, 1])
            b_mean = np.mean(rgb_norm[:, 2])
            grey = (r_mean + g_mean + b_mean) / 3.0

            if r_mean > 0 and g_mean > 0 and b_mean > 0:
                rgb_norm[:, 0] = rgb_norm[:, 0] * (grey / r_mean)
                rgb_norm[:, 1] = rgb_norm[:, 1] * (grey / g_mean)
                rgb_norm[:, 2] = rgb_norm[:, 2] * (grey / b_mean)
                rgb_norm = np.clip(rgb_norm, self.rgb_min, self.rgb_max)

        return rgb_norm

    def check_marking_contrast(
        self,
        intensities: np.ndarray,
        threshold_bright: float = 200.0,
        threshold_dark: float = 50.0
    ) -> Tuple[bool, float]:
        """Check if marking appears bright and asphalt dark in intensity.

        This is a QA check to ensure that road markings (which should
        have high retroreflectivity) are distinguishable from asphalt
        (which should be dark).

        Parameters
        ----------
        intensities : numpy.ndarray
            Compensated intensity values.
        threshold_bright : float, optional
            Intensity threshold for bright markings.
        threshold_dark : float, optional
            Intensity threshold for dark asphalt.

        Returns
        -------
        (bool, float)
            Tuple of (contrast_ok, contrast_ratio).
        """
        # Compute the ratio of bright to dark pixels
        n_bright = np.sum(intensities > threshold_bright)
        n_dark = np.sum(intensities < threshold_dark)

        if n_dark == 0:
            # If no dark pixels, contrast check fails
            return False, 0.0

        contrast_ratio = n_bright / max(n_dark, 1)

        # Expect at least some bright pixels (markings) relative to dark (asphalt)
        contrast_ok = n_bright > 0 and contrast_ratio > 0.01

        return contrast_ok, float(contrast_ratio)

    def normalise(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise intensities, RGB and return material labels.

        Assumes that column 3 of `points` contains range values,
        column 4 contains raw intensity values, and columns 5-7 contain
        RGB values.  Additional columns are preserved in the output array.
        The returned tuple contains the updated points array and an array
        of cluster labels.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with range in col 3, intensity in
            col 4, and optionally RGB in cols 5-7.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple of (updated points, material labels).
        """
        pts = points.copy()

        # If no intensity data present, skip normalisation
        if pts.shape[1] < 5:
            return pts, np.zeros(len(pts), dtype=int)

        distances = pts[:, 3]
        raw_int = pts[:, 4]
        comp_int = self.compensate_range(distances, raw_int)
        comp_int = self.clamp_intensity(comp_int)

        # Normalise RGB if present
        if pts.shape[1] >= 8:
            rgb = pts[:, 5:8]
            rgb_norm = self.normalise_rgb(rgb)
            pts[:, 5:8] = rgb_norm

        # Roughness proxy: local absolute deviation of Z from mean.
        z = pts[:, 2]
        z_mean = np.mean(z)
        roughness = np.abs(z - z_mean)
        labels = self.preclassify(comp_int, roughness)
        pts[:, 4] = comp_int
        return pts, labels