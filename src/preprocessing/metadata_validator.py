"""Metadata validation and CRS checking for LiDAR point clouds.

This module provides the `MetadataValidator` class which performs quality
checks on input LiDAR data to ensure it meets the expected standards for
preprocessing.  It verifies coordinate reference systems (CRS), checks for
the presence of required attributes (RGB, intensity), computes bounding
boxes, and generates a standardised metadata JSON file.

The validator is designed to work with vendor-corrected mobile mapping
data that should already be in RD New + NAP coordinates with RGB and
intensity channels.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np


@dataclass
class ScanMetadata:
    """Structured metadata for a LiDAR scan."""

    crs_ok: bool
    """Whether the CRS is verified as RD New + NAP."""

    has_rgb: bool
    """Whether RGB channels are present."""

    has_intensity: bool
    """Whether intensity channel is present."""

    point_count: int
    """Total number of points in the scan."""

    bbox: Tuple[float, float, float, float, float, float]
    """Bounding box as (x_min, y_min, z_min, x_max, y_max, z_max)."""

    provider: str
    """Data provider or source identifier."""

    crs_epsg: Optional[int] = None
    """EPSG code of the coordinate system if available."""

    rgb_range: Optional[Tuple[int, int]] = None
    """Min and max RGB values across all channels."""

    intensity_range: Optional[Tuple[float, float]] = None
    """Min and max intensity values."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save metadata to JSON file.

        Parameters
        ----------
        path : Path
            Output path for meta_scan.json file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class MetadataValidator:
    """Validate LiDAR point cloud metadata and generate quality flags.

    This validator checks whether input data meets the expected standards:
    - CRS should be RD New (EPSG:28992) + NAP for Z
    - RGB channels should be present (columns 6, 7, 8) in range [0, 255]
    - Intensity should be present (column 4) in reasonable range
    - Bounding box should be within expected Dutch coordinates
    """

    expected_epsg: int = 28992
    """Expected EPSG code for RD New."""

    rd_x_range: Tuple[float, float] = (-7000.0, 300000.0)
    """Valid range for RD X coordinates (metres)."""

    rd_y_range: Tuple[float, float] = (289000.0, 629000.0)
    """Valid range for RD Y coordinates (metres)."""

    nap_z_range: Tuple[float, float] = (-20.0, 400.0)
    """Valid range for NAP Z coordinates (metres)."""

    provider: str = "unknown"
    """Default provider name."""

    def validate_crs(
        self,
        points: np.ndarray,
        epsg: Optional[int] = None
    ) -> bool:
        """Check if coordinates fall within expected RD New + NAP ranges.

        Parameters
        ----------
        points : numpy.ndarray
            Array with XYZ in columns 0-2.
        epsg : int, optional
            EPSG code to verify.  If provided, it is checked against
            the expected EPSG.

        Returns
        -------
        bool
            True if CRS appears to be RD New + NAP.
        """
        if epsg is not None and epsg != self.expected_epsg:
            return False

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Check if coordinates fall within RD New + NAP ranges
        x_ok = (x.min() >= self.rd_x_range[0] and
                x.max() <= self.rd_x_range[1])
        y_ok = (y.min() >= self.rd_y_range[0] and
                y.max() <= self.rd_y_range[1])
        z_ok = (z.min() >= self.nap_z_range[0] and
                z.max() <= self.nap_z_range[1])

        return x_ok and y_ok and z_ok

    def check_rgb(self, points: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Check for presence of RGB channels.

        Parameters
        ----------
        points : numpy.ndarray
            Array that should have RGB in columns 5, 6, 7 (or 6, 7, 8 if
            intensity is in column 4).

        Returns
        -------
        (bool, tuple or None)
            Tuple of (has_rgb, (min_val, max_val)).  If RGB is not
            present, returns (False, None).
        """
        # Assume RGB in columns 5, 6, 7 if point has >= 8 columns
        # (X, Y, Z, range, intensity, R, G, B)
        if points.shape[1] < 8:
            return False, None

        r = points[:, 5]
        g = points[:, 6]
        b = points[:, 7]

        # Check if RGB values are in expected range [0, 255]
        rgb_min = int(min(r.min(), g.min(), b.min()))
        rgb_max = int(max(r.max(), g.max(), b.max()))

        # RGB should be in range [0, 255]
        has_rgb = (rgb_min >= 0 and rgb_max <= 255)

        return has_rgb, (rgb_min, rgb_max)

    def check_intensity(self, points: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Check for presence of intensity channel.

        Parameters
        ----------
        points : numpy.ndarray
            Array that should have intensity in column 4.

        Returns
        -------
        (bool, tuple or None)
            Tuple of (has_intensity, (min_val, max_val)).  If intensity
            is not present, returns (False, None).
        """
        if points.shape[1] < 5:
            return False, None

        intensity = points[:, 4]
        int_min = float(intensity.min())
        int_max = float(intensity.max())

        # Intensity should be positive
        has_intensity = (int_min >= 0.0 and int_max > 0.0)

        return has_intensity, (int_min, int_max)

    def compute_bbox(self, points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Compute 3D bounding box.

        Parameters
        ----------
        points : numpy.ndarray
            Array with XYZ in columns 0-2.

        Returns
        -------
        tuple
            Bounding box as (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        return (
            float(x.min()),
            float(y.min()),
            float(z.min()),
            float(x.max()),
            float(y.max()),
            float(z.max())
        )

    def validate(
        self,
        points: np.ndarray,
        epsg: Optional[int] = None
    ) -> ScanMetadata:
        """Run all validation checks and generate metadata.

        Parameters
        ----------
        points : numpy.ndarray
            Point cloud array with at least XYZ in columns 0-2.
        epsg : int, optional
            EPSG code if known.

        Returns
        -------
        ScanMetadata
            Structured metadata with validation results.
        """
        crs_ok = self.validate_crs(points, epsg)
        has_rgb, rgb_range = self.check_rgb(points)
        has_intensity, intensity_range = self.check_intensity(points)
        bbox = self.compute_bbox(points)
        point_count = len(points)

        return ScanMetadata(
            crs_ok=crs_ok,
            has_rgb=has_rgb,
            has_intensity=has_intensity,
            point_count=point_count,
            bbox=bbox,
            provider=self.provider,
            crs_epsg=epsg if epsg else (self.expected_epsg if crs_ok else None),
            rgb_range=rgb_range,
            intensity_range=intensity_range
        )
