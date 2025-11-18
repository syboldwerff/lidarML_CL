"""Trajectory smoothing and alignment.

This module implements a very simple trajectory correction scheme for
mobile mapping systems.  A trajectory is represented as a sequence of
3D positions (x, y, z) sampled over time.  The `TrajectoryCorrector`
class applies a moving average filter to smooth the trajectory and
optionally applies a constant offset to correct for boresight or
lever‑arm errors.

In a production system you might replace this implementation with a
Kalman or Rauch–Tung–Striebel smoother that uses the sensor's IMU and
GNSS error models.  The moving average filter used here is intended
only as a placeholder to demonstrate the expected interface.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class TrajectoryCorrector:
    """Apply simple smoothing and boresight correction to a trajectory."""

    window: int = 11
    """Window length for the moving average filter.  Should be an odd
    integer to make the filter symmetric.  Larger windows produce
    smoother trajectories but may oversmooth sharp manoeuvres."""

    def smooth(self, positions: Iterable[Tuple[float, float, float]]) -> np.ndarray:
        """Smooth a sequence of 3D positions using a moving average.

        Parameters
        ----------
        positions : iterable of (x, y, z)
            The raw trajectory samples.

        Returns
        -------
        numpy.ndarray
            Smoothed trajectory of shape (N, 3).
        """
        coords = np.asarray(positions, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("positions must be an iterable of 3D coordinates")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        # Pad the sequence at both ends to handle boundary effects.
        pad = self.window // 2
        padded = np.pad(coords, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(self.window) / self.window
        # Convolve each coordinate separately.
        smoothed = np.vstack([
            np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)
        ]).T
        return smoothed

    def apply_offset(self, positions: Iterable[Tuple[float, float, float]], offsets: Tuple[float, float, float]) -> np.ndarray:
        """Apply a fixed lever‑arm or boresight offset to the trajectory.

        The offset is added to each point in the trajectory.  In
        practise you might estimate this offset from calibration data
        (e.g. boresight calibration between LiDAR and GNSS/IMU frame).

        Parameters
        ----------
        positions : iterable of (x, y, z)
            The input trajectory samples.
        offsets : tuple of (dx, dy, dz)
            The offset to apply.

        Returns
        -------
        numpy.ndarray
            Offset corrected trajectory.
        """
        coords = np.asarray(positions, dtype=float)
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (3,):
            raise ValueError("offsets must be a 3‑tuple")
        return coords + offsets