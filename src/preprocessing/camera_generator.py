"""Virtual camera projection for LiDAR points.

This module defines a simple `CameraGenerator` that projects 3D points
onto a 2D image plane using a pinhole camera model.  The class
computes pixel coordinates from world coordinates given the camera
position, orientation (yaw) and focal length.  The output consists of
an image-sized mapping from pixels to point indices.  Intensity and
depth images can be generated similarly.

In a real system you would use the actual calibration parameters of
the MMS cameras, including intrinsic matrix, distortion coefficients
and extrinsic transforms from the LiDAR to the camera frame.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class CameraGenerator:
    """Project LiDAR points onto a virtual camera image plane."""

    image_width: int = 1024
    image_height: int = 768
    focal_length: float = 800.0
    """Focal length in pixels.  A larger value yields a narrower field of view."""

    sensor_yaw: float = 0.0
    """Yaw angle of the camera in radians relative to the world X axis."""

    def project(self, points: np.ndarray) -> Dict[Tuple[int, int], list]:
        """Project 3D points onto the image plane.

        A simple pinhole model is used: X and Y are rotated by the
        camera's yaw, X is treated as depth, and Y,Z are projected
        onto the image plane.  Points behind the camera or outside the
        image boundaries are discarded.  A mapping from pixel indices
        `(row, col)` to point indices is returned.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, 3) or (N, M) containing at least XYZ in
            columns 0–2.

        Returns
        -------
        dict
            Mapping from `(row, col)` pixel indices to lists of point
            indices.
        """
        coords = points[:, :3]
        # Rotate points around Z axis (yaw)
        c = np.cos(-self.sensor_yaw)
        s = np.sin(-self.sensor_yaw)
        rot = np.array([[c, -s], [s, c]])
        xy = coords[:, :2].dot(rot)
        z = coords[:, 2]
        x_rot = xy[:, 0]
        y_rot = xy[:, 1]
        # Use X as depth (distance along camera forward axis)
        depth = x_rot
        # Prevent division by zero or projecting points behind the camera
        valid = depth > 0
        mapping: Dict[Tuple[int, int], list] = {}
        if not np.any(valid):
            return mapping
        # Normalised image coordinates
        u = self.focal_length * (y_rot[valid] / depth[valid]) + self.image_width / 2
        v = self.focal_length * (z[valid] / depth[valid]) + self.image_height / 2
        cols = u.astype(int)
        rows = v.astype(int)
        # Collect mapping
        valid_indices = np.nonzero(valid)[0]
        for pix_r, pix_c, pt_idx in zip(rows, cols, valid_indices):
            if pix_r < 0 or pix_r >= self.image_height or pix_c < 0 or pix_c >= self.image_width:
                continue
            key = (pix_r, pix_c)
            if key in mapping:
                mapping[key].append(int(pt_idx))
            else:
                mapping[key] = [int(pt_idx)]
        return mapping