"""Preprocessing package.

This package contains modules that prepare raw mobile mapping data for
downstream analysis.  It includes trajectory smoothing, LiDAR
filtering, ground and intensity normalisation, tiling, BEV and
camera view generation, and quality assurance checks.  Each module
exposes one or more classes or functions that perform a specific
preprocessing step.
"""

from .trajectory_corrector import TrajectoryCorrector
from .lidar_filter import filter_range, filter_intensity_low, spike_filter_z, uniformize_density
from .ground_normalizer import GroundNormalizer
from .intensity_normalizer import IntensityNormalizer
from .tiler import Tiler
from .qa_engine import QAEngine
from .bev_generator import BEVGenerator
from .camera_generator import CameraGenerator

__all__ = [
    "TrajectoryCorrector",
    "filter_range",
    "filter_intensity_low",
    "spike_filter_z",
    "uniformize_density",
    "GroundNormalizer",
    "IntensityNormalizer",
    "Tiler",
    "QAEngine",
    "BEVGenerator",
    "CameraGenerator",
]