"""Machine learning pipelines for BEV and camera modalities.

This package contains subpackages for BEV semantic segmentation (`ml/bev`)
and camera object detection (`ml/cam`).  Each subpackage defines
dataset loaders, model architectures, training scripts and
inference scripts.  The implementations are intentionally minimal
and serve as a starting point; you are expected to modify them to
suit your datasets and research goals.
"""

from . import bev, cam

__all__ = ["bev", "cam"]