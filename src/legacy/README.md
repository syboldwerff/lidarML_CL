# Legacy preprocessing scripts

This folder contains the original standalone preprocessing scripts that were used before the modular `src/preprocessing/` pipeline was introduced. They are retained for reference and historical context but are not part of the current workflow.

## Scripts
- `preprocess_bev.py`: Generates BEV tiles and mappings from individual LAZ files.
- `preprocess_bev_batch.py`: Batch wrapper that calls `preprocess_bev.py` for multiple inputs.
- `preprocess_cam.py`: Produces virtual camera views from one LAZ file and road axis.
- `preprocess_cam_batch.py`: Batch runner for camera preprocessing.
- `preprocess_cam_axis_multi.py`: Variant that handles multiple LAZ files with axis data.

For production use, prefer the modular components under `src/preprocessing/` (e.g., `BEVGenerator`, `CameraGenerator`, tiling, filtering, and QC utilities). If you need functionality from these legacy scripts, consider porting it into the modular pipeline instead of invoking them directly.
