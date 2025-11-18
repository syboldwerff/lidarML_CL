"""Tile point clouds into spatially indexed chunks.

This module partitions a point cloud into a regular grid of square
tiles.  For each tile we record its bounding box, basic QA metrics
such as point count and density, and optionally compute roll/pitch
stability using trajectory data.  Tiles are saved as separate
structures to facilitate parallel processing in downstream steps
(e.g. BEV generation and machine learning).

Tiles can be exported to Parquet format for efficient storage and
later retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Tile:
    """Represents a square tile of a point cloud."""
    idx: Tuple[int, int]
    bbox: Tuple[float, float, float, float]
    points: np.ndarray
    qa_flags: Dict[str, bool]

    def to_metadata_dict(self) -> Dict:
        """Convert tile metadata to dictionary (without points array).

        Returns
        -------
        dict
            Metadata dictionary containing idx, bbox, point_count, density
            and QA flags.
        """
        area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        density = len(self.points) / area if area > 0 else 0.0

        return {
            "tile_idx_i": self.idx[0],
            "tile_idx_j": self.idx[1],
            "bbox_x_min": self.bbox[0],
            "bbox_y_min": self.bbox[1],
            "bbox_x_max": self.bbox[2],
            "bbox_y_max": self.bbox[3],
            "point_count": len(self.points),
            "density": density,
            **self.qa_flags
        }


@dataclass
class Tiler:
    """Partition a point cloud into square tiles and compute QA flags."""

    size: float = 20.0
    """Tile size in metres."""

    density_threshold: float = 50.0
    """Minimum acceptable point density (points per square metre)."""

    compute_intensity_histogram: bool = False
    """Whether to compute intensity histograms per tile."""

    compute_ground_fraction: bool = False
    """Whether to compute the fraction of ground points per tile.
    Requires a 'is_ground' column in the points array."""

    def tile(self, points: np.ndarray) -> List[Tile]:
        """Split the point cloud into a grid of tiles.

        The algorithm computes the bounding box of the point cloud in the
        XY plane and then divides it into a grid of square cells of
        side length `size`.  All points falling within each cell are
        grouped into a `Tile` object.  A simple QA flag is computed
        based on point density.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, M) with XYZ in columns 0–2.

        Returns
        -------
        list of Tile
            All non‑empty tiles.
        """
        # Handle empty point cloud
        if len(points) == 0:
            return []

        x = points[:, 0]
        y = points[:, 1]
        x_min, y_min = x.min(), y.min()
        x_max, y_max = x.max(), y.max()

        # Compute tile indices for each point
        i = np.floor((x - x_min) / self.size).astype(int)
        j = np.floor((y - y_min) / self.size).astype(int)

        # Clamp points at max boundary to the last tile
        # (Points at exactly x_max or y_max should go in the last tile, not create a new one)
        if x_max > x_min:
            max_i = int(np.ceil((x_max - x_min) / self.size)) - 1
            i = np.minimum(i, max_i)
        if y_max > y_min:
            max_j = int(np.ceil((y_max - y_min) / self.size)) - 1
            j = np.minimum(j, max_j)
        tiles: Dict[Tuple[int, int], List[int]] = {}
        for idx, (ti, tj) in enumerate(zip(i, j)):
            key = (ti, tj)
            tiles.setdefault(key, []).append(idx)
        result: List[Tile] = []
        for key, indices in tiles.items():
            subset = points[indices]
            # Bounding box of this tile
            bbox = (
                x_min + key[0] * self.size,
                y_min + key[1] * self.size,
                x_min + (key[0] + 1) * self.size,
                y_min + (key[1] + 1) * self.size,
            )
            # Compute density
            area = self.size * self.size
            density = len(subset) / area
            qa_flags = {
                "density_ok": density >= self.density_threshold
            }

            # Optional: compute intensity histogram
            if self.compute_intensity_histogram and subset.shape[1] > 4:
                intensity = subset[:, 4]
                qa_flags["intensity_mean"] = float(np.mean(intensity))
                qa_flags["intensity_std"] = float(np.std(intensity))

            # Optional: compute ground fraction
            if self.compute_ground_fraction and subset.shape[1] > 8:
                # Assume is_ground is in column 8 (after X,Y,Z,range,int,R,G,B)
                is_ground = subset[:, 8].astype(bool)
                ground_fraction = np.sum(is_ground) / len(subset)
                qa_flags["ground_fraction"] = float(ground_fraction)
                qa_flags["ground_ok"] = 0.1 < ground_fraction < 0.9

            result.append(Tile(key, bbox, subset, qa_flags))
        return result

    def export_metadata_to_parquet(
        self,
        tiles: List[Tile],
        output_dir: Path
    ) -> None:
        """Export tile metadata to Parquet files.

        This creates one Parquet file per tile containing metadata and
        QA flags.  The point arrays themselves are not saved in this
        metadata export; use `export_tiles_to_parquet` for full tile
        data.

        Parameters
        ----------
        tiles : list of Tile
            Tiles to export.
        output_dir : Path
            Directory where Parquet files will be written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_records = []
        for tile in tiles:
            metadata = tile.to_metadata_dict()
            metadata_records.append(metadata)

        df = pd.DataFrame(metadata_records)
        output_path = output_dir / "tiles_metadata.parquet"
        df.to_parquet(output_path, index=False)

    def export_tiles_to_parquet(
        self,
        tiles: List[Tile],
        output_dir: Path
    ) -> None:
        """Export full tile data (including points) to Parquet files.

        Each tile is saved as a separate Parquet file containing the
        point cloud data.

        Parameters
        ----------
        tiles : list of Tile
            Tiles to export.
        output_dir : Path
            Directory where Parquet files will be written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for tile in tiles:
            tile_id = f"tile_{tile.idx[0]:04d}_{tile.idx[1]:04d}"

            # Convert points array to DataFrame
            n_cols = tile.points.shape[1]
            col_names = ["x", "y", "z"]
            if n_cols > 3:
                col_names.append("range")
            if n_cols > 4:
                col_names.append("intensity")
            if n_cols > 5:
                col_names += ["r", "g", "b"]
            if n_cols > 8:
                col_names.append("is_ground")
            # Add generic names for any additional columns
            for i in range(len(col_names), n_cols):
                col_names.append(f"col_{i}")

            df = pd.DataFrame(tile.points, columns=col_names[:n_cols])

            # Add tile metadata as columns
            for key, value in tile.qa_flags.items():
                df[f"qa_{key}"] = value

            output_path = output_dir / f"{tile_id}.parquet"
            df.to_parquet(output_path, index=False)

    @staticmethod
    def load_tiles_from_parquet(input_dir: Path) -> List[Dict]:
        """Load tile metadata from Parquet files.

        Parameters
        ----------
        input_dir : Path
            Directory containing tiles_metadata.parquet.

        Returns
        -------
        list of dict
            List of tile metadata dictionaries.
        """
        metadata_path = input_dir / "tiles_metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_parquet(metadata_path)
        return df.to_dict('records')
