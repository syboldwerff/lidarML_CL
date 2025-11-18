"""Complete preprocessing pipeline for LiDAR point clouds.

This module provides an integrated preprocessing workflow that
orchestrates all preprocessing steps from metadata validation through
to QA-approved tiling and export.  It implements the roadmap Sectie 1
(Preprocessing & QA) workflow.

Usage:
    python -m src.preprocessing.pipeline --input scan.laz --output output_dir/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd

from .metadata_validator import MetadataValidator, ScanMetadata
from .lidar_filter import filter_range, filter_intensity_low, spike_filter_z
from .ground_normalizer import GroundNormalizer
from .intensity_normalizer import IntensityNormalizer
from .tiler import Tiler, Tile
from .qa_engine import QAEngine


class PreprocessingPipeline:
    """Complete preprocessing pipeline for vendor-corrected LiDAR data.

    This pipeline assumes input data has already been corrected by the
    vendor (GNSS/IMU correction, RD/NAP coordinates, RGB and intensity
    channels present).  It performs light cleaning, ground segmentation,
    intensity/RGB normalisation, tiling and QA flagging.
    """

    def __init__(
        self,
        output_dir: Path,
        provider: str = "unknown",
        tile_size: float = 20.0,
        density_threshold: float = 50.0
    ):
        """Initialize the preprocessing pipeline.

        Parameters
        ----------
        output_dir : Path
            Directory for output files.
        provider : str, optional
            Data provider identifier.
        tile_size : float, optional
            Tile size in metres (default 20m).
        density_threshold : float, optional
            Minimum point density threshold (default 50 pts/m²).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.metadata_validator = MetadataValidator(provider=provider)
        self.ground_normalizer = GroundNormalizer(
            grid_resolution=2.0,
            ground_threshold=0.2
        )
        self.intensity_normalizer = IntensityNormalizer(
            intensity_min=1.0,
            intensity_max=255.0,
            apply_white_balance=False
        )
        self.tiler = Tiler(
            size=tile_size,
            density_threshold=density_threshold,
            compute_intensity_histogram=True,
            compute_ground_fraction=True
        )
        self.qa_engine = QAEngine(
            density_threshold=density_threshold
        )

        # Pipeline state
        self.metadata: Optional[ScanMetadata] = None
        self.points: Optional[np.ndarray] = None
        self.tiles: List[Tile] = []

    def step_1_validate_metadata(self, points: np.ndarray, epsg: Optional[int] = None) -> ScanMetadata:
        """Step 1.1: Validate metadata and generate meta_scan.json.

        Parameters
        ----------
        points : numpy.ndarray
            Input point cloud.
        epsg : int, optional
            EPSG code if known.

        Returns
        -------
        ScanMetadata
            Validation results.
        """
        print("Step 1.1: Validating metadata and CRS...")
        self.metadata = self.metadata_validator.validate(points, epsg)

        # Save metadata
        meta_path = self.output_dir / "meta_scan.json"
        self.metadata.save(meta_path)
        print(f"  ✓ Metadata saved to {meta_path}")
        print(f"  - CRS OK: {self.metadata.crs_ok}")
        print(f"  - Has RGB: {self.metadata.has_rgb}")
        print(f"  - Has intensity: {self.metadata.has_intensity}")
        print(f"  - Point count: {self.metadata.point_count}")

        return self.metadata

    def step_2_filter_points(self, points: np.ndarray) -> np.ndarray:
        """Step 1.2: Basic filtering (light cleaning).

        Parameters
        ----------
        points : numpy.ndarray
            Input points.

        Returns
        -------
        numpy.ndarray
            Filtered points.
        """
        print("\nStep 1.2: Applying basic filters...")
        n_original = len(points)

        # Filter by range if range column is present
        if points.shape[1] > 3:
            points = filter_range(points, max_range=120.0)
            print(f"  - Range filter: {len(points)}/{n_original} points retained")

        # Filter low intensity
        if points.shape[1] > 4:
            points = filter_intensity_low(points, threshold=0.0)
            print(f"  - Intensity filter: {len(points)}/{n_original} points retained")

        # Spike filter on Z
        points = spike_filter_z(points, kernel=5, threshold=1.0)
        print(f"  ✓ Basic filtering complete: {len(points)}/{n_original} points retained")

        return points

    def step_3_normalize_ground(self, points: np.ndarray) -> tuple:
        """Step 1.3: Ground segmentation and height normalisation.

        Parameters
        ----------
        points : numpy.ndarray
            Input points.

        Returns
        -------
        tuple
            (normalised points, is_ground mask)
        """
        print("\nStep 1.3: Ground segmentation and normalisation...")
        points_norm, is_ground = self.ground_normalizer.normalise(points)

        ground_fraction = np.sum(is_ground) / len(is_ground)
        print(f"  ✓ Ground fraction: {ground_fraction:.2%}")
        print(f"  - Ground points: {np.sum(is_ground)}")
        print(f"  - Non-ground points: {np.sum(~is_ground)}")

        return points_norm, is_ground

    def step_4_normalize_attributes(self, points: np.ndarray) -> np.ndarray:
        """Step 1.4: Intensity and RGB normalisation.

        Parameters
        ----------
        points : numpy.ndarray
            Input points.

        Returns
        -------
        numpy.ndarray
            Points with normalised attributes.
        """
        print("\nStep 1.4: Normalising intensity and RGB...")
        points_norm, material_labels = self.intensity_normalizer.normalise(points)

        if points.shape[1] > 4:
            intensity = points_norm[:, 4]
            print(f"  - Intensity range: [{intensity.min():.1f}, {intensity.max():.1f}]")

            # Check marking contrast
            contrast_ok, contrast_ratio = self.intensity_normalizer.check_marking_contrast(intensity)
            print(f"  - Marking contrast OK: {contrast_ok} (ratio: {contrast_ratio:.3f})")

        if points.shape[1] >= 8:
            rgb = points_norm[:, 5:8]
            print(f"  - RGB range: [{rgb.min():.0f}, {rgb.max():.0f}]")

        print(f"  ✓ Material preclassification: {self.intensity_normalizer.n_clusters} classes")

        return points_norm

    def step_5_tile_and_qa(self, points: np.ndarray, is_ground: np.ndarray) -> List[Tile]:
        """Step 1.5: Tiling and QA per tile.

        Parameters
        ----------
        points : numpy.ndarray
            Input points.
        is_ground : numpy.ndarray
            Ground labels.

        Returns
        -------
        list of Tile
            Generated tiles with QA flags.
        """
        print("\nStep 1.5: Tiling and computing QA metrics...")

        # Add is_ground as column 8
        if points.shape[1] == 8:
            # Already has 8 columns (X,Y,Z,range,int,R,G,B), add is_ground as 9th
            points_with_ground = np.column_stack([points, is_ground.astype(float)])
        else:
            points_with_ground = points

        # Generate tiles
        self.tiles = self.tiler.tile(points_with_ground)
        print(f"  ✓ Generated {len(self.tiles)} tiles")

        # Compute detailed QA for each tile
        for tile in self.tiles:
            qa_results = self.qa_engine.run(
                tile.points,
                self.tiler.size,
                is_ground=tile.points[:, 8].astype(bool) if tile.points.shape[1] > 8 else None
            )
            tile.qa_flags.update(qa_results)

        return self.tiles

    def step_6_export_tiles(self, tiles: List[Tile]) -> None:
        """Step 1.6: Export QA-flagged tiles to Parquet.

        Parameters
        ----------
        tiles : list of Tile
            Tiles to export.
        """
        print("\nStep 1.6: Exporting tiles...")

        # Filter QA-approved tiles
        qa_approved = [
            t for t in tiles
            if t.qa_flags.get("density_ok", False) and
               t.qa_flags.get("intensity_ok", True) and
               t.qa_flags.get("ground_ok", True) and
               t.qa_flags.get("crs_ok", True)
        ]

        qa_review = [t for t in tiles if t not in qa_approved]

        print(f"  - QA approved tiles: {len(qa_approved)}")
        print(f"  - QA review tiles: {len(qa_review)}")

        # Export metadata
        pre_tiles_dir = self.output_dir / "pre_tiles"
        self.tiler.export_metadata_to_parquet(tiles, pre_tiles_dir)
        print(f"  ✓ Metadata exported to {pre_tiles_dir}/tiles_metadata.parquet")

        # Export approved tiles
        if qa_approved:
            approved_dir = self.output_dir / "tiles_approved"
            self.tiler.export_tiles_to_parquet(qa_approved, approved_dir)
            print(f"  ✓ Approved tiles exported to {approved_dir}/")

        # Export review tiles
        if qa_review:
            review_dir = self.output_dir / "qa_review"
            self.tiler.export_tiles_to_parquet(qa_review, review_dir)
            print(f"  ✓ Review tiles exported to {review_dir}/")

    def run(self, points: np.ndarray, epsg: Optional[int] = None) -> Dict:
        """Run the complete preprocessing pipeline.

        Parameters
        ----------
        points : numpy.ndarray
            Input point cloud.
        epsg : int, optional
            EPSG code if known.

        Returns
        -------
        dict
            Summary statistics.
        """
        print("="*60)
        print("LiDAR Preprocessing Pipeline - Roadmap Sectie 1")
        print("="*60)

        # Step 1: Metadata validation
        metadata = self.step_1_validate_metadata(points, epsg)

        if not metadata.crs_ok:
            print("\n⚠ WARNING: CRS validation failed. Results may be unreliable.")

        # Step 2: Basic filtering
        points_filtered = self.step_2_filter_points(points)

        # Step 3: Ground normalisation
        points_norm, is_ground = self.step_3_normalize_ground(points_filtered)

        # Step 4: Attribute normalisation
        points_final = self.step_4_normalize_attributes(points_norm)

        # Step 5: Tiling and QA
        tiles = self.step_5_tile_and_qa(points_final, is_ground)

        # Step 6: Export
        self.step_6_export_tiles(tiles)

        # Summary
        summary = {
            "input_points": len(points),
            "filtered_points": len(points_final),
            "total_tiles": len(tiles),
            "qa_approved_tiles": len([t for t in tiles if all(t.qa_flags.values())]),
            "crs_ok": metadata.crs_ok,
            "has_rgb": metadata.has_rgb,
            "has_intensity": metadata.has_intensity
        }

        print("\n" + "="*60)
        print("Pipeline complete!")
        print("="*60)
        print(f"Input points:      {summary['input_points']:,}")
        print(f"Filtered points:   {summary['filtered_points']:,}")
        print(f"Total tiles:       {summary['total_tiles']}")
        print(f"QA approved:       {summary['qa_approved_tiles']}")
        print(f"\nOutput directory: {self.output_dir}")

        return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run LiDAR preprocessing pipeline (Roadmap Sectie 1)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input LAZ file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="unknown",
        help="Data provider name"
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=20.0,
        help="Tile size in metres (default: 20)"
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=50.0,
        help="Minimum point density (default: 50 pts/m²)"
    )

    args = parser.parse_args()

    # Note: In a real implementation, you would load the LAZ file here
    # using laspy or pdal. For now, we show the structure.
    print(f"Error: LAZ file loading not implemented yet.")
    print(f"To use this pipeline, integrate with laspy:")
    print(f"")
    print(f"  import laspy")
    print(f"  las = laspy.read('{args.input}')")
    print(f"  points = np.column_stack([las.x, las.y, las.z, ...])")
    print(f"")
    print(f"Then call:")
    print(f"  pipeline = PreprocessingPipeline('{args.output}', '{args.provider}')")
    print(f"  pipeline.run(points)")

    return 1


if __name__ == "__main__":
    sys.exit(main())
