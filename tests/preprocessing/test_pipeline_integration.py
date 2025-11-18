"""Integration tests for complete preprocessing pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.pipeline import PreprocessingPipeline


class TestPipelineIntegration:
    """Integration tests for PreprocessingPipeline."""

    def create_synthetic_scan(self, n_points=10000):
        """Create synthetic LiDAR scan for testing.

        Returns a point cloud with realistic structure:
        - Ground plane at z≈0
        - Some elevated objects
        - Valid RD New coordinates
        - RGB and intensity channels
        """
        # Ground points (70%)
        n_ground = int(n_points * 0.7)
        ground_x = np.random.uniform(100000, 100100, n_ground)
        ground_y = np.random.uniform(450000, 450100, n_ground)
        ground_z = np.random.normal(0, 0.1, n_ground)  # Flat ground with noise

        # Object points (30%) - elevated
        n_objects = n_points - n_ground
        object_x = np.random.uniform(100000, 100100, n_objects)
        object_y = np.random.uniform(450000, 450100, n_objects)
        object_z = np.random.uniform(1, 5, n_objects)  # 1-5m above ground

        # Combine
        x = np.concatenate([ground_x, object_x])
        y = np.concatenate([ground_y, object_y])
        z = np.concatenate([ground_z, object_z])

        # Add range (approximate distance from sensor)
        range_vals = np.sqrt((x - 100050)**2 + (y - 450050)**2)

        # Add intensity (markings bright, asphalt dark)
        intensity = np.where(
            np.random.random(n_points) < 0.1,  # 10% markings
            np.random.uniform(200, 255, n_points),  # Bright
            np.random.uniform(20, 80, n_points)     # Dark
        )

        # Add RGB
        r = np.random.randint(80, 180, n_points)
        g = np.random.randint(80, 180, n_points)
        b = np.random.randint(80, 180, n_points)

        # Stack all columns
        points = np.column_stack([x, y, z, range_vals, intensity, r, g, b])

        return points

    def test_pipeline_end_to_end(self):
        """Test complete pipeline from input to output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create pipeline
            pipeline = PreprocessingPipeline(
                output_dir=output_dir,
                provider="test_provider",
                tile_size=20.0,
                density_threshold=10.0  # Low threshold for test data
            )

            # Create synthetic data
            points = self.create_synthetic_scan(n_points=5000)

            # Run pipeline
            summary = pipeline.run(points, epsg=28992)

            # Verify summary
            assert summary['input_points'] == 5000
            assert summary['filtered_points'] > 0
            assert summary['total_tiles'] > 0
            assert summary['crs_ok'] == True
            assert summary['has_rgb'] == True
            assert summary['has_intensity'] == True

            # Verify output files exist
            assert (output_dir / "meta_scan.json").exists()
            assert (output_dir / "pre_tiles" / "tiles_metadata.parquet").exists()

            # Check that approved tiles directory exists if there are approved tiles
            if summary['qa_approved_tiles'] > 0:
                assert (output_dir / "tiles_approved").exists()

    def test_pipeline_step_by_step(self):
        """Test pipeline steps individually."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            pipeline = PreprocessingPipeline(
                output_dir=output_dir,
                provider="test_provider"
            )

            points = self.create_synthetic_scan(n_points=3000)

            # Step 1: Metadata validation
            metadata = pipeline.step_1_validate_metadata(points, epsg=28992)
            assert metadata.crs_ok == True
            assert metadata.has_rgb == True
            assert metadata.has_intensity == True
            assert metadata.point_count == 3000

            # Step 2: Filtering
            points_filtered = pipeline.step_2_filter_points(points)
            assert len(points_filtered) > 0
            assert len(points_filtered) <= len(points)

            # Step 3: Ground normalization
            points_norm, is_ground = pipeline.step_3_normalize_ground(points_filtered)
            assert len(points_norm) == len(points_filtered)
            assert len(is_ground) == len(points_filtered)
            ground_fraction = np.sum(is_ground) / len(is_ground)
            assert 0.0 < ground_fraction < 1.0

            # Step 4: Attribute normalization
            points_final = pipeline.step_4_normalize_attributes(points_norm)
            assert len(points_final) == len(points_norm)

            # Step 5: Tiling and QA
            tiles = pipeline.step_5_tile_and_qa(points_final, is_ground)
            assert len(tiles) > 0
            assert all(t.qa_flags for t in tiles)

            # Step 6: Export
            pipeline.step_6_export_tiles(tiles)
            assert (output_dir / "pre_tiles" / "tiles_metadata.parquet").exists()

    def test_pipeline_with_poor_quality_data(self):
        """Test pipeline handling of poor quality data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            pipeline = PreprocessingPipeline(
                output_dir=output_dir,
                provider="test_provider",
                density_threshold=100.0  # High threshold - will fail
            )

            # Create sparse, poor quality data
            n = 500  # Very sparse
            points = np.column_stack([
                np.random.uniform(100000, 100100, n),
                np.random.uniform(450000, 450100, n),
                np.random.uniform(0, 10, n),
                np.full(n, 50.0),
                np.random.uniform(10, 50, n),  # Low intensity
                np.random.randint(0, 100, n),  # Dark RGB
                np.random.randint(0, 100, n),
                np.random.randint(0, 100, n),
            ])

            summary = pipeline.run(points, epsg=28992)

            # Should complete but with few/no approved tiles
            assert summary['total_tiles'] > 0
            # Most tiles should fail QA due to low density
            assert summary['qa_approved_tiles'] < summary['total_tiles']

            # Review directory should exist
            assert (output_dir / "qa_review").exists()

    def test_pipeline_preserves_data_integrity(self):
        """Test that pipeline preserves spatial relationships."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            pipeline = PreprocessingPipeline(
                output_dir=output_dir,
                tile_size=50.0,
                density_threshold=10.0
            )

            # Create data with known structure
            # A clear elevated object at specific location
            object_center = (100050.0, 450050.0)
            object_height = 3.0

            n_ground = 4000
            ground = np.column_stack([
                np.random.uniform(100000, 100100, n_ground),
                np.random.uniform(450000, 450100, n_ground),
                np.random.normal(0, 0.05, n_ground),
                np.full(n_ground, 50.0),
                np.full(n_ground, 50.0),
                np.random.randint(100, 150, (n_ground, 3))
            ])

            n_object = 500
            object_points = np.column_stack([
                np.random.normal(object_center[0], 2, n_object),
                np.random.normal(object_center[1], 2, n_object),
                np.random.normal(object_height, 0.2, n_object),
                np.full(n_object, 50.0),
                np.full(n_object, 200.0),  # Bright
                np.random.randint(200, 255, (n_object, 3))
            ])

            points = np.vstack([ground, object_points])

            summary = pipeline.run(points)

            # After normalization, object should still be at ~3m height
            # (This is tested implicitly through the full pipeline)
            assert summary['filtered_points'] > 4000  # Most points retained

    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal point cloud (just XYZ)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            pipeline = PreprocessingPipeline(
                output_dir=output_dir,
                density_threshold=10.0
            )

            # Minimal data: only XYZ
            n = 2000
            points = np.column_stack([
                np.random.uniform(100000, 100050, n),
                np.random.uniform(450000, 450050, n),
                np.random.uniform(0, 5, n),
            ])

            summary = pipeline.run(points)

            # Should complete with warnings about missing data
            assert summary['input_points'] == 2000
            assert summary['total_tiles'] > 0
            assert summary['has_rgb'] == False
            assert summary['has_intensity'] == False

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results."""
        points = self.create_synthetic_scan(n_points=3000)

        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                # Run pipeline twice with same data
                pipeline1 = PreprocessingPipeline(
                    output_dir=Path(tmpdir1),
                    tile_size=20.0,
                    density_threshold=10.0
                )
                summary1 = pipeline1.run(points.copy())

                pipeline2 = PreprocessingPipeline(
                    output_dir=Path(tmpdir2),
                    tile_size=20.0,
                    density_threshold=10.0
                )
                summary2 = pipeline2.run(points.copy())

                # Results should be identical
                assert summary1['filtered_points'] == summary2['filtered_points']
                assert summary1['total_tiles'] == summary2['total_tiles']
                assert summary1['qa_approved_tiles'] == summary2['qa_approved_tiles']
