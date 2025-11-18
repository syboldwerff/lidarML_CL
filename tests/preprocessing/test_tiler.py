"""Unit tests for tiler."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.tiler import Tiler, Tile


class TestTiler:
    """Test suite for Tiler class."""

    def test_tile_simple_grid(self):
        """Test tiling with a simple regular grid."""
        tiler = Tiler(size=10.0, density_threshold=10.0)

        # Create a 20x20 grid of points (should create 4 tiles)
        x = np.linspace(0, 20, 100)
        y = np.linspace(0, 20, 100)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(10000)])

        tiles = tiler.tile(points)

        # Should create 4 tiles (2x2 grid)
        assert len(tiles) == 4

        # All tiles should be valid Tile objects
        for tile in tiles:
            assert isinstance(tile, Tile)
            assert len(tile.points) > 0
            assert tile.idx[0] >= 0
            assert tile.idx[1] >= 0

    def test_tile_density_check(self):
        """Test density QA flagging."""
        tiler = Tiler(size=10.0, density_threshold=50.0)

        # Create sparse and dense regions
        # Sparse: 100 points in 10x10 = 1 pt/m²
        sparse_x = np.random.uniform(0, 10, 100)
        sparse_y = np.random.uniform(0, 10, 100)
        sparse_z = np.zeros(100)

        # Dense: 10000 points in 10x10 = 100 pt/m²
        dense_x = np.random.uniform(10, 20, 10000)
        dense_y = np.random.uniform(0, 10, 10000)
        dense_z = np.zeros(10000)

        points = np.column_stack([
            np.concatenate([sparse_x, dense_x]),
            np.concatenate([sparse_y, dense_y]),
            np.concatenate([sparse_z, dense_z])
        ])

        tiles = tiler.tile(points)

        # Check density flags
        density_flags = [t.qa_flags.get('density_ok', False) for t in tiles]

        # At least one tile should fail density (sparse region)
        assert not all(density_flags)

        # At least one tile should pass density (dense region)
        assert any(density_flags)

    def test_tile_bbox_computation(self):
        """Test bounding box computation for tiles."""
        tiler = Tiler(size=10.0)

        points = np.array([
            [0, 0, 0],
            [9, 9, 0],
            [15, 5, 0],
        ])

        tiles = tiler.tile(points)

        # Should create at least 2 tiles
        assert len(tiles) >= 2

        for tile in tiles:
            bbox = tile.bbox
            # Bbox should be valid
            assert bbox[2] >= bbox[0]  # x_max >= x_min
            assert bbox[3] >= bbox[1]  # y_max >= y_min
            # Bbox should match tile size
            assert abs((bbox[2] - bbox[0]) - tiler.size) < 0.01
            assert abs((bbox[3] - bbox[1]) - tiler.size) < 0.01

    def test_tile_with_intensity_histogram(self):
        """Test tiling with intensity histogram computation."""
        tiler = Tiler(size=10.0, compute_intensity_histogram=True)

        # Create points with intensity (column 4)
        n = 1000
        points = np.column_stack([
            np.random.uniform(0, 20, n),
            np.random.uniform(0, 20, n),
            np.zeros(n),
            np.full(n, 50.0),  # range
            np.random.uniform(50, 150, n)  # intensity
        ])

        tiles = tiler.tile(points)

        # All tiles should have intensity statistics
        for tile in tiles:
            assert 'intensity_mean' in tile.qa_flags
            assert 'intensity_std' in tile.qa_flags
            assert isinstance(tile.qa_flags['intensity_mean'], float)

    def test_tile_with_ground_fraction(self):
        """Test tiling with ground fraction computation."""
        tiler = Tiler(size=10.0, compute_ground_fraction=True)

        # Create points with is_ground flag (column 8)
        n = 1000
        is_ground = np.random.choice([0, 1], n, p=[0.3, 0.7])

        points = np.column_stack([
            np.random.uniform(0, 20, n),
            np.random.uniform(0, 20, n),
            np.zeros(n),
            np.full(n, 50.0),  # range
            np.full(n, 100.0),  # intensity
            np.zeros((n, 3)),   # RGB
            is_ground
        ])

        tiles = tiler.tile(points)

        # All tiles should have ground fraction
        for tile in tiles:
            assert 'ground_fraction' in tile.qa_flags
            assert 'ground_ok' in tile.qa_flags
            assert 0.0 <= tile.qa_flags['ground_fraction'] <= 1.0

    def test_export_metadata_to_parquet(self):
        """Test metadata export to Parquet."""
        tiler = Tiler(size=10.0)

        points = np.random.uniform(0, 30, (1000, 3))
        tiles = tiler.tile(points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            tiler.export_metadata_to_parquet(tiles, output_dir)

            # Check file exists
            metadata_file = output_dir / "tiles_metadata.parquet"
            assert metadata_file.exists()

            # Load and verify
            df = pd.read_parquet(metadata_file)
            assert len(df) == len(tiles)
            assert 'tile_idx_i' in df.columns
            assert 'tile_idx_j' in df.columns
            assert 'point_count' in df.columns
            assert 'density' in df.columns

    def test_export_tiles_to_parquet(self):
        """Test full tile export to Parquet."""
        tiler = Tiler(size=10.0)

        points = np.random.uniform(0, 20, (500, 5))  # X, Y, Z, range, intensity
        tiles = tiler.tile(points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            tiler.export_tiles_to_parquet(tiles, output_dir)

            # Check files exist
            parquet_files = list(output_dir.glob("tile_*.parquet"))
            assert len(parquet_files) == len(tiles)

            # Load one tile and verify
            df = pd.read_parquet(parquet_files[0])
            assert 'x' in df.columns
            assert 'y' in df.columns
            assert 'z' in df.columns
            assert len(df) > 0

    def test_load_tiles_from_parquet(self):
        """Test loading tile metadata from Parquet."""
        tiler = Tiler(size=10.0)

        points = np.random.uniform(0, 20, (500, 3))
        tiles = tiler.tile(points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            tiler.export_metadata_to_parquet(tiles, output_dir)

            # Load back
            loaded_metadata = Tiler.load_tiles_from_parquet(output_dir)

            assert len(loaded_metadata) == len(tiles)
            assert isinstance(loaded_metadata[0], dict)
            assert 'tile_idx_i' in loaded_metadata[0]

    def test_tile_to_metadata_dict(self):
        """Test Tile.to_metadata_dict conversion."""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        tile = Tile(
            idx=(5, 10),
            bbox=(0.0, 0.0, 10.0, 10.0),
            points=points,
            qa_flags={'density_ok': True, 'test_flag': False}
        )

        metadata = tile.to_metadata_dict()

        assert metadata['tile_idx_i'] == 5
        assert metadata['tile_idx_j'] == 10
        assert metadata['bbox_x_min'] == 0.0
        assert metadata['bbox_y_max'] == 10.0
        assert metadata['point_count'] == 2
        assert 'density' in metadata
        assert metadata['density_ok'] == True
        assert metadata['test_flag'] == False

    def test_empty_points(self):
        """Test handling of empty point cloud."""
        tiler = Tiler(size=10.0)

        points = np.array([]).reshape(0, 3)
        tiles = tiler.tile(points)

        # Should return empty list
        assert len(tiles) == 0
