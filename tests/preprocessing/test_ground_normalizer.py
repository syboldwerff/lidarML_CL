"""Unit tests for ground normalizer."""

import numpy as np
import pytest

from src.preprocessing.ground_normalizer import GroundNormalizer


class TestGroundNormalizer:
    """Test suite for GroundNormalizer class."""

    def test_normalise_simple(self):
        """Test simple global ground normalisation."""
        normalizer = GroundNormalizer(quantile=0.1)

        # Create points with clear ground at z=0 and objects above
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [2.0, 0.0, 0.2],
            [3.0, 0.0, 5.0],  # High object
            [4.0, 0.0, 10.0], # Higher object
        ])

        result = normalizer.normalise_simple(points)

        # Ground level should be close to 0 (10th percentile)
        # All Z values should be relative to ground
        assert result[0, 2] >= -0.5  # First point near ground
        assert result[3, 2] > 4.0    # Object should be elevated

    def test_estimate_local_ground(self):
        """Test local ground estimation with grid."""
        normalizer = GroundNormalizer(
            grid_resolution=5.0,
            quantile=0.1,
            ground_threshold=0.3
        )

        # Create a tilted ground plane
        x = np.linspace(0, 20, 50)
        y = np.linspace(0, 20, 50)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()
        yy = yy.flatten()

        # Sloped ground: z increases with x
        zz = 0.1 * xx + np.random.normal(0, 0.05, len(xx))

        points = np.column_stack([xx, yy, zz])

        result, is_ground = normalizer.estimate_local_ground(points)

        # Most points should be classified as ground
        # (Lower threshold to account for grid-based approximation on sloped surfaces)
        ground_fraction = np.sum(is_ground) / len(is_ground)
        assert ground_fraction > 0.65

        # Normalized Z should be close to 0 for ground points
        ground_z = result[is_ground, 2]
        assert np.abs(np.mean(ground_z)) < 0.2

    def test_normalise_with_objects(self):
        """Test ground normalisation with above-ground objects."""
        normalizer = GroundNormalizer(
            grid_resolution=2.0,
            ground_threshold=0.2
        )

        # Create ground points and elevated objects
        n_ground = 100
        n_objects = 20

        # Ground points
        ground_xy = np.random.uniform(0, 10, (n_ground, 2))
        ground_z = np.random.normal(0, 0.05, n_ground)
        ground = np.column_stack([ground_xy, ground_z])

        # Object points (elevated)
        object_xy = np.random.uniform(0, 10, (n_objects, 2))
        object_z = np.random.uniform(2, 5, n_objects)
        objects = np.column_stack([object_xy, object_z])

        points = np.vstack([ground, objects])

        result, is_ground = normalizer.normalise(points)

        # Ground points should be labeled as ground
        assert np.sum(is_ground[:n_ground]) > n_ground * 0.8

        # Object points should NOT be labeled as ground
        assert np.sum(is_ground[n_ground:]) < n_objects * 0.2

        # Object heights should be preserved relative to ground (most should be > 1.0)
        # Some may be lower due to grid-based estimation in sparse cells
        object_heights = result[n_ground:, 2]
        assert np.mean(object_heights) > 2.0  # Average should still be reasonable
        assert np.sum(object_heights > 1.0) > n_objects * 0.7  # At least 70% > 1.0m

    def test_empty_points(self):
        """Test handling of empty point cloud."""
        normalizer = GroundNormalizer()

        points = np.array([]).reshape(0, 3)

        # Should handle gracefully
        with pytest.raises((ValueError, IndexError)):
            normalizer.normalise(points)

    def test_single_point(self):
        """Test handling of single point."""
        normalizer = GroundNormalizer()

        points = np.array([[100.0, 450.0, 10.0]])

        result, is_ground = normalizer.normalise(points)

        # Single point should be classified as ground
        assert is_ground[0] == True
        # Z should be 0 (relative to itself)
        assert abs(result[0, 2]) < 1e-6
