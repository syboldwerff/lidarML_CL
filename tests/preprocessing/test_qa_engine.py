"""Unit tests for QA engine."""

import numpy as np
import pytest

from src.preprocessing.qa_engine import QAEngine


class TestQAEngine:
    """Test suite for QAEngine class."""

    def test_test_density_pass(self):
        """Test density check with sufficient density."""
        qa = QAEngine(density_threshold=50.0)

        # 1000 points in 20x20m tile = 2.5 pts/m² (should fail)
        points = np.random.uniform(0, 20, (1000, 3))
        assert qa.test_density(points, 20.0) == False

        # 25000 points in 20x20m tile = 62.5 pts/m² (should pass)
        points = np.random.uniform(0, 20, (25000, 3))
        assert qa.test_density(points, 20.0) == True

    def test_test_intensity_pass(self):
        """Test intensity check."""
        qa = QAEngine(intensity_threshold=10.0)

        # Mean intensity above threshold
        intensities = np.array([50.0, 100.0, 150.0])
        assert qa.test_intensity(intensities) == True

        # Mean intensity below threshold
        intensities = np.array([1.0, 2.0, 3.0])
        assert qa.test_intensity(intensities) == False

    def test_test_intensity_empty(self):
        """Test intensity check with empty array."""
        qa = QAEngine()

        intensities = np.array([])
        assert qa.test_intensity(intensities) == False

    def test_test_ground_fraction_valid(self):
        """Test ground fraction check with valid fraction."""
        qa = QAEngine(ground_fraction_min=0.1, ground_fraction_max=0.9)

        # 50% ground (should pass)
        is_ground = np.array([True] * 50 + [False] * 50)
        assert qa.test_ground_fraction(is_ground) == True

        # 95% ground (should fail - too high)
        is_ground = np.array([True] * 95 + [False] * 5)
        assert qa.test_ground_fraction(is_ground) == False

        # 2% ground (should fail - too low)
        is_ground = np.array([True] * 2 + [False] * 98)
        assert qa.test_ground_fraction(is_ground) == False

    def test_test_rgb_valid(self):
        """Test RGB validation."""
        qa = QAEngine(rgb_min=0, rgb_max=255)

        # Valid RGB
        rgb = np.array([
            [100, 150, 200],
            [50, 100, 150],
            [0, 128, 255]
        ])
        assert qa.test_rgb(rgb) == True

        # Invalid RGB (out of range)
        rgb = np.array([
            [100, 150, 300],  # Blue > 255
            [50, 100, 150],
        ])
        assert qa.test_rgb(rgb) == False

        rgb = np.array([
            [100, -10, 200],  # Green < 0
            [50, 100, 150],
        ])
        assert qa.test_rgb(rgb) == False

    def test_test_rgb_empty(self):
        """Test RGB check with empty array."""
        qa = QAEngine()

        rgb = np.array([]).reshape(0, 3)
        assert qa.test_rgb(rgb) == False

    def test_test_crs_valid_rd(self):
        """Test CRS validation with valid RD New coordinates."""
        qa = QAEngine()

        # Valid RD New + NAP coordinates
        points = np.array([
            [100000.0, 450000.0, 10.0],
            [100100.0, 450100.0, 12.0],
        ])
        assert qa.test_crs(points) == True

    def test_test_crs_invalid_coordinates(self):
        """Test CRS validation with invalid coordinates."""
        qa = QAEngine()

        # Invalid coordinates (outside RD range)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        assert qa.test_crs(points) == False

        # Invalid Z (too high for NAP)
        points = np.array([
            [100000.0, 450000.0, 500.0],
            [100100.0, 450100.0, 600.0],
        ])
        assert qa.test_crs(points) == False

    def test_run_complete_qa(self):
        """Test complete QA workflow."""
        qa = QAEngine(
            density_threshold=50.0,
            intensity_threshold=10.0
        )

        # Create complete point cloud
        # Columns: X, Y, Z, range, intensity, R, G, B, is_ground
        n = 3000
        points = np.column_stack([
            np.random.uniform(100000, 100020, n),  # X in RD range
            np.random.uniform(450000, 450020, n),  # Y in RD range
            np.random.uniform(0, 20, n),           # Z in NAP range
            np.full(n, 50.0),                      # range
            np.random.uniform(50, 150, n),         # intensity
            np.random.randint(0, 256, n),          # R
            np.random.randint(0, 256, n),          # G
            np.random.randint(0, 256, n),          # B
            np.random.choice([0, 1], n, p=[0.3, 0.7])  # is_ground
        ])

        results = qa.run(points, tile_size=20.0)

        # All flags should be present
        assert 'crs_ok' in results
        assert 'density_ok' in results
        assert 'intensity_ok' in results
        assert 'rgb_ok' in results
        assert 'ground_ok' in results

        # With good data, most should pass
        assert results['crs_ok'] == True
        assert results['intensity_ok'] == True
        assert results['rgb_ok'] == True

    def test_run_without_optional_data(self):
        """Test QA with minimal point cloud (no intensity/RGB)."""
        qa = QAEngine(density_threshold=50.0)

        # Only XYZ - use enough points to meet density threshold
        # 20x20 area * 50 pts/m² = 20,000 points needed
        n = 20000
        points = np.column_stack([
            np.random.uniform(100000, 100020, n),
            np.random.uniform(450000, 450020, n),
            np.random.uniform(0, 20, n),
        ])

        results = qa.run(points, tile_size=20.0)

        # Should still work, optional checks default to True
        assert results['crs_ok'] == True
        assert results['density_ok'] == True
        assert results['intensity_ok'] == True  # No data = pass
        assert results['rgb_ok'] == True        # No data = pass
        assert results['ground_ok'] == True     # No data = pass

    def test_run_with_is_ground_parameter(self):
        """Test QA with is_ground passed as separate parameter."""
        qa = QAEngine()

        points = np.random.uniform(100000, 100020, (1000, 3))
        is_ground = np.random.choice([True, False], 1000, p=[0.6, 0.4])

        results = qa.run(points, tile_size=20.0, is_ground=is_ground)

        assert 'ground_ok' in results
        # With 60% ground, should pass
        assert results['ground_ok'] == True

    def test_custom_thresholds(self):
        """Test QA with custom thresholds."""
        qa = QAEngine(
            density_threshold=100.0,
            intensity_threshold=50.0,
            ground_fraction_min=0.4,
            ground_fraction_max=0.6
        )

        # This should fail strict thresholds
        n = 1000
        points = np.column_stack([
            np.random.uniform(100000, 100020, n),
            np.random.uniform(450000, 450020, n),
            np.random.uniform(0, 20, n),
            np.full(n, 50.0),
            np.full(n, 30.0),  # Low intensity
        ])
        is_ground = np.array([True] * 700 + [False] * 300)  # 70% ground

        results = qa.run(points, tile_size=20.0, is_ground=is_ground)

        # Should fail density (1000 pts in 400m² = 2.5 pts/m²)
        assert results['density_ok'] == False

        # Should fail intensity (30 < 50 threshold)
        assert results['intensity_ok'] == False

        # Should fail ground (70% > 60% max)
        assert results['ground_ok'] == False
