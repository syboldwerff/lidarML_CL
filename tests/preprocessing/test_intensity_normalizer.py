"""Unit tests for intensity normalizer."""

import numpy as np
import pytest

from src.preprocessing.intensity_normalizer import IntensityNormalizer


class TestIntensityNormalizer:
    """Test suite for IntensityNormalizer class."""

    def test_compensate_range(self):
        """Test range compensation with inverse square law."""
        normalizer = IntensityNormalizer(reference_range=50.0)

        distances = np.array([25.0, 50.0, 100.0])
        intensities = np.array([100.0, 100.0, 100.0])

        compensated = normalizer.compensate_range(distances, intensities)

        # At 25m (half reference): intensity should be 4x (2^2)
        assert compensated[0] == pytest.approx(25.0, rel=0.01)

        # At 50m (reference): intensity should be unchanged
        assert compensated[1] == pytest.approx(100.0, rel=0.01)

        # At 100m (double reference): intensity should be 4x
        assert compensated[2] == pytest.approx(400.0, rel=0.01)

    def test_clamp_intensity(self):
        """Test intensity clamping to valid range."""
        normalizer = IntensityNormalizer(
            intensity_min=1.0,
            intensity_max=255.0
        )

        intensities = np.array([0.0, 50.0, 200.0, 300.0, -10.0])
        clamped = normalizer.clamp_intensity(intensities)

        assert clamped[0] == 1.0    # Clamped to min
        assert clamped[1] == 50.0   # Unchanged
        assert clamped[2] == 200.0  # Unchanged
        assert clamped[3] == 255.0  # Clamped to max
        assert clamped[4] == 1.0    # Clamped to min

    def test_normalise_rgb(self):
        """Test RGB normalisation."""
        normalizer = IntensityNormalizer(
            rgb_min=0,
            rgb_max=255,
            apply_white_balance=False
        )

        rgb = np.array([
            [100, 150, 200],
            [50, 100, 150],
            [300, -10, 255]  # Out of range values
        ])

        normalized = normalizer.normalise_rgb(rgb)

        # In-range values should be unchanged
        assert np.array_equal(normalized[0], [100, 150, 200])
        assert np.array_equal(normalized[1], [50, 100, 150])

        # Out-of-range values should be clamped
        assert normalized[2, 0] == 255  # Clamped to max
        assert normalized[2, 1] == 0    # Clamped to min
        assert normalized[2, 2] == 255  # Unchanged

    def test_normalise_rgb_white_balance(self):
        """Test RGB normalisation with white balance."""
        normalizer = IntensityNormalizer(
            apply_white_balance=True
        )

        # Create RGB with color cast (too much red)
        rgb = np.array([
            [200, 100, 100],
            [180, 90, 90],
            [220, 110, 110]
        ])

        normalized = normalizer.normalise_rgb(rgb)

        # After white balance, channels should be more balanced
        mean_r = np.mean(normalized[:, 0])
        mean_g = np.mean(normalized[:, 1])
        mean_b = np.mean(normalized[:, 2])

        # All channels should be closer to each other
        assert abs(mean_r - mean_g) < abs(200 - 100)  # More balanced than input

    def test_check_marking_contrast(self):
        """Test marking contrast check."""
        normalizer = IntensityNormalizer()

        # Create intensity data with clear bright markings and dark asphalt
        intensities = np.concatenate([
            np.full(80, 30.0),   # Dark asphalt
            np.full(20, 220.0)   # Bright markings
        ])

        contrast_ok, ratio = normalizer.check_marking_contrast(
            intensities,
            threshold_bright=200.0,
            threshold_dark=50.0
        )

        assert contrast_ok == True
        assert ratio > 0.1  # 20 bright vs 80 dark = 0.25 ratio

    def test_check_marking_contrast_no_markings(self):
        """Test marking contrast with no bright markings."""
        normalizer = IntensityNormalizer()

        # Only dark values (no markings)
        intensities = np.full(100, 30.0)

        contrast_ok, ratio = normalizer.check_marking_contrast(
            intensities,
            threshold_bright=200.0,
            threshold_dark=50.0
        )

        assert contrast_ok == False
        assert ratio == 0.0

    def test_normalise_full_workflow(self):
        """Test complete normalisation workflow."""
        normalizer = IntensityNormalizer(
            reference_range=50.0,
            n_clusters=3
        )

        # Create points with range, intensity, and RGB
        # Columns: X, Y, Z, range, intensity, R, G, B
        points = np.array([
            [0, 0, 0, 25.0, 50.0, 100, 150, 200],
            [1, 0, 0, 50.0, 100.0, 110, 160, 210],
            [2, 0, 0, 100.0, 150.0, 120, 170, 220],
            [3, 0, 1, 75.0, 80.0, 130, 180, 230],
        ])

        result, labels = normalizer.normalise(points)

        # Check intensity is compensated and clamped
        intensities = result[:, 4]
        assert np.all(intensities >= 1.0)
        assert np.all(intensities <= 255.0)

        # Check RGB is normalized
        rgb = result[:, 5:8]
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 255)

        # Check material labels are assigned
        assert len(labels) == len(points)
        assert np.all(labels >= 0)
        assert np.all(labels < normalizer.n_clusters)

    def test_preclassify(self):
        """Test material preclassification."""
        normalizer = IntensityNormalizer(n_clusters=2)

        # Create two distinct material types
        intensities = np.concatenate([
            np.full(50, 50.0),   # Dark material
            np.full(50, 200.0)   # Bright material
        ])
        roughness = np.random.uniform(0, 1, 100)

        labels = normalizer.preclassify(intensities, roughness)

        # Should create 2 distinct clusters
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2

        # Dark materials should be in one cluster
        dark_labels = labels[:50]
        assert len(np.unique(dark_labels)) == 1

        # Bright materials should be in another cluster
        bright_labels = labels[50:]
        assert len(np.unique(bright_labels)) == 1

        # Clusters should be different
        assert dark_labels[0] != bright_labels[0]
