"""Unit tests for metadata validator."""

import numpy as np
import pytest

from src.preprocessing.metadata_validator import MetadataValidator, ScanMetadata


class TestMetadataValidator:
    """Test suite for MetadataValidator class."""

    def test_validate_crs_valid_rd_coordinates(self):
        """Test CRS validation with valid RD New coordinates."""
        validator = MetadataValidator()

        # Valid RD New + NAP coordinates (somewhere in Netherlands)
        points = np.array([
            [100000.0, 450000.0, 10.0],
            [100100.0, 450100.0, 12.0],
            [100200.0, 450200.0, 15.0],
        ])

        assert validator.validate_crs(points) == True

    def test_validate_crs_invalid_coordinates(self):
        """Test CRS validation with invalid coordinates."""
        validator = MetadataValidator()

        # Invalid coordinates (outside RD New range)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        assert validator.validate_crs(points) == False

    def test_check_rgb_present(self):
        """Test RGB check when RGB is present."""
        validator = MetadataValidator()

        # Points with RGB (X, Y, Z, range, intensity, R, G, B)
        points = np.array([
            [100000.0, 450000.0, 10.0, 50.0, 128.0, 100, 150, 200],
            [100100.0, 450100.0, 12.0, 55.0, 130.0, 110, 160, 210],
        ])

        has_rgb, rgb_range = validator.check_rgb(points)
        assert has_rgb == True
        assert rgb_range[0] == 100
        assert rgb_range[1] == 210

    def test_check_rgb_absent(self):
        """Test RGB check when RGB is absent."""
        validator = MetadataValidator()

        # Points without RGB
        points = np.array([
            [100000.0, 450000.0, 10.0, 50.0, 128.0],
        ])

        has_rgb, rgb_range = validator.check_rgb(points)
        assert has_rgb == False
        assert rgb_range is None

    def test_check_intensity_present(self):
        """Test intensity check when intensity is present."""
        validator = MetadataValidator()

        points = np.array([
            [100000.0, 450000.0, 10.0, 50.0, 128.0],
            [100100.0, 450100.0, 12.0, 55.0, 130.0],
        ])

        has_intensity, int_range = validator.check_intensity(points)
        assert has_intensity == True
        assert int_range[0] == 128.0
        assert int_range[1] == 130.0

    def test_compute_bbox(self):
        """Test bounding box computation."""
        validator = MetadataValidator()

        points = np.array([
            [100000.0, 450000.0, 10.0],
            [100100.0, 450100.0, 12.0],
            [100200.0, 450200.0, 15.0],
        ])

        bbox = validator.compute_bbox(points)
        assert bbox[0] == 100000.0  # x_min
        assert bbox[1] == 450000.0  # y_min
        assert bbox[2] == 10.0       # z_min
        assert bbox[3] == 100200.0  # x_max
        assert bbox[4] == 450200.0  # y_max
        assert bbox[5] == 15.0       # z_max

    def test_validate_full(self):
        """Test full validation workflow."""
        validator = MetadataValidator(provider="test_provider")

        # Complete point cloud with all attributes
        points = np.array([
            [100000.0, 450000.0, 10.0, 50.0, 128.0, 100, 150, 200],
            [100100.0, 450100.0, 12.0, 55.0, 130.0, 110, 160, 210],
            [100200.0, 450200.0, 15.0, 60.0, 132.0, 120, 170, 220],
        ])

        metadata = validator.validate(points)

        assert isinstance(metadata, ScanMetadata)
        assert metadata.crs_ok == True
        assert metadata.has_rgb == True
        assert metadata.has_intensity == True
        assert metadata.point_count == 3
        assert metadata.provider == "test_provider"
        assert metadata.bbox == (100000.0, 450000.0, 10.0, 100200.0, 450200.0, 15.0)
