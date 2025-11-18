"""Demo script for preprocessing pipeline with synthetic data.

This script demonstrates the complete preprocessing workflow using
synthetic LiDAR data.  It creates realistic point clouds, runs the
full pipeline, and generates QA visualizations.

Usage:
    python examples/demo_preprocessing_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline


def create_synthetic_lidar_scan(
    n_points: int = 50000,
    area_size: float = 100.0,
    rd_origin: tuple = (100000.0, 450000.0)
) -> np.ndarray:
    """Create a realistic synthetic LiDAR scan.

    Creates a point cloud with:
    - Ground plane with slight variations
    - Road markings (bright intensity strips)
    - Elevated objects (poles, buildings)
    - Realistic RGB and intensity values
    - Valid RD New + NAP coordinates

    Parameters
    ----------
    n_points : int
        Total number of points.
    area_size : float
        Size of scanned area in metres.
    rd_origin : tuple
        Origin in RD New coordinates (x, y).

    Returns
    -------
    np.ndarray
        Point cloud with columns [X, Y, Z, range, intensity, R, G, B].
    """
    print("Creating synthetic LiDAR scan...")

    # Ground points (60%)
    n_ground = int(n_points * 0.6)
    print(f"  - Ground points: {n_ground:,}")

    x_ground = np.random.uniform(0, area_size, n_ground) + rd_origin[0]
    y_ground = np.random.uniform(0, area_size, n_ground) + rd_origin[1]

    # Create slightly sloped ground
    z_ground = (
        0.01 * (x_ground - rd_origin[0]) +  # Slight slope
        np.random.normal(0, 0.05, n_ground)  # Natural variation
    )

    # Road markings (10%) - bright strips
    n_markings = int(n_points * 0.1)
    print(f"  - Road markings: {n_markings:,}")

    # Create lane markings as strips
    marking_positions = [20, 40, 60, 80]  # Y positions of markings
    x_markings = []
    y_markings = []
    z_markings = []

    for y_pos in marking_positions:
        n_strip = n_markings // len(marking_positions)
        x_strip = np.random.uniform(0, area_size, n_strip) + rd_origin[0]
        y_strip = np.random.normal(y_pos, 0.2, n_strip) + rd_origin[1]
        z_strip = 0.01 * (x_strip - rd_origin[0]) + np.random.normal(0, 0.01, n_strip)

        x_markings.extend(x_strip)
        y_markings.extend(y_strip)
        z_markings.extend(z_strip)

    x_markings = np.array(x_markings)
    y_markings = np.array(y_markings)
    z_markings = np.array(z_markings)

    # Poles/masts (5%)
    n_poles = int(n_points * 0.05)
    print(f"  - Poles/masts: {n_poles:,}")

    pole_positions = [(10, 10), (10, 90), (90, 10), (90, 90)]
    x_poles = []
    y_poles = []
    z_poles = []

    for px, py in pole_positions:
        n_pole = n_poles // len(pole_positions)
        x_pole = np.random.normal(px, 0.1, n_pole) + rd_origin[0]
        y_pole = np.random.normal(py, 0.1, n_pole) + rd_origin[1]
        z_pole = np.random.uniform(0, 8, n_pole)  # 0-8m height

        x_poles.extend(x_pole)
        y_poles.extend(y_pole)
        z_poles.extend(z_pole)

    x_poles = np.array(x_poles)
    y_poles = np.array(y_poles)
    z_poles = np.array(z_poles)

    # Vegetation/objects (25%)
    n_objects = n_points - n_ground - n_markings - n_poles
    print(f"  - Other objects: {n_objects:,}")

    x_objects = np.random.uniform(0, area_size, n_objects) + rd_origin[0]
    y_objects = np.random.uniform(0, area_size, n_objects) + rd_origin[1]
    z_objects = np.random.uniform(0.5, 3, n_objects)

    # Combine all points
    x = np.concatenate([x_ground, x_markings, x_poles, x_objects])
    y = np.concatenate([y_ground, y_markings, y_poles, y_objects])
    z = np.concatenate([z_ground, z_markings, z_poles, z_objects])

    # Calculate range (distance from sensor at center)
    sensor_x, sensor_y = rd_origin[0] + area_size/2, rd_origin[1] + area_size/2
    range_vals = np.sqrt((x - sensor_x)**2 + (y - sensor_y)**2)

    # Assign intensity
    intensity = np.zeros(n_points)

    # Ground: dark asphalt
    intensity[:n_ground] = np.random.uniform(20, 50, n_ground)

    # Markings: very bright
    intensity[n_ground:n_ground+n_markings] = np.random.uniform(200, 255, n_markings)

    # Poles: medium-bright (metal/painted)
    intensity[n_ground+n_markings:n_ground+n_markings+n_poles] = np.random.uniform(100, 150, n_poles)

    # Objects: varied
    intensity[n_ground+n_markings+n_poles:] = np.random.uniform(40, 120, n_objects)

    # Assign RGB
    r = np.zeros(n_points, dtype=np.uint8)
    g = np.zeros(n_points, dtype=np.uint8)
    b = np.zeros(n_points, dtype=np.uint8)

    # Ground: gray
    r[:n_ground] = np.random.randint(80, 120, n_ground)
    g[:n_ground] = np.random.randint(80, 120, n_ground)
    b[:n_ground] = np.random.randint(80, 120, n_ground)

    # Markings: white
    r[n_ground:n_ground+n_markings] = np.random.randint(220, 255, n_markings)
    g[n_ground:n_ground+n_markings] = np.random.randint(220, 255, n_markings)
    b[n_ground:n_ground+n_markings] = np.random.randint(220, 255, n_markings)

    # Poles: gray metal
    r[n_ground+n_markings:n_ground+n_markings+n_poles] = np.random.randint(100, 150, n_poles)
    g[n_ground+n_markings:n_ground+n_markings+n_poles] = np.random.randint(100, 150, n_poles)
    b[n_ground+n_markings:n_ground+n_markings+n_poles] = np.random.randint(100, 150, n_poles)

    # Objects: varied (greenish for vegetation)
    r[n_ground+n_markings+n_poles:] = np.random.randint(60, 120, n_objects)
    g[n_ground+n_markings+n_poles:] = np.random.randint(100, 180, n_objects)
    b[n_ground+n_markings+n_poles:] = np.random.randint(60, 120, n_objects)

    # Stack into point cloud
    points = np.column_stack([x, y, z, range_vals, intensity, r, g, b])

    print(f"✓ Created {len(points):,} points")
    print(f"  Area: {area_size}x{area_size} m")
    print(f"  Intensity range: [{intensity.min():.1f}, {intensity.max():.1f}]")
    print(f"  Height range: [{z.min():.2f}, {z.max():.2f}] m")

    return points


def main():
    """Run demo pipeline."""
    print("="*70)
    print("LiDAR Preprocessing Pipeline - Demo")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path("output/demo_preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic data
    points = create_synthetic_lidar_scan(
        n_points=50000,
        area_size=100.0,
        rd_origin=(100000.0, 450000.0)
    )

    print()

    # Initialize pipeline
    pipeline = PreprocessingPipeline(
        output_dir=output_dir,
        provider="synthetic_demo",
        tile_size=20.0,
        density_threshold=50.0
    )

    # Run pipeline
    summary = pipeline.run(points, epsg=28992)

    print()
    print("="*70)
    print("Demo Complete!")
    print("="*70)
    print()
    print("Output files:")
    print(f"  • Metadata: {output_dir / 'meta_scan.json'}")
    print(f"  • Tiles metadata: {output_dir / 'pre_tiles' / 'tiles_metadata.parquet'}")

    if (output_dir / "tiles_approved").exists():
        n_approved = len(list((output_dir / "tiles_approved").glob("*.parquet")))
        print(f"  • Approved tiles: {n_approved} files in tiles_approved/")

    if (output_dir / "qa_review").exists():
        n_review = len(list((output_dir / "qa_review").glob("*.parquet")))
        print(f"  • Review tiles: {n_review} files in qa_review/")

    print()
    print("Next steps:")
    print("  1. Visualize BEV tiles:")
    print(f"     python -m src.preprocessing.qa_bev_visualizer \\")
    print(f"       --input {output_dir}/tiles_approved \\")
    print(f"       --output {output_dir}/qa_visualizations")
    print()
    print("  2. Visualize camera views (demo):")
    print(f"     python -m src.preprocessing.qa_cam_visualizer \\")
    print(f"       --output {output_dir}/qa_camera --demo")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
