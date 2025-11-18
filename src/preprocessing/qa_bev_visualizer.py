"""QA visualizer for BEV (Bird's Eye View) tiles.

This module provides visualization tools to inspect BEV tiles and
validate preprocessing quality.  It can generate diagnostic images
showing intensity, RGB, density and QA flags for visual inspection.

Usage:
    python -m src.preprocessing.qa_bev_visualizer --input tiles_dir/ --output qa_vis/
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BEVVisualizer:
    """Visualize BEV tiles for QA purposes."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Parameters
        ----------
        output_dir : Path
            Directory for output visualizations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

    def load_tile_data(self, tile_file: Path) -> pd.DataFrame:
        """Load tile data from Parquet file.

        Parameters
        ----------
        tile_file : Path
            Path to tile Parquet file.

        Returns
        -------
        pd.DataFrame
            Tile data.
        """
        return pd.read_parquet(tile_file)

    def create_bev_image(
        self,
        tile_data: pd.DataFrame,
        resolution: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create BEV images from tile data.

        Parameters
        ----------
        tile_data : pd.DataFrame
            Tile point cloud data.
        resolution : float
            Pixel resolution in metres.

        Returns
        -------
        tuple
            (height_map, intensity_map, rgb_image)
        """
        x = tile_data['x'].values
        y = tile_data['y'].values
        z = tile_data['z'].values

        x_min, y_min = x.min(), y.min()
        x_max, y_max = x.max(), y.max()

        width = int(np.ceil((x_max - x_min) / resolution)) + 1
        height = int(np.ceil((y_max - y_min) / resolution)) + 1

        # Initialize images
        height_map = np.full((height, width), -np.inf)
        intensity_map = np.zeros((height, width))
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        count = np.zeros((height, width), dtype=int)

        # Rasterize
        col = ((x - x_min) / resolution).astype(int)
        row = ((y - y_min) / resolution).astype(int)

        for idx in range(len(tile_data)):
            r, c = row[idx], col[idx]
            if 0 <= r < height and 0 <= c < width:
                # Max height
                if z[idx] > height_map[r, c]:
                    height_map[r, c] = z[idx]

                # Mean intensity
                if 'intensity' in tile_data.columns:
                    intensity_map[r, c] += tile_data['intensity'].iloc[idx]

                # Mean RGB
                if all(col in tile_data.columns for col in ['r', 'g', 'b']):
                    rgb_image[r, c, 0] += tile_data['r'].iloc[idx]
                    rgb_image[r, c, 1] += tile_data['g'].iloc[idx]
                    rgb_image[r, c, 2] += tile_data['b'].iloc[idx]

                count[r, c] += 1

        # Compute means
        mask = count > 0
        intensity_map[mask] /= count[mask]
        for c in range(3):
            rgb_image[mask, c] = (rgb_image[mask, c] / count[mask]).astype(np.uint8)

        # Replace inf with 0
        height_map[~np.isfinite(height_map)] = 0

        return height_map, intensity_map, rgb_image

    def visualize_tile(
        self,
        tile_file: Path,
        tile_name: str
    ) -> Path:
        """Create comprehensive visualization for a single tile.

        Parameters
        ----------
        tile_file : Path
            Path to tile Parquet file.
        tile_name : str
            Tile identifier.

        Returns
        -------
        Path
            Path to output visualization.
        """
        # Load data
        tile_data = self.load_tile_data(tile_file)

        # Create BEV images
        height_map, intensity_map, rgb_image = self.create_bev_image(tile_data)

        # Extract QA flags
        qa_flags = {
            col.replace('qa_', ''): tile_data[col].iloc[0]
            for col in tile_data.columns
            if col.startswith('qa_')
        }

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Tile: {tile_name}', fontsize=16, fontweight='bold')

        # Height map
        ax = axes[0, 0]
        im = ax.imshow(height_map, cmap='terrain', origin='lower')
        ax.set_title('Height Map (Z)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Height (m)')

        # Intensity map
        ax = axes[0, 1]
        im = ax.imshow(intensity_map, cmap='gray', origin='lower')
        ax.set_title('Intensity Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Intensity')

        # RGB image
        ax = axes[1, 0]
        ax.imshow(rgb_image, origin='lower')
        ax.set_title('RGB Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # QA Summary
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('QA Summary', fontweight='bold')

        y_pos = 0.9
        for key, value in qa_flags.items():
            color = 'green' if value else 'red'
            symbol = '✓' if value else '✗'
            ax.text(0.1, y_pos, f'{symbol} {key}: {value}',
                   fontsize=12, color=color, verticalalignment='top')
            y_pos -= 0.1

        # Add statistics
        y_pos -= 0.1
        ax.text(0.1, y_pos, f'Points: {len(tile_data):,}',
               fontsize=10, verticalalignment='top')
        y_pos -= 0.08

        if 'intensity' in tile_data.columns:
            ax.text(0.1, y_pos, f'Intensity: [{tile_data["intensity"].min():.1f}, {tile_data["intensity"].max():.1f}]',
                   fontsize=10, verticalalignment='top')
            y_pos -= 0.08

        if 'z' in tile_data.columns:
            ax.text(0.1, y_pos, f'Height: [{tile_data["z"].min():.2f}, {tile_data["z"].max():.2f}] m',
                   fontsize=10, verticalalignment='top')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f'{tile_name}_qa.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def create_overview_report(
        self,
        tiles_metadata: pd.DataFrame
    ) -> Path:
        """Create overview report of all tiles.

        Parameters
        ----------
        tiles_metadata : pd.DataFrame
            Metadata for all tiles.

        Returns
        -------
        Path
            Path to overview report.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Preprocessing QA Overview', fontsize=16, fontweight='bold')

        # QA pass rates
        ax = axes[0, 0]
        qa_columns = [col for col in tiles_metadata.columns if col.startswith('qa_') or col.endswith('_ok')]
        if qa_columns:
            pass_rates = []
            labels = []
            for col in qa_columns:
                if col in tiles_metadata.columns:
                    rate = tiles_metadata[col].sum() / len(tiles_metadata) * 100
                    pass_rates.append(rate)
                    labels.append(col.replace('qa_', '').replace('_ok', ''))

            ax.barh(labels, pass_rates, color=['green' if r > 80 else 'orange' if r > 50 else 'red' for r in pass_rates])
            ax.set_xlabel('Pass Rate (%)')
            ax.set_title('QA Pass Rates')
            ax.set_xlim([0, 100])
            ax.axvline(80, color='green', linestyle='--', alpha=0.3)
            ax.axvline(50, color='orange', linestyle='--', alpha=0.3)

        # Density distribution
        ax = axes[0, 1]
        if 'density' in tiles_metadata.columns:
            densities = tiles_metadata['density']
            ax.hist(densities, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Point Density (pts/m²)')
            ax.set_ylabel('Frequency')
            ax.set_title('Density Distribution')
            ax.axvline(densities.mean(), color='red', linestyle='--', label=f'Mean: {densities.mean():.1f}')
            ax.legend()

        # Spatial distribution
        ax = axes[1, 0]
        if all(col in tiles_metadata.columns for col in ['bbox_x_min', 'bbox_y_min', 'density_ok']):
            for _, row in tiles_metadata.iterrows():
                color = 'green' if row.get('density_ok', False) else 'red'
                rect = Rectangle(
                    (row['bbox_x_min'], row['bbox_y_min']),
                    20,  # Assuming 20m tiles
                    20,
                    facecolor=color,
                    alpha=0.5,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(rect)
            ax.set_xlabel('X (RD)')
            ax.set_ylabel('Y (RD)')
            ax.set_title('Tile QA Status (Green=Pass, Red=Fail)')
            ax.set_aspect('equal')
            ax.autoscale_view()

        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Summary Statistics', fontweight='bold')

        stats_text = f"""
Total Tiles: {len(tiles_metadata)}
QA Approved: {tiles_metadata.get('density_ok', pd.Series([False])).sum()}
QA Review: {(~tiles_metadata.get('density_ok', pd.Series([True]))).sum()}

Density Stats:
  Mean: {tiles_metadata.get('density', pd.Series([0])).mean():.1f} pts/m²
  Min: {tiles_metadata.get('density', pd.Series([0])).min():.1f} pts/m²
  Max: {tiles_metadata.get('density', pd.Series([0])).max():.1f} pts/m²

Point Count:
  Total: {tiles_metadata.get('point_count', pd.Series([0])).sum():,}
  Mean per tile: {tiles_metadata.get('point_count', pd.Series([0])).mean():.0f}
        """

        ax.text(0.1, 0.9, stats_text, fontsize=11,
               verticalalignment='top', family='monospace')

        plt.tight_layout()

        output_path = self.output_dir / 'qa_overview.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    """Command-line entry point."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return 1

    parser = argparse.ArgumentParser(
        description="Visualize BEV tiles for QA"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing tile Parquet files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--tiles",
        type=int,
        default=10,
        help="Number of tiles to visualize (default: 10)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    visualizer = BEVVisualizer(output_dir)

    # Load metadata
    metadata_file = input_dir / "tiles_metadata.parquet"
    if metadata_file.exists():
        metadata = pd.read_parquet(metadata_file)
        print(f"Loaded metadata for {len(metadata)} tiles")

        # Create overview
        overview = visualizer.create_overview_report(metadata)
        print(f"Created overview: {overview}")
    else:
        print(f"Warning: Metadata file not found: {metadata_file}")

    # Visualize individual tiles
    tile_files = list(input_dir.glob("tile_*.parquet"))[:args.tiles]

    if not tile_files:
        print(f"No tile files found in {input_dir}")
        return 1

    print(f"Visualizing {len(tile_files)} tiles...")
    for tile_file in tile_files:
        tile_name = tile_file.stem
        output = visualizer.visualize_tile(tile_file, tile_name)
        print(f"  ✓ {output}")

    print(f"\nVisualization complete! Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
