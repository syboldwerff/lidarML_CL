"""QA visualizer for camera views.

This module provides visualization tools to inspect camera-projected
point cloud views and validate quality.  It shows point projections,
intensity overlays, and QA metrics.

Usage:
    python -m src.preprocessing.qa_cam_visualizer --input views_dir/ --output qa_vis/
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class CameraViewVisualizer:
    """Visualize camera views for QA purposes."""

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

    def create_placeholder_view_image(
        self,
        width: int = 1024,
        height: int = 768,
        view_name: str = "Front"
    ) -> np.ndarray:
        """Create placeholder camera view image.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        view_name : str
            Name of the view (Front, Left, Right, etc.)

        Returns
        -------
        np.ndarray
            RGB image array.
        """
        # For demonstration, create a synthetic view
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add gradient to simulate depth
        for i in range(height):
            value = int(255 * (1.0 - i / height))
            image[i, :] = [value // 3, value // 2, value]

        return image

    def visualize_camera_qa(
        self,
        view_name: str,
        qa_metrics: Dict[str, float],
        point_count: int,
        image: np.ndarray = None
    ) -> Path:
        """Create camera view QA visualization.

        Parameters
        ----------
        view_name : str
            View identifier (e.g., "view_front_0001").
        qa_metrics : dict
            QA metrics for this view.
        point_count : int
            Number of points in view.
        image : np.ndarray, optional
            Camera view image. If None, creates placeholder.

        Returns
        -------
        Path
            Path to output visualization.
        """
        if image is None:
            image = self.create_placeholder_view_image(view_name=view_name)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Camera View QA: {view_name}', fontsize=14, fontweight='bold')

        # Camera view image
        ax = axes[0]
        ax.imshow(image)
        ax.set_title('Camera View Projection')
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')
        ax.grid(True, alpha=0.3)

        # Add point count overlay
        ax.text(0.02, 0.98, f'Points: {point_count:,}',
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # QA metrics panel
        ax = axes[1]
        ax.axis('off')
        ax.set_title('QA Metrics', fontweight='bold')

        y_pos = 0.9
        for key, value in qa_metrics.items():
            if isinstance(value, bool):
                color = 'green' if value else 'red'
                symbol = '✓' if value else '✗'
                ax.text(0.1, y_pos, f'{symbol} {key}: {value}',
                       fontsize=11, color=color, verticalalignment='top')
            else:
                ax.text(0.1, y_pos, f'{key}: {value:.2f}',
                       fontsize=11, verticalalignment='top')
            y_pos -= 0.08

        # Add general info
        y_pos -= 0.05
        ax.text(0.1, y_pos, f'Total Points: {point_count:,}',
               fontsize=10, verticalalignment='top',
               fontweight='bold')

        plt.tight_layout()

        output_path = self.output_dir / f'{view_name}_qa.png'
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

        return output_path

    def create_overview_report(
        self,
        views_summary: Dict[str, Dict]
    ) -> Path:
        """Create overview report for all camera views.

        Parameters
        ----------
        views_summary : dict
            Summary statistics for all views.

        Returns
        -------
        Path
            Path to overview report.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Camera Views QA Overview', fontsize=16, fontweight='bold')

        # Point count distribution
        ax = axes[0, 0]
        view_names = list(views_summary.keys())
        point_counts = [views_summary[v]['point_count'] for v in view_names]

        ax.bar(range(len(view_names)), point_counts, alpha=0.7, edgecolor='black')
        ax.set_xlabel('View Index')
        ax.set_ylabel('Point Count')
        ax.set_title('Points per View')
        ax.set_xticks(range(len(view_names)))
        ax.set_xticklabels([f'V{i}' for i in range(len(view_names))], rotation=45)

        # QA pass/fail summary
        ax = axes[0, 1]
        # Collect QA status
        qa_pass = sum(1 for v in views_summary.values() if v.get('qa_ok', False))
        qa_fail = len(views_summary) - qa_pass

        ax.pie([qa_pass, qa_fail],
               labels=['Pass', 'Fail'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title('QA Pass/Fail Distribution')

        # View direction distribution (if available)
        ax = axes[1, 0]
        directions = {}
        for name, data in views_summary.items():
            direction = data.get('direction', 'unknown')
            directions[direction] = directions.get(direction, 0) + 1

        if directions:
            ax.bar(directions.keys(), directions.values(), alpha=0.7, edgecolor='black')
            ax.set_xlabel('View Direction')
            ax.set_ylabel('Count')
            ax.set_title('Views by Direction')
            ax.tick_params(axis='x', rotation=45)

        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Summary Statistics', fontweight='bold')

        total_points = sum(point_counts)
        mean_points = np.mean(point_counts)

        stats_text = f"""
Total Views: {len(views_summary)}
QA Approved: {qa_pass}
QA Review: {qa_fail}

Point Statistics:
  Total Points: {total_points:,}
  Mean per View: {mean_points:.0f}
  Min: {min(point_counts):,}
  Max: {max(point_counts):,}
        """

        ax.text(0.1, 0.9, stats_text, fontsize=11,
               verticalalignment='top', family='monospace')

        plt.tight_layout()

        output_path = self.output_dir / 'camera_qa_overview.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    """Command-line entry point."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return 1

    parser = argparse.ArgumentParser(
        description="Visualize camera views for QA"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo visualizations"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    visualizer = CameraViewVisualizer(output_dir)

    if args.demo:
        print("Creating demo camera view visualizations...")

        # Create demo views
        views_summary = {}

        for i, direction in enumerate(['front', 'front_left', 'front_right',
                                       'back', 'back_left', 'back_right']):
            view_name = f"view_{direction}_{i:04d}"

            qa_metrics = {
                'depth_ok': np.random.choice([True, False], p=[0.8, 0.2]),
                'point_density': np.random.uniform(50, 200),
                'coverage': np.random.uniform(0.6, 0.95),
            }

            point_count = int(np.random.uniform(1000, 5000))

            # Create visualization
            output = visualizer.visualize_camera_qa(
                view_name,
                qa_metrics,
                point_count
            )
            print(f"  ✓ {output}")

            views_summary[view_name] = {
                'point_count': point_count,
                'qa_ok': qa_metrics['depth_ok'],
                'direction': direction
            }

        # Create overview
        overview = visualizer.create_overview_report(views_summary)
        print(f"\nCreated overview: {overview}")
        print(f"\nDemo visualization complete! Output: {output_dir}")

    else:
        print("No input data provided. Use --demo for demonstration, or provide actual camera view data.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
