import argparse
from pathlib import Path

from tqdm import tqdm

# we gebruiken de bestaande as-based functie
from preprocess_cam import make_virtual_cameras_along_axis


def main():
    parser = argparse.ArgumentParser(
        description="Batch camera-preprocessing langs as voor meerdere LAZ-bestanden."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory met .laz bestanden",
    )
    parser.add_argument(
        "--axis-dir",
        required=True,
        help="Directory met axis CSV bestanden (zelfde basename als LAZ)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory voor camera views",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="beeldbreedte (px, default 1920)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="beeldhoogte (px, default 1080)",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=60.0,
        help="Field of View in graden (default 60)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Afstand tussen camera-poses langs as (meter, default 10)",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=30.0,
        help="Straal rond de camera waar punten worden meegenomen (meter, default 30)",
    )
    parser.add_argument(
        "--pattern",
        default="*.laz",
        help="Glob pattern voor input (default: *.laz)",
    )
    parser.add_argument(
        "--axis-suffix",
        default="_axis.csv",
        help="Suffix voor axis-bestanden (default: _axis.csv)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    axis_dir = Path(args.axis_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(in_dir.glob(args.pattern))
    if not laz_files:
        raise FileNotFoundError(f"Geen LAZ-bestanden gevonden in {in_dir} met pattern {args.pattern}")

    print(f"Gevonden {len(laz_files)} LAZ-bestanden in {in_dir}")

    for laz_path in tqdm(laz_files, desc="CAM batch"):
        input_name = laz_path.stem
        axis_path = axis_dir / f"{input_name}{args.axis_suffix}"

        if not axis_path.exists():
            print(f"[SKIP] Geen axis CSV gevonden voor {laz_path.name}: verwacht {axis_path.name}")
            continue

        print(f"\n[CAM] Verwerken: {laz_path.name} met as {axis_path.name}")
        try:
            make_virtual_cameras_along_axis(
                input_path=str(laz_path),
                axis_csv=str(axis_path),
                out_dir=str(out_dir),
                img_width=args.width,
                img_height=args.height,
                fov_deg=args.fov_deg,
                step=args.step,
                search_radius=args.search_radius,
            )
        except Exception as e:
            print(f"[ERROR] Fout bij {laz_path.name}: {e}")


if __name__ == "__main__":
    main()
