import argparse
from pathlib import Path

from tqdm import tqdm

# we gebruiken de bestaande single-file functie
from preprocess_bev import make_bev_for_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch BEV-preprocessing voor meerdere LAZ-bestanden."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory met .laz bestanden",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory voor BEV tiles",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Meters per pixel (default 0.05)",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=100.0,
        help="Tile grootte in meters (default 100)",
    )
    parser.add_argument(
        "--pattern",
        default="*.laz",
        help="Glob pattern voor input (default: *.laz)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(in_dir.glob(args.pattern))
    if not laz_files:
        raise FileNotFoundError(f"Geen LAZ-bestanden gevonden in {in_dir} met pattern {args.pattern}")

    print(f"Gevonden {len(laz_files)} LAZ-bestanden in {in_dir}")

    for laz_path in tqdm(laz_files, desc="BEV batch"):
        print(f"\n[BEV] Verwerken: {laz_path.name}")
        try:
            make_bev_for_file(
                input_path=str(laz_path),
                out_dir=str(out_dir),
                resolution=args.resolution,
                tile_size=args.tile_size,
            )
        except Exception as e:
            print(f"[ERROR] Fout bij {laz_path.name}: {e}")


if __name__ == "__main__":
    main()
