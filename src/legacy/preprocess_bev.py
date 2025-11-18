import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from common.lidar_io import load_laz_points


def make_bev_for_file(
    input_path,
    out_dir,
    resolution=0.05,   # meters per pixel
    tile_size=100.0,   # meters
):
    """
    Maak BEV-tiles met 6 kanalen per pixel:

      0: z_max       (max Z per pixel, meter)
      1: int_mean    (gemiddelde intensiteit)
      2: r_mean
      3: g_mean
      4: b_mean
      5: density     (aantal punten in pixel)

    Output per tile:
      - PNG voor visualisatie (RGB)
      - .npz met array 'bev' shape (6, H, W), float32
      - mapping parquet met punt-index en pixelpositie
    """
    pts, header = load_laz_points(input_path)
    x, y, z = pts["x"], pts["y"], pts["z"]
    intensity = pts["intensity"]
    r, g, b = pts["r"], pts["g"], pts["b"]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    width = xmax - xmin
    height = ymax - ymin

    nx_tiles = int(np.ceil(width / tile_size))
    ny_tiles = int(np.ceil(height / tile_size))

    input_name = Path(input_path).stem

    mapping_rows = []

    for ix in tqdm(range(nx_tiles), desc=f"Tiles X for {input_name}"):
        for iy in range(ny_tiles):
            tile_xmin = xmin + ix * tile_size
            tile_xmax = tile_xmin + tile_size
            tile_ymin = ymin + iy * tile_size
            tile_ymax = tile_ymin + tile_size

            # selecteer punten in deze tile
            mask = (
                (x >= tile_xmin) & (x < tile_xmax) &
                (y >= tile_ymin) & (y < tile_ymax)
            )
            if not np.any(mask):
                continue

            x_t = x[mask]
            y_t = y[mask]
            z_t = z[mask]
            i_t = intensity[mask] if intensity is not None else None
            r_t = r[mask] if r is not None else None
            g_t = g[mask] if g is not None else None
            b_t = b[mask] if b is not None else None

            # grid dim
            gw = int(np.ceil(tile_size / resolution))
            gh = int(np.ceil(tile_size / resolution))

            # 6 kanalen + teller
            z_max = np.full((gh, gw), -9999.0, dtype=np.float32)
            int_sum = np.zeros((gh, gw), dtype=np.float32)
            r_sum = np.zeros((gh, gw), dtype=np.float32)
            g_sum = np.zeros((gh, gw), dtype=np.float32)
            b_sum = np.zeros((gh, gw), dtype=np.float32)
            count = np.zeros((gh, gw), dtype=np.int32)

            # pixel indices (x → rechts, y → omhoog; maar img-index y groeit naar beneden)
            u = ((x_t - tile_xmin) / resolution).astype(int)  # kolom
            v = ((y_t - tile_ymin) / resolution).astype(int)  # rij in “wiskunde-y”

            # clip
            u = np.clip(u, 0, gw - 1)
            v = np.clip(v, 0, gh - 1)

            # north-up: grotere Y = noord = omhoog in kaart → in beeld is dat kleinere row-index
            v_flipped = (gh - 1) - v

            orig_indices = np.nonzero(mask)[0]

            # vul rasters + mapping
            for idx in range(len(x_t)):
                uu = u[idx]
                vv = v_flipped[idx]

                zz = z_t[idx]
                if zz > z_max[vv, uu]:
                    z_max[vv, uu] = zz

                if i_t is not None:
                    int_sum[vv, uu] += i_t[idx]
                if r_t is not None:
                    r_sum[vv, uu] += r_t[idx]
                    g_sum[vv, uu] += g_t[idx]
                    b_sum[vv, uu] += b_t[idx]

                count[vv, uu] += 1

                mapping_rows.append({
                    "input_file": input_name,
                    "tile_ix": ix,
                    "tile_iy": iy,
                    "point_index": int(orig_indices[idx]),
                    "px": int(uu),
                    "py": int(vv),
                })

            valid = count > 0

            # gemiddelde intensiteit & RGB
            int_mean = np.zeros_like(int_sum, dtype=np.float32)
            r_mean = np.zeros_like(r_sum, dtype=np.float32)
            g_mean = np.zeros_like(g_sum, dtype=np.float32)
            b_mean = np.zeros_like(b_sum, dtype=np.float32)

            if np.any(valid):
                # alleen delen waar count > 0
                int_mean[valid] = int_sum[valid] / count[valid]
                r_mean[valid] = r_sum[valid] / count[valid]
                g_mean[valid] = g_sum[valid] / count[valid]
                b_mean[valid] = b_sum[valid] / count[valid]

            # density = count
            density = count.astype(np.float32)

            # z_max: maak no-data 0, laat rest als hoogte in meters
            z_max[~valid] = 0.0

            # ---------- PNG voor visualisatie ----------
            # simpele normalisatie per tile:
            if np.any(valid):
                z_valid = z_max[valid]
                z_min_tile = float(np.min(z_valid))
                z_max_tile = float(np.max(z_valid))
                denom = max((z_max_tile - z_min_tile), 1e-3)
                z_norm = (z_max - z_min_tile) / denom
            else:
                z_norm = np.zeros_like(z_max, dtype=np.float32)

            z_img = (np.clip(z_norm, 0.0, 1.0) * 255.0).astype(np.uint8)

            if i_t is not None and np.any(valid):
                i_img = int_mean.copy()
                i_img_valid = i_img[valid]
                imax = float(np.max(i_img_valid)) if i_img_valid.size > 0 else 1.0
                if imax <= 0:
                    imax = 1.0
                i_img = i_img / imax
                i_img[~valid] = 0.0
                i_img = (np.clip(i_img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                i_img = z_img.copy()

            if r_t is not None and np.any(valid):
                r_vis = r_mean.copy()
                g_vis = g_mean.copy()
                b_vis = b_mean.copy()

                # veronderstel 0–65535 in LAS → schalen naar 0–1 per kanaal
                for arr in (r_vis, g_vis, b_vis):
                    vmax = float(np.max(arr[valid])) if np.any(valid) else 1.0
                    if vmax <= 0:
                        vmax = 1.0
                    arr /= vmax
                    arr[~valid] = 0.0

                r_img = (np.clip(r_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
                g_img = (np.clip(g_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
                b_img = (np.clip(b_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                # fallback: pseudo-RGB op basis van hoogte/intensiteit
                r_img = z_img.copy()
                g_img = i_img.copy()
                b_img = z_img.copy()

            bev_bgr = cv2.merge([b_img, g_img, r_img])

            tile_base = f"{input_name}_tx{ix}_ty{iy}"
            tile_png_path = Path(out_dir) / f"{tile_base}.png"
            tile_png_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tile_png_path), bev_bgr)

            # ---------- 6-kanaals BEV tensor opslaan ----------
            bev_stack = np.stack(
                [
                    z_max,        # 0: hoogte (m)
                    int_mean,     # 1: intensiteit
                    r_mean,       # 2: R
                    g_mean,       # 3: G
                    b_mean,       # 4: B
                    density,      # 5: density (#points)
                ],
                axis=0,  # shape (6, H, W)
            ).astype(np.float32)

            tile_npz_path = Path(out_dir) / f"{tile_base}_bev6.npz"
            np.savez(tile_npz_path, bev=bev_stack)
            # eventueel later extra meta meegooien: np.savez(tile_npz_path, bev=bev_stack, resolution=resolution, ...)

    # mapping voor alle tiles in één file
    mapping_df = pd.DataFrame(mapping_rows)
    map_path = Path(out_dir) / f"{input_name}_bev_mapping.parquet"
    mapping_df.to_parquet(map_path, index=False)
    print(f"Saved BEV mapping: {map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input LAZ file")
    parser.add_argument("--out-dir", required=True, help="Output dir for BEV tiles")
    parser.add_argument("--resolution", type=float, default=0.05, help="meters per pixel")
    parser.add_argument("--tile-size", type=float, default=100.0, help="tile size in meters")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_bev_for_file(
        args.input,
        args.out_dir,
        resolution=args.resolution,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
