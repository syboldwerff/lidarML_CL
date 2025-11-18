import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

from common.lidar_io import load_laz_points


def load_axis_csv(path):
    """
    Lees een as uit een CSV met minimaal kolommen: x,y
    Optioneel mag er een kolom z in zitten.
    """
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("axis CSV moet kolommen 'x' en 'y' bevatten.")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float) if "z" in df.columns else None
    return x, y, z


def compute_axis_chainage(ax, ay):
    """
    Bereken cumulatieve afstand (chainage) langs de as.
    ax, ay: 1D arrays met as-coördinaten.
    Retourneert: s (chainage per as-vertex)
    """
    dx = np.diff(ax)
    dy = np.diff(ay)
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.zeros_like(ax)
    s[1:] = np.cumsum(ds)
    return s


def sample_axis(ax, ay, az, s, step):
    """
    Sample as op een vaste interval 'step' (meters).
    Retourneert:
      xs, ys, zs, ss (sample-coords en chainages)
    """
    s_total = s[-1]
    if s_total <= 0:
        raise ValueError("Totale as-lengte is 0, kan niet sampelen.")

    sample_s = np.arange(0.0, s_total + step * 0.5, step)
    sample_s = sample_s[sample_s <= s_total]

    xs = []
    ys = []
    zs = []

    for sk in sample_s:
        # segment zoeken waarin sk ligt
        idx = np.searchsorted(s, sk) - 1
        if idx < 0:
            idx = 0
        if idx >= len(s) - 1:
            idx = len(s) - 2

        s0 = s[idx]
        s1 = s[idx + 1]
        t = 0.0 if s1 == s0 else (sk - s0) / (s1 - s0)

        x0, x1 = ax[idx], ax[idx + 1]
        y0, y1 = ay[idx], ay[idx + 1]

        xk = (1.0 - t) * x0 + t * x1
        yk = (1.0 - t) * y0 + t * y1

        if az is not None:
            z0, z1 = az[idx], az[idx + 1]
            zk = (1.0 - t) * z0 + t * z1
        else:
            zk = None

        xs.append(xk)
        ys.append(yk)
        zs.append(zk)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs) if az is not None else None

    return xs, ys, zs, sample_s


def compute_axis_tangent(ax, ay, s, sample_s):
    """
    Bepaal voor elke sample chainage een tangentrichting (dx, dy).
    Gebruik nabije segment.
    Retourneert arrays tx, ty (genormaliseerde tangent).
    """
    tx_list = []
    ty_list = []

    for sk in sample_s:
        idx = np.searchsorted(s, sk) - 1
        if idx < 0:
            idx = 0
        if idx >= len(s) - 1:
            idx = len(s) - 2

        dx = ax[idx + 1] - ax[idx]
        dy = ay[idx + 1] - ay[idx]
        norm = np.hypot(dx, dy)
        if norm == 0:
            # fallback: als segment lengte 0 is, zoek andere
            tx_list.append(1.0)
            ty_list.append(0.0)
        else:
            tx_list.append(dx / norm)
            ty_list.append(dy / norm)

    return np.array(tx_list), np.array(ty_list)


def make_virtual_cameras_along_axis(
    input_path,
    axis_csv,
    out_dir,
    img_width=1920,
    img_height=1080,
    fov_deg=60.0,
    step=10.0,
    search_radius=30.0,
):
    """
    Genereer virtuele camera-beelden langs een aangeleverde as.

    - As: axis_csv (x,y[,z]) in zelfde CRS als pointcloud.
    - Om de 'step' meters een camera-positie.
    - Voor elke camera-positie 6 richtingen van 60° rondom:
        front       : yaw_offset =   0°
        front_right : yaw_offset = -60°
        back_right  : yaw_offset = -120°
        back        : yaw_offset =  180°
        back_left   : yaw_offset = +120°
        front_left  : yaw_offset =  +60°
    - search_radius: alleen punten binnen deze straal rond de camera worden gebruikt.
    """

    # Laad puntenwolk
    pts, header = load_laz_points(input_path)
    x_all, y_all, z_all = pts["x"], pts["y"], pts["z"]
    r_all, g_all, b_all = pts["r"], pts["g"], pts["b"]

    input_name = Path(input_path).stem

    # Laad as
    ax, ay, az = load_axis_csv(axis_csv)
    s = compute_axis_chainage(ax, ay)
    xs, ys, zs, sample_s = sample_axis(ax, ay, az, s, step)
    tx, ty = compute_axis_tangent(ax, ay, s, sample_s)

    # Hoogte voor camera: als as z heeft, gebruik die + offset,
    # anders gemiddelde z van puntenwolk + offset.
    if zs is not None:
        cam_zs = zs + 2.0
    else:
        z_mean = np.mean(z_all)
        cam_zs = np.full_like(xs, z_mean + 2.0)

    # camera intrinsics (pinhole)
    fov_rad = np.deg2rad(fov_deg)
    fx = img_width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = img_width / 2.0
    cy = img_height / 2.0

    # Richtingen rondom (t.o.v. as-tangent)
    directions = [
        ("front",        0.0),
        ("front_right", -60.0),
        ("back_right",  -120.0),
        ("back",        180.0),
        ("back_left",   120.0),
        ("front_left",   60.0),
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Voor elke sample langs de as
    for i in range(len(xs)):
        cam_x = xs[i]
        cam_y = ys[i]
        cam_z = cam_zs[i]

        # as-tangent richting bij deze positie
        tdx = tx[i]
        tdy = ty[i]

        # yaw van de as t.o.v. wereld (X,Y)
        # yaw=0 als as langs +X loopt
        base_yaw = np.arctan2(tdy, tdx)  # rad

        # selecteer alleen punten binnen search_radius rond de camera
        dx_all = x_all - cam_x
        dy_all = y_all - cam_y
        dz_all = z_all - cam_z
        dist2 = dx_all * dx_all + dy_all * dy_all
        mask = dist2 <= (search_radius * search_radius)
        if not np.any(mask):
            print(f"[pose {i}] Geen punten binnen search_radius, overslaan.")
            continue

        x = x_all[mask]
        y = y_all[mask]
        z = z_all[mask]
        if r_all is not None:
            r = r_all[mask]
            g = g_all[mask]
            b = b_all[mask]
        else:
            r = g = b = None
        orig_indices_all = np.nonzero(mask)[0]

        # relatieve coords t.o.v. camera
        dx = x - cam_x
        dy = y - cam_y
        dz = z - cam_z

        for dir_name, yaw_offset_deg in directions:
            yaw = base_yaw + np.deg2rad(yaw_offset_deg)

            # R_yaw (rotatie rond Z-as)
            # We willen Xc = "vooruit", Yc = "rechts", Zc = "omhoog"
            # Xc =  cos(yaw)*dx + sin(yaw)*dy
            # Yc = -sin(yaw)*dx + cos(yaw)*dy
            # Zc =  dz
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)

            Xc =  cos_y * dx + sin_y * dy
            Yc = -sin_y * dx + cos_y * dy
            Zc =  dz

            # alleen punten vóór de camera
            front_mask = Xc > 0
            if not np.any(front_mask):
                print(f"[pose {i} dir {dir_name}] Geen punten voor camera, overslaan.")
                continue

            Xv = Xc[front_mask]
            Yv = Yc[front_mask]
            Zv = Zc[front_mask]
            if r is not None:
                rv = r[front_mask]
                gv = g[front_mask]
                bv = b[front_mask]
            else:
                rv = gv = bv = None
            orig_indices = orig_indices_all[front_mask]

            # projectie naar beeld
            u = fx * (Yv / Xv) + cx
            v = fy * (-Zv / Xv) + cy

            u_px = np.round(u).astype(int)
            v_px = np.round(v).astype(int)

            valid = (
                (u_px >= 0) & (u_px < img_width) &
                (v_px >= 0) & (v_px < img_height)
            )
            if not np.any(valid):
                print(f"[pose {i} dir {dir_name}] Geen punten in beeldveld, overslaan.")
                continue

            u_px = u_px[valid]
            v_px = v_px[valid]
            if rv is not None:
                rv = rv[valid]
                gv = gv[valid]
                bv = bv[valid]
            orig_indices = orig_indices[valid]
            depth = Xv[valid]  # diepte langs kijkrichting

            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            depth_img = np.full((img_height, img_width), np.inf, dtype=np.float32)

            mapping_rows = []

            # z-buffering: dichtbij eerst
            order = np.argsort(depth)
            u_px = u_px[order]
            v_px = v_px[order]
            depth = depth[order]
            orig_indices = orig_indices[order]
            if rv is not None:
                rv = rv[order]
                gv = gv[order]
                bv = bv[order]

            for idx in range(len(u_px)):
                uu = u_px[idx]
                vv = v_px[idx]
                d = depth[idx]

                if d < depth_img[vv, uu]:
                    depth_img[vv, uu] = d
                    if rv is not None:
                        R = rv[idx]
                        G = gv[idx]
                        B = bv[idx]
                        max_rgb = max(R, G, B, 1)
                        if max_rgb > 255:
                            scale = 255.0 / max_rgb
                            R = int(R * scale)
                            G = int(G * scale)
                            B = int(B * scale)
                        img[vv, uu, :] = [B, G, R]
                    else:
                        depth_norm = np.clip(d / 100.0, 0, 1)
                        val = int((1.0 - depth_norm) * 255)
                        img[vv, uu, :] = [val, val, val]

                    mapping_rows.append({
                        "input_file": input_name,
                        "pose_idx": int(i),
                        "s": float(sample_s[i]),
                        "direction": dir_name,
                        "yaw_deg": float(np.rad2deg(yaw)),
                        "point_index": int(orig_indices[idx]),
                        "px": int(uu),
                        "py": int(vv),
                        "depth": float(d),
                    })

            if len(mapping_rows) == 0:
                print(f"[pose {i} dir {dir_name}] Geen zichtbare punten na z-buffer, overslaan.")
                continue

            img_name = f"{input_name}_s{int(sample_s[i]):06d}_{dir_name}.png"
            img_path = out_dir / img_name
            cv2.imwrite(str(img_path), img)
            print(f"Saved virtual camera image: {img_path}")

            map_df = pd.DataFrame(mapping_rows)
            map_path = out_dir / f"{input_name}_s{int(sample_s[i]):06d}_{dir_name}_mapping.parquet"
            map_df.to_parquet(map_path, index=False)
            print(f"Saved camera mapping: {map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input LAZ file")
    parser.add_argument("--axis-csv", required=True, help="Axis CSV met kolommen x,y[,z]")
    parser.add_argument("--out-dir", required=True, help="Output dir for camera images")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--step", type=float, default=10.0, help="Afstand tussen camera-poses (m)")
    parser.add_argument("--search-radius", type=float, default=30.0, help="Straal rond camera (m)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    make_virtual_cameras_along_axis(
        args.input,
        args.axis_csv,
        args.out_dir,
        img_width=args.width,
        img_height=args.height,
        fov_deg=args.fov_deg,
        step=args.step,
        search_radius=args.search_radius,
    )


if __name__ == "__main__":
    main()
