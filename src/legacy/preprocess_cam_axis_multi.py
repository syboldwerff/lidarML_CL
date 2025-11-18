import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from common.lidar_io import load_laz_points


def load_axis_csv(path):
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("axis CSV moet kolommen 'x' en 'y' bevatten.")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float) if "z" in df.columns else None
    return x, y, z


def compute_axis_chainage(ax, ay):
    dx = np.diff(ax)
    dy = np.diff(ay)
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.zeros_like(ax)
    s[1:] = np.cumsum(ds)
    return s


def sample_axis(ax, ay, az, s, step):
    s_total = s[-1]
    if s_total <= 0:
        raise ValueError("Totale as-lengte is 0, kan niet sampelen.")

    sample_s = np.arange(0.0, s_total + step * 0.5, step)
    sample_s = sample_s[sample_s <= s_total]

    xs, ys, zs = [], [], []

    for sk in sample_s:
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
    tx_list, ty_list = [], []

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
            tx_list.append(1.0)
            ty_list.append(0.0)
        else:
            tx_list.append(dx / norm)
            ty_list.append(dy / norm)

    return np.array(tx_list), np.array(ty_list)


def make_virtual_cameras_along_axis_multi_laz(
    laz_paths,
    axis_csv,
    out_dir,
    img_width=1920,
    img_height=1080,
    fov_deg=60.0,
    step=10.0,
    search_radius=30.0,
):
    """
    Genereer virtuele camera-beelden langs een as over meerdere LAZ-bestanden.

    - laz_paths: lijst van LAZ-bestanden (Path/str)
    - axis_csv: CSV met kolommen x,y[,z] voor de weg-as
    - alle punten uit alle LAZ worden gecombineerd tot één wolk
    - om de 'step' meter langs de as wordt een camera-positie geplaatst
    - per positie 6 richtingen (0, ±60, ±120, 180 graden)
    """

    # Laad alle LAZ-bestanden en concateneer
    x_list, y_list, z_list = [], [], []
    r_list, g_list, b_list = [], [], []
    input_names = []

    for p in laz_paths:
        pts, header = load_laz_points(p)
        x_list.append(pts["x"])
        y_list.append(pts["y"])
        z_list.append(pts["z"])
        input_names.append(Path(p).stem)

        # RGB kan None zijn
        if pts["r"] is not None:
            r_list.append(pts["r"])
            g_list.append(pts["g"])
            b_list.append(pts["b"])

    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    z_all = np.concatenate(z_list)

    if r_list:
        r_all = np.concatenate(r_list)
        g_all = np.concatenate(g_list)
        b_all = np.concatenate(b_list)
    else:
        r_all = g_all = b_all = None

    # Input-naam voor output-bestanden (kan bv. combo zijn)
    input_name = Path(axis_csv).stem

    # As inlezen
    ax, ay, az = load_axis_csv(axis_csv)
    s = compute_axis_chainage(ax, ay)
    xs, ys, zs, sample_s = sample_axis(ax, ay, az, s, step)
    tx, ty = compute_axis_tangent(ax, ay, s, sample_s)

    # camera Z
    if zs is not None:
        cam_zs = zs + 2.0
    else:
        z_mean = np.mean(z_all)
        cam_zs = np.full_like(xs, z_mean + 2.0)

    # intrinsics
    fov_rad = np.deg2rad(fov_deg)
    fx = img_width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = img_width / 2.0
    cy = img_height / 2.0

    # 6 richtingen rondom as-tangent
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

    for i in tqdm(range(len(xs)), desc="Axis poses"):
        cam_x = xs[i]
        cam_y = ys[i]
        cam_z = cam_zs[i]

        tdx = tx[i]
        tdy = ty[i]
        base_yaw = np.arctan2(tdy, tdx)

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

        # relatieve coords t.o.v. camera
        dx = x - cam_x
        dy = y - cam_y
        dz = z - cam_z

        for dir_name, yaw_offset_deg in directions:
            yaw = base_yaw + np.deg2rad(yaw_offset_deg)

            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)

            Xc =  cos_y * dx + sin_y * dy
            Yc = -sin_y * dx + cos_y * dy
            Zc =  dz

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

            # projectie
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
            depth = Xv[valid]

            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            depth_img = np.full((img_height, img_width), np.inf, dtype=np.float32)

            mapping_rows = []

            order = np.argsort(depth)
            u_px = u_px[order]
            v_px = v_px[order]
            depth = depth[order]
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
                        "input_name": input_name,
                        "pose_idx": int(i),
                        "s": float(sample_s[i]),
                        "direction": dir_name,
                        "yaw_deg": float(np.rad2deg(yaw)),
                        "px": int(uu),
                        "py": int(vv),
                        "depth": float(d),
                    })

            if not mapping_rows:
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
    parser = argparse.ArgumentParser(
        description="Virtuele camera's langs as over meerdere LAZ-bestanden."
    )
    parser.add_argument(
        "--laz-dir",
        required=True,
        help="Directory met LAZ-bestanden (meerdere tiles van dezelfde regio)",
    )
    parser.add_argument(
        "--pattern",
        default="*.laz",
        help="Glob pattern voor LAZ (default: *.laz)",
    )
    parser.add_argument(
        "--axis-csv",
        required=True,
        help="Axis CSV (x,y[,z]) voor één weg-as",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory voor camera views",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--step", type=float, default=10.0, help="Afstand tussen poses (m)")
    parser.add_argument("--search-radius", type=float, default=30.0, help="Straal rondom camera (m)")
    args = parser.parse_args()

    laz_dir = Path(args.laz_dir).expanduser().resolve()
    laz_paths = sorted(laz_dir.glob(args.pattern))
    if not laz_paths:
        raise FileNotFoundError(f"Geen LAZ-bestanden gevonden in {laz_dir} met pattern {args.pattern}")

    print(f"Gevonden {len(laz_paths)} LAZ-bestanden in {laz_dir}")

    make_virtual_cameras_along_axis_multi_laz(
        laz_paths=[str(p) for p in laz_paths],
        axis_csv=args.axis_csv,
        out_dir=args.out_dir,
        img_width=args.width,
        img_height=args.height,
        fov_deg=args.fov_deg,
        step=args.step,
        search_radius=args.search_radius,
    )


if __name__ == "__main__":
    main()
