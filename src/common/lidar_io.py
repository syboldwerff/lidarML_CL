import laspy
from pathlib import Path

def load_laz_points(path):
    path = Path(path)
    with laspy.open(path) as f:
        las = f.read()

    x = las.x
    y = las.y
    z = las.z

    intensity = getattr(las, "intensity", None)

    r = getattr(las, "red", None)
    g = getattr(las, "green", None)
    b = getattr(las, "blue", None)

    pts = {
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "r": r,
        "g": g,
        "b": b,
    }

    return pts, las.header
