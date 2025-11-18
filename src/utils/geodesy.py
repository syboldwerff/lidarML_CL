"""Geodesic utilities.

Provides functions for geodesic calculations such as the haversine
formula to compute distances between latitude/longitude coordinates.
"""

import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great‑circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 in degrees.
    lat2, lon2 : float
        Latitude and longitude of point 2 in degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_earth = 6371000.0
    return radius_earth * c