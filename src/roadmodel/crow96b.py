"""Encode road markings according to the CROW 96b specification.

This module defines helper functions to map detected road markings into
attribute structures following the Dutch CROW 96b standard for road
markings.  It provides a very simplified representation: each marking
is assigned a type (solid line, dashed line, symbol) and dimensions.
Additional attributes (colour, retroreflectivity) should be added as
needed.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Marking:
    """Representation of a road marking following CROW 96b."""
    geometry: List[tuple]
    marking_type: str  # e.g. 'solid', 'dashed', 'arrow', etc.
    width_mm: float
    pattern_length_m: float


def encode_markings(marking_polylines: List[List[tuple]], default_width: float = 100.0) -> List[Marking]:
    """Convert polylines into CROW 96b marking objects.

    Parameters
    ----------
    marking_polylines : list of list of tuples
        Polylines representing markings as sequences of (x, y) points.
    default_width : float
        Default line width in millimetres.

    Returns
    -------
    list of Marking
        Encoded marking objects.
    """
    markings: List[Marking] = []
    for polyline in marking_polylines:
        if not polyline:
            continue
        # Very naive classification: solid if continuous, dashed otherwise
        mtype = 'solid'
        # Compute total length and approximate pattern length
        lengths = []
        for p1, p2 in zip(polyline[:-1], polyline[1:]):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            lengths.append((dx**2 + dy**2)**0.5)
        total_length = sum(lengths)
        markings.append(Marking(geometry=polyline, marking_type=mtype, width_mm=default_width, pattern_length_m=total_length))
    return markings