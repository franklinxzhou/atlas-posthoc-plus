"""Reock district compactness, following Greg's ``pairDownReock`` method."""

from math import isfinite, pi

from shapely import minimum_bounding_radius
from shapely.geometry.base import BaseGeometry


def reock_score(district: BaseGeometry) -> float:
    """Divide projected district area by the area of its smallest enclosing circle."""
    area = district.area
    if not isfinite(area) or area <= 0:
        raise ValueError("Reock requires a district with finite, positive area.")

    # GEOS finds the same circle as Greg's enclosing-circle helper on the convex hull.
    radius = float(minimum_bounding_radius(district))
    if not isfinite(radius) or radius <= 0:
        raise ValueError("Reock requires a finite, positive enclosing-circle radius.")
    return area / (pi * radius**2)
