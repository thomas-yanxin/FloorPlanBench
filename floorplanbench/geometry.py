"""Metric drawing geometry, derived without snapping or changing model coordinates."""

from collections import defaultdict
from itertools import combinations
from math import dist, hypot

from shapely.geometry import GeometryCollection, LineString, Polygon
from shapely.ops import unary_union

from .schema import FloorPlan, Opening, valid_opening


def opening_frame(plan: FloorPlan, opening: Opening):
    wall = plan.walls[opening.wallIndex]
    length = dist(wall.start, wall.end)
    ux, uy = ((b - a) / length for a, b in zip(wall.start, wall.end, strict=True))
    a = (wall.start[0] + ux * opening.offset, wall.start[1] + uy * opening.offset)
    b = (a[0] + ux * opening.width, a[1] + uy * opening.width)
    return a, b, (-uy, ux)


def opening_cut(plan: FloorPlan, opening: Opening, thickness: float) -> Polygon:
    a, b, (nx, ny) = opening_frame(plan, opening)
    half = thickness / 2 + 1e-7
    return Polygon(
        [
            (a[0] + nx * half, a[1] + ny * half),
            (b[0] + nx * half, b[1] + ny * half),
            (b[0] - nx * half, b[1] - ny * half),
            (a[0] - nx * half, a[1] - ny * half),
        ]
    )


def wall_surface(plan: FloorPlan):
    """Unite wall solids with mitred junctions and flat, unextended free ends.

    Only exactly shared endpoints are joined. Near misses and old-format gaps
    remain evidence in the drawing instead of being snapped or filled in.
    """
    pieces = []
    junctions = defaultdict(list)
    for wall in plan.walls:
        if not wall.thickness or wall.thickness <= 0 or wall.start == wall.end:
            continue
        pieces.append(LineString([wall.start, wall.end]).buffer(wall.thickness / 2, cap_style=2))
        for a, b in ((wall.start, wall.end), (wall.end, wall.start)):
            length = dist(a, b)
            junctions[a].append(
                ((b[0] - a[0]) / length, (b[1] - a[1]) / length, wall.thickness / 2)
            )
    for (x, y), incident in junctions.items():
        for (ux, uy, half_u), (vx, vy, half_v) in combinations(incident, 2):
            cross = ux * vy - uy * vx
            if abs(cross) < 1e-8:
                continue
            for side in (-1, 1):
                # Intersect the two offset edges of the connected wall faces.
                px, py = side * uy * half_u, -side * ux * half_u
                qx, qy = -side * vy * half_v, side * vx * half_v
                along = ((qx - px) * vy - (qy - py) * vx) / cross
                ix, iy = px + along * ux, py + along * uy
                joint = [(x, y), (x + px, y + py)]
                if hypot(ix, iy) <= 4 * max(half_u, half_v):
                    joint.append((x + ix, y + iy))
                joint.append((x + qx, y + qy))
                pieces.append(Polygon(joint))
    surface = unary_union(pieces) if pieces else GeometryCollection()
    cuts = [
        opening_cut(plan, opening, plan.walls[opening.wallIndex].thickness)
        for opening in plan.openings
        if valid_opening(plan, opening) and (plan.walls[opening.wallIndex].thickness or 0) > 0
    ]
    return surface.difference(unary_union(cuts)) if cuts else surface


def polygon_path(geometry) -> str:
    """One SVG path, including holes; Plotly uses the even-odd fill rule."""
    if geometry.is_empty:
        return ""
    polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
    paths = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            paths.append("M " + " L ".join(f"{x:.8g},{y:.8g}" for x, y in ring.coords) + " Z")
    return " ".join(paths)
