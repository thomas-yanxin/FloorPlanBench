"""Interactive 2D reconstruction using Plotly, without a custom scene engine."""

import html
import re
from collections import Counter
from math import cos, dist, pi, sin

import plotly.graph_objects as go

from .geometry import opening_frame, polygon_path, wall_surface
from .labels import fit_label
from .schema import FloorPlan, room_polygon, valid_opening

PALETTE = ["#e8eef2", "#f1e9dc", "#e4eeea", "#ece8df"]
INK = "#424b50"


def room_label(name: str) -> str:
    """Short display names; the model's full, unchanged label stays in the hover text."""
    if re.search(r"[\u4e00-\u9fff]", name):
        for label in re.split(r"[（）()]", name):
            if re.search(r"[A-Za-z]", label) and not re.search(r"[\u4e00-\u9fff]", label):
                return label.strip()
    return name.strip()


def room_color(name: str, index: int) -> str:
    name = name.lower()
    for words, color in [
        (("卧", "bedroom"), "#f1e9dc"),
        (("卫生", "浴", "bath", "toilet"), "#e2ede9"),
        (("厨", "kitchen"), "#efe8db"),
        (("阳台", "balcony", "terrace"), "#e9eddc"),
        (("客", "餐", "living", "dining"), "#e8eef2"),
    ]:
        if any(word in name for word in words):
            return color
    return PALETTE[index % len(PALETTE)]


def draw_opening(fig, plan, opening, span):
    a, b, (nx, ny) = opening_frame(plan, opening)
    thickness = plan.walls[opening.wallIndex].thickness
    half = thickness / 2 if thickness and thickness > 0 else span * 0.003

    def line(points, color="#72817f", width=0.8):
        fig.add_shape(
            type="path",
            path="M " + " L ".join(f"{x},{y}" for x, y in points),
            line=dict(color=color, width=width),
            layer="above",
        )

    for x, y in (a, b):
        line([(x - nx * half, y - ny * half), (x + nx * half, y + ny * half)], INK)
    if opening.kind == "window":
        for offset in (-half, -half / 3, half / 3, half):
            line(
                [
                    (a[0] + nx * offset, a[1] + ny * offset),
                    (b[0] + nx * offset, b[1] + ny * offset),
                ],
                "#82979f",
                0.65,
            )
    elif opening.kind == "door" and opening.hinge and opening.swing:
        hinge, closed = (a, b) if opening.hinge == "start" else (b, a)
        side = 1 if opening.swing == "left" else -1
        dx, dy = closed[0] - hinge[0], closed[1] - hinge[1]
        ox, oy = nx * opening.width * side, ny * opening.width * side
        line([hinge, (hinge[0] + ox, hinge[1] + oy)], "#697975", 1)
        arc = [
            (hinge[0] + dx * cos(t) + ox * sin(t), hinge[1] + dy * cos(t) + oy * sin(t))
            for t in (i * pi / 64 for i in range(33))
        ]
        line(arc, "#9aa7a1", 0.65)


def plot_plan(plan: FloorPlan) -> go.Figure:
    fig = go.Figure()
    polygons = [room_polygon(room) for room in plan.rooms]
    points = [p for wall in plan.walls for p in (wall.start, wall.end)] + [
        p
        for room, polygon in zip(plan.rooms, polygons, strict=True)
        if polygon is not None
        for p in room.polygon
    ]
    span = (
        max((max(p[i] for p in points) - min(p[i] for p in points) for i in (0, 1)), default=1)
        if points
        else 1
    )
    span = max(span, 1e-3)
    labels = [room_label(room.name) for room in plan.rooms]
    counts = Counter(labels)
    seen = Counter()
    for index, room in enumerate(plan.rooms):
        pts = room.polygon
        polygon = polygons[index]
        if polygon is None:
            continue
        closed = pts + pts[:1]
        # Labels and model-supplied names are text, never HTML.
        name = html.escape(room.name)
        area = room.approximateAreaSqM
        area_text = f"≈ {area:g} m²" if area is not None and area > 0 else ""
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in closed],
                y=[p[1] for p in closed],
                fill="toself",
                fillcolor=room_color(room.name, index),
                line=dict(width=0),
                mode="lines",
                name=name,
                hoveron="fills",
                hovertemplate=f"<b>{name}</b><br>{area_text}<extra></extra>",
            )
        )
        label = labels[index]
        seen[label] += 1
        if counts[label] > 1:
            label += f" {seen[label]}"
        layout = fit_label(polygon, label, area, span)
        if layout:
            # A vector label in world units fits its polygon at every viewport and zoom level.
            fig.add_layout_image(
                source=layout.svg_source(),
                xref="x",
                yref="y",
                x=layout.x,
                y=layout.y,
                sizex=layout.width,
                sizey=layout.height,
                xanchor="center",
                yanchor="middle",
                sizing="contain",
                layer="above",
                name=html.escape(label),
            )
    surface = wall_surface(plan)
    if not surface.is_empty:
        fig.add_shape(
            type="path",
            path=polygon_path(surface),
            fillrule="evenodd",
            fillcolor=INK,
            line=dict(color=INK, width=0.35),
        )
    # Unknown thickness remains a display hairline rather than an invented metric wall.
    for index, wall in enumerate(plan.walls):
        a, b = wall.start, wall.end
        length = dist(a, b)
        if length == 0 or (wall.thickness or 0) > 0:
            continue
        intervals = sorted(
            (o.offset, o.offset + o.width)
            for o in plan.openings
            if o.wallIndex == index and valid_opening(plan, o)
        )
        cursor = 0
        for start, end in [*intervals, (length, length)]:
            if start > cursor:
                start_point = [a[i] + (b[i] - a[i]) * cursor / length for i in (0, 1)]
                end_point = [a[i] + (b[i] - a[i]) * start / length for i in (0, 1)]
                fig.add_shape(
                    type="line",
                    x0=start_point[0],
                    y0=start_point[1],
                    x1=end_point[0],
                    y1=end_point[1],
                    line=dict(color=INK, width=1.5),
                )
            cursor = max(cursor, end)
    for opening in plan.openings:
        if valid_opening(plan, opening):
            draw_opening(fig, plan, opening, span)
    # Frame the physical wall faces, not just their centrelines.
    if not surface.is_empty:
        x0, y0, x1, y1 = surface.bounds
        points.extend([(x0, y0), (x1, y1)])
    if points:
        xs, ys = zip(*points, strict=True)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
        pad = span * 0.045
        fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad])
        fig.update_yaxes(range=[min(ys) - span * 0.12, max(ys) + pad])
        scale = next((n for n in (10, 5, 2, 1, 0.5, 0.2, 0.1) if n <= span * 0.18), 0.1)
        scale_y = min(ys) - span * 0.065
        fig.add_shape(
            type="line",
            x0=min(xs),
            x1=min(xs) + scale,
            y0=scale_y,
            y1=scale_y,
            line=dict(color="#7d868b", width=1),
        )
        for x in (min(xs), min(xs) + scale):
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=scale_y - span * 0.006,
                y1=scale_y + span * 0.006,
                line=dict(color="#7d868b", width=1),
            )
        fig.add_annotation(
            x=min(xs) + scale / 2,
            y=scale_y - span * 0.022,
            text=f"{scale:g} m",
            showarrow=False,
            font=dict(size=10, color="#69747c"),
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        dragmode="pan",
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", color="#6d7a8b"),
        hoverlabel=dict(bgcolor="white", bordercolor="#dce3e5", font_size=12),
        modebar_remove=["select2d", "lasso2d", "autoScale2d"],
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title=None,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        title=None,
    )
    return fig
