"""Interactive 2D reconstruction using Plotly, without a custom scene engine."""

import html
import re
import textwrap
from collections import Counter

import plotly.graph_objects as go
from shapely.ops import polylabel

from .schema import FloorPlan, room_polygon

PALETTE = ["#e8eef2", "#f1e9dc", "#e4eeea", "#ece8df"]
INK = "#39434d"


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


def plot_plan(plan: FloorPlan) -> go.Figure:
    fig = go.Figure()
    labels = [room_label(room.name) for room in plan.rooms]
    counts = Counter(labels)
    seen = Counter()
    for index, room in enumerate(plan.rooms):
        pts = room.polygon
        polygon = room_polygon(room)
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
                line=dict(color="rgba(57,67,77,0.12)", width=0.7),
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
        label = html.escape(label)
        if not re.search(r"[\u4e00-\u9fff]", label):
            label = "<br>".join(textwrap.wrap(label, width=14))
        label_point = polylabel(polygon, tolerance=0.03)
        label_radius = label_point.distance(polygon.boundary)
        text = f"<b>{label}</b>"
        if area_text and label_radius >= 1.35:
            display_area = f"{area:.1f}".removesuffix(".0")
            text += f'<br><span style="font-size:10px;color:#65716d">≈ {display_area} m²</span>'
        fig.add_annotation(
            x=label_point.x,
            y=label_point.y,
            text=text,
            showarrow=False,
            font=dict(size=11 if label_radius < 1.35 else 12, color="#34413f"),
            align="center",
            borderpad=0,
        )
    # Scatter line widths are pixels. Plotly shapes display the actual wall thickness in metres.
    for wall in plan.walls:
        a, b = wall.start, wall.end
        length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        if length == 0:
            continue
        if wall.thickness and wall.thickness > 0:
            dx = -(b[1] - a[1]) / length * wall.thickness / 2
            dy = (b[0] - a[0]) / length * wall.thickness / 2
            pts = [
                (a[0] + dx, a[1] + dy),
                (b[0] + dx, b[1] + dy),
                (b[0] - dx, b[1] - dy),
                (a[0] - dx, a[1] - dy),
            ]
            fig.add_shape(
                type="path",
                path="M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z",
                fillcolor=INK,
                line=dict(width=0),
            )
        else:
            fig.add_shape(
                type="line",
                x0=a[0],
                y0=a[1],
                x1=b[0],
                y1=b[1],
                line=dict(color=INK, width=3),
            )
    # Always use coordinates rather than the model's possibly inconsistent dimension estimates.
    all_points = [p for w in plan.walls for p in (w.start, w.end)] + [
        p for r in plan.rooms for p in r.polygon
    ]
    if all_points:
        xs, ys = zip(*all_points, strict=True)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        pad = span * 0.06
        fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad])
        fig.update_yaxes(range=[min(ys) - span * 0.14, max(ys) + pad])
        scale = next((n for n in (10, 5, 2, 1, 0.5, 0.2, 0.1) if n <= span * 0.18), 0.1)
        scale_y = min(ys) - span * 0.08
        fig.add_shape(
            type="line",
            x0=min(xs),
            x1=min(xs) + scale,
            y0=scale_y,
            y1=scale_y,
            line=dict(color="#7d868b", width=1.5),
        )
        for x in (min(xs), min(xs) + scale):
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=scale_y - span * 0.008,
                y1=scale_y + span * 0.008,
                line=dict(color="#7d868b", width=1.5),
            )
        fig.add_annotation(
            x=min(xs) + scale / 2,
            y=scale_y - span * 0.025,
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
