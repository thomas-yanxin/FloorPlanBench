"""Geometry and rendering checks; synthetic plans are not recognition evidence."""

import base64
import json
from math import sqrt

import plotly.graph_objects as go
import pytest
from shapely.geometry import Point, box

from floorplanbench.geometry import opening_frame, wall_surface
from floorplanbench.labels import fit_label
from floorplanbench.render import draw_opening, plot_plan
from floorplanbench.schema import floorplan_json_schema, parse_plan, room_polygon

from .samples import EXAMPLE


def plan_with(walls, openings=()):
    return parse_plan(json.dumps(dict(EXAMPLE, rooms=[], walls=walls, openings=list(openings))))[0]


def wall(a, b, thickness=0.2):
    return {"start": a, "end": b, "thickness": thickness}


def opening(**changes):
    return (
        dict(wallIndex=0, offset=1, width=0.8, kind="door", hinge="start", swing="left") | changes
    )


def test_mitred_corner_has_no_pinholes_and_free_ends_do_not_extend():
    plan = plan_with([wall([0, 0], [2, 0]), wall([2, 0], [2, 2])])
    before = plan.model_dump()
    surface = wall_surface(plan)
    assert surface.is_valid and len(surface.interiors) == 0
    assert surface.area == pytest.approx(0.8)
    assert surface.covers(box(1.95, -0.09, 2.09, 0.09))
    assert surface.bounds == pytest.approx((0, -0.1, 2.1, 2))
    assert plan.model_dump() == before


def test_variable_thickness_junction_retains_both_measured_faces():
    plan = plan_with([wall([0, 0], [2, 0], 0.4), wall([2, 0], [2, 2], 0.2)])
    surface = wall_surface(plan)
    assert len(surface.interiors) == 0
    assert surface.covers(box(1.99, -0.19, 2.09, -0.01))
    assert surface.bounds == pytest.approx((0, -0.2, 2.1, 2))


def test_old_format_gaps_and_near_miss_junctions_are_not_repaired():
    plan = plan_with([wall([0, 0], [1, 0]), wall([1.02, 0], [2, 0])])
    surface = wall_surface(plan)
    assert surface.geom_type == "MultiPolygon"
    assert not surface.intersects(box(1.001, -0.09, 1.019, 0.09))
    assert plan.openings == []
    assert not any(shape.line.color == "#9aa7a1" for shape in plot_plan(plan).layout.shapes)


@pytest.mark.parametrize("end", [[4, 0], [0, 4], [-4, 0], [0, -4], [sqrt(8), sqrt(8)]])
def test_hosted_opening_cuts_exact_metric_width_in_any_wall_direction(end):
    plan = plan_with([wall([0, 0], end)], [opening()])
    original = wall_surface(plan_with([wall([0, 0], end)]))
    surface = wall_surface(plan)
    assert original.area - surface.area == pytest.approx(0.8 * 0.2)
    a, b, _ = opening_frame(plan, plan.openings[0])
    assert Point(a).distance(Point(b)) == pytest.approx(0.8)
    assert not surface.contains(Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))


@pytest.mark.parametrize("hinge", ["start", "end"])
@pytest.mark.parametrize("swing,sign", [("left", 1), ("right", -1)])
def test_door_hinge_and_swing_use_the_directed_host_wall(hinge, swing, sign):
    plan = plan_with([wall([0, 0], [4, 0])], [opening(hinge=hinge, swing=swing)])
    figure = go.Figure()
    draw_opening(figure, plan, plan.openings[0], 4)
    leaf, arc = figure.layout.shapes[-2:]
    hinge_x = 1 if hinge == "start" else 1.8
    leaf_points = [tuple(map(float, p.split(","))) for p in leaf.path[2:].split(" L ")]
    arc_points = [tuple(map(float, p.split(","))) for p in arc.path[2:].split(" L ")]
    assert leaf_points[0] == pytest.approx((hinge_x, 0))
    assert leaf_points[1] == pytest.approx((hinge_x, 0.8 * sign))
    assert arc_points[0] == pytest.approx((1.8 if hinge == "start" else 1, 0))
    assert arc_points[-1] == pytest.approx(leaf_points[1])
    assert all(Point(p).distance(Point(hinge_x, 0)) == pytest.approx(0.8) for p in arc_points)


def test_unknown_door_swing_and_window_do_not_invent_door_arcs():
    for kind in ("door", "window", "opening"):
        plan = plan_with([wall([0, 0], [4, 0])], [opening(kind=kind, hinge=None, swing=None)])
        figure = go.Figure()
        draw_opening(figure, plan, plan.openings[0], 4)
        assert not any(shape.line.color == "#9aa7a1" for shape in figure.layout.shapes)
        assert len(figure.layout.shapes) == (6 if kind == "window" else 2)


def test_invalid_openings_are_warned_and_not_cut_or_clamped():
    sample = dict(
        EXAMPLE,
        rooms=[],
        walls=[wall([0, 0], [4, 0])],
        openings=[opening(wallIndex=9), opening(offset=3.5)],
    )
    plan, notes = parse_plan(json.dumps(sample))
    assert len(notes) == 2 and all("Skipped opening" in note for note in notes)
    assert len(plan.openings) == 2
    assert wall_surface(plan).area == pytest.approx(0.8)


def test_unknown_thickness_wall_is_split_at_openings_without_inventing_a_solid():
    plan = plan_with([wall([0, 0], [4, 0], None)], [opening()])
    assert wall_surface(plan).is_empty
    lines = [s for s in plot_plan(plan).layout.shapes if s.type == "line"][:2]
    assert [(s.x0, s.x1) for s in lines] == [(0, 1), (1.8, 4)]


@pytest.mark.parametrize("name", ["Children's Bedroom", "卫生间", "<script> Hall"])
def test_vector_label_fits_its_entire_box_and_keeps_text_as_escaped_data(name):
    polygon = box(0, 0, 2.4, 2)
    label = fit_label(polygon, name, 4.67, 12)
    assert label is not None
    assert polygon.contains(
        box(
            label.x - label.width / 2,
            label.y - label.height / 2,
            label.x + label.width / 2,
            label.y + label.height / 2,
        )
    )
    svg = base64.b64decode(label.svg_source().split(",")[1]).decode()
    assert "<script>" not in svg
    assert "4.67" in svg
    if name == "Children's Bedroom":
        assert "Children's" in label.lines


def test_labels_remain_inside_concave_rooms_and_scale_with_geometry():
    sample = json.loads(json.dumps(EXAMPLE))
    original = parse_plan(json.dumps(sample))[0]
    sample["walls"] = []
    for room in sample["rooms"]:
        room["polygon"] = [[x * 2, y * 2] for x, y in room["polygon"]]
    enlarged = parse_plan(json.dumps(sample))[0]
    a, b = plot_plan(original), plot_plan(enlarged)
    for room, label in zip(enlarged.rooms, b.layout.images, strict=True):
        assert room_polygon(room).contains(
            box(
                label.x - label.sizex / 2,
                label.y - label.sizey / 2,
                label.x + label.sizex / 2,
                label.y + label.sizey / 2,
            )
        )
    assert b.layout.images[0].sizex == pytest.approx(a.layout.images[0].sizex * 2)


def test_strict_contract_requires_opening_fields_but_old_saved_json_still_parses():
    schema = floorplan_json_schema()
    assert "openings" in schema["required"]
    definition = schema["$defs"]["Opening"]
    assert set(definition["required"]) == {"wallIndex", "offset", "width", "kind", "hinge", "swing"}
    assert definition["additionalProperties"] is False
    assert {"type": "null"} in definition["properties"]["hinge"]["anyOf"]
    assert parse_plan(json.dumps(EXAMPLE))[0].openings == []
