"""Floor-plan contract, extending Pascal's format with hosted openings."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon

PROMPTS = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = (PROMPTS / "reconstruction_floorplan.txt").read_text().rstrip()
CONTRACT_VERSION = "floorplan-v3"
RENDERER_VERSION = "plotly-v2"


class PascalModel(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="ignore")


class Wall(PascalModel):
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float | None = None


class Room(PascalModel):
    name: str = Field(max_length=200)
    polygon: list[tuple[float, float]] = Field(max_length=2000)
    approximateAreaSqM: float | None = None


class Dimensions(PascalModel):
    widthM: float
    depthM: float


class Opening(PascalModel):
    wallIndex: int = Field(ge=0)
    offset: float = Field(ge=0)
    width: float = Field(gt=0)
    kind: Literal["door", "window", "opening"]
    hinge: Literal["start", "end"] | None = None
    swing: Literal["left", "right"] | None = None


class FloorPlan(PascalModel):
    walls: list[Wall] = Field(max_length=2000)
    rooms: list[Room] = Field(max_length=500)
    openings: list[Opening] = Field(default_factory=list, max_length=2000)
    approximateDimensions: Dimensions
    confidence: float = Field(ge=0, le=1)


def floorplan_json_schema() -> dict:
    """Express the existing wire format in OpenAI's strict JSON Schema subset."""
    schema = FloorPlan.model_json_schema()

    def constrain(node):
        if isinstance(node, list):
            for item in node:
                constrain(item)
        elif isinstance(node, dict):
            # Coordinates are homogeneous two-number arrays, not heterogeneous tuples.
            if "prefixItems" in node:
                node["items"] = node.pop("prefixItems")[0]
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            # Optional values stay nullable and required; length is still checked locally.
            node.pop("default", None)
            node.pop("maxLength", None)
            for value in node.values():
                constrain(value)

    constrain(schema)
    return schema


def parse_plan(raw: str) -> tuple[FloorPlan, list[str]]:
    if len(raw) > 2_000_000:
        raise ValueError("Response exceeds 2 MB.")
    text = raw.strip()
    warnings = []
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
        warnings.append("Removed Markdown fences around JSON.")
    plan = FloorPlan.model_validate_json(text)
    supplied = json.loads(text)
    extras = set(supplied) - set(FloorPlan.model_fields)
    if extras:
        warnings.append("Fields not rendered: " + ", ".join(sorted(extras)))
    if any(len(room.polygon) < 3 for room in plan.rooms):
        warnings.append("Skipped rooms with too few vertices.")
    if any(len(room.polygon) >= 3 and room_polygon(room) is None for room in plan.rooms):
        warnings.append("Skipped invalid room polygons.")
    if plan.approximateDimensions.widthM <= 0 or plan.approximateDimensions.depthM <= 0:
        warnings.append("Invalid estimated dimensions; the view uses coordinates.")
    for index, opening in enumerate(plan.openings):
        if not valid_opening(plan, opening):
            warnings.append(f"Skipped opening {index}: outside its host wall.")
    return plan, warnings


def valid_opening(plan: FloorPlan, opening: Opening) -> bool:
    if opening.wallIndex >= len(plan.walls):
        return False
    wall = plan.walls[opening.wallIndex]
    length = sum((b - a) ** 2 for a, b in zip(wall.start, wall.end, strict=True)) ** 0.5
    return length > 0 and opening.offset + opening.width <= length + 1e-8


def renderable(plan: FloorPlan) -> bool:
    return any(w.start != w.end for w in plan.walls) or any(
        room_polygon(r) is not None for r in plan.rooms
    )


def room_polygon(room: Room) -> Polygon | None:
    if len(room.polygon) < 3:
        return None
    polygon = Polygon(room.polygon)
    return polygon if polygon.is_valid and polygon.area > 1e-8 else None
