"""Room labels sized in drawing units, so they fit and scale with the plan."""

import base64
import html
import textwrap
import unicodedata
from dataclasses import dataclass

from shapely.geometry import Polygon, box
from shapely.ops import polylabel


def text_width(text: str) -> float:
    # SVG textLength enforces these widths, independent of installed fonts.
    return sum(
        1
        if unicodedata.east_asian_width(c) in {"W", "F"}
        else 0.28
        if c in " ilI.,'’!:;|"
        else 0.85
        if c in "MW@%"
        else 0.62
        if c.isupper()
        else 0.54
        for c in text
    )


@dataclass(frozen=True)
class Label:
    x: float
    y: float
    width: float
    height: float
    font_size: float
    lines: tuple[str, ...]
    area: str

    def svg_source(self) -> str:
        # All text is escaped; no model-supplied markup enters the SVG.
        width, height = self.width / self.font_size * 100, self.height / self.font_size * 100
        lines = []
        for index, line in enumerate(self.lines):
            lines.append(
                f'<text x="{width / 2}" y="{82 + index * 120}" font-weight="500" '
                f'textLength="{text_width(line) * 100}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(line)}</text>"
            )
        if self.area:
            lines.append(
                f'<text x="{width / 2}" y="{len(self.lines) * 120 + 70}" font-size="76" '
                f'fill="#77817f" textLength="{text_width(self.area) * 76}" '
                f'lengthAdjust="spacingAndGlyphs">{html.escape(self.area)}</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            '<g font-family="Inter,Arial,PingFang SC,Microsoft YaHei,sans-serif" '
            'font-size="100" text-anchor="middle" fill="#34413f">' + "".join(lines) + "</g></svg>"
        )
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def fit_label(polygon: Polygon, name: str, area: float | None, span: float) -> Label | None:
    point = polylabel(polygon, tolerance=max(span * 0.001, 1e-5))
    area_text = f"≈ {area:g} m²" if area is not None and area > 0 else ""
    # Wrap only at word boundaries; CJK text can wrap between characters.
    cjk = any(unicodedata.east_asian_width(c) in {"W", "F"} for c in name)
    variants = set()
    for width in range(1, min(len(name), 80) + 1):
        lines = tuple(
            textwrap.wrap(name, width=width, break_long_words=cjk, break_on_hyphens=False)
        )
        if lines:
            variants.add(lines)
    variants = sorted(variants, key=lambda lines: (len(lines), max(map(text_width, lines)), lines))
    for with_area in (bool(area_text), False) if area_text else (False,):
        for factor in (1, 0.9, 0.8, 0.7):
            size = span * 0.026 * factor
            for lines in variants:
                width = (
                    max(
                        max(map(text_width, lines)),
                        text_width(area_text) * 0.76 if with_area else 0,
                    )
                    * size
                    + size * 0.25
                )
                height = (len(lines) * 1.2 + (0.85 if with_area else 0)) * size
                margin = size * 0.28
                rectangle = box(
                    point.x - width / 2 - margin,
                    point.y - height / 2 - margin,
                    point.x + width / 2 + margin,
                    point.y + height / 2 + margin,
                )
                if polygon.covers(rectangle):
                    return Label(
                        point.x, point.y, width, height, size, lines, area_text if with_area else ""
                    )
    return None
