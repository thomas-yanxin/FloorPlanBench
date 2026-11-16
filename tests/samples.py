"""Synthetic image and mock output for local tests, not model predictions."""

from PIL import Image, ImageDraw

EXAMPLE = {
    "walls": [
        {"start": a, "end": b, "thickness": 0.16}
        for a, b in [
            ([0, 0], [10, 0]),
            ([10, 0], [10, 8]),
            ([10, 8], [0, 8]),
            ([0, 8], [0, 0]),
            ([4, 0], [4, 3.2]),
            ([4, 4.2], [4, 8]),
            ([4, 4.8], [6.1, 4.8]),
            ([7, 4.8], [10, 4.8]),
            ([7.2, 4.8], [7.2, 8]),
            ([0, 4.8], [2.6, 4.8]),
            ([3.5, 4.8], [4, 4.8]),
        ]
    ],
    "rooms": [
        {
            "name": "Living / Dining",
            "polygon": [[4, 0], [10, 0], [10, 4.8], [4, 4.8]],
            "approximateAreaSqM": 28.8,
        },
        {
            "name": "Bedroom 01",
            "polygon": [[0, 0], [4, 0], [4, 4.8], [0, 4.8]],
            "approximateAreaSqM": 19.2,
        },
        {
            "name": "Bedroom 02",
            "polygon": [[0, 4.8], [4, 4.8], [4, 8], [0, 8]],
            "approximateAreaSqM": 12.8,
        },
        {
            "name": "Kitchen",
            "polygon": [[4, 4.8], [7.2, 4.8], [7.2, 8], [4, 8]],
            "approximateAreaSqM": 10.24,
        },
        {
            "name": "Bath",
            "polygon": [[7.2, 4.8], [10, 4.8], [10, 8], [7.2, 8]],
            "approximateAreaSqM": 8.96,
        },
    ],
    "approximateDimensions": {"widthM": 10, "depthM": 8},
    "confidence": 1,
}


def example_image() -> Image.Image:
    image = Image.new("RGB", (320, 256), "white")
    draw = ImageDraw.Draw(image)
    for wall in EXAMPLE["walls"]:
        draw.line(
            [(20 + x * 28, 240 - y * 28) for x, y in (wall["start"], wall["end"])],
            fill="#263346",
            width=4,
        )
    return image
