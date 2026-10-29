"""Assign detections to the 4 virtual regions."""
from __future__ import annotations


def region_of(box: tuple[float, float, float, float], regions: dict[str, list[float]],
              width: int, height: int) -> str | None:
    """Region containing the bottom-centre of the box (where the person stands)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = y2 / height
    cy = min(cy, 0.9999)
    for name, (rx1, ry1, rx2, ry2) in regions.items():
        if rx1 <= cx < rx2 and ry1 <= cy < ry2:
            return name
    return None


def count_by_region(boxes: list, regions: dict[str, list[float]], width: int, height: int) -> dict[str, int]:
    counts = {name: 0 for name in regions}
    for b in boxes:
        r = region_of(b, regions, width, height)
        if r:
            counts[r] += 1
    return counts
