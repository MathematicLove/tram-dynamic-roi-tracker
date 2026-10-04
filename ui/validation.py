"""Input validation for the launcher UI (no tkinter, so it is easy to test)."""
from __future__ import annotations

from pathlib import Path

MAX_SPEED_KMH = 200


def parse_speed(raw: str) -> int:
    """Tram speed in km/h as a positive integer (detection.py expects an int)."""
    text = raw.strip()
    if not text:
        raise ValueError("Enter the tram speed")
    try:
        speed = int(text)
    except ValueError:
        raise ValueError("Speed must be a whole number, e.g. 40") from None
    if not 0 < speed <= MAX_SPEED_KMH:
        raise ValueError(f"Speed must be between 1 and {MAX_SPEED_KMH} km/h")
    return speed


def parse_camera_index(raw: str) -> str:
    """Camera index as a non-negative integer; empty input means camera 0."""
    text = raw.strip()
    if not text:
        return "0"
    try:
        index = int(text)
    except ValueError:
        raise ValueError("Camera index must be a whole number, e.g. 0") from None
    if index < 0:
        raise ValueError("Camera index cannot be negative")
    return str(index)


def check_input_file(path: str | None, kind: str) -> str:
    """Make sure the chosen file is still there before starting the process."""
    if not path:
        raise ValueError(f"Select {kind}")
    if not Path(path).is_file():
        raise ValueError(f"File not found: {path}")
    return path
