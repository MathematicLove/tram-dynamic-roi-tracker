from __future__ import annotations

import os

# macOS: PyTorch + NumPy/OpenCV may load libomp twice; avoid abort on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from datetime import datetime

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ultralytics import YOLO

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
MODEL_PATH = _ROOT / "/Users/monadayzek/Desktop/Ayzek/1-spbstu/diploma/objects-recognation/yolo/best_fold_3.pt"
VIDEO_PATH = str(_ROOT / "/Users/monadayzek/Desktop/Ayzek/1-spbstu/diploma/objects-recognation/yolo/a.mp4")
OUTPUT_PATH = str(_DIR / "roi_zones_stable___3.mp4")

CONF = 0.325
DEVICE = "mps"

LOW_CONF_DIR = _DIR / "low-conf-data"
LOW_CONF_NORM = 0.5

RAIL_CLS = {0, 1, 3, 4}

RAIL_BRANCH = 0
_MAX_CENTER_JUMP_PX = 120

GAUGE_MM = 1524
DEFAULT_FOV_DEG = 70.0
ZONE_DISTS_MM = {
    "red": 500,
    "orange": 1000,
    "yellow": 1500,
}
GRID_STEP_MM = 2000

ZONE_ALPHA = 0.30
ZONE_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 255),
}
GRID_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)

MORPH_KERNEL = (5, 5)
MORPH_CLOSE_ITER = 1
MORPH_OPEN_ITER = 1
USE_MORPH = False

# Mask extraction: soft (probabilistic) accumulation + sub-pixel resize.
MASK_PROB_THRESH = 0.5
# Always bridge tiny seg holes (cheap, stabilizes per-row edges).
_BASE_CLOSE_KERNEL = (3, 3)
# rail_rows: bridge gaps up to this many px when finding the contiguous bed run.
_ROW_GAP_PX = 14
# Line rendering style (lightweight: outlines only, no filled polygons).
_ZONE_LINE_THICK = 2
_RAIL_LINE_THICK = 2

# Distance model: minimum rows for a trustworthy line fit, robust-trim params.
_FIT_MIN_ROWS = 6
_FIT_ITERS = 4
_FIT_TRIM_SIGMA = 2.0


def load_model(path: Optional[Path] = None) -> YOLO:
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        print(f"[ERROR] Модель не найдена: {p}")
        sys.exit(1)
    return YOLO(str(p))


def _max_rail_confidence(res) -> Optional[float]:
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    rail = np.isin(cls, list(RAIL_CLS))
    if not np.any(rail):
        return None
    return float(np.max(conf[rail]))


def _next_low_conf_path() -> Path:
    LOW_CONF_DIR.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%d_%m_%Y")
    p = LOW_CONF_DIR / f"{stem}.png"
    if not p.exists():
        return p
    n = 1
    while True:
        p = LOW_CONF_DIR / f"{stem}_{n}.png"
        if not p.exists():
            return p
        n += 1


def _maybe_save_low_conf_frame(frame: np.ndarray, max_rail_conf: Optional[float]) -> None:
    if max_rail_conf is None or max_rail_conf >= LOW_CONF_NORM:
        return
    path = _next_low_conf_path()
    cv2.imwrite(str(path), frame)


def get_focal_px(frame_w: int, fov_deg: float = DEFAULT_FOV_DEG) -> float:
    return frame_w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))


def extract_rail_mask(results, h: int, w: int, morph: bool = False) -> np.ndarray:
    """Accumulate rail-class masks as probabilities, then threshold.

    Soft (float) accumulation + bilinear upscale yields sub-pixel-accurate,
    less jagged boundaries than per-mask NEAREST + binary OR. A light close is
    always applied to bridge tiny segmentation holes for stable per-row edges.
    """
    if results.masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    prob = np.zeros((h, w), dtype=np.float32)
    has_any = False
    for m, c in zip(
        results.masks.data.cpu().numpy(),
        results.boxes.cls.cpu().numpy().astype(int),
    ):
        if c in RAIL_CLS:
            mr = cv2.resize(
                m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            np.maximum(prob, mr, out=prob)
            has_any = True

    if not has_any:
        return np.zeros((h, w), dtype=np.uint8)

    mask = (prob >= MASK_PROB_THRESH).astype(np.uint8) * 255
    base_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _BASE_CLOSE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, base_k, iterations=1)
    return _clean_rail_mask(mask) if morph else mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background holes inside the rail blob (flood from border)."""
    if not np.any(mask):
        return mask
    h, w = mask.shape
    ff = mask.copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def _clean_rail_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
    mask = _fill_holes(mask)
    return mask


def _medfilt1d(a: np.ndarray, k: int = 7) -> np.ndarray:
    n = len(a)
    if n < k:
        return a.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(a.astype(np.float64), pad, mode="edge")
    out = np.empty(n, dtype=a.dtype)
    for i in range(n):
        out[i] = np.median(padded[i : i + k])
    return out


def _smooth1d(a: np.ndarray, k: int = 21) -> np.ndarray:
    n = len(a)
    if n < 5:
        return a.copy()
    k = min(k, n // 2 * 2 + 1)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(a.astype(np.float64), pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[:n].astype(a.dtype)


def _row_extent(cols: np.ndarray, gap_px: int = _ROW_GAP_PX) -> Tuple[int, int]:
    """Edges of the dominant contiguous run in a row (bridging small gaps).

    Robust to stray segmentation pixels far from the rail bed: instead of the
    global min/max (which any speck corrupts), pick the run holding the most
    pixels and return its span.
    """
    if len(cols) == 1:
        return int(cols[0]), int(cols[0])
    splits = np.where(np.diff(cols) > gap_px)[0]
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [len(cols) - 1]))
    counts = ends - starts + 1
    best = int(np.argmax(counts))
    return int(cols[starts[best]]), int(cols[ends[best]])


def rail_rows(
    mask: np.ndarray,
    min_px: int = 3,
    smooth: int = 21,
    max_w_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    ys, ls, rs = [], [], []
    for y in range(h):
        cols = np.where(mask[y] > 0)[0]
        if len(cols) < min_px:
            continue
        l, r = _row_extent(cols)
        if (r - l) > max_w_frac * w:
            continue
        ys.append(y)
        ls.append(l)
        rs.append(r)
    if len(ys) < 4:
        return (
            np.array(ys, dtype=int),
            np.array(ls, dtype=int),
            np.array(rs, dtype=int),
        )
    ys = np.array(ys, dtype=int)
    ls_a = np.array(ls, dtype=int)
    rs_a = np.array(rs, dtype=int)

    # Width must shrink monotonically with distance (rows higher up are
    # narrower). Reject rows whose width deviates grossly from the local
    # median before smoothing — kills single-row blow-ups from merged blobs.
    widths = np.maximum(rs_a - ls_a, 1)
    wmed = _medfilt1d(widths, 9)
    good = np.abs(widths - wmed) <= np.maximum(0.6 * wmed, 8)
    if int(good.sum()) >= 4:
        ys, ls_a, rs_a = ys[good], ls_a[good], rs_a[good]

    ls = _smooth1d(_medfilt1d(ls_a, 9), smooth)
    rs = _smooth1d(_medfilt1d(rs_a, 9), smooth)
    return ys, ls, rs


def zone_bounds(
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    w: int,
) -> dict:
    if len(ys) == 0:
        return {}
    widths = np.maximum(rs - ls, 1).astype(float)
    scale = GAUGE_MM / widths
    out = {}
    for name in ("yellow", "orange", "red"):
        off = (ZONE_DISTS_MM[name] / scale).astype(int)
        out[name] = (
            np.clip(ls - off, 0, w - 1),
            np.clip(rs + off, 0, w - 1),
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
#  ADAPTIVE, GEOMETRY-EXACT DISTANCE MODEL
#
#  For a planar track under a pinhole camera the projected rail pixel-width is
#  EXACTLY linear in the image row:   w(y) = a·(y − y_vp),  with y_vp the row of
#  the vanishing point (where the rails meet, w → 0). Combined with the known
#  gauge G and focal length f, the ground distance is a smooth, monotonic,
#  closed-form function of the row:
#
#       Z(y) = f·G / w(y) = (f·G / a) / (y − y_vp) = C / (y − y_vp)
#       y(Z) = y_vp + C / Z          (exact inverse → exact grid rows)
#
#  We recover (a, y_vp) per frame with a robust (outlier-trimmed) line fit, so
#  the model self-calibrates to the current geometry, denoises the per-row
#  width noise (which explodes far away), and works on straights and gentle
#  curves alike. A width-table fallback keeps it working when the fit is weak.
# ─────────────────────────────────────────────────────────────────────────


def _fit_width_line(
    ys: np.ndarray, ls: np.ndarray, rs: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Robust linear fit width(y)=a*y+b via iterative residual trimming."""
    if len(ys) < _FIT_MIN_ROWS:
        return None
    yf = ys.astype(np.float64)
    wf = np.maximum(rs - ls, 1).astype(np.float64)
    try:
        a, b = np.polyfit(yf, wf, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    for _ in range(_FIT_ITERS):
        res = wf - (a * yf + b)
        sigma = float(np.std(res))
        if sigma < 1e-6:
            break
        keep = np.abs(res) <= _FIT_TRIM_SIGMA * sigma
        if int(keep.sum()) < max(4, _FIT_MIN_ROWS // 2):
            break
        try:
            a, b = np.polyfit(yf[keep], wf[keep], 1)
        except (np.linalg.LinAlgError, ValueError):
            break
    # Width must grow toward the bottom of the image (a > 0) for a valid model.
    if a <= 1e-6:
        return None
    return float(a), float(b)


class _DistanceModel:
    """Closed-form row<->metric-distance mapping from the fitted width line."""

    __slots__ = ("a", "b", "y_vp", "C")

    def __init__(self, a: float, b: float, f_px: float) -> None:
        self.a = a
        self.b = b
        self.y_vp = -b / a
        self.C = f_px * GAUGE_MM / a  # mm * px

    def width_at(self, y: float) -> float:
        return self.a * float(y) + self.b

    def dist_mm(self, y: float) -> float:
        dy = max(float(y) - self.y_vp, 1e-3)
        return self.C / dy

    def row_at_dist(self, dist_mm: float) -> float:
        return self.y_vp + self.C / max(dist_mm, 1e-3)


def build_distance_model(
    ys: np.ndarray, ls: np.ndarray, rs: np.ndarray, f_px: float
) -> Optional[_DistanceModel]:
    fit = _fit_width_line(ys, ls, rs)
    if fit is None:
        return None
    a, b = fit
    return _DistanceModel(a, b, f_px)


def estimate_distance_m(
    y_px: int,
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    f_px: float,
) -> Optional[float]:
    """Ground distance (m) of image row y_px, relative to nearest rail row."""
    if len(ys) < 2:
        return None
    y_bot = float(np.max(ys))
    if y_px >= y_bot:
        return 0.0

    model = build_distance_model(ys, ls, rs, f_px)
    if model is not None:
        d_obj = model.dist_mm(float(y_px))
        d_bot = model.dist_mm(y_bot)
        return max(0.0, float(d_obj - d_bot) / 1000.0)

    widths = np.maximum(rs - ls, 1).astype(float)
    w_at_y = float(np.interp(y_px, ys, widths))
    if w_at_y < 1:
        return None
    d_obj = f_px * GAUGE_MM / w_at_y
    d_bot = f_px * GAUGE_MM / float(widths[int(np.argmax(ys))])
    return max(0.0, (d_obj - d_bot) / 1000.0)


def _grid_positions_fallback(
    ys: np.ndarray, ls: np.ndarray, rs: np.ndarray, f_px: float
) -> list:
    widths = np.maximum(rs - ls, 1).astype(float)
    d_mm = f_px * GAUGE_MM / widths
    d_ref = d_mm[int(np.argmax(ys))]
    marks = []
    for i in range(1, 50):
        target = d_ref + i * GRID_STEP_MM
        above = np.where(d_mm >= target)[0]
        if len(above) == 0:
            break
        best = above[int(np.argmin(np.abs(d_mm[above] - target)))]
        marks.append((int(ys[best]), i * GRID_STEP_MM / 1000.0))
    return marks


def grid_positions(
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    f_px: Optional[float] = None,
) -> list:
    """Rows for distance grid lines. Returns (row_px, distance_m) per mark."""
    if len(ys) < 2:
        return []
    if f_px is None:
        f_px = get_focal_px(1920)

    model = build_distance_model(ys, ls, rs, f_px)
    if model is None:
        return _grid_positions_fallback(ys, ls, rs, f_px)

    y_bot = float(np.max(ys))
    y_top = float(np.min(ys))
    d_ref = model.dist_mm(y_bot)

    marks = []
    for i in range(1, 80):
        target = d_ref + i * GRID_STEP_MM
        yy = model.row_at_dist(target)
        # Stop once the mark climbs past the visible rail / approaches horizon.
        if yy <= y_top - 1.0 or yy <= model.y_vp + 1.0:
            break
        if yy >= y_bot:
            continue
        marks.append((int(round(yy)), i * GRID_STEP_MM / 1000.0))
    return marks


def _outline_pts(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Single boundary polyline (one side) for crisp anti-aliased outlines."""
    return np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)


def _put_label(frame: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    """Distance label with a dark outline so it stays readable on any zone."""
    x, y = org
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1,
               (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1,
               TEXT_COLOR, 1, cv2.LINE_AA)


def draw_zones(
    frame: np.ndarray,
    ys: np.ndarray,
    zones: dict,
    ls: np.ndarray,
    rs: np.ndarray,
    grid: list,
) -> None:
    """Lightweight rendering: zone boundary lines + rails + distance markup.

    No filled / translucent polygons — only crisp anti-aliased outlines drawn
    directly on the frame, plus the distance grid lines and labels.
    """
    # Zone boundary lines (left + right edge of each zone).
    for name in ("yellow", "orange", "red"):
        if name not in zones:
            continue
        zl, zr = zones[name]
        c = ZONE_COLORS[name]
        cv2.polylines(frame, [_outline_pts(ys, zl)], False, c,
                     _ZONE_LINE_THICK, cv2.LINE_AA)
        cv2.polylines(frame, [_outline_pts(ys, zr)], False, c,
                     _ZONE_LINE_THICK, cv2.LINE_AA)

    # Rail edges.
    cv2.polylines(frame, [_outline_pts(ys, ls)], False, ZONE_COLORS["red"],
                 _RAIL_LINE_THICK, cv2.LINE_AA)
    cv2.polylines(frame, [_outline_pts(ys, rs)], False, ZONE_COLORS["red"],
                 _RAIL_LINE_THICK, cv2.LINE_AA)

    # Distance grid lines + labels across the outer zone.
    outer = zones.get("yellow") or zones.get("orange") or zones.get("red")
    if outer and grid:
        gy_arr = np.fromiter((g[0] for g in grid), dtype=np.int32, count=len(grid))
        ii = np.argmin(np.abs(ys[:, None] - gy_arr[None, :]), axis=0)
        z0, z1 = outer[0], outer[1]
        for k, idx in enumerate(ii):
            gy = int(gy_arr[k])
            dist_m = grid[k][1]
            x1, x2 = int(z0[idx]), int(z1[idx])
            cv2.line(frame, (x1, gy), (x2, gy), GRID_COLOR, 1, cv2.LINE_AA)
            _put_label(frame, f"{dist_m:.0f} m", (x2 + 5, gy + 4))


def draw_legend(frame: np.ndarray) -> None:
    h = frame.shape[0]
    lx, ly = 10, h - 90
    lt = cv2.LINE_8
    for i, name in enumerate(("red", "orange", "yellow")):
        yy = ly + i * 24
        cv2.rectangle(frame, (lx, yy), (lx + 18, yy + 16), ZONE_COLORS[name], -1)
        dist = ZONE_DISTS_MM[name] / 1000
        if name == "red":
            label = f"Red: rails + {dist:.1f} m"
        elif name == "orange":
            label = f"Orange: {dist:.1f} m"
        else:
            label = f"Yellow: {dist:.1f} m"
        cv2.putText(
            frame,
            label,
            (lx + 24, yy + 13),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            TEXT_COLOR,
            1,
            lt,
        )


def find_groups(mask: np.ndarray, min_area: int = 300) -> list:
    n, labels = cv2.connectedComponents(mask)
    groups = []
    for lab in range(1, n):
        comp = (labels == lab).astype(np.uint8) * 255
        if np.sum(comp > 0) < min_area:
            continue
        groups.append(comp)
    return groups if groups else [mask]


def _branch_sort_key(branch: int):
    if branch == 0:
        return lambda c: (-c[1], -c[2])
    return lambda c: (c[1], -c[2])


def _select_rail_group(
    groups: list, h: int, prev_center: Optional[float] = None
) -> np.ndarray:
    if len(groups) == 1:
        return groups[0]
    band = slice(int(0.55 * h), h)
    candidates: list[Tuple[np.ndarray, float, float]] = []
    for g in groups:
        gy, gx = np.where(g > 0)
        if len(gx) == 0:
            continue
        candidates.append((g, float(np.median(gx)), float(np.sum(g[band] > 0))))
    if not candidates:
        return groups[0]

    if prev_center is not None:
        candidates.sort(key=lambda c: abs(c[1] - prev_center))
        if abs(candidates[0][1] - prev_center) <= _MAX_CENTER_JUMP_PX:
            return candidates[0][0]

    candidates.sort(key=_branch_sort_key(RAIL_BRANCH))
    return candidates[0][0]


class _TemporalState:
    EMA_ALPHA = 0.30
    MAX_AREA_RATIO = 1.8
    FALLBACK_FRAMES = 10
    DECAY_SHRINK = 0.97
    TURN_SHIFT_THRESHOLD = 30
    TURN_ALPHA = 0.50

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ys: Optional[np.ndarray] = None
        self._ls: Optional[np.ndarray] = None
        self._rs: Optional[np.ndarray] = None
        self._area: float = 0.0
        self._miss: int = 0

    @property
    def active(self) -> bool:
        return self._ys is not None and len(self._ys) >= 4

    def center(self) -> Optional[float]:
        if not self.active:
            return None
        return float(np.median((self._ls + self._rs) / 2.0))

    def _area_of(self, ls: np.ndarray, rs: np.ndarray) -> float:
        return float(np.sum(np.maximum(rs - ls, 0)))

    def update(
        self,
        ys: np.ndarray,
        ls: np.ndarray,
        rs: np.ndarray,
        w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(ys) < 4:
            return self._fallback(w)

        area = self._area_of(ls, rs)

        if not self.active:
            self._ys, self._ls, self._rs = ys.copy(), ls.copy(), rs.copy()
            self._area = area
            self._miss = 0
            return ys, ls, rs

        prev_c = self.center()
        cur_c = float(np.median((ls + rs) / 2.0))
        shift = abs(cur_c - prev_c) if prev_c is not None else 0.0

        alpha = self.TURN_ALPHA if shift > self.TURN_SHIFT_THRESHOLD else self.EMA_ALPHA

        if self._area > 0:
            ratio = area / self._area
            if ratio > self.MAX_AREA_RATIO:
                alpha *= 0.2
            elif ratio < 1.0 / self.MAX_AREA_RATIO:
                alpha *= 0.4

        ls_prev = np.interp(ys, self._ys, self._ls).astype(np.float64)
        rs_prev = np.interp(ys, self._ys, self._rs).astype(np.float64)

        ls_new = np.clip(
            (alpha * ls.astype(np.float64) + (1 - alpha) * ls_prev), 0, w - 1
        ).astype(np.int32)
        rs_new = np.clip(
            (alpha * rs.astype(np.float64) + (1 - alpha) * rs_prev), 0, w - 1
        ).astype(np.int32)

        bad = ls_new >= rs_new
        if np.any(bad):
            ls_new[bad] = np.minimum(ls[bad], ls_prev[bad].astype(np.int32))
            rs_new[bad] = np.maximum(rs[bad], rs_prev[bad].astype(np.int32))

        self._ys, self._ls, self._rs = ys.copy(), ls_new.copy(), rs_new.copy()
        self._area = self._area_of(ls_new, rs_new)
        self._miss = 0
        return ys, ls_new, rs_new

    def _fallback(self, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = np.array([], dtype=np.int32)
        if not self.active:
            return empty, empty, empty
        self._miss += 1
        if self._miss > self.FALLBACK_FRAMES:
            self.reset()
            return empty, empty, empty
        c = (self._ls + self._rs) / 2.0
        hw = (self._rs - self._ls) / 2.0 * self.DECAY_SHRINK
        self._ls = np.clip((c - hw).astype(np.int32), 0, w - 1)
        self._rs = np.clip((c + hw).astype(np.int32), 0, w - 1)
        self._area = self._area_of(self._ls, self._rs)
        return self._ys.copy(), self._ls.copy(), self._rs.copy()


_temporal = _TemporalState()


def process_frame(frame: np.ndarray, model: YOLO) -> np.ndarray:
    h, w = frame.shape[:2]
    res = model(frame, conf=CONF, device=DEVICE, verbose=False)[0]
    _maybe_save_low_conf_frame(frame, _max_rail_confidence(res))
    full_mask = extract_rail_mask(res, h, w, morph=USE_MORPH)

    if np.any(full_mask):
        groups = find_groups(full_mask)
        gmask = _select_rail_group(groups, h, _temporal.center())
        ys, ls, rs = rail_rows(gmask)
    else:
        ys = np.array([], dtype=np.int32)
        ls = np.array([], dtype=np.int32)
        rs = np.array([], dtype=np.int32)

    ys, ls, rs = _temporal.update(ys, ls, rs, w)

    if len(ys) < 4:
        return frame

    f_px = get_focal_px(w)
    zones = zone_bounds(ys, ls, rs, w)
    grid = grid_positions(ys, ls, rs, f_px)
    draw_zones(frame, ys, zones, ls, rs, grid)
    return frame


def reset_temporal_state() -> None:
    _temporal.reset()


def main() -> None:
    global USE_MORPH
    ap = argparse.ArgumentParser(
        description="Building danger zones around rails"
    )
    ap.add_argument(
        "-m", "--morph", action="store_true", dest="morph",
        help="Enable morphological cleaning of the segmentation mask",
    )
    args = ap.parse_args()
    USE_MORPH = args.morph

    if USE_MORPH:
        print("Morphology enabled: clean rail mask after segmentation")

    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {VIDEO_PATH}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if OUTPUT_PATH:
        out = cv2.VideoWriter(
            OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )

    print(f"Video {W}×{H} @ {fps} fps  -  'q' for exit")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(frame, model)
        cv2.imshow("Rail Danger Zones", frame)
        if out:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        n += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Done - {n} frames processed.")


if __name__ == "__main__":
    main()
