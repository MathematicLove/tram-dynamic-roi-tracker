from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

from ultralytics import YOLO
import final_roi_3 as roi

_DIR = Path(__file__).resolve().parent
DET_MODEL_PATH = "yolov8n.pt"
DET_CONF = 0.35
DET_CLASSES_DEFAULT = ""
# Outside yellow polygon: 1× gauge width on each side (not drawn)
_ROI_BUFFER_GAUGE_FRAC = 5.5
OUTPUT_PATH = str(_DIR / "all_detection.mp4")
ORANGE_RED_SHOTS_DIR = _DIR / "orange_red_crossings"

_RU_ALERT_YELLOW = "Сигналь (предупреждение)"
_RU_ALERT_ORANGE = "Снизь скорость и сигналь"
_RU_ALERT_RED = "Тормози (аварийная остановка)"
_RU_ALERT_TRAIN = "Внимание на трамвай/поезд"

ALERT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 255),
}
_NEUTRAL_COLOR = (180, 180, 180)

_ALERT_HOLD_SEC = 1.0
_TRACK_GRID_PX = 80

# --- трекинг траекторий и предупреждение WATCH THE OBJECT ---
HISTORY = 5          # окно кадров для оценки тренда движения
MIN_MOVE_PX = 1.5    # мин. горизонтальное смещение, чтобы считать движение
DIST_EPS_PX = 1.0    # насколько должно упасть расстояние до ROI, чтобы «сокращается»
TRACK_TTL = 30       # через сколько кадров без объекта забыть его историю
TRAIL_LEN = 30       # длина рисуемого следа (в кадрах)
_TRAIL_COLOR = (0, 255, 0)
_TRAIL_WATCH_COLOR = (0, 0, 255)

# --- компенсация движения камеры (ego-motion) ---
# Нетто-смещение рамки за окно HISTORY ПОСЛЕ вычета глобального движения камеры.
# Больше порога => объект реально движется; меньше => «плывёт» из-за камеры.
MIN_REAL_MOVE_PX = 6.0
EGO_MIN_FEATURES = 12     # мин. число фоновых точек для надёжной оценки
EGO_MAX_CORNERS = 600

_TRAIN_CLASS_NAME = "train"
_HAZARD_BASE_NAMES: frozenset[str] = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "bench",
    }
)
_HAZARD_NAME_SUBSTR_EXTRA: Tuple[str, ...] = (
    "tree",
    "trash",
    "garbage",
    "waste",
    "dumpster",
)

# --- классы, для которых ведём трекинг: транспорт + люди + животные ---
_TRACK_PERSON_NAMES = {"person"}
_TRACK_VEHICLE_NAMES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "skateboard", "skis", "snowboard", "surfboard",
}
_TRACK_ANIMAL_NAMES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}
_TRACK_BASE_NAMES: frozenset[str] = frozenset(
    _TRACK_PERSON_NAMES | _TRACK_VEHICLE_NAMES | _TRACK_ANIMAL_NAMES
)
_TRACK_NAME_SUBSTR: Tuple[str, ...] = ("bike", "scooter", "skate")


def _norm_det_name(model: YOLO, cls_id: int) -> str:
    return str(model.names.get(cls_id, f"class_{cls_id}")).strip().lower()


def _build_track_class_names(model: YOLO) -> frozenset[str]:
    """Transport (incl. ridables), people and animals — the trackable set."""
    extra: set[str] = set()
    for v in model.names.values():
        lv = str(v).strip().lower()
        if lv and any(s in lv for s in _TRACK_NAME_SUBSTR):
            extra.add(lv)
    return frozenset(_TRACK_BASE_NAMES | extra)


def _build_hazard_class_names(model: YOLO) -> frozenset[str]:
    extra: set[str] = set()
    for v in model.names.values():
        lv = str(v).strip().lower()
        if not lv or lv == _TRAIN_CLASS_NAME:
            continue
        if any(s in lv for s in _HAZARD_NAME_SUBSTR_EXTRA):
            extra.add(lv)
    return frozenset(_HAZARD_BASE_NAMES | extra)


def _det_class_ids(model: YOLO, names_csv: str) -> list[int]:
    want = {s.strip().lower() for s in names_csv.split(",") if s.strip()}
    if not want:
        raise ValueError("Empty class list")
    found: dict[str, int] = {}
    for k, v in model.names.items():
        key = str(v).strip().lower()
        if key in want:
            found[key] = int(k)
    missing = want - set(found.keys())
    if missing:
        sys.stderr.write(
            f"Classes not found in model: {sorted(missing)}. "
            f"Model names: {list(model.names.values())[:30]}\n"
        )
        sys.exit(1)
    return sorted(found.values())

class _AlertTracker:
    def __init__(self, fps: int, hold_sec: float) -> None:
        self._fps = max(1, fps)
        self._min_frames = max(1, int(fps * hold_sec))
        self._timers: Dict[Tuple[int, int, int], int] = {}

    @staticmethod
    def _key(cls_id: int, cx: int, cy: int) -> Tuple[int, int, int]:
        return (cls_id, cx // _TRACK_GRID_PX, cy // _TRACK_GRID_PX)

    def tick(self, cls_id: int, cx: int, cy: int) -> Tuple[bool, int]:
        k = self._key(cls_id, cx, cy)
        self._timers[k] = self._timers.get(k, 0) + 1
        nf = self._timers[k]
        return nf >= self._min_frames, nf

    def dwell_sec(self, cls_id: int, cx: int, cy: int) -> float:
        k = self._key(cls_id, cx, cy)
        return self._timers.get(k, 0) / float(self._fps)

    def end_frame(self, seen: set) -> None:
        for k in list(self._timers):
            if k not in seen:
                del self._timers[k]


class _CrossingTracker:
    def __init__(self, fps: int) -> None:
        self._fps = max(1, fps)
        self._cnt: Dict[Tuple[int, int, int], int] = {}

    @staticmethod
    def _key(cls_id: int, cx: int, cy: int) -> Tuple[int, int, int]:
        return (cls_id, cx // _TRACK_GRID_PX, cy // _TRACK_GRID_PX)

    def tick(self, cls_id: int, cx: int, cy: int, crossing: bool) -> int:
        k = self._key(cls_id, cx, cy)
        if crossing:
            self._cnt[k] = self._cnt.get(k, 0) + 1
            return self._cnt[k]
        self._cnt.pop(k, None)
        return 0

    def dwell_sec(self, cls_id: int, cx: int, cy: int) -> float:
        k = self._key(cls_id, cx, cy)
        return self._cnt.get(k, 0) / float(self._fps)

    def end_frame(self, seen: set) -> None:
        for k in list(self._cnt):
            if k not in seen:
                del self._cnt[k]

def estimate_distance_m(
    y_px: int,
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    f_px: float,
) -> Optional[float]:
    if len(ys) < 2:
        return None
    widths = np.maximum(rs - ls, 1).astype(float)

    if y_px >= ys[np.argmax(ys)]:
        return 0.0

    w_at_y = float(np.interp(y_px, ys, widths))
    if w_at_y < 1:
        return None

    d_obj = f_px * roi.GAUGE_MM / w_at_y
    d_bot = f_px * roi.GAUGE_MM / float(widths[np.argmax(ys)])
    return max(0.0, (d_obj - d_bot) / 1000.0)

def classify_zone(
    x: int,
    y: int,
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    zones: dict,
) -> Optional[str]:
    if len(ys) == 0:
        return None
    i = int(np.argmin(np.abs(ys - y)))
    if abs(int(ys[i]) - y) > 15:
        return None
    if int(ls[i]) <= x <= int(rs[i]):
        return "red"
    for name in ("red", "orange", "yellow"):
        if name not in zones:
            continue
        zl, zr = zones[name]
        if int(zl[i]) <= x <= int(zr[i]):
            return name
    return None


def _outer_bounds(
    zones: dict, ls: np.ndarray, rs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    for name in ("yellow", "orange", "red"):
        if name in zones:
            return zones[name]
    return ls, rs


def _roi_band_at_y(
    y: int, ys: np.ndarray, L_arr: np.ndarray, R_arr: np.ndarray
) -> Tuple[float, float]:
    yy = float(np.clip(y, int(ys.min()), int(ys.max())))
    ysf = ys.astype(np.float64)
    L = float(np.interp(yy, ysf, L_arr.astype(np.float64)))
    R = float(np.interp(yy, ysf, R_arr.astype(np.float64)))
    if L > R:
        L, R = R, L
    return L, R


def _horiz_gap(x1: int, x2: int, L: float, R: float) -> float:
    if x2 < L:
        return L - x2
    if x1 > R:
        return x1 - R
    return 0.0


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _warp_affine_pt(
    M: Optional[np.ndarray], x: float, y: float
) -> Tuple[float, float]:
    """Куда уехала бы точка (x, y) при движении камеры (M: prev->cur).

    M is None => считаем камеру неподвижной (тождество).
    """
    if M is None:
        return float(x), float(y)
    nx = float(M[0, 0] * x + M[0, 1] * y + M[0, 2])
    ny = float(M[1, 0] * x + M[1, 1] * y + M[1, 2])
    return nx, ny


class _EgoMotion:
    """Глобальная модель движения камеры между соседними кадрами.

    По фоновым точкам (вне рамок объектов) оцениваем аффинное преобразование
    prev->cur (сдвиг + поворот + масштаб) методом RANSAC. Масштаб критичен:
    при движении камеры вперёд статичные объекты «расплываются» от центра
    кадра — это ловится коэффициентом масштаба, а не только сдвигом.
    """

    _QUALITY = 0.01
    _MIN_DIST = 8
    _BOX_MARGIN = 12
    _LK = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None

    def update(
        self,
        gray: np.ndarray,
        exclude_boxes: List[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        prev = self.prev_gray
        self.prev_gray = gray
        if prev is None:
            return None

        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        m = self._BOX_MARGIN
        for (x1, y1, x2, y2) in exclude_boxes:
            xa, ya = max(0, int(x1) - m), max(0, int(y1) - m)
            xb, yb = min(w, int(x2) + m), min(h, int(y2) + m)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] = 0

        p0 = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=EGO_MAX_CORNERS,
            qualityLevel=self._QUALITY,
            minDistance=self._MIN_DIST,
            mask=mask,
        )
        if p0 is None or len(p0) < EGO_MIN_FEATURES:
            return None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **self._LK)
        if p1 is None or st is None:
            return None
        st = st.reshape(-1).astype(bool)
        a = p0.reshape(-1, 2)[st]
        b = p1.reshape(-1, 2)[st]
        if len(a) < EGO_MIN_FEATURES:
            return None

        M, inl = cv2.estimateAffinePartial2D(
            a, b,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if M is None or inl is None or int(inl.sum()) < EGO_MIN_FEATURES // 2:
            return None
        return M


def _bbox_spans_red_and_orange(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    zones: dict,
) -> bool:
    if y2 <= y1 or "orange" not in zones or "red" not in zones:
        return False
    seen: set[str] = set()
    cx = (x1 + x2) // 2
    for t in (0.0, 0.12, 0.28, 0.45, 0.62, 0.78, 0.92, 1.0):
        yy = int(y1 + (y2 - y1) * t)
        z = classify_zone(cx, yy, ys, ls, rs, zones)
        if z:
            seen.add(z)
        if "red" in seen and "orange" in seen:
            return True
    y_foot = max(y1, y2 - 3)
    for xx in (x1 + 2, cx, x2 - 2):
        xx = int(min(max(xx, x1), x2))
        z = classify_zone(xx, y_foot, ys, ls, rs, zones)
        if z:
            seen.add(z)
        if "red" in seen and "orange" in seen:
            return True
    return "red" in seen and "orange" in seen

def _expanded_interest_band(
    ys: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
    zones: dict,
    w: int,
    buffer_gauge_frac: float = _ROI_BUFFER_GAUGE_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Yellow outer edge + buffer_gauge_frac * gauge width OUTSIDE on left/right."""
    widths = np.maximum(rs - ls, 1).astype(float)
    buf_px = (buffer_gauge_frac * widths).astype(np.int32)
    yl, yr = _outer_bounds(zones, ls, rs)
    el = np.clip(yl.astype(np.int32) - buf_px, 0, w - 1)
    er = np.clip(yr.astype(np.int32) + buf_px, 0, w - 1)
    return el, er


def _expanded_band_poly(
    ys: np.ndarray,
    exp_l: np.ndarray,
    exp_r: np.ndarray,
) -> np.ndarray:
    lp = np.column_stack([exp_l, ys])
    rp = np.column_stack([exp_r, ys])
    return np.vstack([lp, rp[::-1]]).astype(np.int32)


def _mask_frame_for_roi_det(
    frame: np.ndarray,
    ys: np.ndarray,
    exp_l: np.ndarray,
    exp_r: np.ndarray,
) -> np.ndarray:
    """Black-out everything outside the expanded band (invisible, not drawn)."""
    poly = _expanded_band_poly(ys, exp_l, exp_r)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_8)
    out = np.zeros_like(frame)
    out[mask > 0] = frame[mask > 0]
    return out


def _bbox_in_expanded_roi(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ys: np.ndarray,
    exp_l: np.ndarray,
    exp_r: np.ndarray,
) -> bool:
    if len(ys) < 2:
        return False
    ymn, ymx = int(ys.min()), int(ys.max())
    y_lo, y_hi = int(min(y1, y2)), int(max(y1, y2))
    for yy in (y_lo, (y_lo + y_hi) // 2, y_hi):
        yy = int(np.clip(yy, ymn, ymx))
        fl = float(np.interp(yy, ys.astype(np.float64), exp_l.astype(np.float64)))
        fr = float(np.interp(yy, ys.astype(np.float64), exp_r.astype(np.float64)))
        el, er = int(round(fl)), int(round(fr))
        if el > er:
            el, er = er, el
        if not (x2 < el or x1 > er):
            return True
    return False


def alert_distance_in_band(speed_kmh: int, distance_m: Optional[float]) -> bool:
    if distance_m is None:
        return False
    d = max(0.0, float(distance_m))
    v = max(0, min(75, int(speed_kmh)))
    if v <= 30:
        return d <= 15.0
    if v < 44:
        return 2.0 <= d <= 10.0
    return d >= 10.0


def compute_alert(
    zone: Optional[str],
    obj_name: str,
    distance_m: Optional[float],
    speed: int,
) -> Optional[Tuple[str, str]]:
    if zone is None:
        return None
    if not alert_distance_in_band(speed, distance_m):
        return None

    d = distance_m
    if d is None:
        return None
    dist_human = f"{d:.1f} м"

    if zone == "red":
        ru = _RU_ALERT_RED
        console = (
            f"{ru} | объект={obj_name} | дистанция={dist_human} | скорость={speed} км/ч"
        )
        return (console, ru)
    if zone == "orange":
        ru = _RU_ALERT_ORANGE
        console = (
            f"{ru} | объект={obj_name} | дистанция={dist_human} | скорость={speed} км/ч"
        )
        return (console, ru)
    if zone == "yellow":
        ru = _RU_ALERT_YELLOW
        console = (
            f"{ru} | объект={obj_name} | дистанция={dist_human} | скорость={speed} км/ч"
        )
        return (console, ru)
    return None


def _write_danger_log_line(
    danger_f,
    ts_iso: str,
    obj_name: str,
    distance_m: Optional[float],
    speed_kmh: int,
    action_ru: str,
    frame_idx: int,
) -> None:
    date_s, _sep, time_s = ts_iso.partition(" ")
    dist_s = f"{distance_m:.2f}" if distance_m is not None else "na"
    danger_f.write(
        f"{ts_iso}\tdate={date_s}\ttime={time_s}\tclass={obj_name}\t"
        f"dist_m={dist_s}\tspeed_kmh={speed_kmh}\taction={action_ru}\tframe={frame_idx}\n"
    )
    danger_f.flush()


_LT = cv2.LINE_8
_FONT = cv2.FONT_HERSHEY_PLAIN
_FONT_SCALE = 1


def _draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    obj_name: str,
    zone: Optional[str],
    distance_m: Optional[float],
    override_color: Optional[Tuple[int, int, int]] = None,
    override_thickness: Optional[int] = None,
    track_id: Optional[int] = None,
) -> None:
    if override_color is not None:
        color = override_color
        thickness = override_thickness if override_thickness is not None else 3
    else:
        color = ALERT_COLORS.get(zone, _NEUTRAL_COLOR) if zone else _NEUTRAL_COLOR
        thickness = 2 if zone else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = obj_name
    if track_id is not None and track_id >= 0:
        label += f" #{int(track_id)}"
    if distance_m is not None:
        label += f" {distance_m:.1f}m"
    (tw, th), bl = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
    pad_y = th + bl + 6
    cv2.rectangle(frame, (x1, y1 - pad_y), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - bl - 3),
        _FONT, _FONT_SCALE, (255, 255, 255), 1, _LT,
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--speed", type=int, required=True)
    ap.add_argument("-v", "--video", type=str, default=roi.VIDEO_PATH)
    ap.add_argument("-o", "--output", type=str, default=OUTPUT_PATH)
    ap.add_argument("--det-conf", type=float, default=DET_CONF)
    ap.add_argument("--det-model", type=str, default=str(DET_MODEL_PATH))
    ap.add_argument(
        "--det-classes",
        type=str,
        default=DET_CLASSES_DEFAULT
    )
    ap.add_argument(
        "--roi",
        action="store_true",
        help=(
            "Detect/track in colored zones + "
            f"{_ROI_BUFFER_GAUGE_FRAC:.1f}× gauge width outside yellow (masked, not drawn)"
        ),
    )
    ap.add_argument("--seg-model", type=str, default=None)
    ap.add_argument(
        "-m", "--morph", action="store_true", dest="morph",
    )
    ap.add_argument("--hold-sec", type=float, default=_ALERT_HOLD_SEC)
    ap.add_argument("--fov", type=float, default=roi.DEFAULT_FOV_DEG)
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    args = ap.parse_args()

    speed = max(1, min(75, args.speed))

    seg_model = roi.load_model(args.seg_model)
    det_model = YOLO(args.det_model)
    hazard_class_names = _build_hazard_class_names(det_model)
    track_class_names = _build_track_class_names(det_model)
    _dc = (args.det_classes or "").strip().lower()
    if not _dc or _dc == "all":
        det_class_ids = None
    else:
        det_class_ids = _det_class_ids(det_model, args.det_classes)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hold_frames = max(1, int(fps * args.hold_sec))
    _log_date = datetime.now().strftime("%d_%m_%Y")
    log_path = _DIR / f"detection_log_{_log_date}.txt"
    danger_log_path = _DIR / f"DANGER_LOG_{_log_date}.txt"

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H),
        )

    f_px = roi.get_focal_px(W, args.fov)

    ORANGE_RED_SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    temporal = roi._TemporalState()
    tracker = _AlertTracker(fps, args.hold_sec)
    cross_tracker = _CrossingTracker(fps)
    train_warn_last: Dict[Tuple[int, int, int], int] = {}
    traj_hist: Dict[int, deque] = {}
    trail_pts: Dict[int, deque] = {}
    track_last_seen: Dict[int, int] = {}
    # ego-motion: оценщик движения камеры + пер-трековые компенсированные смещения
    ego = _EgoMotion()
    prev_obj_boxes: List[Tuple[int, int, int, int]] = []
    track_prev_pos: Dict[int, Tuple[int, int]] = {}
    track_resid: Dict[int, deque] = {}
    n = 0
    shot_n = 0

    with open(log_path, "a", encoding="utf-8") as log_f, open(
        danger_log_path, "a", encoding="utf-8"
    ) as danger_f:
        _run_hdr = (
            f"\n!!!!!! run {datetime.now().isoformat()} speed_kmh={speed} "
            f"video={args.video} hold_sec={args.hold_sec} fps={fps} "
            f"roi_filter={args.roi} roi_buffer_gauge_frac={_ROI_BUFFER_GAUGE_FRAC} !!!!!!!\n"
        )
        log_f.write(_run_hdr)
        log_f.flush()
        danger_f.write(
            f"{_run_hdr}"
            "ts\tdate\ttime\tclass\tdist_m\tspeed_kmh\taction\tframe\n"
        )
        danger_f.flush()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]

            # Движение камеры prev->cur (по чистому кадру, до отрисовки зон).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ego_M = ego.update(gray, prev_obj_boxes)

            seg_res = seg_model(frame, conf=roi.CONF, device=roi.DEVICE, verbose=False)[0]
            roi._maybe_save_low_conf_frame(frame, roi._max_rail_confidence(seg_res))
            full_mask = roi.extract_rail_mask(seg_res, h, w, morph=args.morph)

            if np.any(full_mask):
                groups = roi.find_groups(full_mask)
                gmask = roi._select_rail_group(groups, h, temporal.center())
                ys, ls, rs = roi.rail_rows(gmask)
            else:
                ys = np.array([], dtype=np.int32)
                ls = np.array([], dtype=np.int32)
                rs = np.array([], dtype=np.int32)

            ys, ls, rs = temporal.update(ys, ls, rs, w)
            has_roi = len(ys) >= 4

            zones: dict = {}
            exp_l: Optional[np.ndarray] = None
            exp_r: Optional[np.ndarray] = None
            outer_l: Optional[np.ndarray] = None
            outer_r: Optional[np.ndarray] = None
            if has_roi:
                zones = roi.zone_bounds(ys, ls, rs, w)
                grid = roi.grid_positions(ys, ls, rs, f_px)
                roi.draw_zones(frame, ys, zones, ls, rs, grid)
                outer_l, outer_r = _outer_bounds(zones, ls, rs)
                if args.roi:
                    exp_l, exp_r = _expanded_interest_band(ys, ls, rs, zones, w)

            _dkw: dict = {
                "conf": args.det_conf,
                "device": roi.DEVICE,
                "verbose": False,
            }
            if det_class_ids is not None:
                _dkw["classes"] = det_class_ids

            det_res = None
            if not args.roi or has_roi:
                det_frame = frame
                if (
                    args.roi
                    and has_roi
                    and exp_l is not None
                    and exp_r is not None
                ):
                    det_frame = _mask_frame_for_roi_det(frame, ys, exp_l, exp_r)
                det_res = det_model.track(
                    det_frame, tracker=args.tracker, persist=True, **_dkw
                )[0]

            seen_keys: set = set()
            all_det_keys: set = set()
            seen_ids: set = set()
            going_now: set = set()
            cur_obj_boxes: List[Tuple[int, int, int, int]] = []
            save_orange_red_shot = False
            ts_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            if det_res is not None and det_res.boxes is not None:
                boxes_xyxy = det_res.boxes.xyxy.cpu().numpy()
                classes = det_res.boxes.cls.cpu().numpy().astype(int)
                if det_res.boxes.id is not None:
                    track_ids = det_res.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.full(len(classes), -1, dtype=int)

                for box, cls_id, tid in zip(boxes_xyxy, classes, track_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    obj_name = det_model.names.get(cls_id, f"class_{cls_id}")
                    cx = (x1 + x2) // 2
                    cy_foot = int(y2)
                    # все рамки маскируем при оценке ego-motion следующего кадра
                    cur_obj_boxes.append((int(x1), int(y1), int(x2), int(y2)))

                    cls_name_l = _norm_det_name(det_model, cls_id)
                    # Трекинг (ID + траектория + след) только для транспорта,
                    # людей и животных. Остальное — без трекинга.
                    is_trackable = cls_name_l in track_class_names
                    eff_tid = int(tid) if (is_trackable and tid >= 0) else -1

                    # --- ego-motion: реальное движение объекта vs «плывущая» рамка ---
                    # Предсказываем, куда уехала бы рамка, будь объект статичным
                    # (только из-за камеры). Остаток = собственное движение объекта.
                    really_moving = False
                    net_dx = 0.0
                    if eff_tid >= 0:
                        prev_pos = track_prev_pos.get(eff_tid)
                        if prev_pos is not None:
                            pcx, pcy = _warp_affine_pt(
                                ego_M, prev_pos[0], prev_pos[1]
                            )
                            rdx = cx - pcx
                            rdy = cy_foot - pcy
                        else:
                            rdx = rdy = 0.0
                        rb = track_resid.setdefault(eff_tid, deque(maxlen=HISTORY))
                        rb.append((rdx, rdy))
                        track_prev_pos[eff_tid] = (cx, cy_foot)
                        # нетто-смещение за окно (джиттер взаимно гасится)
                        net_dx = float(sum(d[0] for d in rb))
                        net_dy = float(sum(d[1] for d in rb))
                        really_moving = (
                            (net_dx * net_dx + net_dy * net_dy) ** 0.5
                            >= MIN_REAL_MOVE_PX
                        )

                    # --- траектория к ROI: ТОЛЬКО если объект реально движется ---
                    going = False
                    if really_moving and has_roi and outer_l is not None and eff_tid >= 0:
                        L, R = _roi_band_at_y(cy_foot, ys, outer_l, outer_r)
                        dist = _horiz_gap(x1, x2, L, R)
                        roi_cx = 0.5 * (L + R)
                        hist = traj_hist.setdefault(eff_tid, deque(maxlen=HISTORY))
                        if len(hist) >= 1:
                            ref_cx, ref_cy, ref_dist, ref_roi_cx = hist[0]
                            # направление «к ROI» — по геометрии текущего кадра
                            h_dir = _sign(ref_roi_cx - ref_cx)
                            # движение берём КОМПЕНСИРОВАННОЕ (без сдвига камеры)
                            if h_dir == 0:
                                moving_toward = True
                            else:
                                moving_toward = (
                                    _sign(net_dx) == h_dir and abs(net_dx) >= MIN_MOVE_PX
                                )
                            if moving_toward and dist < ref_dist - DIST_EPS_PX:
                                going = True
                                print(f"WATCH THE OBJECT {obj_name}")
                        hist.append((cx, cy_foot, dist, roi_cx))

                    if eff_tid >= 0:
                        track_last_seen[eff_tid] = n
                        seen_ids.add(eff_tid)
                        # след движения рисуем только для реально движущихся объектов;
                        # для статичных (дрейф камеры) трекинг-след игнорируем
                        if really_moving:
                            trail_pts.setdefault(
                                eff_tid, deque(maxlen=TRAIL_LEN)
                            ).append((cx, cy_foot))
                            if going:
                                going_now.add(eff_tid)

                    if args.roi:
                        if (
                            exp_l is None
                            or exp_r is None
                            or not _bbox_in_expanded_roi(
                                x1, y1, x2, y2, ys, exp_l, exp_r
                            )
                        ):
                            continue

                    if cls_name_l == _TRAIN_CLASS_NAME:
                        all_det_keys.add(tracker._key(cls_id, cx, cy_foot))
                        zone_t: Optional[str] = None
                        dist_t: Optional[float] = None
                        crossing_t = False
                        if has_roi and zones:
                            zone_t = classify_zone(
                                cx, cy_foot, ys, ls, rs, zones
                            )
                            if zone_t is not None:
                                dist_t = estimate_distance_m(
                                    cy_foot, ys, ls, rs, f_px
                                )
                            crossing_t = _bbox_spans_red_and_orange(
                                x1, y1, x2, y2, ys, ls, rs, zones
                            )
                            if crossing_t:
                                save_orange_red_shot = True
                        nf_cross_t = cross_tracker.tick(
                            cls_id, cx, cy_foot, crossing_t
                        )
                        cross_sec_t = nf_cross_t / float(fps)
                        d_ts = (
                            f"{dist_t:.2f}" if dist_t is not None else "na"
                        )
                        z_ts = zone_t if zone_t is not None else "na"
                        log_f.write(
                            f"[TRAIN] {ts_log} frame={n + 1} obj={obj_name} "
                            f"zone={z_ts} dist_m={d_ts} crossing={crossing_t} "
                            f"dwell_in_crossing_s={cross_sec_t:.3f} "
                            f"speed_kmh={speed}\n"
                        )
                        log_f.flush()
                        wk = tracker._key(cls_id, cx, cy_foot)
                        _tw_gap = max(1, fps // 2)
                        if n - train_warn_last.get(wk, -10**9) >= _tw_gap:
                            if alert_distance_in_band(speed, dist_t):
                                _dist_h = (
                                    f"{dist_t:.1f} м"
                                    if dist_t is not None
                                    else "н/д"
                                )
                                _tcons = (
                                    f"{_RU_ALERT_TRAIN} | объект={obj_name} | "
                                    f"дистанция={_dist_h} | скорость={speed} км/ч"
                                )
                                print(
                                    f"[DANGER {ts_log}] frame={n + 1} {_tcons}"
                                )
                                train_warn_last[wk] = n
                                _write_danger_log_line(
                                    danger_f,
                                    ts_log,
                                    str(obj_name),
                                    dist_t,
                                    speed,
                                    _RU_ALERT_TRAIN,
                                    n + 1,
                                )
                        _draw_detection(
                            frame,
                            x1,
                            y1,
                            x2,
                            y2,
                            obj_name,
                            zone_t,
                            dist_t,
                            override_color=(255, 0, 255),
                            track_id=eff_tid,
                        )
                        continue

                    if cls_name_l not in hazard_class_names:
                        continue

                    all_det_keys.add(tracker._key(cls_id, cx, cy_foot))

                    zone: Optional[str] = None
                    dist_m: Optional[float] = None
                    crossing = False

                    if has_roi and zones:
                        zone = classify_zone(cx, cy_foot, ys, ls, rs, zones)
                        if zone is not None:
                            dist_m = estimate_distance_m(cy_foot, ys, ls, rs, f_px)
                        crossing = _bbox_spans_red_and_orange(
                            x1, y1, x2, y2, ys, ls, rs, zones
                        )
                        if crossing:
                            save_orange_red_shot = True

                    nf_cross = cross_tracker.tick(cls_id, cx, cy_foot, crossing)
                    cross_sec = nf_cross / float(fps)

                    if zone is not None:
                        key = tracker._key(cls_id, cx, cy_foot)
                        seen_keys.add(key)
                        _confirmed, nf_zone = tracker.tick(cls_id, cx, cy_foot)
                        dwell_zone_s = nf_zone / float(fps)
                        msg_pair = compute_alert(zone, obj_name, dist_m, speed)
                        d_s = f"{dist_m:.2f}" if dist_m is not None else "na"

                        if crossing:
                            log_f.write(
                                f"[CROSS] {ts_log} frame={n + 1} obj={obj_name} "
                                f"dist_m={d_s} zone={zone} dwell_in_crossing_s={cross_sec:.3f} "
                                f"dwell_in_zone_s={dwell_zone_s:.3f} speed_kmh={speed}\n"
                            )
                        else:
                            log_f.write(
                                f"[ZONE] {ts_log} frame={n + 1} obj={obj_name} "
                                f"zone={zone} dist_m={d_s} dwell_in_zone_s={dwell_zone_s:.3f} "
                                f"speed_kmh={speed}\n"
                            )
                        log_f.flush()

                        if (
                            msg_pair is not None
                            and nf_zone == hold_frames
                        ):
                            console_line, action_ru = msg_pair
                            print(
                                f"[DANGER {ts_log}] frame={n + 1} {console_line} "
                                f"(зона ≥{args.hold_sec:.1f} с)"
                            )
                            _write_danger_log_line(
                                danger_f,
                                ts_log,
                                str(obj_name),
                                dist_m,
                                speed,
                                action_ru,
                                n + 1,
                            )

                    _draw_detection(
                        frame, x1, y1, x2, y2, obj_name, zone, dist_m, track_id=eff_tid
                    )

            cross_tracker.end_frame(all_det_keys)

            if save_orange_red_shot:
                shot_n += 1
                ts = datetime.now().strftime("%d_%m_%Y_%H%M%S_%f")[:-3]
                out_png = ORANGE_RED_SHOTS_DIR / f"frame{n + 1:06d}_{shot_n:04d}_{ts}.png"
                cv2.imwrite(str(out_png), frame)

            tracker.end_frame(seen_keys)

            # визуализация следов трекинга (красный = идёт к ROI)
            for tid in seen_ids:
                pts = trail_pts.get(tid)
                if not pts or len(pts) < 2:
                    continue
                arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                tc = _TRAIL_WATCH_COLOR if tid in going_now else _TRAIL_COLOR
                cv2.polylines(frame, [arr], False, tc, 2, cv2.LINE_AA)
                px, py = pts[-1]
                cv2.circle(frame, (int(px), int(py)), 3, tc, -1)

            # забываем старые треки
            for tid in list(track_last_seen.keys()):
                if n - track_last_seen[tid] > TRACK_TTL:
                    track_last_seen.pop(tid, None)
                    traj_hist.pop(tid, None)
                    trail_pts.pop(tid, None)
                    track_prev_pos.pop(tid, None)
                    track_resid.pop(tid, None)

            # рамки этого кадра -> маска фона для ego-motion следующего кадра
            prev_obj_boxes = cur_obj_boxes

            cv2.imshow("All detection + zones", frame)
            if writer:
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            n += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
