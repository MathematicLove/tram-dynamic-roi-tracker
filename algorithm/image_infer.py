"""Run the segment -> ROI -> detection pipeline on a single image.

Usage:
    python algorithm/image_infer.py -i path/to/image.jpg --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import detection as det_mod
import roi

_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _ROOT / "results" / "images"


def process_image(
    image_path: str,
    output_path: str,
    seg_model: YOLO,
    det_model: YOLO,
    det_conf: float,
    fov_deg: float,
    morph: bool,
) -> str:
    frame = cv2.imread(image_path)
    if frame is None:
        raise OSError(f"Failed to read image: {image_path}")
    h, w = frame.shape[:2]

    seg_res = seg_model(frame, conf=roi.CONF, device=roi.DEVICE, verbose=False)[0]
    full_mask = roi.extract_rail_mask(seg_res, h, w, morph=morph)

    ys = np.array([], dtype=np.int32)
    ls = np.array([], dtype=np.int32)
    rs = np.array([], dtype=np.int32)
    if np.any(full_mask):
        groups = roi.find_groups(full_mask)
        gmask = roi._select_rail_group(groups, h, None)
        ys, ls, rs = roi.rail_rows(gmask)

    has_roi = len(ys) >= 4
    zones: dict = {}
    f_px = roi.get_focal_px(w, fov_deg)
    if has_roi:
        zones = roi.zone_bounds(ys, ls, rs, w)
        grid = roi.grid_positions(ys, ls, rs, f_px)
        roi.draw_zones(frame, ys, zones, ls, rs, grid)

    hazard_names = det_mod._build_hazard_class_names(det_model)
    det_res = det_model(frame, conf=det_conf, device=roi.DEVICE, verbose=False)[0]

    if det_res.boxes is not None:
        boxes_xyxy = det_res.boxes.xyxy.cpu().numpy()
        classes = det_res.boxes.cls.cpu().numpy().astype(int)
        for box, cls_id in zip(boxes_xyxy, classes):
            x1, y1, x2, y2 = box.astype(int)
            obj_name = det_model.names.get(cls_id, f"class_{cls_id}")
            cls_name_l = det_mod._norm_det_name(det_model, cls_id)
            is_train = cls_name_l == det_mod._TRAIN_CLASS_NAME
            if not is_train and cls_name_l not in hazard_names:
                continue

            cx = (x1 + x2) // 2
            cy_foot = int(y2)
            zone = None
            dist_m = None
            if has_roi and zones:
                zone = det_mod.classify_zone(cx, cy_foot, ys, ls, rs, zones)
                if zone is not None:
                    dist_m = det_mod.estimate_distance_m(cy_foot, ys, ls, rs, f_px)

            override = (255, 0, 255) if is_train else None
            det_mod._draw_detection(
                frame, x1, y1, x2, y2, obj_name, zone, dist_m,
                override_color=override,
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment rails, build ROI zones and detect hazards on a single image"
    )
    ap.add_argument("-i", "--image", type=str, required=True)
    ap.add_argument("-o", "--output", type=str, default=None)
    ap.add_argument("--det-conf", type=float, default=det_mod.DET_CONF)
    ap.add_argument("--det-model", type=str, default=str(det_mod.DET_MODEL_PATH))
    ap.add_argument("--seg-model", type=str, default=None)
    ap.add_argument("--fov", type=float, default=roi.DEFAULT_FOV_DEG)
    ap.add_argument("-m", "--morph", action="store_true", dest="morph")
    ap.add_argument("--show", action="store_true", help="Open a window with the result")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    output_path = args.output or str(OUTPUT_DIR / f"{image_path.stem}_result.png")

    seg_model = roi.load_model(args.seg_model)
    det_model_path = Path(args.det_model)
    if not det_model_path.exists() and args.det_model == str(det_mod.DET_MODEL_PATH):
        print(
            f"[ERROR] Detector model not found: {det_model_path}\n"
            "Run: python scripts/download_models.py"
        )
        sys.exit(1)
    det_model = YOLO(str(det_model_path))

    out_path = process_image(
        str(image_path), output_path, seg_model, det_model,
        args.det_conf, args.fov, args.morph,
    )
    print(f"Saved: {out_path}")

    if args.show:
        result = cv2.imread(out_path)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
