from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _ROOT / "models"

SEGMENTOR_REPO_ID = "ayzeksalimli/tram-dynamic-roi-tracker-yolo11s"
SEGMENTOR_FILENAME = "tram-dynamic-roi-tracker-yolo11s.pt"
SEGMENTOR_DIR = MODELS_DIR / "segmentor"

DETECTOR_FILENAME = "yolov8s.pt"
DETECTOR_DIR = MODELS_DIR / "detector"

TRACKER_FILENAME = "bytetrack.yaml"
TRACKER_DIR = MODELS_DIR / "trackers"


def download_segmentor() -> Path:
    from huggingface_hub import hf_hub_download

    SEGMENTOR_DIR.mkdir(parents=True, exist_ok=True)
    dest = SEGMENTOR_DIR / SEGMENTOR_FILENAME
    if dest.exists():
        print(f"[segmentor] already present: {dest}")
        return dest

    print(f"[segmentor] downloading {SEGMENTOR_REPO_ID} ...")
    local_path = hf_hub_download(repo_id=SEGMENTOR_REPO_ID, filename=SEGMENTOR_FILENAME)
    shutil.copy2(local_path, dest)
    print(f"[segmentor] saved to {dest}")
    return dest


def download_detector() -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    dest = DETECTOR_DIR / DETECTOR_FILENAME
    if dest.exists():
        print(f"[detector] already present: {dest}")
        return dest

    print(f"[detector] downloading {DETECTOR_FILENAME} ...")
    attempt_download_asset(str(dest))
    print(f"[detector] saved to {dest}")
    return dest


def download_tracker() -> Path:
    import ultralytics

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRACKER_DIR / TRACKER_FILENAME
    if dest.exists():
        print(f"[tracker] already present: {dest}")
        return dest

    src = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / TRACKER_FILENAME
    if not src.exists():
        print(f"[tracker] ERROR: bundled config not found at {src}", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(src, dest)
    print(f"[tracker] saved to {dest}")
    return dest


def main() -> None:
    download_segmentor()
    download_detector()
    download_tracker()
    print("\nAll models are ready in", MODELS_DIR)


if __name__ == "__main__":
    main()
