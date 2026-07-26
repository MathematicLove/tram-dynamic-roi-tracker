# Tram Dynamic ROI Tracker

Algorithm for detecting foreign objects on tram/train tracks based on dynamic
ROI segmentation (**YOLOv11s-seg**), object detection (**YOLOv8s**) and
tracking (**ByteTrack**).

Full project source: https://github.com/MathematicLove/tram-dynamic-roi-tracker

---

## What's in the image

The container starts a small desktop launcher UI (`ui/app.py`) with three modes:

- **Camera Mode** — run the pipeline on a live camera feed.
- **Video Mode** — drop/select a video file and run the pipeline on it.
- **Image Mode** — drop/select a single image and get back a segmented + annotated result.

The UI is a Tkinter window, so it needs a display to render to (see below).

---

## Running

### Prerequisites: X11 display

The container renders windows via X11, so the host needs an X server and the
container needs `DISPLAY` + the X11 socket.

**Linux:**
```bash
xhost +local:docker
```

**macOS** (via [XQuartz](https://www.xquartz.org/)):
```bash
xhost + 127.0.0.1
```
and set `DISPLAY=host.docker.internal:0` instead of the default below.

### docker run

```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/videos:/app/videos:ro \
  --network host \
  ayzeksalimli/tram-dynamic-roi-tracker:latest
```

### docker compose

```bash
docker compose up
```

Uses `DISPLAY` from your shell (defaults to `:0`), mounts `./models`,
`./results` and `./videos` into the container, and launches the UI by
default. See `docker-compose.yml` in the repo for the full definition.

### Camera passthrough (Linux only)

Docker Desktop on macOS/Windows cannot pass a host camera into a container.
On Linux, uncomment the `devices` section in `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
```

On macOS, use **Video Mode** or **Image Mode** instead, or run `ui/app.py`
natively on the host for Camera Mode.

---

## Volumes

| Path in container | Purpose |
|---|---|
| `/app/models` | Segmentation/detection/tracker model weights, auto-downloaded on first start |
| `/app/results` | Output videos, images, logs, crossing snapshots |
| `/app/videos` | Read-only input videos for Video Mode |

---

## Running the CLI directly (no UI)

The launcher just wraps these scripts — you can call them directly too:

```bash
# Video / camera, full pipeline (segmentation + detection + tracking)
docker run ... ayzeksalimli/tram-dynamic-roi-tracker:latest \
  python algorithm/detection.py --speed 40 --video /app/videos/input.mp4

# Single image
docker run ... ayzeksalimli/tram-dynamic-roi-tracker:latest \
  python algorithm/image_infer.py --image /app/videos/frame.jpg --show
```

---

## Models

Weights are hosted on Hugging Face and downloaded automatically on container
start (`scripts/download_models.py`, run by the entrypoint):

- Segmentor: [tram-dynamic-roi-tracker-yolo11s](https://huggingface.co/ayzeksalimli/tram-dynamic-roi-tracker-yolo11s)
- Detector: YOLOv8s (Ultralytics)
- Tracker: ByteTrack

---

by Salimli Ayzek (Салимли Айзек): https://mathematiclove.github.io
