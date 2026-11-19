# %%
# CELL 1 - install
!pip -q install -U supervision ultralytics yt-dlp
import supervision as sv, ultralytics
print("supervision", sv.__version__, "| ultralytics", ultralytics.__version__)

# %%
# CELL 2 - get video
import glob, os, subprocess
WORK = "/kaggle/working"
VIDEO_PATH = f"{WORK}/traffic.mp4"
URL = "https://youtu.be/wqctLW0Hb_0"

if not os.path.exists(VIDEO_PATH):
    r = subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]/b[ext=mp4]/b",
                        "-o", VIDEO_PATH, URL], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])

if not os.path.exists(VIDEO_PATH):
    found = glob.glob("/kaggle/input/**/*.mp4", recursive=True)
    print("YouTube download failed. mp4 files in /kaggle/input:", found)
    assert found, "Upload the video as a Kaggle dataset (Add Input -> Upload) and re-run."
    VIDEO_PATH = found[0]

info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(VIDEO_PATH, info)

# %%
# CELL 3 - show frame with pixel grid
import cv2, matplotlib.pyplot as plt
frame = next(sv.get_video_frames_generator(VIDEO_PATH))
H, W = frame.shape[:2]
plt.figure(figsize=(14, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.xticks(range(0, W, 100)); plt.yticks(range(0, H, 100)); plt.grid(alpha=.5, color="yellow")
plt.title(f"{W}x{H} - read pixel coords of 4 road corners"); plt.show()

# %%
# CELL 4 - config (EDIT the SOURCE points after looking at the grid)
import numpy as np

# Pixel corners: top-left, top-right, bottom-right, bottom-left. These defaults are a ROUGH GUESS
# (fractions of the frame) - replace with your own pixel values from the grid.
SOURCE = np.array([[0.46*W, 0.25*H], [0.62*W, 0.25*H], [0.95*W, 0.85*H], [0.05*W, 0.85*H]], dtype=np.float32)

# Real-world size (metres) of that rectangle. Estimate from lane width (~3.5 m per lane)
# and lane dashes (check local standard; commonly ~3 m dash + ~9 m gap).
TARGET_WIDTH_M  = 14.0
TARGET_LENGTH_M = 40.0

# The video description says it "uses time-lapse". If it is sped up N times, real speed = shown speed / N.
# Set N here once you know it (1.0 = not time-lapsed).
TIMELAPSE_FACTOR = 1.0

MODEL = "yolov8m.pt"      # accuracy ladder: yolov8s < yolov8m < yolov8l < yolov8x (slower)
CONF, IOU, IMGSZ = 0.30, 0.50, 1280
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)

# %%
# CELL 5 - check the calibration visually
prev = frame.copy()
cv2.polylines(prev, [SOURCE.astype(np.int32)], True, (0, 255, 0), 4)
plt.figure(figsize=(14, 8)); plt.imshow(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)); plt.title("Green = measured zone"); plt.show()

# %%
# CELL 6 - pipeline
from collections import defaultdict, deque
from ultralytics import YOLO
import pandas as pd

class ViewTransformer:
    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
    def transform(self, pts):
        if pts.size == 0: return pts
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2)

TARGET = np.array([[0, 0], [TARGET_WIDTH_M, 0], [TARGET_WIDTH_M, TARGET_LENGTH_M], [0, TARGET_LENGTH_M]], dtype=np.float32)
vt = ViewTransformer(SOURCE, TARGET)

model = YOLO(MODEL)
fps = info.fps
tracker = sv.ByteTrack(frame_rate=int(round(fps)))
zone = sv.PolygonZone(polygon=SOURCE.astype(np.int32))
box_ann = sv.BoxAnnotator(thickness=2)
lab_ann = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_position=sv.Position.BOTTOM_CENTER)
trace_ann = sv.TraceAnnotator(thickness=2, trace_length=int(fps * 2), position=sv.Position.BOTTOM_CENTER)

WIN = max(int(fps), 5)                       # ~1 second window of positions
coords = defaultdict(lambda: deque(maxlen=WIN))
speed_hist = defaultdict(list)               # tracker_id -> list of km/h
OUT = "/kaggle/working/speed_result.mp4"

with sv.VideoSink(OUT, info) as sink:
    for f in sv.get_video_frames_generator(VIDEO_PATH):
        res = model(f, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=VEHICLE_CLASSES, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        det = det[zone.trigger(det)]
        det = tracker.update_with_detections(det)

        pts = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pts = vt.transform(pts)
        labels = []
        for tid, (x, y) in zip(det.tracker_id, pts):
            coords[tid].append((x, y))
            if len(coords[tid]) < WIN // 2:
                labels.append(f"#{tid}")
                continue
            (x0, y0), (x1, y1) = coords[tid][0], coords[tid][-1]
            dist = float(np.hypot(x1 - x0, y1 - y0))
            secs = (len(coords[tid]) - 1) / fps
            kmh = dist / secs * 3.6 / TIMELAPSE_FACTOR
            speed_hist[tid].append(kmh)
            labels.append(f"#{tid} {int(kmh)} km/h")

        out = trace_ann.annotate(f.copy(), det)
        out = box_ann.annotate(out, det)
        out = lab_ann.annotate(out, det, labels=labels)
        sink.write_frame(out)

rows = [{"id": k, "median_kmh": float(np.median(v)), "max_kmh": float(np.max(v)), "n": len(v)}
        for k, v in speed_hist.items() if len(v) >= 5]
df = pd.DataFrame(rows).sort_values("id")
df.to_csv("/kaggle/working/vehicle_speeds.csv", index=False)
print(df.describe().round(1)); df.head(15)

# %%
# CELL 7 - preview the result video inside the notebook
!ffmpeg -loglevel error -y -i /kaggle/working/speed_result.mp4 -vcodec libx264 -pix_fmt yuv420p /kaggle/working/speed_result_web.mp4
from IPython.display import Video
Video("/kaggle/working/speed_result_web.mp4", embed=True, width=800)

# %%
# CELL 8 - sanity checks
import matplotlib.pyplot as plt
print("Median of vehicle medians (km/h):", round(df.median_kmh.median(), 1))
plt.hist(df.median_kmh, bins=20); plt.xlabel("km/h"); plt.ylabel("vehicles"); plt.title("Estimated speeds"); plt.show()

# Known-distance check: put two pixel points (x, y) that are a known real distance apart ALONG the road
P1, P2, KNOWN_M = (900, 400), (910, 520), 12.0     # EDIT: e.g. start of one lane dash to start of the next = 12 m
a, b = vt.transform(np.array([P1, P2], dtype=np.float32))
print(f"Estimated {np.hypot(*(a-b)):.1f} m vs known {KNOWN_M} m")

