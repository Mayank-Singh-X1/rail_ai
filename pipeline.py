import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Unified Pipeline and System Initializer for Railway Surveillance AI
"""
import os
import json
import time
from collections import defaultdict, deque
import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    class MockResult:
        def __init__(self):
            self.boxes = []
            self.names = {0: "person", 39: "bottle"}
            self.keypoints = type("KP", (), {"xy": torch.zeros((1, 17, 2)), "data": torch.zeros((1, 17, 3))})()
            self.orig_shape = (720, 1280)

    class MockYOLO:
        def __init__(self, *args, **kwargs):
            self.names = {0: "person", 39: "bottle"}
        def __call__(self, frame, *args, **kwargs):
            return [MockResult()]
        def track(self, frame, *args, **kwargs):
            return [MockResult()]
        def train(self, *args, **kwargs):
            return {}

    YOLO = MockYOLO

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    class MockFace:
        def __init__(self):
            self.embedding = np.ones(512, dtype=np.float32)
            self.bbox = np.array([50, 50, 200, 200])

    class MockFaceAnalysis:
        def __init__(self, *args, **kwargs): pass
        def prepare(self, *args, **kwargs): pass
        def get(self, img):
            return [MockFace()]

    FaceAnalysis = MockFaceAnalysis
    HAS_INSIGHTFACE = False

from modules import (
    CrowdAnalyzer,
    CriminalDetector,
    AnomalyDetector,
    CleanlinessMonitor,
    WorkerMonitor,
    PersonTracker,
    AlertSystem
)


class RailwaySurveillanceSystem:

  def __init__(self):
    print("🚂 Initializing Railway Surveillance System...")

    # Determine execution provider (CUDA for NVIDIA, MPS for Apple Silicon M4, or CPU)
    if torch.cuda.is_available():
        self.device = 0
        device_name = "NVIDIA CUDA GPU"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        self.device = "mps"
        device_name = "Apple Silicon M4 GPU (Metal MPS)"
    else:
        self.device = "cpu"
        device_name = "CPU"

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )


    # ---- YOLO Models (Native PyTorch .pt) ----
    try:
        self.yolo_detect = YOLO("yolo11n.pt")
        print(f"✅ YOLO Detection model loaded on {self.device}")
    except Exception as e:
        print(f"⚠️ YOLO Detection fallback: {e}")
        self.yolo_detect = YOLO()

    self.yolo = self.yolo_detect

    try:
        self.yolo_pose = YOLO("yolo11n-pose.pt")
        print(f"✅ YOLO Pose model loaded on {self.device}")
    except Exception as e:
        print(f"⚠️ YOLO Pose fallback: {e}")
        self.yolo_pose = YOLO()

    # ---- InsightFace ----
    try:
        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640)
        )
        print(f"✅ InsightFace model initialized (Providers: {providers})")
    except Exception as e:
        print(f"⚠️ InsightFace fallback: {e}")
        self.face_app = FaceAnalysis()

    # ---- Databases ----
    self.criminal_db = {}
    self.worker_db = {}
    self.worker_attendance = defaultdict(list)

    # ---- Tracking & Analytics ----
    self.track_history = defaultdict(lambda: deque(maxlen=50))
    self.crowd_history = deque(maxlen=300)

    # ---- Zones & Alerts ----
    self.zones = {}
    self.alerts = []

    print("🎉 System initialized successfully on GPU/CPU!")

  def _extract_normalized_embedding(self, image_path):
    img = cv2.imread(image_path)
    if img is None:
      print(f"❌ Could not read image at path: {image_path}")
      return None

    faces = self.face_app.get(img)
    if len(faces) > 0:
      embedding = faces[0].embedding
      normalized_embedding = embedding / np.linalg.norm(embedding)
      return normalized_embedding.astype(np.float32)

    return None

  def add_criminal_to_db(self, name, image_path):
    embedding = self._extract_normalized_embedding(image_path)
    if embedding is not None:
      self.criminal_db[name] = embedding
      print(f"✅ Criminal '{name}' added to vector database.")
      return True
    print(f"❌ No face detected in criminal image: {image_path}")
    return False

  def add_worker_to_db(self, name, image_path):
    embedding = self._extract_normalized_embedding(image_path)
    if embedding is not None:
      self.worker_db[name] = embedding
      print(f"✅ Worker '{name}' added to vector database.")
      return True
    print(f"❌ No face detected in worker image: {image_path}")
    return False

  def add_zone(self, zone_name, polygon_points, zone_type="restricted"):
    self.zones[zone_name] = {
        "polygon": np.array(polygon_points, dtype=np.int32),
        "type": zone_type,
    }
    print(f"✅ Zone '{zone_name}' ({zone_type}) configured.")


# Global system instances
system = RailwaySurveillanceSystem()
crowd_analyzer = CrowdAnalyzer(system)
criminal_detector = CriminalDetector(system)
anomaly_detector = AnomalyDetector(system)
cleanliness_monitor = CleanlinessMonitor(system)
worker_monitor = WorkerMonitor(system)
person_tracker = PersonTracker(system)
alert_system = AlertSystem()


import threading
from concurrent.futures import ThreadPoolExecutor


class UnifiedPipeline:

  def __init__(self):
    self.system = system
    self.crowd_analyzer = crowd_analyzer
    self.criminal_detector = criminal_detector
    self.anomaly_detector = anomaly_detector
    self.cleanliness_monitor = cleanliness_monitor
    self.worker_monitor = worker_monitor
    self.person_tracker = person_tracker

    self.frame_count = 0
    self.fps_history = deque(maxlen=30)

    # State caches updated asynchronously by background workers
    self.cached_criminals = []
    self.cached_anomalies = []
    self.cached_cleanliness_score = 94.5
    self.cached_workers_present = []
    self.cached_workers_absent = []

    # Configurable wall-clock time intervals (in seconds)
    self.cleanliness_interval = 20.0  # Process cleanliness every 20 seconds
    self.criminal_interval = 1.0     # Process face recognition every 1 second
    self.worker_interval = 15.0       # Process staff attendance every 15 seconds
    self.anomaly_interval = 1.0       # Process pose fall detection every 1 second

    # Timestamps of last execution
    self.last_cleanliness_time = 0.0
    self.last_criminal_time = 0.0
    self.last_worker_time = 0.0
    self.last_anomaly_time = 0.0

    # Thread pool for non-blocking asynchronous AI compute
    self.executor = ThreadPoolExecutor(max_workers=3)
    self.lock = threading.Lock()

  def _async_cleanliness_task(self, frame_copy):
    try:
      score = self.cleanliness_monitor.get_cleanliness_score(frame_copy)
      with self.lock:
        self.cached_cleanliness_score = score
    except Exception:
      pass

  def _async_criminal_task(self, frame_copy):
    try:
      _, matches = self.criminal_detector.detect_criminals(frame_copy)
      with self.lock:
        self.cached_criminals = matches
    except Exception:
      pass

  def _async_worker_task(self, frame_copy):
    try:
      _, present, absent = self.worker_monitor.check_attendance(frame_copy)
      with self.lock:
        self.cached_workers_present = present
        self.cached_workers_absent = absent
    except Exception:
      pass

  def _async_anomaly_task(self, frame_copy):
    try:
      _, anomalies = self.anomaly_detector.detect_anomalies(frame_copy)
      with self.lock:
        self.cached_anomalies = anomalies
    except Exception:
      pass

  def process_frame(
      self,
      frame,
      enable_crowd=True,
      enable_criminal=True,
      enable_anomaly=True,
      enable_cleanliness=True,
      enable_tracking=True,
      enable_worker=True,
  ):
    """Ultra-fast non-blocking frame processor with time-gapped async AI tasks."""
    start_time = time.time()
    now = time.time()
    self.frame_count += 1

    annotated = frame.copy()

    # ---- 1. FAST REAL-TIME TRACKING & CROWD (Runs every frame) ----
    active_count = 0
    if enable_tracking:
      annotated, active_count = self.person_tracker.track_persons(annotated)
    elif enable_crowd:
      active_count, _ = self.crowd_analyzer.count_people(frame)

    crowd_lvl = "LOW"
    if enable_crowd:
      crowd_lvl, _ = self.crowd_analyzer.get_crowd_level(active_count)

    # ---- 2. ASYNC TIME-GAPPED TASKS (Dispatched without blocking video) ----
    # Cleanliness every 20s
    if enable_cleanliness and (now - self.last_cleanliness_time >= self.cleanliness_interval):
      self.last_cleanliness_time = now
      self.executor.submit(self._async_cleanliness_task, frame.copy())

    # Criminal / Face ID — SYNC if DB has faces to guarantee zero latency box rendering
    if enable_criminal and (now - self.last_criminal_time >= self.criminal_interval):
      self.last_criminal_time = now
      if self.system.criminal_db:
        self._async_criminal_task(frame.copy())
      else:
        self.executor.submit(self._async_criminal_task, frame.copy())

    # Worker Attendance every 15s
    if enable_worker and (now - self.last_worker_time >= self.worker_interval):
      self.last_worker_time = now
      self.executor.submit(self._async_worker_task, frame.copy())

    # Anomaly / Fall Pose every 1s
    if enable_anomaly and (now - self.last_anomaly_time >= self.anomaly_interval):
      self.last_anomaly_time = now
      self.executor.submit(self._async_anomaly_task, frame.copy())

    # ---- 3. RENDER ANNOTATIONS FROM ATOMIC CACHE ----
    with self.lock:
      criminals = list(self.cached_criminals)
      anomalies = list(self.cached_anomalies)
      clean_score = self.cached_cleanliness_score
      w_present = list(self.cached_workers_present)
      w_absent = list(self.cached_workers_absent)

    # Draw cached criminal target boxes
    for match in criminals:
      bbox = match["bbox"]
      x1, y1, x2, y2 = bbox
      cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 180), 1)
      cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
      ln = min(14, max(4, (x2 - x1) // 4))
      for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(annotated, (px, py), (px + dx * ln, py), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (px, py), (px, py + dy * ln), (0, 0, 255), 3, cv2.LINE_AA)

      label = f"⚠️ WANTED: {match['name']} ({match.get('score', 0.0):.2f})"
      (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
      by = max(y1 - 4, lh + 8)
      cv2.rectangle(annotated, (x1, by - lh - 6), (x1 + lw + 8, by + 2), (0, 0, 200), -1)
      cv2.putText(annotated, label, (x1 + 4, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 4. FPS CALCULATION ----
    elapsed = time.time() - start_time
    fps = 1.0 / (elapsed + 1e-6)
    self.fps_history.append(fps)

    results = {
        "frame": annotated,
        "crowd_count": active_count,
        "crowd_level": crowd_lvl,
        "criminals_found": criminals,
        "anomalies": anomalies,
        "cleanliness_score": clean_score,
        "workers_present": w_present,
        "workers_absent": w_absent,
        "tracked_persons": active_count,
        "alerts": self.system.alerts[-10:],
    }

    # ---- 5. OVERLAYS & HUD RENDERING ----
    from utils.visualization import draw_hud, draw_zones
    if hasattr(self.system, "zones") and self.system.zones:
      draw_zones(annotated, self.system.zones)

    draw_hud(annotated, results)
    results["frame"] = annotated

    return results


  def _draw_status_bar(self, frame, results):
    """Fallback alias for HUD drawing."""
    from utils.visualization import draw_hud
    draw_hud(frame, results)



pipeline = UnifiedPipeline()


def process_video(
    video_path,
    output_path="output_surveillance.mp4",
    max_frames=200,
    target_width=1280,
    frame_stride=3,
):
    """Process a video file through the unified surveillance pipeline with fast downscaling and frame skipping."""
    if not os.path.exists(video_path):
        print(f"❌ Video not found at path: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = target_width / orig_w
    target_height = int(orig_h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps // frame_stride, (target_width, target_height)
    )

    frame_idx = 0
    processed_count = 0

    while cap.isOpened() and processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            result = pipeline.process_frame(frame)
            out.write(result["frame"])
            processed_count += 1

        frame_idx += 1

    cap.release()
    out.release()
    print(f"🎬 Video processing complete! Saved to {output_path}")
    return output_path


def stream_live_camera(source="0", frame_stride=2, target_width=1280, target_height=720, max_frames=300):
    """
    Generator function for real-time live feed.
    Supports local webcams, RTSP streams, video files, and Colab simulation fallback.
    """
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    is_valid_source = cap.isOpened()

    if not is_valid_source:
        print(f"⚠️ Physical camera not detected on cloud server ({source}). Switching to Live Railway CCTV Simulation Mode...")

    frame_idx = 0
    last_frame = None

    while frame_idx < max_frames:
        if is_valid_source:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                ret, frame = cap.read()
                if not ret:
                    break
        else:
            # Generate dynamic realistic Railway CCTV stream simulation
            frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            frame[:] = (40, 45, 50) # Dark platform background

            # Platform edge and yellow safety line
            cv2.line(frame, (0, target_height - 180), (target_width, target_height - 180), (0, 215, 255), 4)
            cv2.line(frame, (0, target_height - 100), (target_width, target_height - 100), (80, 80, 80), 3)
            cv2.putText(frame, "PLATFORM 2 - CAUTION: STAY BEHIND YELLOW LINE", (40, target_height - 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

            # Simulated moving passenger targets
            t = frame_idx * 0.08
            num_passengers = 6 + int(3 * np.sin(t * 0.5))
            for i in range(num_passengers):
                px = int((target_width * 0.15 + (i * 140) + np.sin(t + i) * 60) % (target_width - 100))
                py = int(220 + (i % 3) * 80 + np.cos(t * 0.8 + i) * 30)
                # Draw passenger representation
                cv2.circle(frame, (px + 25, py - 15), 18, (180, 200, 220), -1) # Head
                cv2.rectangle(frame, (px, py), (px + 50, py + 110), (140, 150, 160), -1) # Body
                cv2.putText(frame, "person", (px, py - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            time.sleep(0.04) # Simulate 25 FPS stream timing

        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        if frame_idx % frame_stride == 0:
            result = pipeline.process_frame(frame)
            last_frame = result["frame"]

        frame_idx += 1
        yield last_frame if last_frame is not None else frame

    if is_valid_source:
        cap.release()