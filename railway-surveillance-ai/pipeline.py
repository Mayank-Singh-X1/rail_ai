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

    # Determine execution provider
    self.device = 0 if torch.cuda.is_available() else "cpu"
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

    # State caches to prevent UI flickering during frame-skipping
    self.cached_criminals = []
    self.cached_anomalies = []
    self.cached_cleanliness_score = 100.0
    self.cached_workers_present = []
    self.cached_workers_absent = []

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
    """Process a single frame through enabled modules efficiently."""

    start_time = time.time()
    self.frame_count += 1

    annotated = frame.copy()

    results = {
        "frame": None,
        "crowd_count": 0,
        "crowd_level": "N/A",
        "criminals_found": [],
        "anomalies": [],
        "cleanliness_score": self.cached_cleanliness_score,
        "workers_present": [],
        "workers_absent": [],
        "tracked_persons": 0,
        "alerts": [],
    }

    # ---- 1. TRACKING & CROWD (Every Frame - Combined YOLO11 Pass) ----
    if enable_tracking:
      annotated, active_count = self.person_tracker.track_persons(annotated)
      results["tracked_persons"] = active_count
      results["crowd_count"] = active_count
    elif enable_crowd:
      count, detections = self.crowd_analyzer.count_people(frame)
      results["crowd_count"] = count

    if enable_crowd:
      level, color = self.crowd_analyzer.get_crowd_level(
          results["crowd_count"]
      )
      results["crowd_level"] = level


    # ---- 2. CRIMINAL DETECTION (Every 5th Frame) ----
    if enable_criminal and (self.frame_count % 5 == 0):
      _, matches = self.criminal_detector.detect_criminals(frame)
      self.cached_criminals = matches

    results["criminals_found"] = self.cached_criminals

    # Render cached criminal alerts
    for match in self.cached_criminals:
      bbox = match["bbox"]
      cv2.rectangle(
          annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3
      )
      cv2.putText(
          annotated,
          f"🚨 {match['name']} ({match['score']:.2f})",
          (bbox[0], bbox[1] - 15),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 0, 255),
          2,
      )

    # ---- 3. ANOMALY DETECTION (Every 3rd Frame) ----
    if enable_anomaly and (self.frame_count % 3 == 0):
      _, anomalies = self.anomaly_detector.detect_anomalies(frame)
      self.cached_anomalies = anomalies

    results["anomalies"] = self.cached_anomalies

    # ---- 4. CLEANLINESS MONITORING (Every 30th Frame) ----
    if enable_cleanliness and (self.frame_count % 30 == 0):
      self.cached_cleanliness_score = (
          self.cleanliness_monitor.get_cleanliness_score(frame)
      )

    results["cleanliness_score"] = self.cached_cleanliness_score

    # ---- 5. WORKER MONITORING (Every 10th Frame) ----
    if enable_worker and (self.frame_count % 10 == 0):
      _, present, absent = self.worker_monitor.check_attendance(frame)
      self.cached_workers_present = present
      self.cached_workers_absent = absent

    results["workers_present"] = self.cached_workers_present
    results["workers_absent"] = self.cached_workers_absent

    # ---- 6. FPS CALCULATION ----
    elapsed = time.time() - start_time
    fps = 1.0 / (elapsed + 1e-6)
    self.fps_history.append(fps)
    avg_fps = np.mean(self.fps_history)

    # ---- 7. OVERLAYS & HUD RENDERING ----
    from utils.visualization import draw_hud, draw_zones
    if hasattr(self.system, 'zones') and self.system.zones:
        draw_zones(annotated, self.system.zones)

    draw_hud(annotated, results)

    results["frame"] = annotated
    results["alerts"] = self.system.alerts[-10:]

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