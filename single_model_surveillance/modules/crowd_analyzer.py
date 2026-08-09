# ============================================
# Crowd Analysis Module (YOLO11 Ready)
# ============================================
import time
from collections import Counter
import cv2
import numpy as np


class CrowdAnalyzer:

  def __init__(self, system):
    self.system = system
    self.density_thresholds = {
        "low": 20,
        "medium": 50,
        "high": 100,
        "critical": 200,
    }

  def count_people(self, frame):
    """Count people in real-time using YOLO11 detection model."""
    results = self.system.yolo_detect(
        frame,
        classes=[0],  # Class 0 = person in COCO dataset
        conf=0.3,
        verbose=False,
        device=self.system.device,
    )

    detections = results[0].boxes
    count = len(detections)

    # Store time-series entry for dynamic crowd trend analysis
    if hasattr(self.system, "crowd_history"):
      self.system.crowd_history.append({"count": count, "timestamp": time.time()})

    return count, detections

  def generate_density_heatmap(self, frame, detections):
    """Generate Gaussian crowd density heatmap overlaid on frame."""
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

    for box in detections:
      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
      cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
      cv2.circle(heatmap, (cx, cy), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)

    if heatmap.max() > 0:
      heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    return overlay, heatmap

  def get_crowd_level(self, count):
    """Classify station crowd density state and assign status color."""
    if count < self.density_thresholds["low"]:
      return "LOW", (0, 255, 0)  # Green
    elif count < self.density_thresholds["medium"]:
      return "MEDIUM", (0, 255, 255)  # Yellow
    elif count < self.density_thresholds["high"]:
      return "HIGH", (0, 165, 255)  # Orange
    else:
      return "CRITICAL", (0, 0, 255)  # Red

  def get_zone_wise_count(self, frame, detections):
    """Evaluate occupant density across configured station polygons."""
    zone_counts = {}

    if hasattr(self.system, "zones"):
      for zone_name, zone_info in self.system.zones.items():
        count = 0
        polygon = zone_info["polygon"]

        for box in detections:
          x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
          cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

          if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
            count += 1

        zone_counts[zone_name] = count

    return zone_counts

  def predict_crowd_trend(self):
    """Forecast short-term crowd inflow/outflow trajectory."""
    if not hasattr(self.system, "crowd_history") or len(self.system.crowd_history) < 10:
      return "INSUFFICIENT DATA"

    recent = [h["count"] for h in list(self.system.crowd_history)[-10:]]
    older = [h["count"] for h in list(self.system.crowd_history)[-20:-10]]

    if not older:
      return "INSUFFICIENT DATA"

    recent_avg = np.mean(recent)
    older_avg = np.mean(older)

    if recent_avg > older_avg * 1.2:
      return "📈 INCREASING"
    elif recent_avg < older_avg * 0.8:
      return "📉 DECREASING"
    else:
      return "📊 STABLE"
