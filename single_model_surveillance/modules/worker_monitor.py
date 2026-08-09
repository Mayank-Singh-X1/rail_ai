# ============================================
# Worker Monitoring Module (Vector Accelerated)
# ============================================
import time
from collections import defaultdict
import cv2
import numpy as np


class WorkerMonitor:

  def __init__(self, system):
    self.system = system
    self.attendance_log = defaultdict(list)
    self.zone_time_tracker = defaultdict(lambda: defaultdict(float))
    self.last_seen = {}

  def check_attendance(self, frame):
    """Detect workers and maintain real-time attendance logs."""
    faces = self.system.face_app.get(frame)
    present_workers = []
    annotated_frame = frame.copy()

    if not faces or not self.system.worker_db:
      all_workers = set(self.system.worker_db.keys())
      return annotated_frame, [], list(all_workers)

    db_names = list(self.system.worker_db.keys())
    db_matrix = np.array(
        list(self.system.worker_db.values()), dtype=np.float32
    )

    for face in faces:
      bbox = face.bbox.astype(int)
      live_embedding = face.embedding / np.linalg.norm(face.embedding)

      similarities = np.dot(db_matrix, live_embedding)
      best_idx = np.argmax(similarities)
      best_score = similarities[best_idx]
      best_match = db_names[best_idx]

      if best_score > 0.4:
        present_workers.append(best_match)
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if (
            best_match not in self.last_seen
            or (time.time() - self.last_seen.get(best_match, 0)) > 3600
        ):
          self.attendance_log[best_match].append(current_timestamp)

        self.last_seen[best_match] = time.time()
        self.track_zone_presence(best_match, bbox)

        cv2.rectangle(
            annotated_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (255, 200, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"👷 {best_match} ({best_score:.2f})",
            (bbox[0], max(14, bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

    all_workers = set(self.system.worker_db.keys())
    absent_workers = all_workers - set(present_workers)

    return annotated_frame, present_workers, list(absent_workers)

  def track_zone_presence(self, worker_name, worker_bbox):
    """Track time spent by staff inside designated station zones."""
    cx = (worker_bbox[0] + worker_bbox[2]) // 2
    cy = (worker_bbox[1] + worker_bbox[3]) // 2

    if hasattr(self.system, "zones"):
      for zone_name, zone_info in self.system.zones.items():
        if (
            cv2.pointPolygonTest(
                zone_info["polygon"], (float(cx), float(cy)), False
            )
            >= 0
        ):
          self.zone_time_tracker[worker_name][zone_name] += 0.033

  def get_attendance_report(self):
    """Summarize staff presence and zone activity."""
    report = {}
    if hasattr(self.system, "worker_db"):
      for name in self.system.worker_db.keys():
        report[name] = {
            "status": "Present" if name in self.last_seen else "Absent",
            "check_in_times": self.attendance_log.get(name, []),
            "zones_visited_seconds": {
                k: round(v, 1)
                for k, v in self.zone_time_tracker.get(name, {}).items()
            },
        }
    return report
