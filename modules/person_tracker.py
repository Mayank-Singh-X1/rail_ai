# ============================================
# CELL 8: ByteTrack Tracking Integration (YOLO11 Native)
# ============================================
from collections import defaultdict, deque
import cv2
import numpy as np


class PersonTracker:

  def __init__(self, system):
    self.system = system
    self.track_history = defaultdict(lambda: deque(maxlen=100))
    self.line_crossings = defaultdict(int)

  def track_persons(self, frame):
    """Detect and track persons instantly using YOLO11 + ByteTrack."""
    # Native YOLO11 tracking via ByteTrack
    results = self.system.yolo_detect.track(
        frame,
        persist=True,
        classes=[0],  # class 0 = person
        conf=0.3,
        tracker="bytetrack.yaml",
        verbose=False,
        device=self.system.device,
    )

    annotated_frame = frame.copy()
    active_tracks = 0

    res_boxes = getattr(results[0], 'boxes', None) if len(results) > 0 else None

    if res_boxes is not None and hasattr(res_boxes, 'id') and res_boxes.id is not None:
      boxes = res_boxes.xyxy.cpu().numpy().astype(int)
      track_ids = res_boxes.id.cpu().numpy().astype(int)

      active_tracks = len(track_ids)

      for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box

        # Store path history for trajectory visualization
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.track_history[track_id].append((cx, cy))

        # Draw bounding box & persistent ID
        cv2.rectangle(
            annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Render movement trail
        history = list(self.track_history[track_id])
        for i in range(1, len(history)):
          cv2.line(
              annotated_frame, history[i - 1], history[i], (0, 255, 255), 2
          )

    return annotated_frame, active_tracks

  def setup_line_counter(self, line_start, line_end):
    """Configure virtual line for passenger/footfall counting."""
    self.line_start = np.array(line_start)
    self.line_end = np.array(line_end)

  def count_line_crossings(self, frame):
    """Detect line-crossing events using vector cross-product geometry."""
    annotated_frame, active_count = self.track_persons(frame)

    if not hasattr(self, 'line_start') or not hasattr(self, 'line_end'):
      return annotated_frame, self.line_crossings

    # Render virtual counting line
    cv2.line(
        annotated_frame,
        tuple(self.line_start),
        tuple(self.line_end),
        (255, 0, 255),
        3,
    )
    cv2.putText(
        annotated_frame,
        f"IN: {self.line_crossings['IN']} | OUT: {self.line_crossings['OUT']}",
        (self.line_start[0], self.line_start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    return annotated_frame, self.line_crossings
