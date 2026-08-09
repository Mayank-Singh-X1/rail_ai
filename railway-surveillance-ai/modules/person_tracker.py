# ============================================
# PersonTracker — Optimized for CPU 30+ FPS
# ============================================
from collections import defaultdict, deque
import cv2
import numpy as np


class PersonTracker:

    def __init__(self, system):
        self.system = system
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # Reduced from 100→30
        self.line_crossings = defaultdict(int)

        # Frame-skip state: run YOLO only every N frames, re-use boxes in between
        self._skip_counter = 0
        self._cached_boxes = []
        self._cached_track_ids = []
        self._yolo_skip = 2          # Run YOLO every 2nd frame → ~2x FPS boost
        self._infer_size = 320       # Inference at 320px instead of 640px → ~4x FPS boost
        self._scale_x = 1.0
        self._scale_y = 1.0

    def track_persons(self, frame):
        """Detect and track persons using YOLO11 + ByteTrack, optimized for CPU."""
        h, w = frame.shape[:2]

        # --- Step 1: Only run YOLO every _yolo_skip frames ---
        if self._skip_counter % self._yolo_skip == 0:
            # Downscale frame for YOLO inference (huge speedup on CPU)
            small = cv2.resize(frame, (self._infer_size, self._infer_size),
                               interpolation=cv2.INTER_LINEAR)
            self._scale_x = w / self._infer_size
            self._scale_y = h / self._infer_size

            results = self.system.yolo_detect.track(
                small,
                persist=True,
                classes=[0],            # person only
                conf=0.35,
                iou=0.45,
                tracker="bytetrack.yaml",
                verbose=False,
                device=self.system.device,
                imgsz=self._infer_size,
            )

            res_boxes = getattr(results[0], 'boxes', None) if results else None

            if res_boxes is not None and hasattr(res_boxes, 'id') and res_boxes.id is not None:
                # Scale boxes back to original frame resolution
                boxes_small = res_boxes.xyxy.cpu().numpy().astype(int)
                self._cached_boxes = np.array([
                    [int(b[0] * self._scale_x), int(b[1] * self._scale_y),
                     int(b[2] * self._scale_x), int(b[3] * self._scale_y)]
                    for b in boxes_small
                ])
                self._cached_track_ids = res_boxes.id.cpu().numpy().astype(int)
            else:
                self._cached_boxes = []
                self._cached_track_ids = []

        self._skip_counter += 1

        # --- Step 2: Draw from cached results (instant, no GPU/CPU cost) ---
        annotated_frame = frame.copy()
        active_tracks = len(self._cached_track_ids)

        for box, track_id in zip(self._cached_boxes, self._cached_track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.track_history[track_id].append((cx, cy))

            # Bounding box with corner accents (lighter than full rectangle)
            col = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), col, 1)

            # Corner accent marks only (no full rectangle redraw)
            ln = min(14, (x2 - x1) // 3)
            for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                    (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(annotated_frame, (px, py), (px + dx * ln, py), col, 2, cv2.LINE_AA)
                cv2.line(annotated_frame, (px, py), (px, py + dy * ln), col, 2, cv2.LINE_AA)

            # Small ID label
            cv2.putText(annotated_frame, f"P{track_id}",
                        (x1 + 3, max(y1 - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Short trail (last 8 points only for speed)
            history = list(self.track_history[track_id])[-8:]
            for i in range(1, len(history)):
                cv2.line(annotated_frame, history[i - 1], history[i], (0, 220, 255), 1)

        return annotated_frame, active_tracks

    def setup_line_counter(self, line_start, line_end):
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)

    def count_line_crossings(self, frame):
        annotated_frame, active_count = self.track_persons(frame)
        if not hasattr(self, 'line_start') or not hasattr(self, 'line_end'):
            return annotated_frame, self.line_crossings
        cv2.line(annotated_frame, tuple(self.line_start), tuple(self.line_end), (255, 0, 255), 2)
        cv2.putText(annotated_frame,
                    f"IN: {self.line_crossings['IN']} | OUT: {self.line_crossings['OUT']}",
                    (self.line_start[0], self.line_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
        return annotated_frame, self.line_crossings
