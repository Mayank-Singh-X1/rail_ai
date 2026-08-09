# ============================================
# Cleanliness Detection & Surface Sanitation Module
# ============================================
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


LITTER_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup / container",
    44: "spoon / utensil",
    45: "bowl / food plate",
    46: "food waste (banana)",
    47: "food waste (apple)",
    48: "food waste (wrapper)",
    49: "food waste (citrus)",
    54: "food waste (donut)",
    55: "food waste (cake)",
    73: "paper / book litter",
    76: "trash / waste item",
    24: "baggage clutter",
    26: "handbag clutter",
    28: "suitcase clutter",
    32: "ball / floor debris",
}


class CleanlinessMonitor:
  """Automated cleanliness scoring and litter detection using YOLO11 + Surface Clutter Analysis."""

  def __init__(self, system):
    self.system = system
    self.dirty_classes = LITTER_CLASSES
    self.custom_model = None

  def train_custom_model(self, dataset_yaml_path):
    """Fine-tune YOLO11 on custom railway waste/litter datasets."""
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=dataset_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name="railway_cleanliness",
        patience=10,
        device=self.system.device,
    )
    self.custom_model = YOLO("runs/detect/railway_cleanliness/weights/best.pt")
    return results

  def detect_uncleanliness(self, frame):
    """Identify waste items and render orange bounding box overlays."""
    detections = []
    annotated_frame = frame.copy()

    if self.custom_model:
      results = self.custom_model(
          frame, conf=0.15, verbose=False, device=self.system.device
      )
    else:
      all_litter_ids = list(self.dirty_classes.keys())
      results = self.system.yolo_detect(
          frame, classes=all_litter_ids, conf=0.15, verbose=False, device=self.system.device
      )

    res_boxes = getattr(results[0], 'boxes', None) if results else None
    if res_boxes is not None and len(res_boxes) > 0:
      for box in res_boxes:
        cls_id = int(box.cls[0])

        if cls_id in self.dirty_classes or self.custom_model:
          x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
          conf = float(box.conf[0])

          if self.custom_model:
            label = results[0].names[cls_id]
          else:
            label = self.dirty_classes.get(cls_id, "litter item")

          detections.append(
              {"type": label, "bbox": (x1, y1, x2, y2), "confidence": conf}
          )

          cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
          badge = f"🗑️ {label.upper()} ({conf:.0%})"
          (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
          by = max(y1 - 4, bh + 6)
          cv2.rectangle(annotated_frame, (x1, by - bh - 4), (x1 + bw + 6, by + 2), (0, 140, 255), -1)
          cv2.putText(
              annotated_frame,
              badge,
              (x1 + 3, by - 2),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.45,
              (255, 255, 255),
              1,
              cv2.LINE_AA,
          )

    h, w = frame.shape[:2]
    floor_roi = frame[int(h * 0.55):, :]
    if floor_roi.size > 0:
      gray_floor = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
      std_dev = float(np.std(gray_floor))

      if std_dev > 48 and len(detections) == 0:
        cx1, cy1 = int(w * 0.3), int(h * 0.65)
        cx2, cy2 = int(w * 0.7), int(h * 0.88)
        detections.append({"type": "floor clutter / debris", "bbox": (cx1, cy1, cx2, cy2), "confidence": 0.65})
        cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 140, 255), 2)
        cv2.putText(annotated_frame, "🗑️ FLOOR CLUTTER DEBRIS (65%)", (cx1, cy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    return annotated_frame, detections

  def get_cleanliness_score(self, frame):
    """Calculate zone cleanliness score on a scale of 0% to 100%."""
    _, detections = self.detect_uncleanliness(frame)

    score = 100.0
    deduction_per_item = 12.0
    score -= len(detections) * deduction_per_item

    if detections:
      avg_conf = float(np.mean([d["confidence"] for d in detections]))
      score -= avg_conf * 10.0

    return float(max(0.0, min(100.0, score)))
