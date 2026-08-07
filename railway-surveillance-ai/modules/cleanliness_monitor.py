# ============================================
# CELL 6: Cleanliness Detection Module (YOLO11)
# ============================================
import cv2
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class CleanlinessMonitor:
  """Automated cleanliness scoring and litter detection using YOLO11."""

  def __init__(self, system):
    self.system = system

    # COCO object classes indicative of platform litter for default demo mode
    self.dirty_classes = {
        39: "bottle",
        40: "wine glass",
        41: "cup",
        73: "paper/book",
        76: "trash item",
    }

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
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        device=self.system.device,
    )
    self.custom_model = YOLO("runs/detect/railway_cleanliness/weights/best.pt")
    return results

  def detect_uncleanliness(self, frame):
    """Identify waste items and render bounding box overlays."""
    detections = []
    annotated_frame = frame.copy()

    # Route inference through fine-tuned model or fall back to system detector
    if self.custom_model:
      results = self.custom_model(
          frame, conf=0.3, verbose=False, device=self.system.device
      )
    else:
      results = self.system.yolo_detect(
          frame, conf=0.3, verbose=False, device=self.system.device
      )

    for box in results[0].boxes:
      cls_id = int(box.cls[0])

      if cls_id in self.dirty_classes or self.custom_model:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])

        if self.custom_model:
          label = results[0].names[cls_id]
        else:
          label = self.dirty_classes.get(cls_id, "litter")

        detections.append(
            {"type": label, "bbox": (x1, y1, x2, y2), "confidence": conf}
        )

        # Draw ORANGE bounding box for uncleanliness detections
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            annotated_frame,
            f"🗑️ {label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    return annotated_frame, detections

  def get_cleanliness_score(self, frame):
    """Calculate zone cleanliness score on a scale of 0 to 100."""
    _, detections = self.detect_uncleanliness(frame)

    # Base score (100 = perfectly clean)
    score = 100.0

    # Deduct points based on detected waste volume and confidence
    deduction_per_item = 8.0
    score -= len(detections) * deduction_per_item

    if detections:
      avg_conf = np.mean([d["confidence"] for d in detections])
      score -= avg_conf * 5.0

    return float(max(0.0, min(100.0, score)))
