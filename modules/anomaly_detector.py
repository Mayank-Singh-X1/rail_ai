# ============================================
# CELL 5: Anomaly Detection Module (YOLO11 Pose)
# ============================================
import time
from collections import defaultdict, deque
import cv2
import numpy as np


class AnomalyDetector:

  def __init__(self, system):
    self.system = system
    self.pose_history = defaultdict(lambda: deque(maxlen=30))

    # COCO Pose keypoint indices
    self.KEYPOINTS = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

  def detect_anomalies(self, frame):
    """Detect falls, rapid violent movements, and track intrusions via YOLO11 pose."""
    results = self.system.yolo_pose(
        frame, verbose=False, device=self.system.device
    )
    anomalies = []
    annotated_frame = frame.copy()

    if results[0].keypoints is not None:
      keypoints_data = results[0].keypoints.data.cpu().numpy()
      boxes = results[0].boxes

      for i, (kps, box) in enumerate(zip(keypoints_data, boxes)):
        person_id = i  # Tracks individual pose index per frame

        # Store keypoint history for velocity tracking
        self.pose_history[person_id].append(kps)

        # ---- FALL DETECTION ----
        if self._detect_fall(kps):
          bbox = box.xyxy[0].cpu().numpy().astype(int)
          anomalies.append({
              "type": "🆘 FALL DETECTED",
              "person_id": person_id,
              "bbox": bbox,
              "severity": "HIGH",
          })
          cv2.rectangle(
              annotated_frame,
              (bbox[0], bbox[1]),
              (bbox[2], bbox[3]),
              (0, 0, 255),
              3,
          )
          cv2.putText(
              annotated_frame,
              "⚠️ FALL DETECTED",
              (bbox[0], bbox[1] - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              (0, 0, 255),
              2,
          )

        # ---- FIGHTING/VIOLENCE DETECTION ----
        if self._detect_violence(kps, person_id):
          bbox = box.xyxy[0].cpu().numpy().astype(int)
          anomalies.append({
              "type": "👊 VIOLENCE DETECTED",
              "person_id": person_id,
              "bbox": bbox,
              "severity": "CRITICAL",
          })
          cv2.rectangle(
              annotated_frame,
              (bbox[0], bbox[1]),
              (bbox[2], bbox[3]),
              (255, 0, 0),
              3,
          )
          cv2.putText(
              annotated_frame,
              "👊 VIOLENCE",
              (bbox[0], bbox[1] - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              (255, 0, 0),
              2,
          )

        # ---- TRACK / ZONE INTRUSION ----
        # Calculate torso midpoint
        if (
            kps[self.KEYPOINTS["left_hip"]][2] > 0.3
            and kps[self.KEYPOINTS["right_hip"]][2] > 0.3
        ):
          cx = int(
              (
                  kps[self.KEYPOINTS["left_hip"]][0]
                  + kps[self.KEYPOINTS["right_hip"]][0]
              )
              / 2
          )
          cy = int(
              (
                  kps[self.KEYPOINTS["left_hip"]][1]
                  + kps[self.KEYPOINTS["right_hip"]][1]
              )
              / 2
          )

          for zone_name, zone_info in self.system.zones.items():
            if zone_info["type"] == "restricted":
              if (
                  cv2.pointPolygonTest(
                      zone_info["polygon"], (float(cx), float(cy)), False
                  )
                  >= 0
              ):
                anomalies.append({
                    "type": f"🚫 INTRUSION: {zone_name}",
                    "person_id": person_id,
                    "severity": "CRITICAL",
                })

    # Log anomalies to global alert queue
    for anomaly in anomalies:
      self.system.alerts.append(
          {**anomaly, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
      )

    return annotated_frame, anomalies

  def _detect_fall(self, keypoints):
    """Detect sudden collapse by measuring torso-to-ankle aspect ratios."""
    left_hip = keypoints[self.KEYPOINTS["left_hip"]]
    right_hip = keypoints[self.KEYPOINTS["right_hip"]]
    left_ankle = keypoints[self.KEYPOINTS["left_ankle"]]
    right_ankle = keypoints[self.KEYPOINTS["right_ankle"]]
    nose = keypoints[self.KEYPOINTS["nose"]]

    # Require minimum detection confidence
    if left_hip[2] < 0.3 or right_hip[2] < 0.3:
      return False

    hip_y = (left_hip[1] + right_hip[1]) / 2
    ankle_y = (
        (left_ankle[1] + right_ankle[1]) / 2
        if (left_ankle[2] > 0.3 and right_ankle[2] > 0.3)
        else hip_y + 10
    )

    vertical_diff = ankle_y - hip_y

    shoulder_l = keypoints[self.KEYPOINTS["left_shoulder"]]
    shoulder_r = keypoints[self.KEYPOINTS["right_shoulder"]]

    body_height = abs(nose[1] - ankle_y) if nose[2] > 0.3 else 0
    body_width = (
        abs(shoulder_l[0] - shoulder_r[0])
        if (shoulder_l[2] > 0.3 and shoulder_r[2] > 0.3)
        else 0
    )

    # Trigger fall alert if body orientation leans horizontal
    if body_height > 0 and body_width > 0:
      aspect_ratio = body_width / (body_height + 1e-6)
      if aspect_ratio > 1.4 or vertical_diff < 25:
        return True

    return False

  def _detect_violence(self, keypoints, person_id):
    """Detect fighting gestures via wrist movement velocity and position relative to shoulders."""
    if len(self.pose_history[person_id]) < 5:
      return False

    recent_poses = list(self.pose_history[person_id])[-5:]

    left_wrist_positions = [
        p[self.KEYPOINTS["left_wrist"]][:2]
        for p in recent_poses
        if p[self.KEYPOINTS["left_wrist"]][2] > 0.3
    ]
    right_wrist_positions = [
        p[self.KEYPOINTS["right_wrist"]][:2]
        for p in recent_poses
        if p[self.KEYPOINTS["right_wrist"]][2] > 0.3
    ]

    def compute_velocity(positions):
      if len(positions) < 3:
        return 0
      velocities = [
          np.sqrt(
              (positions[i][0] - positions[i - 1][0]) ** 2
              + (positions[i][1] - positions[i - 1][1]) ** 2
          )
          for i in range(1, len(positions))
      ]
      return np.mean(velocities) if velocities else 0

    left_vel = compute_velocity(left_wrist_positions)
    right_vel = compute_velocity(right_wrist_positions)

    # Check if wrists extend above shoulder plane
    hands_raised = False
    if (
        keypoints[self.KEYPOINTS["left_wrist"]][2] > 0.3
        and keypoints[self.KEYPOINTS["left_shoulder"]][2] > 0.3
    ):
      if (
          keypoints[self.KEYPOINTS["left_wrist"]][1]
          < keypoints[self.KEYPOINTS["left_shoulder"]][1]
      ):
        hands_raised = True

    if (
        keypoints[self.KEYPOINTS["right_wrist"]][2] > 0.3
        and keypoints[self.KEYPOINTS["right_shoulder"]][2] > 0.3
    ):
      if (
          keypoints[self.KEYPOINTS["right_wrist"]][1]
          < keypoints[self.KEYPOINTS["right_shoulder"]][1]
      ):
        hands_raised = True

    VELOCITY_THRESHOLD = 45  # Velocity threshold for rapid movement
    return (
        left_vel > VELOCITY_THRESHOLD or right_vel > VELOCITY_THRESHOLD
    ) and hands_raised
