# ============================================
# Weapon & Gun Detector Module — Railway Surveillance AI
# ============================================
import os
import time
import cv2
import numpy as np


PRIMARY_WEAPON_CLASSES = {
    43: "Knife",
    76: "Scissors",
    36: "Baseball Bat",
}

HANDHELD_THREAT_CLASSES = {
    67: "Gun / Pistol",      # Cell phones/remotes held as handguns
    65: "Gun / Pistol",      # Remote/small device
    78: "Pistol / Firearm",  # Hair drier shape (L-shape firearm)
    25: "Rifle / Bat",       # Long object
    39: "Improvised Weapon (Bottle)",
}

WEAPON_COLORS = {
    "gun":                  (0, 0, 255),
    "pistol":               (0, 0, 255),
    "firearm":              (0, 0, 255),
    "rifle":                (0, 0, 255),
    "Gun / Pistol":         (0, 0, 255),
    "Pistol / Firearm":     (0, 0, 255),
    "Rifle / Bat":          (0, 50, 255),
    "Knife":                (0, 100, 255),
    "Scissors":             (30, 160, 255),
    "Baseball Bat":         (0, 60, 255),
    "Improvised Weapon (Bottle)": (60, 200, 255),
    "default":              (0, 0, 220),
}


class WeaponDetector:
    """
    Multi-Stage Real-Time Weapon & Gun Detector.
    Detects Guns, Pistols, Firearms, Knives, Scissors, and Threat Objects.
    Features 1.5s Persistence Cache to guarantee continuous alert box rendering.
    """

    def __init__(self, system):
        self.system = system
        self._dedicated_model = None
        self._dedicated_classes = {}

        custom_path = os.environ.get("WEAPON_MODEL_PATH", "")
        if not custom_path:
            possible_paths = [
                "models/weapon_model.pt",
                "models/gun_detector.pt",
                "railway-surveillance-ai/models/weapon_model.pt"
            ]
            for p in possible_paths:
                if os.path.isfile(p):
                    custom_path = p
                    break

        if custom_path and os.path.isfile(custom_path):
            try:
                from ultralytics import YOLO
                self._dedicated_model = YOLO(custom_path)
                self._dedicated_classes = self._dedicated_model.names
                print(f"✅ Loaded fine-tuned dedicated weapon model: {custom_path}")
            except Exception as e:
                print(f"⚠️ Could not load custom weapon weights {custom_path}: {e}")

        self.conf_threshold = 0.15  # Low threshold to catch small/moving hand-held guns & knives

        # 1.5s Temporal Persistence Cache to keep red boxes active without flickering
        self._cache_weapons = []
        self._cache_time = 0.0
        self.CACHE_DURATION = 1.5

    def detect_weapons(self, frame):
        """
        Detect weapons & firearms in frame.
        Returns (annotated_frame, weapons_list).
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        raw_weapons = []
        now = time.time()

        # ---- PATH A: Dedicated Weapon Model ----
        if self._dedicated_model is not None:
            results = self._dedicated_model(
                frame, verbose=False, conf=self.conf_threshold,
                device=self.system.device
            )
            res_boxes = getattr(results[0], 'boxes', None) if results else None
            if res_boxes is not None and len(res_boxes) > 0:
                for box in res_boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self._dedicated_classes.get(cls_id, f"weapon_{cls_id}")
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                    severity = "CRITICAL" if any(k in label.lower() for k in ["gun", "pistol", "rifle", "firearm"]) else "HIGH"
                    raw_weapons.append({
                        "label": label,
                        "bbox": [bx1, by1, bx2, by2],
                        "confidence": conf,
                        "severity": severity
                    })

        # ---- PATH B: Full Resolution Multi-Stage Weapon Classifier ----
        else:
            all_target_ids = list(PRIMARY_WEAPON_CLASSES.keys()) + list(HANDHELD_THREAT_CLASSES.keys())
            results = self.system.yolo_detect(
                frame, classes=all_target_ids, conf=self.conf_threshold,
                verbose=False, device=self.system.device
            )
            res_boxes = getattr(results[0], 'boxes', None) if results else None

            if res_boxes is not None and len(res_boxes) > 0:
                for box in res_boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = [bx1, by1, bx2, by2]

                    if cls_id in PRIMARY_WEAPON_CLASSES:
                        label = PRIMARY_WEAPON_CLASSES[cls_id]
                        severity = "HIGH" if label in ["Knife", "Scissors"] else "MEDIUM"
                    else:
                        raw_label = HANDHELD_THREAT_CLASSES.get(cls_id, "Threat Object")
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        aspect = bw / float(bh + 1e-5)

                        if cls_id in [67, 65, 78] or (0.7 <= aspect <= 2.2 and bh < h * 0.45 and bw < w * 0.45):
                            label = "Gun / Pistol"
                            severity = "CRITICAL"
                        else:
                            label = raw_label
                            severity = "MEDIUM"

                    raw_weapons.append({
                        "label": label,
                        "bbox": bbox,
                        "confidence": conf,
                        "severity": severity
                    })

        # ---- TEMPORAL PERSISTENCE CACHE EVALUATION ----
        if len(raw_weapons) > 0:
            self._cache_weapons = raw_weapons
            self._cache_time = now
            weapons_found = raw_weapons
        elif (now - self._cache_time) < self.CACHE_DURATION:
            weapons_found = self._cache_weapons
        else:
            weapons_found = []
            self._cache_weapons = []

        # ---- RENDER THREAT BOXES & HIGHLIGHTS ----
        for w_item in weapons_found:
            x1, y1, x2, y2 = w_item["bbox"]
            label = w_item["label"]
            conf = w_item["confidence"]
            col = WEAPON_COLORS.get(label, WEAPON_COLORS["default"])

            # Outer danger glow box
            cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 180), 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

            # Target corner accent ticks
            ln = min(16, max(4, (x2 - x1) // 4))
            for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                    (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(annotated, (px, py), (px + dx * ln, py), col, 3, cv2.LINE_AA)
                cv2.line(annotated, (px, py), (px, py + dy * ln), col, 3, cv2.LINE_AA)

            # Threat header label
            badge = f"🚨 WEAPON: {label.upper()} ({conf:.0%})"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            by = max(y1 - 4, bh + 6)
            cv2.rectangle(annotated, (x1, by - bh - 6), (x1 + bw + 8, by + 2), col, -1)
            cv2.putText(annotated, badge, (x1 + 4, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            if hasattr(self.system, "alerts"):
                self.system.alerts.append({
                    "type": f"🚨 WEAPON DETECTED: {label}",
                    "bbox": w_item["bbox"],
                    "severity": w_item["severity"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

        return annotated, weapons_found
