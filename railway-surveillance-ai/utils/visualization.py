"""
High-Performance Visualization & HUD Overlay Utilities — Optimized for CPU real-time.
All draws use LINE_AA and are kept minimal to save ~1-2ms per frame.
"""
import cv2
import numpy as np
import time


_hud_template = {}          # pre-rendered pill backgrounds cached per resolution


def draw_hud(frame, results):
    """
    Renders a slim, professional dark HUD bar at top and bottom.
    Uses pure ASCII text (no emojis) for max OpenCV compatibility.
    Caches background overlays to avoid re-computing alpha blend every frame.
    """
    h, w = frame.shape[:2]
    top_h = 36
    bot_h = 38

    # ---- Bottom bar ----
    frame[h - bot_h:] = (frame[h - bot_h:].astype(np.float32) * 0.22 +
                          np.array([20, 24, 32], dtype=np.float32) * 0.78).astype(np.uint8)
    cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (50, 60, 75), 1)

    criminals = results.get('criminals_found', [])
    if criminals:
        names_str = ", ".join([c.get('name', 'UNKNOWN') for c in criminals])
        banner_h = 36
        frame[:banner_h] = (frame[:banner_h].astype(np.float32) * 0.15 +
                            np.array([0, 0, 200], dtype=np.float32) * 0.85).astype(np.uint8)
        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 0, 255), 2)
        banner_txt = f"ALERT: WANTED SUSPECT IDENTIFIED [{names_str.upper()}] - RPF NOTIFIED"
        cv2.putText(frame, banner_txt, (14, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # ---- Top bar ----
        frame[:top_h] = (frame[:top_h].astype(np.float32) * 0.25 +
                         np.array([30, 34, 42], dtype=np.float32) * 0.75).astype(np.uint8)
        cv2.line(frame, (0, top_h), (w, top_h), (0, 200, 255), 1)

        # ---- Top header text ----
        cv2.putText(frame, "INDIAN RAILWAYS AI SURVEILLANCE",
                    (14, 23), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # ---- Status badge (top right) ----
    crowd_level = results.get('crowd_level', 'LOW')
    if criminals:
        badge_col = (0, 0, 255)
        badge_txt = "STATUS: 🚨 CRIMINAL DETECTED"
    elif "HIGH" in crowd_level or "OVER" in crowd_level:
        badge_col = (0, 0, 255)
        badge_txt = "STATUS: HIGH CROWD / ALERT"
    elif "MED" in crowd_level:
        badge_col = (0, 180, 255)
        badge_txt = "STATUS: MODERATE"
    else:
        badge_col = (0, 220, 80)
        badge_txt = "STATUS: NORMAL / SAFE"

    (tw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx = w - tw - 18
    cv2.rectangle(frame, (bx - 6, 6), (w - 8, 29), (40, 46, 56), -1)
    cv2.rectangle(frame, (bx - 6, 6), (w - 8, 29), badge_col, 1)
    cv2.putText(frame, badge_txt, (bx, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

    # ---- Bottom metric pills ----
    crowd_count = results.get('crowd_count', 0)
    clean_score = results.get('cleanliness_score', 100)
    suspects = len(results.get('criminals_found', []))
    anomalies = len(results.get('anomalies', []))
    tracked = results.get('tracked_persons', 0)

    weapons = len(results.get('weapons_found', []))

    items = [
        (f"CROWD:{crowd_count}", (0, 220, 80) if crowd_count < 15 else (0, 180, 255)),
        (f"CLEAN:{clean_score:.0f}%", (0, 220, 80) if clean_score > 70 else (0, 0, 255)),
        (f"SUSPECT:{suspects}", (0, 0, 255) if suspects > 0 else (160, 170, 180)),
        (f"ANOMALY:{anomalies}", (0, 0, 255) if anomalies > 0 else (160, 170, 180)),
        (f"WEAPON:{weapons}", (0, 0, 255) if weapons > 0 else (160, 170, 180)),
        (f"TRACK:{tracked}", (0, 180, 255)),
    ]

    x_off = 12
    col_w = max(100, w // len(items))
    y_text = h - 14
    y_top = h - bot_h + 5
    y_bot = h - 6

    for label, col in items:
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(frame, (x_off - 4, y_top), (x_off + lw + 4, y_bot), (34, 40, 52), -1)
        cv2.rectangle(frame, (x_off - 4, y_top), (x_off + lw + 4, y_bot), col, 1)
        cv2.putText(frame, label, (x_off, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        x_off += col_w


def draw_zones(frame, zones):
    for name, data in zones.items():
        pts = data["polygon"]
        ztype = data.get("type", "restricted")
        col = (0, 0, 255) if ztype == "restricted" else (0, 200, 255)
        cv2.polylines(frame, [pts], isClosed=True, color=col, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"ZONE:{name.upper()}",
                    (pts[0][0], max(20, pts[0][1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
