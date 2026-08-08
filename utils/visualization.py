"""
Visualization utilities for drawing overlays, bounding boxes, status bars, and zones.
"""
import cv2
import numpy as np

def draw_status_bar(frame, results):
    """Render semi-transparent status bar across bottom edge."""
    h, w = frame.shape[:2]
    bar_height = 50

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    info_items = [
        f"👥 Crowd: {results.get('crowd_count', 0)}",
        f"📊 Level: {results.get('crowd_level', 'N/A')}",
        f"🧹 Clean: {results.get('cleanliness_score', 100):.0f}%",
        f"🚨 Suspects: {len(results.get('criminals_found', []))}",
        f"⚠️ Anomalies: {len(results.get('anomalies', []))}",
        f"📍 Tracked: {results.get('tracked_persons', 0)}"
    ]

    x_offset = 15
    spacing = max(1, w // len(info_items))
    for item in info_items:
        cv2.putText(frame, item, (x_offset, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        x_offset += spacing

def draw_zones(frame, zones):
    """Draw configured polygons and labels on frame."""
    for name, data in zones.items():
        pts = data["polygon"]
        ztype = data.get("type", "restricted")
        color = (0, 0, 255) if ztype == "restricted" else (0, 255, 255)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"ZONE: {name}", (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
