"""
High-Performance Visualization & HUD Overlay Utilities for Railway AI Surveillance.
Provides crisp anti-aliased status badges, clean bounding boxes, and glassmorphic HUD.
"""
import cv2
import numpy as np


def draw_hud(frame, results):
    """
    Renders a modern, professional, high-contrast HUD bar across the top and bottom.
    Uses pure ASCII badges with anti-aliasing to avoid OpenCV unicode emoji artifacting (????).
    """
    h, w = frame.shape[:2]
    
    # 1. Top HUD Header Bar (Height: 38px)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 24, 30), -1)
    # Bottom HUD Status Bar (Height: 45px)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    
    # Top Border Accents
    cv2.line(frame, (0, 40), (w, 40), (0, 215, 255), 2)
    cv2.line(frame, (0, h - 45), (w, h - 45), (50, 60, 75), 1)

    # 2. Top Header Information
    title_text = "INDIAN RAILWAYS AI SURVEILLANCE"
    cv2.putText(frame, title_text, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 215, 255), 1, cv2.LINE_AA)
    
    # Level & Threat Badge (Top Right)
    crowd_level = results.get('crowd_level', 'LOW')
    if "HIGH" in crowd_level or "OVERCROWD" in crowd_level:
        badge_color = (0, 0, 255) # Red
        badge_text = "STATUS: HIGH DENSITY / ALERT"
    elif "MED" in crowd_level:
        badge_color = (0, 215, 255) # Yellow/Orange
        badge_text = "STATUS: MODERATE CROWD"
    else:
        badge_color = (0, 255, 120) # Green
        badge_text = "STATUS: NORMAL / SAFE"
        
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    badge_x = w - tw - 25
    cv2.rectangle(frame, (badge_x - 10, 8), (w - 10, 32), (35, 40, 50), -1)
    cv2.rectangle(frame, (badge_x - 10, 8), (w - 10, 32), badge_color, 1)
    cv2.putText(frame, badge_text, (badge_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 1, cv2.LINE_AA)

    # 3. Bottom HUD Metrics (Crisp, clean, professional layout)
    crowd_count = results.get('crowd_count', 0)
    clean_score = results.get('cleanliness_score', 100)
    suspects_count = len(results.get('criminals_found', []))
    anomalies_count = len(results.get('anomalies', []))
    tracked_count = results.get('tracked_persons', 0)

    clean_str = f"CLEAN: {clean_score:.0f}%"
    suspect_str = f"SUSPECTS: {suspects_count}"
    anom_str = f"ANOMALIES: {anomalies_count}"
    crowd_str = f"CROWD: {crowd_count}"
    track_str = f"TRACKED: {tracked_count}"

    items = [
        (crowd_str, (0, 255, 120) if crowd_count < 10 else (0, 215, 255)),
        (clean_str, (0, 255, 120) if clean_score > 70 else (0, 0, 255)),
        (suspect_str, (0, 0, 255) if suspects_count > 0 else (180, 190, 200)),
        (anom_str, (0, 0, 255) if anomalies_count > 0 else (180, 190, 200)),
        (track_str, (0, 215, 255))
    ]

    x_offset = 18
    col_width = max(110, w // len(items))
    for label, col in items:
        # Mini pill background
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x_offset - 6, h - 34), (x_offset + lw + 6, h - 12), (32, 38, 48), -1)
        cv2.rectangle(frame, (x_offset - 6, h - 34), (x_offset + lw + 6, h - 12), col, 1)
        cv2.putText(frame, label, (x_offset, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        x_offset += col_width


def draw_styled_bbox(frame, bbox, label, color=(0, 255, 0), track_id=None):
    """Draws sleek rounded corner bounding box with solid label banner."""
    x1, y1, x2, y2 = bbox
    # Main Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Corner Accents for high-tech surveillance look
    line_len = min(20, (x2 - x1) // 3)
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 4, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, 4, cv2.LINE_AA)

    # Top Label Tag with background
    display_text = f"ID:{track_id} {label}" if track_id is not None else label
    (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    tag_y = max(18, y1 - 6)
    cv2.rectangle(frame, (x1, tag_y - th - 6), (x1 + tw + 10, tag_y + 2), color, -1)
    cv2.putText(frame, display_text, (x1 + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)


def draw_zones(frame, zones):
    """Draw configured security polygons and labels on frame."""
    for name, data in zones.items():
        pts = data["polygon"]
        ztype = data.get("type", "restricted")
        color = (0, 0, 255) if ztype == "restricted" else (0, 215, 255)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        tag = f"ZONE: {name.upper()}"
        cv2.putText(frame, tag, (pts[0][0], max(20, pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
