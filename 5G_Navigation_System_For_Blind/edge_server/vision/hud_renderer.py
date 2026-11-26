"""
HUD Renderer Module for 5G Assistive Wearable
=============================================
Handles all OpenCV visual overlays on the Edge AI monitor window:
- Translucent glass cards for telemetry and turn-by-turn guidance.
- Bounding boxes and distance labels for detected obstacles.
- QR code overlay for easy mobile pairing.
"""

import cv2
import numpy as np

def draw_navigation_hud(frame, current_gps, current_ultra, nav_state, gps_mode="FALLBACK", current_speed=0.0, cam_source="5G MOBILE PHONE"):
    """
    Renders clean glass cards at top and bottom of the video frame:
    - Top card: Camera source, live GPS coordinates, speed, and destination target.
    - Bottom card: Next turn-by-turn maneuver instruction.
    """
    if frame is None or frame.size == 0:
        return
        
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # Responsive font scaling for mobile and HD feeds
    scale = 0.40 if w < 600 else 0.48
    scale_title = 0.44 if w < 600 else 0.52
    scale_step = 0.44 if w < 00 else 0.54
    max_step_chars = 55 if w < 600 else 75
    
    # 1. Top HUD Card (Telemetry & Destination Target)
    card_h = 70
    cv2.rectangle(overlay, (15, 10), (w - 15, 10 + card_h), (20, 20, 24), -1)
    
    # 2. Bottom HUD Card (Turn-by-turn Next Step)
    bot_card_h = 55
    cv2.rectangle(overlay, (15, h - bot_card_h - 10), (w - 15, h - 10), (20, 20, 24), -1)
    
    # Apply alpha transparency (72% frame, 28% dark glass card)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    
    # Cyan border accents
    cv2.rectangle(frame, (15, 10), (w - 15, 10 + card_h), (0, 229, 255), 1)
    cv2.rectangle(frame, (15, h - bot_card_h - 10), (w - 15, h - 10), (0, 229, 255), 1)
    
    # Format telemetry strings
    if current_gps:
        mode_tag = f" [{gps_mode}]" if gps_mode in ("5G SATELLITE", "LIVE SATELLITE") else " [FALLBACK]"
        gps_display = f"{current_gps[0]:.5f}, {current_gps[1]:.5f}{mode_tag}"
    else:
        gps_display = "Waiting for GPS..."
        
    ultra_display = f"{current_ultra:.1f}cm" if current_ultra is not None else "N/A"
    dest_display = nav_state.get("destination_name", "Standby")
    dist_val = nav_state.get("total_distance_m", 0)
    dist_display = f"{dist_val}m" if dist_val > 0 else "--"
    spd_tag = f"  |  SPD: {current_speed:.1f}km/h" if gps_mode in ("5G SATELLITE", "LIVE SATELLITE") else ""
    cam_tag = f"CAM: {cam_source or '5G PHONE'}"
    
    # Top card Line 1: Camera & Location
    cv2.putText(frame, f"{cam_tag}  |  CURRENT: ({gps_display}){spd_tag}", (25, 36), 
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 200), 1, cv2.LINE_AA)
    
    # Top card Line 2: Target Destination & Distance
    cv2.putText(frame, f"TARGET:  {dest_display}  |  REMAINING: {dist_display}", (25, 64), 
                cv2.FONT_HERSHEY_SIMPLEX, scale_title, (0, 230, 255), 1, cv2.LINE_AA)
    
    # Bottom card: Next maneuver step
    step_text = nav_state.get("current_step", "")
    if len(step_text) > max_step_chars:
        step_text = step_text[:max_step_chars - 3] + "..."
    cv2.putText(frame, f"NEXT: {step_text}", (25, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, scale_step, (255, 255, 255), 2, cv2.LINE_AA)

def draw_threat_boxes(frame, valid_threats, is_walking=False, alert_distance_m=3.5, critical_distance_m=1.5):
    """
    Renders colored bounding boxes and distance tags for active collision threats:
    - Red (thickness 3): Critical stop threat (<= 1.5m)
    - Orange (thickness 2): Caution threat in alert zone (<= 3.5m)
    - Yellow: Approaching object (> 3.5m)
    """
    if not valid_threats or frame is None:
        return
        
    for idx, threat in enumerate(valid_threats):
        bx1, by1, bx2, by2 = threat['bbox']
        dist = threat['distance']
        name = threat['name']
        
        is_critical = (dist <= critical_distance_m)
        in_alert_zone = (dist <= alert_distance_m)
        
        if is_critical:
            box_color = (0, 0, 255)       # Red
            thickness = 3
        elif in_alert_zone:
            box_color = (0, 165, 255)     # Orange
            thickness = 2
        else:
            box_color = (0, 230, 255)     # Yellow
            thickness = 1
            
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, thickness)
        
        # Primary threat gets status tag
        if idx == 0:
            status_tag = " [WALKING - CAUTION ACTIVE]" if is_walking else " [STATIONARY - EVALUATING]"
            label = f"THREAT: {name} {dist:.1f}m{status_tag}"
            cv2.putText(frame, label, (bx1, max(22, by1 - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{name} {dist:.1f}m", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

def draw_qr_overlay(frame, qr_image):
    """
    Draws a mini QR code on the top-right corner of the Edge AI monitor for quick phone connection.
    """
    if frame is None or qr_image is None:
        return
        
    try:
        mini_qr = cv2.resize(qr_image, (160, 160), interpolation=cv2.INTER_NEAREST)
        qh, qw = mini_qr.shape[:2]
        fh, fw = frame.shape[:2]
        
        if fh > qh + 80 and fw > qw + 20:
            x_pos = fw - qw - 15
            y_pos = 80
            frame[y_pos:y_pos+qh, x_pos:x_pos+qw] = mini_qr
            cv2.rectangle(frame, (x_pos - 1, y_pos - 1), (x_pos + qw + 1, y_pos + qh + 1), (0, 229, 255), 2)
    except Exception:
        pass
