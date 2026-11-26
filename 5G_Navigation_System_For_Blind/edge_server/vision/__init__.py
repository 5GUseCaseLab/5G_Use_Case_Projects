"""
Vision Module Package for 5G Assistive Wearable
===============================================
Encapsulates real-time computer vision capabilities:
- YoloDetector: Fast object and obstacle bounding box detection.
- DepthEstimator: MiDaS monocular depth estimation.
- MotionDetector: Frame-to-frame optical flow & pedestrian movement tracking.
- HUDRenderer: Visual telemetry overlays, threat bounding boxes, and QR codes.
"""

from .yolo_detector import YoloDetector
from .depth_estimator import DepthEstimator
from .motion_detector import MotionDetector
from .hud_renderer import draw_navigation_hud, draw_threat_boxes, draw_qr_overlay
from .threat_tracker import ThreatTracker
from .collision_predictor import assess, phrase

__all__ = [
    'YoloDetector',
    'DepthEstimator',
    'MotionDetector',
    'ThreatTracker',
    'assess',
    'phrase',
    'draw_navigation_hud',
    'draw_threat_boxes',
    'draw_qr_overlay'
]
