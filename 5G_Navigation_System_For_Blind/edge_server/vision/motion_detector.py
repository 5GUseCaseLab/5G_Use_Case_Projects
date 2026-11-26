"""
Motion Detector Module for 5G Assistive Wearable
================================================
Calculates optical motion energy from video frames and tracks pedestrian 
displacements to determine whether the user is actively walking or stationary.
"""

import cv2
import numpy as np

class MotionDetector:
    def __init__(self, energy_decay=0.65, energy_gain=0.35, walk_threshold=4.5):
        """
        Initializes the motion detector.
        - energy_decay / energy_gain: Exponential smoothing factors for camera motion.
        - walk_threshold: Minimum motion energy to classify camera as walking.
        """
        self.energy_decay = energy_decay
        self.energy_gain = energy_gain
        self.walk_threshold = walk_threshold
        
        self.prev_gray_frame = None
        self.prev_person_centers = []
        self.motion_energy = 0.0

    def update_frame(self, frame):
        """
        Calculates optical motion energy between the current frame and the previous frame.
        Uses downsampled, blurred grayscale frames for fast, noise-resistant processing.
        """
        if frame is None or frame.size == 0:
            return 0.0
            
        try:
            # Resize to small thumbnail (120x90) for lightning-fast computation
            gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 90))
            gray_blur = cv2.GaussianBlur(gray_small, (15, 15), 0)
            
            if self.prev_gray_frame is not None:
                frame_diff = float(np.mean(cv2.absdiff(self.prev_gray_frame, gray_blur)))
            else:
                frame_diff = 0.0
                
            self.prev_gray_frame = gray_blur
            self.motion_energy = (self.energy_decay * self.motion_energy) + (self.energy_gain * frame_diff)
            return self.motion_energy
        except Exception:
            return 0.0

    def track_obstacles(self, detections):
        """
        Analyzes bounding boxes of detected obstacles across consecutive frames.
        Returns True if any obstacle (person, chair, bicycle, vehicle, etc.) 
        is moving towards/away or displacing significantly.
        """
        current_centers = []
        obstacle_moving = False
        
        if detections:
            for det in detections:
                bx1, by1, bx2, by2 = det['bbox']
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                bw = bx2 - bx1
                bh = by2 - by1
                current_centers.append((cx, cy, bw, bh))
                
                # Check displacement against previous frame's obstacle positions
                if self.prev_person_centers:
                    for pcx, pcy, pbw, pbh in self.prev_person_centers:
                        dist_moved = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
                        size_diff = abs(bw - pbw)
                        # Noticeable positional displacement or bounding box expansion/contraction
                        # (>= 15px shift or >= 12px size expansion indicates approaching obstacle)
                        if dist_moved >= 15.0 or size_diff >= 12.0:
                            obstacle_moving = True
                            break
                            
        self.prev_person_centers = current_centers
        return obstacle_moving

    def track_pedestrians(self, detections):
        """Backward-compatible alias for pedestrian-specific tracking."""
        return self.track_obstacles(detections)

    def evaluate_walking(self, gps_speed=0.0, is_phone_walking=False, detections=None, has_gps=False):
        """
        Consolidates GPS speed, phone pedometer / displacement, camera optical flow,
        and obstacle movements into a single robust walking & threat determination.
        Rule:
        - If user is stationary (GPS speed < 1.0 km/h and phone not walking):
          * Caution triggers if an obstacle is actively moving or approaching.
        - If user is actively walking (GPS speed >= 1.0 km/h or phone walking):
          * Caution triggers for any hazard within warning range.
        - If GPS has no fix (indoor navigation): uses camera optical flow as fallback.
        """
        gps_walking = (gps_speed >= 1.0) or bool(is_phone_walking)
        camera_walking = (self.motion_energy >= self.walk_threshold)
        
        person_present = False
        if detections is not None:
            for det in detections:
                c_name = str(det.get('class_name') or det.get('name') or '').lower().strip()
                if c_name == 'person':
                    person_present = True
                    break
                    
        obstacle_moving = self.track_obstacles(detections)
        person_moving = obstacle_moving if person_present else False
            
        if person_present:
            # Person ahead: caution triggers if the other person is walking OR user is walking
            is_walking = person_moving or gps_walking
        else:
            # General obstacle (chair, table, pole, vehicle):
            # When GPS is locked and active, GPS is ground truth for user walking.
            if has_gps:
                is_walking = gps_walking or obstacle_moving
            else:
                # Fallback when indoor without GPS fix: require sustained camera optical flow or obstacle motion
                is_walking = camera_walking or obstacle_moving

        return {
            "is_walking": is_walking,
            "motion_energy": round(self.motion_energy, 2),
            "camera_walking": camera_walking,
            "gps_walking": gps_walking,
            "person_moving": person_moving,
            "obstacle_moving": obstacle_moving,
            "person_present": person_present
        }

    def reset(self):
        """Resets all historical frame memory."""
        self.prev_gray_frame = None
        self.prev_person_centers = []
        self.motion_energy = 0.0
