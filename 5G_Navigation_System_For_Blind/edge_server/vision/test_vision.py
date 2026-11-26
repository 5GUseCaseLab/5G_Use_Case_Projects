import cv2
import numpy as np
from yolo_detector import YoloDetector
from depth_estimator import DepthEstimator

def run_sensor_fusion():
    print("=========================================")
    print(" 5G NAVIGATION - DYNAMIC THREAT TEST     ")
    print("=========================================")
    
    yolo = YoloDetector('yolov8m.pt')
    depth_model = DepthEstimator()
    
    CALIBRATION_CONSTANT = 3000.0 
    
    cap = cv2.VideoCapture(0)
    print("\n🟢 Dynamic Threat Detection active! Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_height, frame_width = frame.shape[:2]
        
        # --- THE 3D DANGER ZONE ---
        # 1. Horizontal: Middle 50% (The path ahead)
        zone_left = int(frame_width * 0.25)
        zone_right = int(frame_width * 0.75)
        
        # 2. Vertical: Bottom 70% (Ignore the ceiling / high shelves)
        zone_top = int(frame_height * 0.30)
        
        # Draw Danger Zone Boundaries on screen
        cv2.line(frame, (zone_left, 0), (zone_left, frame_height), (255, 100, 0), 2) # Left wall
        cv2.line(frame, (zone_right, 0), (zone_right, frame_height), (255, 100, 0), 2) # Right wall
        cv2.line(frame, (0, zone_top), (frame_width, zone_top), (255, 100, 0), 2) # The Ceiling line
        
        cv2.putText(frame, "CEILING (IGNORED)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.putText(frame, "DANGER ZONE", (zone_left + 10, zone_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        
        raw_depth_map, depth_colormap = depth_model.estimate_depth(frame)
        detections = yolo.detect_objects(frame, confidence_threshold=0.3)
        
        valid_threats = []
        screen_area = frame_width * frame_height
        
        for det in detections:
            name = det['class_name']
            x1, y1, x2, y2 = det['bbox']
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            
            # --- DYNAMIC PHYSICAL FILTERS (No hardcoded dictionaries!) ---
            
            # Filter A: Verticality (Is it on the ceiling/high shelf?)
            # If the BOTTOM edge of the object (y2) is above our cutoff line, it is hanging over our heads.
            if y2 < zone_top:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue
                
            # Filter B: Trajectory (Is it directly in our path?)
            center_x = (x1 + x2) // 2
            if not (zone_left < center_x < zone_right):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue
                
            # Filter C: Size/Mass (Is it large enough to trip over?)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < (screen_area * 0.02): # Less than 2% of screen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue
            
            # If it survives all physical filters, it IS a physical threat. Calculate distance.
            object_depth_pixels = raw_depth_map[y1:y2, x1:x2]
            
            if object_depth_pixels.size > 0:
                raw_depth_score = np.median(object_depth_pixels)
                distance_meters = CALIBRATION_CONSTANT / (raw_depth_score + 0.0001)
                
                valid_threats.append({
                    'name': name,
                    'distance': distance_meters,
                    'bbox': (x1, y1, x2, y2)
                })
        
        # Sort threats by closest distance
        valid_threats = sorted(valid_threats, key=lambda t: t['distance'])
        
        if valid_threats:
            primary = valid_threats[0]
            px1, py1, px2, py2 = primary['bbox']
            dist = primary['distance']
            p_name = primary['name']
            
            # Trigger warning if closer than 15 meters
            if dist < 15.0:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 4)
                cv2.putText(frame, f"THREAT: {p_name} {dist:.1f}m", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                warning_text = f"AUDIO OUTPUT: '{p_name} ahead, {dist:.1f} meters'"
                cv2.putText(frame, warning_text, (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Threat Detection System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sensor_fusion()
