from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_size='yolov8n.pt'):
        """
        Initializes the pre-trained YOLOv8 model.
        'yolov8n.pt' is the Nano model (fastest, great for real-time edge processing).
        """
        print(f"⏳ Loading YOLOv8 model ({model_size})...")
        self.model = YOLO(model_size)
        print("✅ YOLOv8 Model loaded successfully!")

    def detect_objects(self, frame, confidence_threshold=0.5, classes=None, imgsz=416):
        """
        Takes an OpenCV video frame, runs inference, and returns detected objects.
        """
        # Run YOLOv8 inference on the frame (verbose=False hides the console spam)
        results = self.model(frame, imgsz=imgsz, classes=classes, verbose=False)[0]
        
        detections = []
        # Loop through all the bounding boxes YOLO found
        for box in results.boxes:
            conf = float(box.conf[0])
            
            # Only keep detections we are highly confident in
            if conf > confidence_threshold:
                # Get bounding box coordinates (x_min, y_min, x_max, y_max)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get the class name (e.g., 'person', 'car')
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                detections.append({
                    'class_name': class_name,
                    'name': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detections

    def detect(self, frame, confidence_threshold=0.35):
        """Alias for detect_objects for cross-compatibility."""
        return self.detect_objects(frame, confidence_threshold=confidence_threshold)
