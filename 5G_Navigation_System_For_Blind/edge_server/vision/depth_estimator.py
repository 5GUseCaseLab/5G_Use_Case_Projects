import torch
import cv2
import numpy as np
import sys

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initializes the pre-trained MiDaS depth estimation model.
        We use 'MiDaS_small' by default to ensure real-time speed alongside YOLO.
        """
        print(f"⏳ Loading Depth model ({model_type})... This will take a moment on the first run.")
        
        # Automatically use a Graphics Card (GPU) if available, otherwise use the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Processing Depth on device: {self.device}")
        
        # Load the MiDaS model directly from PyTorch Hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode (not training mode)
        
        # Load the transforms (these format the webcam image perfectly for the AI)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
            
        print("✅ MiDaS Depth Model loaded successfully!")

    def estimate_depth(self, frame):
        """
        Takes a standard BGR OpenCV frame, processes it, and returns a 2D depth map.
        """
        # OpenCV captures in BGR, but MiDaS expects RGB colors
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms and move the image to the CPU/GPU
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Run the AI model (torch.no_grad() speeds it up by disabling training math)
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # The AI outputs a tiny image. We resize it back to match your webcam's resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Move the data back to the CPU as a standard numpy array
        raw_depth_map = prediction.cpu().numpy()
        
        # --- FOR VISUALIZATION ONLY ---
        # The AI outputs raw mathematical distance scores. 
        # We normalize these scores into pixels (0 to 255) so we can see it on screen.
        depth_min = raw_depth_map.min()
        depth_max = raw_depth_map.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (raw_depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = raw_depth_map
            
        depth_visual = (255 * depth_normalized).astype(np.uint8)
        
        # Apply a heat-map color filter (Inferno). 
        # Objects very close to you will be bright Yellow/White. Far away is dark Purple/Black.
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)
        
        return raw_depth_map, depth_colormap

if __name__ == "__main__":
    # --- LOCAL TESTING BLOCK ---
    print("=========================================")
    print("   5G NAVIGATION - LOCAL DEPTH TEST      ")
    print("=========================================")
    
    estimator = DepthEstimator()
    
    # Open laptop webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
    else:
        print("\n🟢 Live Depth active! Press 'q' on your keyboard to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run the Depth Estimator
            raw_depth, depth_colormap = estimator.estimate_depth(frame)
            
            # Show original and depth side-by-side
            cv2.imshow("Original WebCam", frame)
            cv2.imshow("MiDaS Depth Map", depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
