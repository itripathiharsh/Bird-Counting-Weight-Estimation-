import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime

class BirdAnalyzer:
    def __init__(self):
        # Initialize YOLOv8 model (downloads automatically if not present)
        # Using 'yolov8n.pt' for speed. Use 'yolov8m.pt' for better accuracy.
        self.model = YOLO("yolov8n.pt") 

    def estimate_weight_proxy(self, box):
        """
        Calculates a 'Weight Proxy' based on the bounding box pixel area.
        Assumption: In a fixed camera setup, larger pixel area ~= larger bird.
        To get grams, we need a calibration factor 'k': Weight = k * Area.
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        return float(area)

    def process_video(self, video_path: str, output_path: str, fps_sample: int = 1, conf_thresh: float = 0.3):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        # Video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

        # Data storage for JSON response
        time_series_counts = []
        track_samples = {} # Store info about specific tracks
        weight_data = []

        frame_idx = 0
        
        # Calculate frame skip interval if downsampling
        skip_step = max(1, int(orig_fps / fps_sample)) if fps_sample else 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process if it matches our sampling rate, otherwise write original frame
            # (To keep video smooth, we usually process every frame for the output video 
            # but only run heavy logic if needed. Here we process all for smoothness.)
            
            # Run YOLO Tracking (ByteTrack is default in Ultralytics)
            # classes=2 is 'car' in COCO, bird is 14. 
            # Check COCO classes: 14 = bird.
            results = self.model.track(frame, persist=True, conf=conf_thresh, classes=[14], verbose=False)
            
            current_count = 0
            frame_weights = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                current_count = len(track_ids)

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 1. Weight Estimation (Proxy)
                    weight_index = self.estimate_weight_proxy((x1, y1, x2, y2))
                    frame_weights.append(weight_index)

                    # 2. Annotation
                    # Color for box (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label: ID | Weight Index
                    label = f"ID:{track_id} W:{int(weight_index)}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Store sample track data (just once per ID to keep JSON small)
                    if int(track_id) not in track_samples:
                        track_samples[int(track_id)] = {
                            "first_seen_frame": frame_idx,
                            "confidence": float(conf),
                            "sample_box": [int(x1), int(y1), int(x2), int(y2)]
                        }

            # Draw global counters
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {current_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Write frame to output video
            out.write(frame)

            # Collect Time Series Data (every 'skip_step' frames or every frame)
            # Timestamp calculation
            timestamp = round(frame_idx / orig_fps, 2)
            
            time_series_counts.append({
                "time_sec": timestamp,
                "count": current_count,
                "avg_weight_proxy": float(np.mean(frame_weights)) if frame_weights else 0.0
            })

            frame_idx += 1

        cap.release()
        out.release()

        return {
            "total_frames_processed": frame_idx,
            "counts_timeseries": time_series_counts, # Full timeline
            "unique_birds_tracked": len(track_samples),
            "tracks_sample": track_samples,
            "weight_summary": "Values are pixel area (Width * Height). Calibration required for grams."
        }