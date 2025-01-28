import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import cv2
from ultralytics import YOLO
import requests
from pymongo import MongoClient, errors
from datetime import datetime
import threading
from queue import Queue
import logging
import os
import json
import sys
from typing import Dict, List, Optional, Tuple
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string: str, max_retries: int = 3):
        self.connection_string = connection_string
        self.max_retries = max_retries
        self.client = None
        self.db = None
        
    @contextmanager
    def get_connection(self):
        try:
            if not self.client:
                self.client = MongoClient(self.connection_string, 
                                       serverSelectionTimeoutMS=5000,
                                       connectTimeoutMS=5000)
                self.db = self.client['Traffic']
            yield self.db
        except errors.ServerSelectionTimeoutError:
            logger.error("Database connection timeout")
            raise
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise
            
    def store_plate_data(self, plate_data: Dict, location: str) -> bool:
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as db:
                    db['license_plates'].insert_one({
                        "plate_number": plate_data["plate_number"],
                        "location": location,
                        "confidence": plate_data.get("confidence", 0.0),
                        "timestamp": datetime.utcnow(),
                        "processed": False
                    })
                return True
            except Exception as e:
                logger.error(f"Failed to store plate data (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(1)  # Wait before retry
        return False

class VehicleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        results = self.model(frame, conf=self.confidence_threshold)
        detected_objects = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "bike": 0,
            "emergency": 0
        }
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                if cls in detected_objects:
                    detected_objects[cls] += 1
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{cls} {conf:.2f}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
        
        return detected_objects, annotated_frame

class RealtimeTrafficSystem:
    def __init__(self, 
                 config_path: str,
                 camera_id: int = 0,
                 buffer_size: int = 30):
        self.load_config(config_path)
        self.camera_id = camera_id
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.fps_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config['mongodb_uri'])
        self.vehicle_detector = VehicleDetector(
            self.config['yolo_model_path'],
            confidence_threshold=self.config['detection_threshold']
        )
        
        # Performance monitoring
        self.performance_metrics = {
            'detection_time': [],
            'processing_time': [],
            'database_time': []
        }
        
    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def start_camera_stream(self):
        self.running = True
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
        # Set camera properties for optimal performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        threading.Thread(target=self._capture_frames, daemon=True).start()
        
    def _capture_frames(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()  # Remove oldest frame
                self.frame_queue.put((frame, time.time()))
            else:
                logger.error("Failed to capture frame")
                self.running = False
                break
                
    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
            
    def process_realtime(self):
        if not self.running:
            self.start_camera_stream()
            
        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get()
                    
                    # Process frame
                    start_time = time.time()
                    detected_objects, annotated_frame = self.vehicle_detector.detect(frame)
                    self.performance_metrics['detection_time'].append(time.time() - start_time)
                    
                    # Store results if emergency vehicle detected
                    if detected_objects['emergency'] > 0:
                        db_start = time.time()
                        self.db_manager.store_plate_data({
                            "plate_number": "EMERGENCY",
                            "confidence": 1.0
                        }, self.config['location'])
                        self.performance_metrics['database_time'].append(time.time() - db_start)
                    
                    # Add performance overlay
                    self.update_fps()
                    cv2.putText(annotated_frame, f"FPS: {self.fps}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('Traffic Management System', annotated_frame)
                    
                    # Check for exit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Log performance periodically
                    if len(self.performance_metrics['detection_time']) >= 100:
                        self._log_performance()
                        self.performance_metrics = {k: [] for k in self.performance_metrics}
                        
        except KeyboardInterrupt:
            logger.info("Shutting down due to keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in real-time processing: {str(e)}")
        finally:
            self.stop()
            
    def _log_performance(self):
        metrics = {
            'avg_detection_time': np.mean(self.performance_metrics['detection_time']),
            'avg_database_time': np.mean(self.performance_metrics['database_time']) 
                if self.performance_metrics['database_time'] else 0
        }
        logger.info(f"Performance metrics: {json.dumps(metrics)}")
            
    def stop(self):
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.release()
        cv2.destroyAllWindows()
        logger.info("Traffic management system stopped")

if __name__ == "__main__":
    try:
        # Configuration details
        config = {
            "mongodb_uri": "localhost:27017",
            "yolo_model_path":r"C:\\Users\\msgok\\OneDrive\\Desktop\\projets\\2025\\models\\yolov8n.pt",
            "detection_threshold": 0.5,
            "location": "intersection_name"
        }
        
        # Write configuration to a temporary file
        config_path = "config.json"
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file)

        # Initialize and run the traffic system
        system = RealtimeTrafficSystem(config_path)
        system.process_realtime()
    except Exception as e:
        logger.critical(f"System crashed: {str(e)}")
        sys.exit(1)
