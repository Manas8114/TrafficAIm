import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time
import cv2
from ultralytics import YOLO
import requests
import pytesseract
from pymongo import MongoClient
from datetime import datetime
import re
import threading
from queue import Queue
import sys

# Original TrafficManagementSystem class needs to be defined first
class TrafficManagementSystem:
    def __init__(self):
        self.db = connect_to_mongodb()
        self.traffic_model = build_traffic_model()
        
        # Train the model
        traffic_data, congestion_levels = generate_traffic_data()
        train_size = int(0.8 * len(traffic_data))
        X_train = traffic_data[:train_size]
        y_train = congestion_levels[:train_size]
        self.traffic_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    def process_frame(self, frame, location, weather_api_key, lane_status, priority_route):
        """
        Processes a single frame with all integrated functionalities
        """
        # Vehicle and plate detection
        detected_objects = detect_vehicles_and_priority(frame)
        detected_plates = detect_license_plate(frame)
        
        # Store plates in database
        for plate in detected_plates:
            store_plate_data(self.db, plate, location)
            
            # Visualize detected plates
            x, y, w, h = plate['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, plate['plate_number'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Weather and traffic management
        weather_condition = get_weather_conditions(weather_api_key, location)
        
        # Get current traffic data (replace with real sensor data in production)
        current_traffic = np.random.randint(50, 500, size=(1, 4))
        traffic_forecast = self.traffic_model.predict(current_traffic)
        traffic_forecast = (traffic_forecast > 0.5).astype(int)
        
        # Optimize traffic lights
        timing_allocation = optimize_traffic_lights(
            {f"Lane_{i}": v for i, v in enumerate(current_traffic[0])},
            lane_status
        )
        
        # Handle priority vehicles
        if "PRIORITY_VEHICLE_DETECTED" in detected_objects:
            synchronize_lights(priority_route)
        
        # Update displays
        print(f"Weather Condition: {weather_condition}")
        for lane, timing in timing_allocation.items():
            print(f"{lane}: Green light for {timing} seconds")
            
        return frame, detected_plates, weather_condition, timing_allocation

    def query_plates(self, plate_number=None, start_time=None, end_time=None):
        """
        Query the database for plate records
        """
        query = {}
        if plate_number:
            query['plate_number'] = plate_number
        if start_time and end_time:
            query['timestamp'] = {'$gte': start_time, '$lte': end_time}
        
        try:
            return list(self.db['license_plates'].find(query))
        except Exception as e:
            print(f"Error querying plate data: {e}")
            return []

# Now define the RealtimeTrafficSystem that inherits from TrafficManagementSystem
class RealtimeTrafficSystem(TrafficManagementSystem):
    def __init__(self, camera_source=0):
        """
        Initialize with camera source (0 for default webcam, or IP camera URL/path)
        """
        super().__init__()
        self.camera_source = camera_source
        self.frame_queue = Queue(maxsize=30)  # Buffer for frames
        self.running = False
        self.processed_frames = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def start_camera_stream(self):
        """
        Start the camera stream in a separate thread
        """
        self.running = True
        self.capture = cv2.VideoCapture(self.camera_source)
        
        if not self.capture.isOpened():
            raise ValueError("Could not open camera source")
            
        # Set camera properties for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start frame capture thread
        threading.Thread(target=self._capture_frames, daemon=True).start()
        
    def _capture_frames(self):
        """
        Continuously capture frames from camera
        """
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()  # Remove oldest frame
                self.frame_queue.put(frame)
            else:
                print("Error capturing frame")
                break
                
    def process_realtime(self, location, weather_api_key, lane_status, priority_route):
        """
        Process frames in real-time
        """
        if not self.running:
            self.start_camera_stream()
            
        try:
            while True:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Process frame
                    processed_frame, plates, weather, timings = self.process_frame(
                        frame, location, weather_api_key, lane_status, priority_route
                    )
                    
                    # Calculate FPS
                    self.processed_frames += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.processed_frames
                        self.processed_frames = 0
                        self.last_fps_time = current_time
                    
                    # Display FPS and status
                    cv2.putText(processed_frame, f"FPS: {self.fps}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display weather condition
                    cv2.putText(processed_frame, f"Weather: {weather}", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show the processed frame
                    cv2.imshow('Traffic Management System', processed_frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping traffic management system...")
        finally:
            self.stop()
            
    def stop(self):
        """
        Stop the system and release resources
        """
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.release()
        cv2.destroyAllWindows()

# Make sure all the helper functions are available
# These should be defined before the classes
def connect_to_mongodb():
    """
    Establishes connection to MongoDB database
    """
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['Traffic']
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def generate_traffic_data(samples=1000):
    np.random.seed(42)
    traffic_data = np.random.randint(50, 500, size=(samples, 4))
    congestion_levels = (traffic_data.sum(axis=1) > 1200).astype(int)
    return traffic_data, congestion_levels

def build_traffic_model():
    model = Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Additional helper functions

def detect_vehicles_and_priority(frame):
    """
    Detect vehicles and identify priority vehicles (e.g., emergency vehicles).
    """
    yolo_model = YOLO('yolov8n.pt')  # Assuming YOLOv8 is being used
    results = yolo_model(frame)

    detected_objects = {}
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]
            if cls in ["car", "truck", "bus"]:
                detected_objects[cls] = detected_objects.get(cls, 0) + 1
            if cls == "emergency":
                detected_objects["PRIORITY_VEHICLE_DETECTED"] = True
    
    return detected_objects

def detect_license_plate(frame):
    """
    Detect license plates in the frame and recognize plate numbers using OCR.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = []  # This would normally use a plate detection model

    # Placeholder logic: Detect and recognize plates (dummy data for illustration)
    # Replace with actual model for plate detection
    dummy_plate = {
        "plate_number": "ABC1234",
        "bbox": (100, 100, 200, 50)  # Example bounding box
    }
    plates.append(dummy_plate)

    return plates

def store_plate_data(db, plate, location):
    """
    Store license plate data in MongoDB.
    """
    try:
        db['license_plates'].insert_one({
            "plate_number": plate["plate_number"],
            "location": location,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"Error storing plate data: {e}")

def get_weather_conditions(api_key, location):
    """
    Retrieve weather conditions from a weather API.
    """
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        response = requests.get(url)
        data = response.json()
        return data['current']['condition']['text']
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return "Unknown"

def optimize_traffic_lights(traffic_data, lane_status):
    """
    Optimize traffic light timings based on current traffic data and lane status.
    """
    timings = {}
    for lane, count in traffic_data.items():
        status = lane_status.get(lane, "LOW_TRAFFIC")
        if status == "HIGH_TRAFFIC":
            timings[lane] = 60  # Longer green light
        else:
            timings[lane] = 30  # Default green light
    return timings

def synchronize_lights(priority_route):
    """
    Synchronize traffic lights for priority vehicles along the route.
    """
    print("Synchronizing lights for priority route:", priority_route)

# Example usage remains the same
if __name__ == "__main__":
    try:
        # Initialize system with camera source
        camera_source = 0  # Use 0 for webcam, or RTSP URL for IP camera
        system = RealtimeTrafficSystem(camera_source)

        # Configuration
        location = "New York"
        weather_api_key = "9d467bf6e4c9b18030f2bf6cd74faecd"
        lane_status = {
            "Lane_0": "HIGH_TRAFFIC",
            "Lane_1": "LOW_TRAFFIC",
            "Lane_2": "LOW_TRAFFIC",
            "Lane_3": "HIGH_TRAFFIC"
        }
        priority_route = [
            "Intersection_1",
            "Intersection_2",
            "Intersection_3"
        ]

        print("Starting traffic management system...")
        print("Press 'q' to quit")
        system.process_realtime(location, weather_api_key, lane_status, priority_route)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
