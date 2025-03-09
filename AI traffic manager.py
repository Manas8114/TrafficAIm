import numpy as np
import cv2
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
from queue import Queue
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pytesseract
import urllib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from flask import Flask, render_template, Response
from cryptography.fernet import Fernet
import requests
from twilio.rest import Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
EMISSION_FACTOR = 2.3  # kg CO2 per liter of fuel
LANE_DIVISIONS = 3
MIN_GREEN_TIME = 10
YELLOW_DURATION = 3
RED_DURATION = 2
CAMERA_URLS = [
    "http://192.168.1.9:8080/video",
    "http://192.168.1.9:8080/video"
]

# Security Configuration
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Twilio Alerts Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URI")
db_client = MongoClient(MONGO_URI)
traffic_db = db_client.traffic

class EnhancedVehicleDetector:
    """Advanced vehicle detection and tracking with violation checking"""
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = YOLO(model_path)
        self.vehicle_history = defaultdict(list)
        self.violation_count = 0

    def detect_and_track(self, frame: np.ndarray) -> Tuple[dict, dict, bool, np.ndarray, list]:
        try:
            results = self.model.track(frame, persist=True)
            annotated_frame = frame.copy()
            lane_densities = defaultdict(int)
            vehicle_counts = defaultdict(int)
            emergency_present = False
            violations = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id else []

                for idx, (box, class_id) in enumerate(zip(boxes, class_ids)):
                    cls = self.model.names[class_id]
                    vehicle_counts[cls] += 1
                    track_id = track_ids[idx] if idx < len(track_ids) else None

                    # Update vehicle history
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    if track_id is not None:
                        self.vehicle_history[track_id].append((x_center, y_center, time.time()))

                    # Check for violations
                    if track_id and len(self.vehicle_history[track_id]) > 10:
                        if self._check_speed_violation(track_id):
                            violations.append({
                                "type": "speed",
                                "track_id": track_id,
                                "position": (x_center, y_center),
                                "timestamp": datetime.now()
                            })

                    # Update lane densities
                    lane_idx = int(x_center / (frame.shape[1] / LANE_DIVISIONS))
                    lane_densities[lane_idx] += 1

                    # Draw bounding boxes
                    cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{cls} {track_id}", 
                              (int(box[0]), int(box[1]) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return vehicle_counts, lane_densities, emergency_present, annotated_frame, violations
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return defaultdict(int), defaultdict(int), False, frame, []

    def _check_speed_violation(self, track_id: int) -> bool:
        history = self.vehicle_history[track_id]
        if len(history) < 2:
            return False
        
        # Calculate speed based on pixel movement and time
        dx = history[-1][0] - history[-2][0]
        dt = history[-1][2] - history[-2][2]
        speed = abs(dx) / dt if dt != 0 else 0
        return speed > 50  # Threshold for demonstration

class TrafficPredictor:
    """LSTM-based traffic prediction with adaptive learning"""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_lstm_model()
        self.history = []
        self.update_interval = 60  # Retrain model every minute

    def _build_lstm_model(self) -> Sequential:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 1)),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_and_predict(self, densities: dict) -> dict:
        try:
            current_density = sum(densities.values())
            self.history.append(current_density)
            
            if len(self.history) > 300:
                self.history = self.history[-300:]
            
            if len(self.history) % self.update_interval == 0:
                self._retrain_model()
            
            if len(self.history) >= 30:
                scaled_data = self.scaler.transform(np.array(self.history[-30:]).reshape(-1, 1))
                prediction = self.model.predict(scaled_data.reshape(1, 30, 1))
                return {
                    "current": current_density,
                    "predicted": float(self.scaler.inverse_transform(prediction)[0][0])
                }
            return {"current": current_density}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"current": sum(densities.values())}

    def _retrain_model(self):
        try:
            scaled_data = self.scaler.fit_transform(np.array(self.history).reshape(-1, 1))
            X, y = [], []
            for i in range(30, len(scaled_data)):
                X.append(scaled_data[i-30:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

class IncidentDetector:
    """Real-time incident detection system"""
    def __init__(self):
        self.stopped_vehicles = {}
        self.accident_threshold = 10  # Seconds

    def detect(self, vehicle_history: dict) -> List[dict]:
        incidents = []
        current_time = time.time()
        
        for vid, history in vehicle_history.items():
            if len(history) < 10:
                continue
                
            # Check for stopped vehicles
            if self._is_stopped(history):
                if vid not in self.stopped_vehicles:
                    self.stopped_vehicles[vid] = current_time
                elif current_time - self.stopped_vehicles[vid] > self.accident_threshold:
                    incidents.append({
                        "type": "stopped_vehicle",
                        "track_id": vid,
                        "position": history[-1][:2],
                        "duration": current_time - self.stopped_vehicles[vid]
                    })
            else:
                self.stopped_vehicles.pop(vid, None)
                
        return incidents

    def _is_stopped(self, history: list) -> bool:
        return all(abs(p[0] - history[-1][0]) < 5 for p in history[-10:])

class DynamicTrafficController:
    """Adaptive traffic light control system"""
    def __init__(self):
        self.lane_states = {i: "red" for i in range(LANE_DIVISIONS)}
        self.current_green = None
        self.last_change = time.time()
        self.emergency_mode = False

    def update_lights(self, densities: dict, incidents: list) -> dict:
        if self.emergency_mode:
            return self._handle_emergency()
            
        if any(incident["type"] == "stopped_vehicle" for incident in incidents):
            return self._handle_incident(incidents)
            
        total_density = sum(densities.values())
        if total_density == 0:
            return self.lane_states
            
        # Calculate green times based on density proportions
        green_times = {}
        for lane, density in densities.items():
            green_times[lane] = max(MIN_GREEN_TIME, 
                                  int((density / total_density) * (MIN_GREEN_TIME * LANE_DIVISIONS)))
        
        # Implement phase rotation
        if time.time() - self.last_change > green_times.get(self.current_green, MIN_GREEN_TIME):
            next_lane = max(densities, key=lambda k: densities[k])
            if next_lane != self.current_green:
                self._transition_lights(next_lane)
                self.last_change = time.time()
                
        return self.lane_states

    def _transition_lights(self, new_green: int):
        for lane in self.lane_states:
            self.lane_states[lane] = "green" if lane == new_green else "red"

    def _handle_emergency(self):
        # Prioritize emergency routes
        return {lane: "green" if lane == 0 else "red" for lane in self.lane_states}

    def _handle_incident(self, incidents: list):
        # Flash all lights yellow for attention
        return {lane: "yellow" for lane in self.lane_states}

class EnvironmentalMonitor:
    """CO2 emissions and air quality monitoring"""
    def __init__(self):
        self.emissions = 0.0
        self.idle_vehicles = set()
        self.last_update = time.time()

    def update(self, vehicle_history: dict, lane_states: dict):
        current_time = time.time()
        delta_time = current_time - self.last_update
        
        # Detect idle vehicles
        new_idle = set()
        for vid, history in vehicle_history.items():
            if len(history) > 10 and self._is_idling(history):
                new_idle.add(vid)
                
        # Update emissions
        idle_count = len(new_idle)
        self.emissions += idle_count * EMISSION_FACTOR * delta_time / 3600
        
        self.idle_vehicles = new_idle
        self.last_update = current_time

    def _is_idling(self, history: list) -> bool:
        return all(abs(p[0] - history[-1][0]) < 5 for p in history[-10:])

class TrafficSystem:
    """Integrated traffic management system"""
    def __init__(self):
        self.detector = EnhancedVehicleDetector("yolov8n.pt")
        self.predictor = TrafficPredictor()
        self.incident_detector = IncidentDetector()
        self.controller = DynamicTrafficController()
        self.environment = EnvironmentalMonitor()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.frame_queue = Queue(maxsize=30)
        self.running = True
        self.last_alert = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            # Vehicle detection and tracking
            counts, densities, emergency, annotated, violations = self.detector.detect_and_track(frame)
            
            # Traffic prediction
            prediction = self.predictor.update_and_predict(densities)
            
            # Incident detection
            incidents = self.incident_detector.detect(self.detector.vehicle_history)
            
            # Update traffic lights
            light_states = self.controller.update_lights(densities, incidents)
            
            # Update environmental impact
            self.environment.update(self.detector.vehicle_history, light_states)
            
            # Handle alerts
            if incidents and time.time() - self.last_alert > 300:  # 5 minute cooldown
                self._send_alerts(incidents)
                self.last_alert = time.time()
            
            # Store encrypted data
            self._store_data({
                "timestamp": datetime.now(),
                "counts": dict(counts),
                "densities": dict(densities),
                "incidents": incidents,
                "lights": light_states,
                "emissions": self.environment.emissions,
                "prediction": prediction
            })
            
            # Add UI overlays
            annotated = self._add_overlays(annotated, light_states, prediction)
            return annotated
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return frame

    def _add_overlays(self, frame: np.ndarray, lights: dict, prediction: dict) -> np.ndarray:
        # Add traffic light status
        for lane, state in lights.items():
            color = (0, 255, 0) if state == "green" else (0, 255, 255) if state == "yellow" else (0, 0, 255)
            cv2.circle(frame, (50 + lane*100, 50), 20, color, -1)
        
        # Add prediction text
        cv2.putText(frame, f"Predicted density: {prediction.get('predicted', 'N/A')}", 
                  (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def _store_data(self, data: dict):
        # Encrypt sensitive data
        encrypted = cipher_suite.encrypt(json.dumps(data).encode())
        self.executor.submit(
            traffic_db.traffic_data.insert_one,
            {"encrypted": encrypted, "timestamp": datetime.now()}
        )

    def _send_alerts(self, incidents: list):
        if not twilio_client:
            return
            
        for incident in incidents:
            try:
                message = twilio_client.messages.create(
                    body=f"Traffic Alert: {incident['type']} detected at {datetime.now().strftime('%H:%M:%S')}",
                    from_=TWILIO_PHONE_NUMBER,
                    to=ALERT_PHONE_NUMBER
                )
                logger.info(f"Sent alert: {message.sid}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    def camera_thread(self, url: str):
        """Capture frames from IP camera"""
        while self.running:
            try:
                cap = cv2.VideoCapture(url)
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.frame_queue.qsize() < 30:
                        self.frame_queue.put(frame)
                cap.release()
            except Exception as e:
                logger.error(f"Camera error: {e}")
                time.sleep(1)

    def processing_thread(self):
        """Process frames from queue"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed = self.process_frame(frame)
                cv2.imshow("Traffic Management", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

# Flask Web Interface
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        system = TrafficSystem()
        while system.running:
            if not system.frame_queue.empty():
                frame = system.process_frame(system.frame_queue.get())
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Initialize traffic system
    system = TrafficSystem()
    
    # Start camera threads
    for url in CAMERA_URLS:
        threading.Thread(target=system.camera_thread, args=(url,), daemon=True).start()
    
    # Start processing thread
    processing_thread = threading.Thread(target=system.processing_thread, daemon=True)
    processing_thread.start()
    
    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
    # Cleanup
    system.running = False
    processing_thread.join()
    cv2.destroyAllWindows()
