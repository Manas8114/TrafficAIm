import numpy as np
import cv2
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime
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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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
LANE_DIVISIONS = 3  # Number of lanes to monitor
MIN_GREEN_TIME = 10  # Minimum green light duration (seconds)
YELLOW_DURATION = 3  # Yellow light duration (seconds)
RED_DURATION = 2  # Red light duration (seconds)

# Load Tesseract path
pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")


class EnhancedVehicleDetector:
    """Detects and classifies vehicles using YOLO and tracks their movement."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = YOLO(model_path)
        self.vehicle_history = {}  # Tracks vehicle movement patterns

    def detect_and_track(self, frame: np.ndarray) -> Tuple[dict, dict, bool, np.ndarray]:
        """Detects vehicles, classifies them, and tracks their movement."""
        results = self.model(frame)
        annotated_frame = frame.copy()
        lane_densities = {f"lane_{i+1}": 0 for i in range(LANE_DIVISIONS)}
        vehicle_counts = defaultdict(int)
        emergency_present = False

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, scores, class_ids):
                cls = self.model.names[class_id]
                vehicle_counts[cls] += 1
                if cls == "emergency":
                    emergency_present = True

                # Update position history for movement analysis
                x1, y1, x2, y2 = box
                x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                if class_id not in self.vehicle_history:
                    self.vehicle_history[class_id] = []
                self.vehicle_history[class_id].append((x_center, y_center))

                # Calculate lane density
                lane_idx = int(x_center / (frame.shape[1] / LANE_DIVISIONS))
                lane_densities[f"lane_{lane_idx+1}"] += 1

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{cls} {score:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return vehicle_counts, lane_densities, emergency_present, annotated_frame


class TrafficPredictor:
    """Predicts traffic congestion using LSTM models."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_lstm_model()
        self.history = []

    def _build_lstm_model(self) -> Sequential:
        """Builds an LSTM model for traffic prediction."""
        model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(10, 1)),
                LSTM(50),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def update_and_predict(self, density_data: dict) -> dict:
        """Updates historical data and predicts future traffic density."""
        self.history.append(sum(density_data.values()))
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Generate predictions
        scaled_data = self.scaler.fit_transform(np.array(self.history).reshape(-1, 1))
        if len(scaled_data) >= 10:
            X = np.array([scaled_data[i - 10 : i] for i in range(10, len(scaled_data))])
            y = scaled_data[10:]
            self.model.fit(X, y, epochs=10, verbose=0)
            future_pred = self.model.predict(X[-1].reshape(1, 10, 1))
            return {"predicted_density": float(self.scaler.inverse_transform(future_pred)[0][0])}
        return {}


class IncidentDetector:
    """Detects traffic incidents like stopped vehicles or accidents."""

    def __init__(self):
        self.stopped_vehicles = {}
        self.accident_zones = []

    def detect_incidents(self, vehicle_history: dict, frame: np.ndarray) -> List[dict]:
        """Detects incidents based on vehicle movement patterns."""
        incidents = []
        current_time = time.time()

        # Detect stopped vehicles
        for vid, positions in list(vehicle_history.items()):
            if len(positions) > 10 and self._is_stopped(positions[-10:]):
                incidents.append(
                    {
                        "type": "stopped_vehicle",
                        "position": positions[-1],
                        "duration": current_time - self.stopped_vehicles.get(vid, current_time),
                    }
                )
                self.stopped_vehicles[vid] = current_time

        # Detect accidents using motion patterns
        if self._detect_sudden_stops(vehicle_history):
            incidents.append({"type": "possible_accident", "position": self._get_hotspot(vehicle_history)})

        return incidents

    def _is_stopped(self, positions: list) -> bool:
        """Checks if a vehicle has stopped."""
        return all(abs(positions[i][0] - positions[-1][0]) < 5 for i in range(len(positions)))

    def _detect_sudden_stops(self, vehicle_history: dict) -> bool:
        """Detects sudden stops indicating potential accidents."""
        for vid, positions in vehicle_history.items():
            if len(positions) > 5 and self._is_stopped(positions[-5:]):
                return True
        return False

    def _get_hotspot(self, vehicle_history: dict) -> Tuple[int, int]:
        """Returns the hotspot of an incident."""
        positions = [pos for history in vehicle_history.values() for pos in history]
        return np.mean(positions, axis=0).astype(int)


class DynamicTrafficController:
    """Controls traffic lights dynamically based on real-time data."""

    def __init__(self):
        self.lane_states = {f"lane_{i+1}": "red" for i in range(LANE_DIVISIONS)}
        self.current_green = None
        self.emergency_override = False

    def calculate_timings(self, densities: dict, emergencies: bool) -> dict:
        """Calculates traffic light timings based on lane densities."""
        if emergencies:
            return {lane: ("green" if lane == "emergency_route" else "red") for lane in self.lane_states}

        total = sum(densities.values())
        if total == 0:
            return {lane: "green" for lane in self.lane_states}

        timings = {}
        for lane, density in densities.items():
            timings[lane] = max(MIN_GREEN_TIME, min(60, int((density / total) * 90)))
        return timings


class EnvironmentalMonitor:
    """Monitors environmental impact of traffic."""

    def __init__(self):
        self.total_idle_time = 0
        self.emission_data = []

    def update_emissions(self, idle_vehicles: int, duration: float):
        """Updates CO2 emissions based on idle vehicles."""
        emissions = idle_vehicles * duration * EMISSION_FACTOR
        self.emission_data.append(
            {
                "timestamp": datetime.now(),
                "co2_kg": emissions,
                "vehicles_affected": idle_vehicles,
            }
        )


class IntegratedTrafficSystem:
    """Main class integrating all components."""

    def __init__(self):
        self.detector = EnhancedVehicleDetector("yolov8n.pt")
        self.predictor = TrafficPredictor()
        self.incident_detector = IncidentDetector()
        self.controller = DynamicTrafficController()
        self.emission_monitor = EnvironmentalMonitor()
        self.db = MongoClient(os.getenv("MONGODB_URI")).traffic
        self.executor = ThreadPoolExecutor(max_workers=8)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame and returns annotated results."""
        # Vehicle detection and tracking
        counts, densities, emergency, annotated = self.detector.detect_and_track(frame)

        # Traffic prediction
        prediction = self.predictor.update_and_predict(densities)

        # Incident detection
        incidents = self.incident_detector.detect_incidents(self.detector.vehicle_history, frame)

        # Calculate traffic light timings
        timings = self.controller.calculate_timings(densities, emergency)

        # Environmental impact
        idle_count = sum(1 for v in self.detector.vehicle_history.values() if self._is_idling(v))
        self.emission_monitor.update_emissions(idle_count, 1 / 30)  # Assume 30 FPS

        # Store data and handle incidents
        self._store_data(counts, densities, incidents, timings)
        self._handle_incidents(incidents)

        return self._annotate_frame(annotated, counts, densities, incidents, timings)

    def _store_data(self, counts: dict, densities: dict, incidents: list, timings: dict):
        """Stores data in MongoDB."""
        doc = {
            "timestamp": datetime.now(),
            "counts": counts,
            "densities": densities,
            "incidents": incidents,
            "timings": timings,
            "emissions": self.emission_monitor.emission_data[-1] if self.emission_monitor.emission_data else None,
        }
        self.executor.submit(self.db.traffic_data.insert_one, doc)

    def _handle_incidents(self, incidents: list):
        """Handles detected incidents."""
        for incident in incidents:
            if incident["type"] in ["accident", "emergency"]:
                self._call_emergency_services(incident)
            self.db.incidents.insert_one(incident)

    def _call_emergency_services(self, incident: dict):
        """Calls emergency services (100 for police, 108 for ambulance)."""
        place_name = "Your Location Name"  # Replace with the actual place name or fetch dynamically
        message = f"Accident detected at {place_name}. Please respond immediately."

        # Simulate calling 100 (Police)
        logger.info(f"Calling 100 (Police): {message}")
        # Replace with actual telephony API call to dial 100
        # Example: self._make_phone_call("100", message)

        # Simulate calling 108 (Ambulance)
        logger.info(f"Calling 108 (Ambulance): {message}")
        # Replace with actual telephony API call to dial 108
        # Example: self._make_phone_call("108", message)

    def _make_phone_call(self, number: str, message: str):
        """Placeholder function to make a phone call using a telephony API."""
        # Example implementation using Twilio:
        # from twilio.rest import Client
        # account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        # auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        # client = Client(account_sid, auth_token)
        # call = client.calls.create(
        #     twiml=f'<Response><Say>{message}</Say></Response>',
        #     to=number,
        #     from_=os.getenv("TWILIO_PHONE_NUMBER")
        # )
        # logger.info(f"Call initiated to {number}: {call.sid}")
        pass

    def _annotate_frame(self, frame: np.ndarray, counts: dict, densities: dict, incidents: list, timings: dict) -> np.ndarray:
        """Annotates the frame with traffic data."""
        # Add visualizations for all components
        cv2.putText(frame, f"Densities: {densities}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Timings: {timings}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for incident in incidents:
            cv2.putText(
                frame,
                incident["type"],
                tuple(map(int, incident["position"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return frame


class RealTimeProcessor:
    """Handles real-time video processing."""

    def __init__(self):
        self.system = IntegratedTrafficSystem()
        self.frame_queue = Queue(maxsize=30)

    def start_capture(self):
        """Captures frames from the camera feed."""
        cap = cv2.VideoCapture(os.getenv("http://192.168.29.105:4747/"))
        while True:
            ret, frame = cap.read()
            if ret:
                self.frame_queue.put(frame)
            else:
                logger.error("Frame capture failed")

    def start_processing(self):
        """Processes frames in real-time."""
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed = self.system.process_frame(frame)
                cv2.imshow("Traffic Analytics", processed)
                if cv2.waitKey(1) == ord("q"):
                    break


if __name__ == "__main__":
    processor = RealTimeProcessor()
    threading.Thread(target=processor.start_capture).start()
    processor.start_processing()
