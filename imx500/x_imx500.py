import numpy as np
from multiprocessing import shared_memory, Lock, Queue
import argparse
import copy
from enum import Enum
from dataclasses import dataclass
from PyQt5.QtCore import QThread, pyqtSignal

# Import Sony-specific modules
from imx500.imx500_object_detection_SORT import IMX500Detector
from imx500.imx500_anomaly_detection import IMX500AnomalyDetector
from imx500.imx500_no_model import IMX500NoModel

# Constants
DET_MODEL = "imx500/Models/detection-mixed/network.rpk"
DET_LABEL = "imx500/Models/detection-mixed/labels.txt"
ANOM_MODEL = "imx500/Models/anomaly-250501/network.rpk"
COLORS = ["red", "royalblue"]


class Model(Enum):
    NO_MODEL = 1
    TRAINED = 2


class IMX500CameraSystem(QThread):
    # Define signals to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)
    log_feed_signal = pyqtSignal(tuple)

    def __init__(self, parent=None, model_type=Model.TRAINED, det_label_fn=DET_LABEL):
        super().__init__(parent)  
        self.previous_camera_det = None
        self.previous_camera_anom = None
        self.model_type = model_type
        
        # Initialize the system components
        self.init_camera_system()
        
        if model_type == Model.TRAINED:
            self.det_labels = self.parse_det_labels(det_label_fn)

    def init_camera_system(self):
        CAMERA_DISTANCE_MM = 40
        CAMERA_DISTANCE_PIXELS = -140
        PIXELS_PER_MM = CAMERA_DISTANCE_PIXELS / CAMERA_DISTANCE_MM
        DETECTION_REGION = [0, 70, 465, 340]
        ANOMALY_IMAGE_THRESHOLD = 0.45

        self.bbox_queue = Queue(maxsize=50)
        self.results_queue = Queue()

        detection_args = argparse.Namespace(
            model=DET_MODEL,
            labels=DET_LABEL,
            camera_index=0,
            fps=20,
            max_disappeared=20,
            iou=0.65,
            threshold=0.5,
            max_detections=10,
            pixels_per_mm=PIXELS_PER_MM,
            detection_region=DETECTION_REGION,
            anomaly_image_threshold=ANOMALY_IMAGE_THRESHOLD,
        )

        anomaly_args = argparse.Namespace(
            model=ANOM_MODEL,
            camera_index=1,
            fps=14,
            image_threshold=ANOMALY_IMAGE_THRESHOLD,
            pixel_threshold=0.30,
            constant_offset_in_pixel=CAMERA_DISTANCE_PIXELS,
            roi_box_size=110,
            pixels_per_mm=PIXELS_PER_MM,
        )
        
        if self.model_type == Model.TRAINED:
            # Create shared memory for both cameras
            self.det_camera_shm = self.create_shared_memory(shape=(100, 3))
            self.anom_camera_shm = self.create_shared_memory(shape=(100, 3))
            
            # Initialize detectors
            self.detector = IMX500Detector(detection_args, camera_shm=self.det_camera_shm)
            self.anomaly_detector = IMX500AnomalyDetector(anomaly_args, camera_shm=self.anom_camera_shm)
        elif self.model_type == Model.NO_MODEL:
            self.detector = IMX500NoModel(0)
            self.anomaly_detector = IMX500NoModel(1)
            self.det_camera_shm = None
            self.anom_camera_shm = None

    def create_shared_memory(self, shape):
        shm = shared_memory.SharedMemory(
            create=True, size=np.prod(shape) * np.float16().itemsize
        )
        
        class SharedMemoryContainer:
            def __init__(self, shm, shape):
                self.alg_shm = shm
                self.alg_shape = shape
                self.alg_lock = Lock()
                
        return SharedMemoryContainer(shm, shape)

    def run(self):
        """Main thread execution - process camera outputs"""
        if (self.model_type == Model.TRAINED 
                and hasattr(self, 'det_camera_shm') 
                and hasattr(self, 'anom_camera_shm') 
                and self.det_camera_shm is not None
                and self.anom_camera_shm is not None):
            # Process output from algorithms for log information
            try:
                det_results = self.read_alg_results(self.det_camera_shm)
                anom_results = self.read_alg_results(self.anom_camera_shm)
                if det_results.shape[0] > 0 or anom_results.shape[0] > 0:
                    det_log = self.det_results_to_string(det_results)
                    anom_log = self.anom_results_to_string(anom_results)
                    self.log_feed_signal.emit((det_log, anom_log))
            except Exception as e:
                print(f"Error processing camera results: {e}")

    def read_alg_results(self, camera_shm):
        """Read algorithm results from shared memory"""
        with camera_shm.alg_lock:
            shm = shared_memory.SharedMemory(name=camera_shm.alg_shm.name)
            results = np.ndarray(
                camera_shm.alg_shape, dtype=np.float16, buffer=shm.buf
            )
            alg_results = copy.deepcopy(results)

            # clear the shm
            results[:] = 0.

        # crop the bottom part of the array
        to_remove = np.all(alg_results == 0, axis=1)
        alg_results = alg_results[~to_remove, :]

        return alg_results

    def det_results_to_string(self, det_results):
        """Convert detection results to a formatted string"""
        output_str = ""
        # sort the rows by tracking_id (column 0)
        det_results = det_results[det_results[:, 0].argsort()]

        for row in det_results:
            # buffer is tracking_id, category, confidence
            cat = int(row[1])
            class_name = self.det_labels[cat]
            curr_str = f"ID {int(row[0])}: <b style='color: {COLORS[cat]};'>{class_name}</b>, conf = {row[2]:.3f}<br>"
            output_str += curr_str

        return output_str

    def anom_results_to_string(self, anom_results):
        """Convert anomaly results to a formatted string"""
        output_str = ""
        # ignore the first row -- this holds the index for appending
        for row in anom_results[1:,:]:
            # buffer is bbox_id, category (is_anomaly), confidence
            cat = int(row[1])
            color = "red" if cat else "limegreen"
            class_name = "anomaly" if cat else "normal"
            curr_str = f"ID {int(row[0])}: <b style='color: {color};'>{class_name}</b>, conf = {row[2]:.3f}<br>"
            output_str += curr_str

        # remove extra <br>
        return output_str.rstrip("<br>")

    @staticmethod
    def parse_det_labels(label_fn):
        """Parse detection labels from a file"""
        with open(label_fn, 'r') as f:
            labels = f.read().splitlines()
        return labels

    def cleanup(self):
        """Clean up resources when the object is destroyed"""
        if self.model_type == Model.TRAINED:
            if hasattr(self, 'det_camera_shm') and self.det_camera_shm:
                self.det_camera_shm.alg_shm.close()
                self.det_camera_shm.alg_shm.unlink()
                print(f"Cleaning shared memory: det_camera_shm")
            
            if hasattr(self, 'anom_camera_shm') and self.anom_camera_shm:
                self.anom_camera_shm.alg_shm.close()
                self.anom_camera_shm.alg_shm.unlink()
                print(f"Cleaning shared memory: anom_camera_shm")