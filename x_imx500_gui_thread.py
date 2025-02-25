# import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QDesktopWidget,QHBoxLayout,QComboBox,QTextEdit

from dataclasses import dataclass
from multiprocessing import shared_memory, Lock
# from picamera2 import Picamera2
# from  adxl359  import ADXL359
# from io import BytesIO
# import cv2
import copy
# import matplotlib.pyplot as plt
# import multiprocessing
# import time
# import pyqtgraph as pg
# import argparse
# from multiprocessing import Process, Queue
# from sony_code.imx500_object_detection_SORT import IMX500Detector
# from sony_code.imx500_object_detection_demo import IMX500ObjectDetector
# import sony_code.imx500_object_detection_demo as ob_det
# from enum import Enum
# import json


@dataclass
class CameraShm:
    im_shape: list[int]
    alg_shape: list[int]

    def __post_init__(self):
        # Create shared memory and lock for camera image
        self.im_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.im_shape) * np.uint8().itemsize
        )
        self.im_lock = Lock()

        # Create shared memory and lock for algorithm output
        self.alg_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.im_shape) * np.float16().itemsize
        )
        self.alg_lock = Lock()

class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def __init__(
        self,
        parent,
        det_camera_shm: CameraShm,
        anom_camera_shm: CameraShm,
    ):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_det = None
        self.previous_camera_anom = None
        self.det_camera_shm = det_camera_shm
        self.anom_camera_shm = anom_camera_shm
        
    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def run(self):

        # Attach to the shared memory for camera
        camera_image_det = self.readCameraImageShm(self.det_camera_shm)
        camera_image_anom = self.readCameraImageShm(self.anom_camera_shm)
        
        # Check if either camera has new data
        if (self.previous_camera_det is None or not np.array_equal(camera_image_det, self.previous_camera_det)) or \
           (self.previous_camera_anom is None or not np.array_equal(camera_image_anom, self.previous_camera_anom)):
            # Update previous images with current ones
            self.previous_camera_det = camera_image_det
            self.previous_camera_anom = camera_image_anom
            # Emit the signal with the new data
            self.camera_feed_signal.emit((self.numpy_arrray_to_pixmap(camera_image_det), self.numpy_arrray_to_pixmap(camera_image_anom)))

        # Process output from algorithms for log information
        # TODO SUE

    @staticmethod
    def readCameraImageShm(camera_shm: CameraShm):
        with camera_shm.im_lock:
            shm = shared_memory.SharedMemory(name=camera_shm.im_shm.name)
            image = np.ndarray(
                camera_shm.im_shape, dtype=np.uint8, buffer=shm.buf
            )
            camera_image = copy.deepcopy(image)

        return camera_image
