import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QDesktopWidget,QHBoxLayout,QComboBox,QTextEdit

from dataclasses import dataclass
from multiprocessing import shared_memory, Lock
from picamera2 import Picamera2
from  adxl359  import ADXL359
from io import BytesIO
import cv2
import copy
import matplotlib.pyplot as plt
import multiprocessing
import time
import pyqtgraph as pg
import argparse
from multiprocessing import Process, Queue
from sony_code.imx500_object_detection_SORT import IMX500Detector
from sony_code.imx500_object_detection_demo import IMX500ObjectDetector
import sony_code.imx500_object_detection_demo as ob_det
import time
from enum import Enum
import json


@dataclass
class CameraShm:
    shape: list[int]

    def __post_init__(self):
        self.shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.shape) * np.uint8().itemsize
        )
        self.lock = Lock()


class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def __init__(
        self,
        parent,
        camera_one_shm: CameraShm,
        camera_two_shm: CameraShm,
    ):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_one = None
        self.previous_camera_two = None
        self.camera_one_shm = camera_one_shm
        self.camera_two_shm = camera_two_shm
        
    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def run(self):

        # Attach to the shared memory for camera
        camera_one = self.readCameraShm(self.camera_one_shm)
        camera_two = self.readCameraShm(self.camera_two_shm)

        
        # Check if either camera has new data
        if (self.previous_camera_one is None or not np.array_equal(camera_one, self.previous_camera_one)) or \
           (self.previous_camera_two is None or not np.array_equal(camera_two, self.previous_camera_two)):
            # Update previous images with current ones
            self.previous_camera_one = camera_one
            self.previous_camera_two = camera_two
            # Emit the signal with the new data
            self.camera_feed_signal.emit((self.numpy_arrray_to_pixmap(camera_one), self.numpy_arrray_to_pixmap(camera_two)))

    @staticmethod
    def readCameraShm(camera_shm: CameraShm):
        with camera_shm.lock:
            shm = shared_memory.SharedMemory(name=camera_shm.shm.name)
            image = np.ndarray(
                camera_shm.shape, dtype=np.uint8, buffer=shm.buf
            )
            camera_data = copy.deepcopy(image)

        return camera_data
