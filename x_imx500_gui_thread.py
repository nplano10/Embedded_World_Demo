import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QDesktopWidget,QHBoxLayout,QComboBox,QTextEdit

from multiprocessing import shared_memory
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


class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def __init__(self, parent, camera_one_shm_name,camera_two_shm_name,camera_shape,camera_one_lock,camera_two_lock):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_one = None
        self.previous_camera_two = None
        self.camera_one_shm_name =camera_one_shm_name
        self.camera_two_shm_name =camera_two_shm_name
        self.camera_shape = camera_shape
        self.camera_one_lock = camera_one_lock
        self.camera_two_lock = camera_two_lock

    
    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap


    def run(self):

        # Attach to the shared memory for camera one
        with self.camera_one_lock:
            one_shm = shared_memory.SharedMemory(name=self.camera_one_shm_name)
            one_image = np.ndarray(self.camera_shape, dtype=np.uint8, buffer=one_shm.buf)
            camera_one = copy.deepcopy(one_image)

        # Attach to the shared memory for camera two
        with self.camera_two_lock:
            two_shm = shared_memory.SharedMemory(name=self.camera_two_shm_name)
            two_image = np.ndarray(self.camera_shape, dtype=np.uint8, buffer=two_shm.buf)
            camera_two = copy.deepcopy(two_image)

        # Check if either camera has new data
        if (self.previous_camera_one is None or not np.array_equal(camera_one, self.previous_camera_one)) or \
           (self.previous_camera_two is None or not np.array_equal(camera_two, self.previous_camera_two)):
            # Update previous images with current ones
            self.previous_camera_one = camera_one.copy()  # TODO: do we need to do another copy after the prev deepcopy?
            self.previous_camera_two = camera_two.copy()
            # Emit the signal with the new data 
            self.camera_feed_signal.emit((self.numpy_arrray_to_pixmap(camera_one), self.numpy_arrray_to_pixmap(camera_two)))