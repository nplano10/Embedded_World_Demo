
import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QDesktopWidget,QHBoxLayout,QComboBox,QTextEdit

from multiprocessing import shared_memory
from picamera2 import Picamera2
import adxl359
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
from sony_code.imx500_anomaly_detection import IMX500AnomalyDetector
from sony_code.imx500_object_detection_demo import IMX500ObjectDetector
import sony_code.imx500_object_detection_demo as ob_det
import time
from enum import Enum
import json







picam2_0 = Picamera2(0)
picam2_0.video_configuration.controls.FrameRate = 30.0
picam2_0.video_configuration.size = (640, 480)
picam2_0.start("video")

picam2_1 = Picamera2(1)
picam2_1.video_configuration.controls.FrameRate = 30.0
picam2_0.video_configuration.size = (640, 480)
picam2_1.start("video")

start=0
stop=0
for i in range(100):
    # start = time.time()
    time.sleep(1/30)

    start = time.time()
    t = picam2_0.capture_array()
    y = picam2_1.capture_array()
    print(np.shape(t))
    stop = time.time()
    print("update shared mem  freq ", 1/(stop-start))


