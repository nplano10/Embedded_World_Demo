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
from sony_code.imx500_anomaly_detection import IMX500AnomalyDetector
from sony_code.imx500_object_detection_demo import IMX500ObjectDetector
import sony_code.imx500_object_detection_demo as ob_det
import time
from enum import Enum
import json

def pixmap_to_numpy(pixmap):
    # Convert QPixmap to QImage
    image = pixmap.toImage()
    
    # Ensure the image is in a format compatible with raw data access
    image = image.convertToFormat(QImage.Format_RGB888)
    
    # Extract the width and height
    width = image.width()
    height = image.height()
   # print(f"Width: {width}, Height: {height}")
    
    # Extract the raw pixel data
    ptr = image.bits()
    ptr.setsize(image.byteCount())  # Ensure the byte size is correct
    
    # Calculate bytes per row (includes padding)
    bytes_per_line = image.bytesPerLine()
    
    # Create a raw numpy array from the pixel data (including padding)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
    
    # Remove the padding from each row
    # We only want width * 3 bytes per row (since the image is in RGB format)
    arr = arr[:, :width * 3].reshape((height, width, 3))
    
    return arr


def get_anomaly_scores(self,vibx,viby,vibz):
        vibx_score= np.random.rand(1)[0]
        viby_score= np.random.rand(1)[0]
        vibz_score= np.random.rand(1)[0]
        return vibx_score,viby_score,vibz_score
    
def update_anomaly_score_arrays(self,vibx_data,viby_data,vibz_data):
    vibx_score,viby_score,vibz_score =get_anomaly_scores(vibx_data,viby_data,vibz_data)
    vibx_anomaly_scores = np.roll(vibx_anomaly_scores, 1)  # Shift elements to the right by 1
    vibx_anomaly_scores[0] = vibx_score

    viby_anomaly_scores = np.roll(viby_anomaly_scores, 1)  # Shift elements to the right by 1
    viby_anomaly_scores[0] = viby_score

    vibz_anomaly_scores = np.roll(vibz_anomaly_scores, 1)  # Shift elements to the right by 1
    vibz_anomaly_scores[0] = vibz_score

def update_adxl359_vib_data_shm(adxl359_vib_data_shm_name,adxl359_temp_shm_name, adxl359_vib_data_shape,adxl359_lock):

    adxl359 = ADXL359()  # Adjust according to your actual initialization code
    adxl359._initialize()

    plot1 = pg.PlotWidget(title="Vibration X Axis")
    plot2 = pg.PlotWidget(title="Vibration Y Axis")
    plot3 = pg.PlotWidget(title="Vibration Z Axis")
    plot1.setFixedWidth(850)
    plot1.setFixedHeight(320)
    plot2.setFixedWidth(850)
    plot2.setFixedHeight(320)
    plot3.setFixedWidth(850)
    plot3.setFixedHeight(320)

    plot1_item = plot1.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='b')
    plot2_item =plot2.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='g')
    plot3_item =plot3.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='r')


    # Create the three anomaly plots
    anomaly_score_plot1 = pg.PlotWidget(title="Anomaly Plot 1")

    vibx_anomaly_scores = np.zeros(20)


    anomaly_score_plot1_item = anomaly_score_plot1.plot(np.arange(20), vibx_anomaly_scores, pen='orange', name="Anomaly Score 1")


    # existing_shm = shared_memory.SharedMemory(name=adxl359_vib_data_shm.name)
    # shm_array = np.ndarray((1,), dtype=np.float16, buffer=adxl359_temp_shm.buf)

    existing_adxl359_vib_data_shm = shared_memory.SharedMemory(name=adxl359_vib_data_shm_name)
    existing_adxl359_temp_shm = shared_memory.SharedMemory(name=adxl359_temp_shm_name)
    shared_adxl359_vib_data = np.ndarray(adxl359_vib_data_shape, dtype=np.int8, buffer=existing_adxl359_vib_data_shm.buf)
    shared_adxl359_temp_data = np.ndarray((1,), dtype=np.float16, buffer=existing_adxl359_temp_shm.buf)


    

    while True:
        x_data,y_data,z_data,temp_data = adxl359.collect_data() # Example method from adxl359 object
        time.sleep(1)
        plot1_item.setData(np.linspace(0, 1000, 1000).tolist(),x_data)
        plot2_item.setData(np.linspace(0, 1000, 1000).tolist(),y_data)
        plot3_item.setData(np.linspace(0, 1000, 1000).tolist(),z_data)

        #temperature_label.setText(f"Temperature: {temp:.2f} °C")
        # update_anomaly_score_arrays(x_data,y_data,z_data)
        # index = np.arange(20)
        # anomaly_score_plot1_item.setData(index, vibx_anomaly_scores)
        # anomaly_score_plot2_item.setData(index, viby_anomaly_scores)
        # anomaly_score_plot3_item.setData(index, vibz_anomaly_scores)   

        with adxl359_lock:                                        
            shared_adxl359_vib_data[:, :, :, 0] =pixmap_to_numpy(plot1.grab())
            shared_adxl359_vib_data[:, :, :, 1] =pixmap_to_numpy(plot2.grab()) 
            shared_adxl359_vib_data[:, :, :, 2] =pixmap_to_numpy(plot3.grab())
            shared_adxl359_temp_data[0] = temp_data[0]