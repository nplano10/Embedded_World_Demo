
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


from x_shared_mem_adxl import update_adxl359_vib_data_shm
#from x_camera_thread_gui import CameraThread
from x_adxl_thread_gui import Adxl359Thread


camera_one_lock = multiprocessing.Lock()
camera_two_lock = multiprocessing.Lock()
width,height, channels = 480, 640, 3
camera_shape = (width,height, channels)
camera_one_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)
camera_two_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)




class Model(Enum):
    OBJECT = 1
    SMARTIE = 2
    SONY = 3
    NOMODEL = 4



def anomaly_process(event,bbox_queue, results_queue, args):

    global camera_two_lock
    global camera_two_shm
    global camera_shape 
    detector = IMX500AnomalyDetector(args)
    detector.picam2.pre_callback = lambda req: detector.process_frame(
        req, bbox_queue, results_queue)
    print("Done loading model")
    two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
    two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)

    while not event.is_set():
        time.sleep(1/30)
        with camera_two_lock:
            two_image[:] =detector.picam2.capture_array().astype(np.uint8)[:, :, :3]

def detection_process(event,bbox_queue, results_queue, args):
    global camera_one_lock
    global camera_one_shm
    global camera_shape 

    detector = IMX500Detector(args)
    detector.picam2.pre_callback = lambda req: detector.draw_detections(req, results_queue)
    print("Done loading model")

    one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
    one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
    while not event.is_set():
        time.sleep(1/30)
        with camera_one_lock:  # Ensure exclusive access to the shared memory
            metadata = detector.picam2.capture_metadata()
            detector.last_results = detector.parse_detections(
                metadata,
                args.iou,
                args.max_detections,
                args.threshold
            )
            
            detector.update_bbox_queue(bbox_queue)
            one_image[:] = detector.picam2.capture_array().astype(np.uint8)[:, :, :3]


def no_model_process(event):
    global camera_one_lock
    global camera_one_shm
    global camera_two_lock
    global camera_two_shm
    global camera_shape 
    picam2_0 = Picamera2(0)
    picam2_0.video_configuration.controls.FrameRate = 30.0
    picam2_0.video_configuration.size = (640, 480)
    picam2_0.start("video")
    picam2_1 = Picamera2(1)
    picam2_1.video_configuration.controls.FrameRate = 30.0
    picam2_1.video_configuration.size = (640, 480)
    picam2_1.start("video")
    one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
    one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
    two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
    two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)

    while not event.is_set():
        time.sleep(1/30)
        with camera_one_lock:  
            one_image[:] = picam2_0.capture_array().astype(np.uint8)[:, :, :3]
        with camera_two_lock:
            two_image[:] =picam2_1.capture_array().astype(np.uint8)[:, :, :3]

       

def obj_detection_process(event,mode):

    global camera_one_lock
    global camera_one_shm
    global camera_two_lock
    global camera_two_shm
    global camera_shape 

    one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=shared_memory.SharedMemory(name=camera_one_shm.name).buf)
    two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=shared_memory.SharedMemory(name=camera_two_shm.name).buf)
    

    if(mode == 0):
        camera1 = IMX500ObjectDetector(ob_det.sony_args(),1)
        camera2 = IMX500ObjectDetector(ob_det.sony_args(),0)   
    if(mode ==1 ):
        camera1 = IMX500ObjectDetector(ob_det.sue_args(),1)
        camera2 = IMX500ObjectDetector(ob_det.sue_args(),0)   
    camera1.picam2.pre_callback = camera1.draw_detections
    camera2.picam2.pre_callback = camera2.draw_detections
    while not event.is_set():
        time.sleep(1/30)
        with camera_one_lock:  # Ensure exclusive access to the shared memory
            meta_data = camera1.picam2.capture_metadata()
            camera1.last_results = camera1.parse_detections(meta_data)
            one_image[:] = camera1.picam2.capture_array().astype(np.uint8)[:, :, :3]
        with camera_two_lock:
            camera2.last_results = camera2.parse_detections(camera2.picam2.capture_metadata())
            two_image[:] =camera2.picam2.capture_array().astype(np.uint8)[:, :, :3]

        
def update_imx500_shm(selected_model,event):

    CAMERA_DISTANCE_MM = 40  # Physical distance between cameras in mm
    CAMERA_DISTANCE_PIXELS = -122  # Distance in pixels
    PIXELS_PER_MM = CAMERA_DISTANCE_PIXELS / CAMERA_DISTANCE_MM
    DETECTION_REGION = [20, 70, 600, 410]

    bbox_queue = Queue(maxsize=50)  # Queue for passing bounding boxes
    results_queue = Queue()  # Queue for receiving classification results

    pill_detection_args = argparse.Namespace(
        model="sony_code/Models/detection-imx500/network.rpk", 
        labels="sony_code/Models/detection-imx500/labels.txt",
        camera_index=0,
        fps=20,
        max_disappeared=20,
        iou=0.65,
        threshold=0.5,
        max_detections=10,
        pixels_per_mm=PIXELS_PER_MM,
        detection_region=DETECTION_REGION
    )
    anomaly_detection_args = argparse.Namespace(
        model="sony_code/Models/anomaly-imx500/network.rpk",
        camera_index=1,
        fps=20,
        image_threshold=0.50,
        pixel_threshold=0.30,
        constant_offset_in_pixel=CAMERA_DISTANCE_PIXELS,
        roi_box_size=128,
        pixels_per_mm=PIXELS_PER_MM
    )


    pill_detection_proc = Process(target=detection_process, args=(event,bbox_queue, results_queue, pill_detection_args))
    anomaly_detection_proc = Process(target=anomaly_process, args=(event,bbox_queue, results_queue, anomaly_detection_args))

    if selected_model == Model.SONY:
        pill_detection_proc.start()
        anomaly_detection_proc.start()
        pill_detection_proc.join()
        anomaly_detection_proc.join()
    
    if selected_model == Model.NOMODEL:
       no_model_process(event)

    if selected_model == Model.OBJECT:
        
        obj_detection_process(event,0)

    if selected_model == Model.SMARTIE:
        obj_detection_process(event,1)


class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_one = None
        self.previous_camera_two = None

    
    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap


    def run(self):
        global camera_one_lock
        global camera_one_shm
        global camera_two_lock
        global camera_two_shm
        global camera_shape 

        # Attach to the shared memory for camera one
        with camera_one_lock:
            one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
            one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
            camera_one = copy.deepcopy(one_image)

        # Attach to the shared memory for camera two
        with camera_two_lock:
            two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
            two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)
            camera_two = copy.deepcopy(two_image)

        # Check if either camera has new data
        if (self.previous_camera_one is None or not np.array_equal(camera_one, self.previous_camera_one)) or \
           (self.previous_camera_two is None or not np.array_equal(camera_two, self.previous_camera_two)):
            # Update previous images with current ones
            self.previous_camera_one = camera_one.copy()
            self.previous_camera_two = camera_two.copy()
            # Emit the signal with the new data 
            self.camera_feed_signal.emit((self.numpy_arrray_to_pixmap(camera_one), self.numpy_arrray_to_pixmap(camera_two)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.imx500_height = 480
        self.imx500_height = 640 
        self.imx500_fps = 60
        self.adxl359_sample_rate = 1000
        self.adxl359_sample_length = 1000 


        # Set up the window to match the screen size
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = int(screen.height()*0.90)
        self.setGeometry(0, 0, screen_width, screen_height)

        # Set up the central widget and the main layout
        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: #2f2f2f;")
        self.setCentralWidget(central_widget)

        # Create the main layout (vertical)
        main_layout = QVBoxLayout()

        # Create the top row (camera feeds and logging windows)
        top_layout = QHBoxLayout()

        # Camera feed 1 and its logging window
        camera_feed_1_layout = QHBoxLayout()
        self.camera_feed_1 = QLabel(self)
        self.camera_feed_1.height= int(screen_height/3)
        self.camera_feed_1.width= int(screen_width/4)
        self.camera_feed_1.setFixedSize(self.camera_feed_1.width,self.camera_feed_1.height)
        self.camera_feed_1.setText("Camera Feed 1")  # Placeholder text
        self.camera_feed_1.setStyleSheet("background-color: lightgray;")
        camera_feed_1_layout.addWidget(self.camera_feed_1)

        # Logging window for camera feed 1
        self.log_1 = QTextEdit(self)
        self.log_1.setPlaceholderText("Logging window for Camera Feed 1...")
        self.log_1.setReadOnly(True)
        self.log_1.height= int(screen_height/3)
        self.log_1.width = int(screen_width/8)
        self.log_1.setFixedSize(self.log_1.width,self.log_1.height)
        self.log_1.setStyleSheet("background-color: black; color: white;")
        camera_feed_1_layout.addWidget(self.log_1)
        top_layout.addLayout(camera_feed_1_layout)

        # Camera feed 2 and its logging window
        camera_feed_2_layout = QHBoxLayout()
        self.camera_feed_2 = QLabel(self)
        self.camera_feed_2.height= int(screen_height/3)
        self.camera_feed_2.width= int(screen_width/4)
        self.camera_feed_2.setFixedSize(self.camera_feed_2.width,self.camera_feed_2.height)
        self.camera_feed_2.setText("Camera Feed 2")  # Placeholder text
        self.camera_feed_2.setStyleSheet("background-color: lightgray;")
        camera_feed_2_layout.addWidget(self.camera_feed_2)

        # Logging window for camera feed 2
        self.log_2 = QTextEdit(self)
        self.log_2.height= int(screen_height/3)
        self.log_2.width = int(screen_width/8)
        self.log_2.setFixedSize(self.log_2.width,self.log_2.height)
        self.log_2.setPlaceholderText("Logging window for Camera Feed 2...")
        self.log_2.setReadOnly(True)
        self.log_2.setStyleSheet("background-color: black; color: white;")
        camera_feed_2_layout.addWidget(self.log_2)
        top_layout.addLayout(camera_feed_2_layout)

        # Add the top row to the main layout
        main_layout.addLayout(top_layout)

        # Create the bottom row (buttons, temperature, plots)
        bottom_layout = QHBoxLayout()

        # Create control buttons (Exit, Start, Stop)
        height_buttons = int(screen_height*(1/32))
        width_buttons = int(screen_width/16)
        
        button_exit = QPushButton('Exit', self)
        button_exit.clicked.connect(self.exit_application)
        button_exit.setFixedSize(width_buttons,height_buttons)
        
        button_start = QPushButton('Start', self)
        button_start.setFixedSize(width_buttons,height_buttons)

        button_stop = QPushButton('Stop', self)
        button_stop.setFixedSize(width_buttons,height_buttons)

        button_dispense = QPushButton('Dispense', self)
        button_dispense.setFixedSize(width_buttons,height_buttons)

        # Create button layout and add buttons to it
        button_layout = QVBoxLayout()
        button_layout.addWidget(button_exit)
        button_layout.addWidget(button_start)
        button_layout.addWidget(button_stop)
        button_layout.addWidget(button_dispense)

        # Create Apply Model button and dropdown
        apply_model_button = QPushButton('Apply Model', self)
        apply_model_button.clicked.connect(self.apply_model)  # Connect to a function for applying model
        apply_model_button.setFixedSize(width_buttons,height_buttons)
        self.model = Model.NOMODEL
        
        # Create dropdown for model selection
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.setFixedSize(width_buttons,height_buttons)

        for model in Model:
            self.model_dropdown.addItem(model.name, model)


  
        # Create a horizontal layout for Apply Model button and dropdown
        apply_model_layout = QHBoxLayout()
        apply_model_layout.addWidget(apply_model_button)
        apply_model_layout.addWidget(self.model_dropdown)


        # Create temperature label
        self.temperature_label = QLabel(self)
        
        self.temperature_label.setStyleSheet("font-size: 18px;")

        # Add buttons, apply model controls, and temperature label to the bottom-left layout
        button_layout.addLayout(apply_model_layout)
        button_layout.addWidget(self.temperature_label)

        # Add the buttons, apply model controls, and temperature layout to the bottom-left of the layout
        bottom_layout.addLayout(button_layout)
        bottom_layout.addStretch(1) 

        # Create a vertical layout for the vibration plots (3 plots on the left)
        plot_layout = QVBoxLayout()

        self.vibx_graph = QLabel(self)
        self.vibx_graph.height= int(screen_height*(2/9))
        self.vibx_graph.width= int(screen_width/3)
        self.vibx_graph.setFixedSize(self.vibx_graph.width,self.vibx_graph.height)

        self.viby_graph = QLabel(self)
        self.viby_graph.height= int(screen_height*(2/9))
        self.viby_graph.width= int(screen_width/3)
        self.viby_graph.setFixedSize(self.viby_graph.width,self.viby_graph.height)

        self.vibz_graph = QLabel(self)
        self.vibz_graph.height= int(screen_height*(2/9))
        self.vibz_graph.width= int(screen_width/3)
        self.vibz_graph.setFixedSize(self.vibz_graph.width,self.vibz_graph.height)

        # Update the plot data
        # Add the plots to the vertical layout
        plot_layout.addWidget(self.vibx_graph)
        plot_layout.addWidget(self.viby_graph)
        plot_layout.addWidget(self.vibz_graph)

        # Add the plot layout to the bottom row (left side)
        bottom_layout.addLayout(plot_layout)
        bottom_layout.addStretch(1) 

        # Create a vertical layout for the anomaly plots (far right)
        anomaly_plot_layout = QVBoxLayout()

        self.vibx_anomaly_graph = QLabel(self)
        self.vibx_anomaly_graph.height= int(screen_height*(2/9))
        self.vibx_anomaly_graph.width= int(screen_width/3)
        self.vibx_anomaly_graph.setFixedSize(self.vibx_anomaly_graph.width,self.vibx_anomaly_graph.height)

        self.viby_anomaly_graph = QLabel(self)
        self.viby_anomaly_graph.height= int(screen_height*(2/9))
        self.viby_anomaly_graph.width= int(screen_width/3)
        self.viby_anomaly_graph.setFixedSize(self.viby_anomaly_graph.width,self.viby_anomaly_graph.height)

        self.vibz_anomaly_graph = QLabel(self)
        self.vibz_anomaly_graph.height= int(screen_height*(2/9))
        self.vibz_anomaly_graph.width= int(screen_width/3)
        self.vibz_anomaly_graph.setFixedSize(self.vibz_anomaly_graph.width,self.vibz_anomaly_graph.height)

        # Add the anomaly plots to the vertical layout
        anomaly_plot_layout.addWidget(self.vibx_anomaly_graph)
        anomaly_plot_layout.addWidget(self.viby_anomaly_graph)
        anomaly_plot_layout.addWidget(self.vibz_anomaly_graph)

        # Add the anomaly plot layout to the bottom row (right side)
        bottom_layout.addLayout(anomaly_plot_layout)
        bottom_layout.addStretch(1) 

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)
        # Set the central widget layout
        central_widget.setLayout(main_layout)




        
  





  

        #Set up the QTimer to update the images every 2 seconds (2000ms)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.start_camera_thread)
        self.camera_timer.start(int(1000/25))  # Update every 2000ms (2 seconds)

        self.adxl359_timer= QTimer(self)
        self.adxl359_timer.timeout.connect(self.start_sensor_thread)
        self.adxl359_timer.start(int(1000))  # Update every 2000ms (2 seconds)





        self.adxl359_lock = lock = multiprocessing.Lock()
        self.adxl359_vib_data_shape =(320,850,3,6)
        self.adxl359_vib_data_shm = shared_memory.SharedMemory(create=True,size=np.prod(self.adxl359_vib_data_shape)* np.uint8().itemsize)
        self.adxl359_temp_shm = shared_memory.SharedMemory(create=True,size=np.float16().itemsize)


        self.camera_thread = CameraThread(self)
        self.adxl359_thread = Adxl359Thread(self,self.adxl359_vib_data_shm.name,self.adxl359_temp_shm.name, self.adxl359_vib_data_shape,self.adxl359_lock)
        

        # # Connect the thread signals to slots in the main window
        self.camera_thread.camera_feed_signal.connect(self.update_camera_feed)
        self.adxl359_thread.adxl359_plot_signal.connect(self.update_adxl359_feed)


        self.sensor_processes = multiprocessing.Process(target=update_adxl359_vib_data_shm,args=(self.adxl359_vib_data_shm.name,self.adxl359_temp_shm.name, self.adxl359_vib_data_shape,self.adxl359_lock))
        self.sensor_processes.start()
        self.terminate_event = multiprocessing.Event()
        self.camera_processes = multiprocessing.Process(target=update_imx500_shm,args=(self.model,self.terminate_event))
        self.camera_processes.start()


        
 

    def apply_model(self):
        """Handle the Apply Model button action."""
        selected_model = self.model_dropdown.currentData()

        if self.model== selected_model:
            print("no change")
        else:
            self.model= selected_model
            if self.camera_processes.is_alive():
                self.terminate_event.set()
                self.camera_processes.join()
                self.terminate_event = multiprocessing.Event()
                self.camera_processes = multiprocessing.Process(target=update_imx500_shm,args=(self.model,self.terminate_event))
                self.camera_processes.start()

    def exit_application(self):
        global camera_one_shm 
        global camera_two_shm 
        global adxl359_vib_data_shm

        print(" Cleaning threads")
        if self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait()

        if self.adxl359_thread.isRunning():
            self.adxl359_thread.quit()
            self.adxl359_thread.wait()
        print(" Done Cleaning threads")

        if self.sensor_processes.is_alive():
            print("Terminating sensor process as it is still running...")
            self.sensor_processes.terminate()  # Forcefully terminate the process

        if self.camera_processes.is_alive():
            print("Terminating camera_processes as it is still running...")
            self.terminate_event.set()
            self.camera_processes.join()
            print("child is done")

        print("Cleaning mem")
        camera_one_shm.close()  # Detach from the shared memory
        camera_one_shm.unlink()  # Deallocate the shared memory
        self.adxl359_temp_shm.close()
        self.adxl359_temp_shm.unlink()

        camera_two_shm.close()  # Detach from the shared memory
        camera_two_shm.unlink()  # Deallocate the shared memory

        self.adxl359_vib_data_shm.close()  # Detach from the shared memory
        self.adxl359_vib_data_shm.unlink()  # Deallocate the shared memory
        print("Done Cleaning mem")
        

        print("exiting")
        QApplication.exit()

    def start_camera_thread(self):
        self.camera_thread.start()
        self.camera_thread.setPriority(QThread.TimeCriticalPriority)

    def start_sensor_thread(self):
        self.adxl359_thread.start()
        self.adxl359_thread.setPriority(QThread.LowPriority)

    def update_camera_feed(self,camera_data_tuple):

        camera_one, camera_two = camera_data_tuple
        self.display_image(self.camera_feed_1, camera_one)
        self.display_image(self.camera_feed_2, camera_two)

    def update_adxl359_feed(self,plot_tuple):
        vibx_graph, viby_graph,vibz_graph, temp= plot_tuple
        self.display_image(self.vibx_graph, vibx_graph)
        self.display_image(self.viby_graph, viby_graph)
        self.display_image(self.vibz_graph, vibz_graph)
        self.temperature_label.setText(f"Temperature: {temp[0]:.2f} °C")


    def display_image(self, label, image_pixmap):
        label.setPixmap(image_pixmap.scaled(label.width,label.height))








# Main function to start the application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
