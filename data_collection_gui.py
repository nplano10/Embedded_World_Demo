
import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget , QPushButton ,QLineEdit


from picamera2 import Picamera2
import adxl359
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import cv2
import copy
import matplotlib.pyplot as plt
import time
from PIL import Image
import os

from datetime import datetime




class RandomImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.picam2_0 = Picamera2(0)
        self.picam2_0.start()

        self.picam2_1 = Picamera2(1)
        self.picam2_1.start()

        self.setWindowTitle("Random Image Viewer")
        self.setGeometry(100, 100, 1200, 1200)

        # Set up the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create the layout for the central widget
        layout = QVBoxLayout()

        # Create labels for displaying images
        self.image_label1 = QLabel(self)
        self.image_label2 = QLabel(self)

        # Add the labels to the layout
        layout.addWidget(self.image_label1)
        layout.addWidget(self.image_label2)

        # Set the layout of the central widget
        central_widget.setLayout(layout)

        # Create the button
        self.button = QPushButton('Collect Single Frame', self)
        self.button.clicked.connect(self.collect_frame)

        # Resize and move the button
        self.button.resize(150, 75)  # Resize the button
        self.button.move(1000, 400)  # Move the button to the position (800, 50)
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("label")  # Optional placeholder text
        self.text_input.resize(300, 30)  # Resize the button
        self.text_input.move(800, 510)  # Move the button to the position (800, 50)
        self.label = "unlabeled"

        self.start_collection_button = QPushButton('Start Collection', self)
        self.start_collection_button.clicked.connect(self.set_collect_flag)
        self.start_collection_button.resize(150, 75)  # Resize the button
        self.start_collection_button.move(700, 400)  # Move the button to the position (800, 50)

        self.stop_collection_button = QPushButton('Stop Collection', self)
        self.stop_collection_button.clicked.connect(self.unset_collect_flag)
        self.stop_collection_button.resize(150, 75)  # Resize the button
        self.stop_collection_button.move(850, 400)  # Move the button to the position (800, 50)

        self.collect_flag =False



        self.sample_frequency_label = QLineEdit(self)
        self.sample_frequency_label.setPlaceholderText("Collection sample rate in fps")  # Optional placeholder text
        self.sample_frequency_label.resize(300, 30)  # Resize the button
        self.sample_frequency_label.move(800, 480)  # Move the button to the position (800, 50)
        self.save_sample_rate=1
        self.time_start =datetime.now().timestamp()
        self.time_stop =datetime.now().timestamp()


        self.set_inputs_buttons = QPushButton('Set Inputs', self)
        self.set_inputs_buttons.clicked.connect(self.set_inputs)
        self.set_inputs_buttons.resize(150, 30)  # Resize the button
        self.set_inputs_buttons.move(650, 495)  # Move the button to the position (800, 50)

        # layout.addWidget(self.text_input)

        # Set up the QTimer to update the images every 2 seconds (2000ms)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(int(1000/60))  # Update every 2000ms (2 seconds)
        # Initial image update
        self.update_camera_feed()

    def set_inputs(self):
        
        if self.sample_frequency_label.text().isdigit():
            self.save_sample_rate = int(self.sample_frequency_label.text())
        if(self.text_input.text()!=""):
            self.label = self.text_input.text()

    def set_collect_flag(self):
        self.collect_flag =True
    def unset_collect_flag(self):
        self.collect_flag =False
    def get_file_name(self,dir,prefix):
        index = 0 
        while(True):
            test_string =prefix+"_"+"{:04}".format(index)+".png"
            path= dir+"/"+test_string
            index = index+1
            if os.path.exists(path) ==False:
                return path 



    def collect_frame(self):
        # Update two random images on the first tab
        path = "data/"+self.label

        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(path):
            os.mkdir(path)
        image_one =self.picam2_0.capture_array().astype(np.uint8)
        image_two = self.picam2_1.capture_array().astype(np.uint8)

        image_one_rgba = Image.fromarray(image_one)
        image_one_rgba.save(self.get_file_name(path,self.label), "PNG")

        image_two_rgba = Image.fromarray(image_two)
        image_two_rgba.save(self.get_file_name(path,self.label), "PNG")

        time.sleep(1)

    def save_frame(self,frame1,frame2):
        # Update two random images on the first tab
        path = "data/"+self.label

        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(path):
            os.mkdir(path)

        image_one_rgba = Image.fromarray(frame1)
        image_one_rgba.save(self.get_file_name(path,self.label), "PNG")

        image_two_rgba = Image.fromarray(frame2)
        image_two_rgba.save(self.get_file_name(path,self.label), "PNG")
    

    def update_camera_feed(self):
        # Update two random images on the first tab
        camera_one,camera_two = self.capture_camera_data()
        self.display_image(self.image_label1, camera_one)
        self.display_image(self.image_label2, camera_two)

    def capture_camera_data(self):

        frame1  = self.picam2_0.capture_array().astype(np.uint8)
        frame2 = self.picam2_1.capture_array().astype(np.uint8)
        
        if(self.collect_flag==True):
            self.time_stop =datetime.now().timestamp()
            if(self.time_stop- self.time_start> (1/self.save_sample_rate)):
                self.time_start = self.time_stop
                self.save_frame(frame1,frame2)
        
        frame1 =frame1[:, :, :3]
        frame2 =frame2[:, :, :3]
        return frame1,frame2

    
    def display_image(self, label, image_data):
        # Convert NumPy array to QImage
        height, width, _ = image_data.shape
        q_image = QImage(image_data.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))


# Main function to start the application
def main():
    app = QApplication(sys.argv)
    window = RandomImageWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
