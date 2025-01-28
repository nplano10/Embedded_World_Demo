
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
        self.button = QPushButton('Collect Frame', self)
        self.button.clicked.connect(self.collect_frame)

        # Resize and move the button
        self.button.resize(150, 75)  # Resize the button
        self.button.move(800, 50)  # Move the button to the position (800, 50)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("label")  # Optional placeholder text
        self.text_input.resize(150, 30)  # Resize the button
        self.text_input.move(800, 130)  # Move the button to the position (800, 50)

        # layout.addWidget(self.text_input)

        # Set up the QTimer to update the images every 2 seconds (2000ms)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(int(1000/60))  # Update every 2000ms (2 seconds)
        # Initial image update
        self.update_camera_feed()
    
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
        prefix = self.text_input.text()
        if prefix=="":
            prefix = "unlabeled"
        path = "data/"+prefix

        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(path):
            os.mkdir(path)
        image_one =self.picam2_0.capture_array().astype(np.uint8)
        image_two = self.picam2_1.capture_array().astype(np.uint8)

        image_one_rgba = Image.fromarray(image_one)
        image_one_rgba.save(self.get_file_name(path,prefix), "PNG")

        image_two_rgba = Image.fromarray(image_two)
        image_two_rgba.save(self.get_file_name(path,prefix), "PNG")



        time.sleep(1)
    

    def update_camera_feed(self):
        # Update two random images on the first tab
        camera_one = self.cpature_camera_one_data()
        camera_two = self.cpature_camera_two_data()
        self.display_image(self.image_label1, camera_one)
        self.display_image(self.image_label2, camera_two)

    def cpature_camera_one_data(self):
        # Create a random image (height=240, width=320, RGB)
        frame  = self.picam2_0.capture_array().astype(np.uint8)
        frame =frame[:, :, :3]
        return frame

    def cpature_camera_two_data(self):
        # Create a random image (height=240, width=320, RGB)
        frame  = self.picam2_1.capture_array().astype(np.uint8)
        frame =frame[:, :, :3]
        return frame
    
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
