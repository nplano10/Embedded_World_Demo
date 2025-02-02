
import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton


from multiprocessing import shared_memory
#from picamera2 import Picamera2
import adxl359
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import cv2
import copy
import matplotlib.pyplot as plt
import multiprocessing
import time

camera_one_lock = multiprocessing.Lock()
camera_two_lock = multiprocessing.Lock()
height, width, channels = 480, 640, 3
camera_shape = (height, width, channels)
camera_one_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)
camera_two_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)



fig_lock = lock = multiprocessing.Lock()
fig_height, fig_width = 800, 1000
fig_shape = (fig_height, fig_width, channels)
fig_shm = shared_memory.SharedMemory(create=True, size=np.prod(fig_shape) * np.uint8().itemsize)





# adxl359 = adxl359.ADXL359()  # Adjust according to your actual initialization code
# adxl359._initialize()


def figure_to_numpy(fig):
    # Create a BytesIO buffer to store the image
    buf = BytesIO()
    
    # Render the figure to the buffer (as a PNG image)
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)  # Render the figure into the buffer as PNG
    
    # Move to the beginning of the buffer
    buf.seek(0)
    
    # Convert the buffer to a NumPy array
    img = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    
    # Decode the NumPy array into an image using OpenCV
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    return img


def update_fig():
    # Extract accelerometer data (replace with actual sensor data fetching methods)
    # global fig, axs

    global fig_lock
    global fig_shm
    global fig_shape
    existing_shm = shared_memory.SharedMemory(name=fig_shm.name)
    # Create a NumPy array from the shared memory buffer
    shared_image = np.ndarray(fig_shape, dtype=np.uint8, buffer=existing_shm.buf)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    while True:
        #x_data,y_data,z_data,temp_data = adxl359.collect_data() # Example method from adxl359 object
        x_data = np.random.rand(100)
        y_data = np.random.rand(100)
        z_data = np.random.rand(100)
        temp_data = np.random.rand(100)    
        for ax in axs.flatten():
            ax.clear()
        axs[0, 0].plot(x_data)
        axs[0, 1].plot(y_data)
        axs[1, 0].plot(z_data)
        axs[1, 1].plot(temp_data)      
        with fig_lock:                                                                                 
            shared_image[:] = figure_to_numpy(fig)

def capture_camera_data(camera):
    # Create a random image (height=240, width=320, RGB)
    frame  = camera.capture_array().astype(np.uint8)
    frame =frame[:, :, :3]
    return frame



def update_image():
    global camera_one_lock
    global camera_one_shm
    global camera_two_lock
    global camera_two_shm
    global camera_shape 

    # picam2_0 = Picamera2(0)
    # picam2_0.start()

    # picam2_1 = Picamera2(1)
    # picam2_1.start()

    # Attach to the shared memory block
    one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
    one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
    two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
    two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)
    while True:
        time.sleep(1/60)
        with camera_one_lock:  # Ensure exclusive access to the shared memory
            # one_image[:] = capture_camera_data(picam2_0)
            one_image[:] = np.random.randint(0, 255, camera_shape,dtype=np.uint8)
        with camera_two_lock:
            # two_image[:] =capture_camera_data(picam2_1)
            two_image[:] = np.random.randint(0, 255, camera_shape,dtype=np.uint8)



class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def run(self):
        global camera_one_lock
        global camera_one_shm
        global camera_two_lock
        global camera_two_shm
        global camera_shape 
        # Attach to the shared memory
        with camera_one_lock:
            one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
            # Create a NumPy array from the shared memory buffer
            one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
            camera_one = copy.deepcopy(one_image)

        with camera_two_lock:
            two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
            # Create a NumPy array from the shared memory buffer
            two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)
            camera_two = copy.deepcopy(two_image)
        
        self.camera_feed_signal.emit((camera_one,camera_two))

class Adxl359Thread(QThread):
    # Define a signal to send data to the main thread
    adxl359_plot_signal = pyqtSignal(np.ndarray)
    def run(self):
        # Update one random image on the second tab
        global fig_lock
        global fig_shm
        global fig_shape 

        with fig_lock:
            existing_shm = shared_memory.SharedMemory(name=fig_shm.name)
            # Create a NumPy array from the shared memory buffer
            figure = np.ndarray(fig_shape, dtype=np.uint8, buffer=existing_shm.buf)
            ret = copy.deepcopy(figure)
        self.adxl359_plot_signal.emit(ret)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Random Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Set up the TabWidget
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Create the first tab for displaying two random images
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.image_label1 = QLabel(self)
        self.image_label2 = QLabel(self)
        self.tab1_layout.addWidget(self.image_label1)
        self.tab1_layout.addWidget(self.image_label2)
        self.tab1.setLayout(self.tab1_layout)
        self.tabs.addTab(self.tab1, "Tab 1 - Two Images")

        button = QPushButton('Exit', self.tab1)
        button.clicked.connect(self.exit_application)
        button.resize(100, 50)
        button.move(800,50)

        # Create the second tab for displaying one random image
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        self.image_label3 = QLabel(self)
        self.tab2_layout.addWidget(self.image_label3)
        self.tab2.setLayout(self.tab2_layout)
        self.tabs.addTab(self.tab2, "Tab 2 - One Image")

        # # Set up the QTimer to update the images every 2 seconds (2000ms)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.start_camera_thread)
        self.camera_timer.start(int(1000/60))  # Update every 2000ms (2 seconds)


        self.adxl359_timer= QTimer(self)
        self.adxl359_timer.timeout.connect(self.start_sensor_thread)
        self.adxl359_timer.start(int(1000/60))  # Update every 2000ms (2 seconds)

        self.camera_thread = CameraThread(self)
        self.adxl359_thread = Adxl359Thread(self)

        # # Connect the thread signals to slots in the main window
        self.camera_thread.camera_feed_signal.connect(self.update_camera_label)
        self.adxl359_thread.adxl359_plot_signal.connect(self.update_adxl359_label)

        self.sensor_processes = multiprocessing.Process(target=update_fig)
        self.sensor_processes.start()

        self.camera_processes = multiprocessing.Process(target=update_image)
        self.camera_processes.start()
 

        # self.update_adxl359_plots()

    def exit_application(self):
        global camera_one_shm 
        global camera_two_shm 
        global fig_shm

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
            self.camera_processes.terminate()  # Forcefully terminate the process

        print("Cleaning mem")
        camera_one_shm.close()  # Detach from the shared memory
        camera_one_shm.unlink()  # Deallocate the shared memory

        camera_two_shm.close()  # Detach from the shared memory
        camera_two_shm.unlink()  # Deallocate the shared memory

        fig_shm.close()  # Detach from the shared memory
        fig_shm.unlink()  # Deallocate the shared memory
        print("Done Cleaning mem")
        
 

        print("exiting")
        QApplication.exit()

    def start_camera_thread(self):
        self.camera_thread.start()

    def start_sensor_thread(self):
        self.adxl359_thread.start()

    def update_camera_label(self,camera_data_tuple):
        camera_one, camera_two = camera_data_tuple
        self.display_image(self.image_label1, camera_one)
        self.display_image(self.image_label2, camera_two)

    def update_adxl359_label(self,plot):
        self.display_image(self.image_label3, plot)    

    def display_image(self, label, image_data):
        # Convert NumPy array to QImage
        height, width, _ = image_data.shape
        q_image = QImage(image_data.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))


# Main function to start the application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
