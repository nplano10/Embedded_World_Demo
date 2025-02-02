
import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QDesktopWidget,QHBoxLayout,QComboBox,QTextEdit

from multiprocessing import shared_memory
from picamera2 import Picamera2
import adxl359
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import cv2
import copy
import matplotlib.pyplot as plt
import multiprocessing
import time
import pyqtgraph as pg

camera_one_lock = multiprocessing.Lock()
camera_two_lock = multiprocessing.Lock()
width,height, channels = 480, 640, 3
camera_shape = (width,height, channels)
camera_one_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)
camera_two_shm = shared_memory.SharedMemory(create=True, size=np.prod(camera_shape) * np.uint8().itemsize)



adxl359_data_lock = lock = multiprocessing.Lock()
data_length=1000
adxl359_data_shape =(data_length,4)
adxl359_shm = shared_memory.SharedMemory(create=True,size=np.prod(adxl359_data_shape)* np.float16().itemsize)





# adxl359 = adxl359.ADXL359()  # Adjust according to your actual initialization code
# adxl359._initialize()

def update_adxl359_shm():
    # Extract accelerometer data (replace with actual sensor data fetching methods)
    # global fig, axs
    global adxl359_data_lock
    global adxl359_shm
    global adxl359_data_shape
    existing_shm = shared_memory.SharedMemory(name=adxl359_shm.name)
    # Create a NumPy array from the shared memory buffer
    shared_adxl359_data = np.ndarray(adxl359_data_shape, dtype=np.float16, buffer=existing_shm.buf)
    while True:
        #x_data,y_data,z_data,temp_data = adxl359.collect_data() # Example method from adxl359 object
        time.sleep(1)
        x_data = np.random.rand(1000).astype(np.float16)
        y_data = np.random.rand(1000).astype(np.float16)
        z_data = np.random.rand(1000).astype(np.float16)
        temp = np.random.rand(1000).astype(np.float16)
        with adxl359_data_lock:                                        
            shared_adxl359_data[:] =np.column_stack((x_data, y_data, z_data, temp))     


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

    picam2_0 = Picamera2(0)
    picam2_0.start()

    picam2_1 = Picamera2(1)
    picam2_1.start()

    # Attach to the shared memory block
    one_shm = shared_memory.SharedMemory(name=camera_one_shm.name)
    one_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=one_shm.buf)
    two_shm = shared_memory.SharedMemory(name=camera_two_shm.name)
    two_image = np.ndarray(camera_shape, dtype=np.uint8, buffer=two_shm.buf)
    while True:
        time.sleep(1/60)
        with camera_one_lock:  # Ensure exclusive access to the shared memory
            one_image[:] = capture_camera_data(picam2_0)
            #one_image[:] = np.random.randint(0, 255, camera_shape,dtype=np.uint8)
        with camera_two_lock:
            two_image[:] =capture_camera_data(picam2_1)
            #two_image[:] = np.random.randint(0, 255, camera_shape,dtype=np.uint8)

class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_one = None
        self.previous_camera_two = None

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
            self.camera_feed_signal.emit((camera_one, camera_two))

class Adxl359Thread(QThread):
    # Define a signal to send data to the main thread
    adxl359_plot_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)  # Call the parent constructor
        # Initialize any other variables as needed
        self.previous_data = None

    def run(self):
        # Update data from shared memory
 
        global adxl359_data_lock
        global adxl359_shm
        global adxl359_data_shape

        with adxl359_data_lock:
            existing_shm = shared_memory.SharedMemory(name=adxl359_shm.name)
            # Create a NumPy array from the shared memory buffer
            data = np.ndarray(adxl359_data_shape, dtype=np.float16, buffer=existing_shm.buf)
            ret = copy.deepcopy(data)

        # Emit the signal only if the data has changed
        if self.previous_data is None or not np.array_equal(ret, self.previous_data):
            self.previous_data = ret.copy()  # Update with the new data
            self.adxl359_plot_signal.emit(ret)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the window to match the screen size
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        self.setGeometry(0, 0, screen_width, screen_height)

        # Set up the central widget and the main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create the main layout (vertical)
        main_layout = QVBoxLayout()

        # Create the top row (camera feeds and logging windows)
        top_layout = QHBoxLayout()

        # Camera feed 1 and its logging window
        camera_feed_1_layout = QHBoxLayout()
        self.camera_feed_1 = QLabel(self)
        self.camera_feed_1.setText("Camera Feed 1")  # Placeholder text
        self.camera_feed_1.setStyleSheet("background-color: lightgray;")
        self.camera_feed_1.resize(640, 480)
        camera_feed_1_layout.addWidget(self.camera_feed_1)

        # Logging window for camera feed 1
        self.log_1 = QTextEdit(self)
        self.log_1.setPlaceholderText("Logging window for Camera Feed 1...")
        self.log_1.setReadOnly(True)
        self.log_1.setStyleSheet("background-color: black; color: white;")
        camera_feed_1_layout.addWidget(self.log_1)
        top_layout.addLayout(camera_feed_1_layout)

        # Camera feed 2 and its logging window
        camera_feed_2_layout = QHBoxLayout()
        self.camera_feed_2 = QLabel(self)
        self.camera_feed_2.setText("Camera Feed 2")  # Placeholder text
        self.camera_feed_2.setStyleSheet("background-color: lightgray;")
        self.camera_feed_2.resize(640, 480)
        camera_feed_2_layout.addWidget(self.camera_feed_2)

        # Logging window for camera feed 2
        self.log_2 = QTextEdit(self)
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
        button_exit = QPushButton('Exit', self)
        button_exit.clicked.connect(self.exit_application)
        
        button_start = QPushButton('Start', self)
        button_stop = QPushButton('Stop', self)

        # Create button layout and add buttons to it
        button_layout = QVBoxLayout()
        button_layout.addWidget(button_exit)
        button_layout.addWidget(button_start)
        button_layout.addWidget(button_stop)

        # Create Apply Model button and dropdown
        apply_model_button = QPushButton('Apply Model', self)
        apply_model_button.clicked.connect(self.apply_model)  # Connect to a function for applying model
        
        # Create dropdown for model selection
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.addItem("Model 1")
        self.model_dropdown.addItem("Model 2")
        
        self.model_dropdown.addItem("No Model")

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

        # Create a vertical layout for the vibration plots (3 plots on the left)
        plot_layout = QVBoxLayout()

        # Create the three vibration plots
        self.plot1 = pg.PlotWidget(title="Vibration X Axis")
        self.plot2 = pg.PlotWidget(title="Vibration Y Axis")
        self.plot3 = pg.PlotWidget(title="Vibration Z Axis")

        self.plot1_item = self.plot1.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='b')
        self.plot2_item =self.plot2.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='g')
        self.plot3_item =self.plot3.plot(np.linspace(0, 1000, 1000),np.zeros(1000).astype(np.float16), pen='r')

        # Resize the plots
        plot_width = screen_width // 2  # Half the width of the screen
        plot_height = plot_width * 2 // 3  # Aspect ratio of 3:2 (height:width)
        self.plot1.resize(plot_width, plot_height)
        self.plot2.resize(plot_width, plot_height)
        self.plot3.resize(plot_width, plot_height)

        # Update the plot data
        # Add the plots to the vertical layout
        plot_layout.addWidget(self.plot1)
        plot_layout.addWidget(self.plot2)
        plot_layout.addWidget(self.plot3)

        # Add the plot layout to the bottom row (left side)
        bottom_layout.addLayout(plot_layout)

        # Create a vertical layout for the anomaly plots (far right)
        anomaly_plot_layout = QVBoxLayout()

        # Create the three anomaly plots
        self.anomaly_score_plot1 = pg.PlotWidget(title="Anomaly Plot 1")
        self.anomaly_score_plot2 = pg.PlotWidget(title="Anomaly Plot 2")
        self.anomaly_score_plot3 = pg.PlotWidget(title="Anomaly Plot 3")

        # Resize the anomaly plots
        self.anomaly_score_plot1.resize(plot_width, plot_height)
        self.anomaly_score_plot2.resize(plot_width, plot_height)
        self.anomaly_score_plot3.resize(plot_width, plot_height)

        # Add the anomaly plots to the vertical layout
        anomaly_plot_layout.addWidget(self.anomaly_score_plot1)
        anomaly_plot_layout.addWidget(self.anomaly_score_plot2)
        anomaly_plot_layout.addWidget(self.anomaly_score_plot3)

        # Add the anomaly plot layout to the bottom row (right side)
        bottom_layout.addLayout(anomaly_plot_layout)

        self.vibx_anomaly_scores = np.zeros(20)
        self.viby_anomaly_scores = np.zeros(20)
        self.vibz_anomaly_scores = np.zeros(20)
        self.vibx = None
        self.viby = None
        self.vibz = None
        self.temp = None

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)

        # Set the central widget layout
        central_widget.setLayout(main_layout)
  

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
        self.camera_thread.camera_feed_signal.connect(self.update_camera_feed)
        self.adxl359_thread.adxl359_plot_signal.connect(self.update_adxl359_feed)


        self.sensor_processes = multiprocessing.Process(target=update_adxl359_shm)
        self.sensor_processes.start()

        self.camera_processes = multiprocessing.Process(target=update_image)
        self.camera_processes.start()
 

    def apply_model(self):
        """Handle the Apply Model button action."""
        selected_model = self.model_dropdown.currentText()
        print(f"Applied {selected_model}")

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

        adxl359_shm.close()  # Detach from the shared memory
        adxl359_shm.unlink()  # Deallocate the shared memory
        print("Done Cleaning mem")
        

        print("exiting")
        QApplication.exit()

    def start_camera_thread(self):
        self.camera_thread.start()
        

    def start_sensor_thread(self):
        self.adxl359_thread.start()

    def update_camera_feed(self,camera_data_tuple):

        camera_one, camera_two = camera_data_tuple
        self.display_image(self.camera_feed_1, camera_one)
        self.display_image(self.camera_feed_2, camera_two)

    def get_anomaly_scores(self,vibx,viby,vibz):
        vibx_score= np.random.rand(1)[0]
        viby_score= np.random.rand(1)[0]
        vibz_score= np.random.rand(1)[0]
        return vibx_score,viby_score,vibz_score
    
        
    def update_anomaly_score_arrays(self,vibx_data,viby_data,vibz_data):
        vibx_score,viby_score,vibz_score =self.get_anomaly_scores(vibx_data,viby_data,vibz_data)

        
        self.vibx_anomaly_scores = np.roll(self.vibx_anomaly_scores, 1)  # Shift elements to the right by 1
        self.vibx_anomaly_scores[0] = vibx_score

        self.viby_anomaly_scores = np.roll(self.viby_anomaly_scores, 1)  # Shift elements to the right by 1
        self.viby_anomaly_scores[0] = vibx_score

        self.vibz_anomaly_scores = np.roll(self.vibz_anomaly_scores, 1)  # Shift elements to the right by 1
        self.vibz_anomaly_scores[0] = vibx_score

    def update_adxl359_feed(self,array):
    
        self.vibx = array[:, 0]
        self.viby = array[:, 1]
        self.vibz = array[:, 2]
        self.temp = array[0, 3]
        print(self.temp)


        self.plot1_item.setData(np.linspace(0, 1000, 1000).tolist(),self.vibx)
        self.plot2_item.setData(np.linspace(0, 1000, 1000).tolist(),self.viby)
        self.plot3_item.setData(np.linspace(0, 1000, 1000).tolist(),self.vibz)
        self.temperature_label.setText(f"Temperature: {self.temp:.2f} °C")

        # self.anomaly_score_plot1.clear()
        # self.anomaly_score_plot2.clear()
        # self.anomaly_score_plot3.clear()

        # self.update_anomaly_score_arrays(array[:,0],array[:,1],array[:,2])

        # index = np.arange(20)
        # self.anomaly_score_plot1.setData(index, self.vibx_anomaly_scores, pen='orange', name="Anomaly Score 1")
        # self.anomaly_score_plot2.setData(index, self.viby_anomaly_scores, pen='purple', name="Anomaly Score 2")
        # self.anomaly_score_plot3.setData(index, self.vibz_anomaly_scores, pen='pink', name="Anomaly Score 3")
        # # Define anomaly thresholds
        # threshold = 0.8
        # self.anomaly_score_plot1.scatterPlot(index[self.vibx_anomaly_scores > threshold], 
        #                                 self.vibx_anomaly_scores[self.vibx_anomaly_scores >threshold], 
        #                                 pen=None, symbol='o', symbolBrush='r', symbolSize=6)
        # self.anomaly_score_plot2.scatterPlot(index[self.viby_anomaly_scores > threshold], 
        #                                 self.viby_anomaly_scores[self.viby_anomaly_scores > threshold], 
        #                                 pen=None, symbol='o', symbolBrush='r', symbolSize=6)
        # self.anomaly_score_plot3.scatterPlot(index[self.vibz_anomaly_scores > threshold], 
        #                                 self.vibz_anomaly_scores[self.vibz_anomaly_scores > threshold], 
        #                                 pen=None, symbol='o', symbolBrush='r', symbolSize=6)
        

    def display_image(self, label, image_data):
        # Convert NumPy array to QImage
        height, width, _ = image_data.shape
        q_image = QImage(image_data.tobytes(), width, height, 3 * width, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        label.setPixmap(pixmap)



# Main function to start the application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
