import numpy as np
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    # QTabWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QDesktopWidget,
    QHBoxLayout,
    # QComboBox,
    QTextEdit,
    QButtonGroup,
    QRadioButton,
    QMessageBox,
)
from multiprocessing import shared_memory
import multiprocessing
from x_adxl_process import update_adxl359_vib_data_shm
from x_imx500_process import Model, update_imx500_shm
from x_imx500_gui_thread import CameraThread
from x_adxl_gui_thread import Adxl359Thread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.imx500_height = 480
        self.imx500_width = 640 
        self.adxl359_sample_rate = 1000
        self.adxl359_sample_length = 1000

        self.model = Model.NOMODEL

        # Set up the window to match the screen size
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = int(screen.height()*0.90)
        self.setGeometry(0, 0, screen_width, screen_height)

        # Set up the central widget and the main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create the main layout (vertical)
        main_layout = QVBoxLayout()

        # Create the top row (camera feeds and logging windows)
        top_layout = self.camera_layout(screen_height, screen_width)

        # Add the top row to the main layout
        main_layout.addLayout(top_layout)

        # Create the bottom row (buttons, temperature, plots)
        bottom_layout = QHBoxLayout()

        # Create control buttons (Start, Stop)
        button_layout = self.button_layout(screen_height, screen_width)

        # Add the buttons, apply model controls, and temperature layout to the bottom-left of the layout
        bottom_layout.addLayout(button_layout)
        bottom_layout.addStretch(1)

        # Create a vertical layout for the vibration plots (3 plots on the left)
        vib_layout = self.vibration_layout(screen_height, screen_width)

        # Add the plot layout to the bottom row (left side)
        bottom_layout.addLayout(vib_layout)
        bottom_layout.addStretch(1) 

        # Create a vertical layout for the anomaly plots (far right)
        anomaly_plot_layout = self.vibration_anomaly_layout(screen_height, screen_width)

        # Add the anomaly plot layout to the bottom row (right side)
        bottom_layout.addLayout(anomaly_plot_layout)
        bottom_layout.addStretch(1) 

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)
        # Set the central widget layout
        central_widget.setLayout(main_layout)

        # Set up the QTimer to update the images at 25 fps seconds
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.start_camera_thread)
        self.camera_timer.start(int(1000/25))  # Update 25 times every 1000ms

        self.adxl359_timer= QTimer(self)
        self.adxl359_timer.timeout.connect(self.start_sensor_thread)
        self.adxl359_timer.start(int(1000))  # Update every 1000ms (1 second)

        self.adxl359_lock  = multiprocessing.Lock()
        self.adxl359_vib_data_shape = (180,320,3,4)
        self.adxl359_vib_data_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.adxl359_vib_data_shape) * np.uint8().itemsize
        )
        self.adxl359_temp_shm = shared_memory.SharedMemory(create=True,size=np.float16().itemsize)

        self.adxl359_thread = Adxl359Thread(
            self,
            self.adxl359_vib_data_shm.name,
            self.adxl359_temp_shm.name,
            self.adxl359_vib_data_shape,
            self.adxl359_lock,
        )
        self.adxl359_thread.adxl359_plot_signal.connect(self.update_adxl359_feed)
        self.sensor_processes = multiprocessing.Process(
            target=update_adxl359_vib_data_shm,
            args=(
                self.adxl359_vib_data_shm.name,
                self.adxl359_temp_shm.name,
                self.adxl359_vib_data_shape,
                self.adxl359_lock,
            ),
        )
        self.sensor_processes.start()

        self.camera_one_lock = multiprocessing.Lock()
        self.camera_two_lock = multiprocessing.Lock()
        self.camera_shape = (480, 640, 3)
        self.camera_one_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.camera_shape) * np.uint8().itemsize)
        self.camera_two_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.camera_shape) * np.uint8().itemsize)
        self.camera_thread = CameraThread(
            self,
            self.camera_one_shm.name,
            self.camera_two_shm.name,
            self.camera_shape,
            self.camera_one_lock,
            self.camera_two_lock,
        )

        # # Connect the thread signals to slots in the main window
        self.camera_thread.camera_feed_signal.connect(self.update_camera_feed)
        self.terminate_event = multiprocessing.Event()
        self.camera_processes = multiprocessing.Process(
            target=update_imx500_shm,
            args=(
                self.model,
                self.terminate_event,
                self.camera_one_shm.name,
                self.camera_two_shm.name,
                self.camera_shape,
                self.camera_one_lock,
                self.camera_two_lock,
            ),
        )
        self.camera_processes.start()

    def closeEvent(self, event):
        # This function will be triggered when the window is closed (clicked on "X")
        reply = QMessageBox.question(self, 'Confirm Exit', 
                                     'Are you sure you want to exit?', 
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.exit_application()
            event.accept()  # Accept the event, allowing the window to close
        else:
            event.ignore()  # Ignore the event, preventing the window from closing

    def button_layout(self, screen_height, screen_width):
        height_buttons = int(screen_height * (1 / 32))
        width_buttons = int(screen_width / 16)

        # button_exit = QPushButton("Exit", self)
        # button_exit.clicked.connect(self.exit_application)
        # button_exit.setFixedSize(width_buttons, height_buttons)

        button_start = QPushButton("Start", self)
        button_start.setFixedSize(width_buttons, height_buttons)

        button_stop = QPushButton("Stop", self)
        button_stop.setFixedSize(width_buttons, height_buttons)

        button_dispense = QPushButton("Dispense", self)
        button_dispense.setFixedSize(width_buttons, height_buttons)

        # Create button layout and add buttons to it
        button_layout = QVBoxLayout()
        # button_layout.addWidget(button_exit)
        button_layout.addWidget(button_start)
        button_layout.addWidget(button_stop)
        button_layout.addWidget(button_dispense)

        # Create Apply Model button and dropdown
        apply_model_button = QPushButton("Apply Model", self)
        apply_model_button.clicked.connect(
            self.apply_model
        )  # Connect to a function for applying model
        apply_model_button.setFixedSize(width_buttons, height_buttons)
        apply_model_button.setStyleSheet(
            "background-color: lightblue; border: 1px solid lightgray;"
        )

        # # Create dropdown for model selection
        # self.model_dropdown = QComboBox(self)
        # self.model_dropdown.setFixedSize(width_buttons,height_buttons)

        # for model in Model:
        #     self.model_dropdown.addItem(model.name, model)

        # # Create a horizontal layout for Apply Model button and dropdown
        # apply_model_layout = QHBoxLayout()
        # apply_model_layout.addWidget(apply_model_button)
        # apply_model_layout.addWidget(self.model_dropdown)

        # Create radio buttons for model selection
        self.model_selection = QButtonGroup()
        models = list(Model)
        self.radio_button1 = QRadioButton(models[0].name)
        self.radio_button2 = QRadioButton(models[1].name)
        if models[0] == self.model:
            self.radio_button1.setChecked(True)
        else:
            self.radio_button2.setChecked(True)

        apply_model_layout = QVBoxLayout()
        apply_model_layout.addWidget(self.radio_button1)
        apply_model_layout.addWidget(self.radio_button2)
        self.model_selection.addButton(self.radio_button1)
        self.model_selection.addButton(self.radio_button2)
        apply_model_layout.addWidget(apply_model_button)

        button_layout.addLayout(apply_model_layout)

        # # Create temperature label
        # self.temperature_label = QLabel(self)
        # self.temperature_label.setStyleSheet("font-size: 18px;")

        # # Add buttons, apply model controls, and temperature label to the bottom-left layout
        # button_layout.addWidget(self.temperature_label)

        return button_layout

    def camera_layout(self, screen_height, screen_width):

        cam_layout = QHBoxLayout()
        scale = 1.45
        cam_height = int(screen_height * scale / 3)
        cam_width = int(screen_width * scale / 4)
        log_height = cam_height
        log_width = int(screen_width / 8)

        # Camera feed 1 and its logging window
        cam_1_layout = QHBoxLayout()
        self.camera_feed_1 = self.camera_feed(
            cam_height, cam_width, title="Camera Feed 1"
        )
        self.log_1 = self.camera_log(
            log_height, log_width, title="Camera 1 Logging"
        )

        cam_1_layout.addWidget(self.log_1)
        cam_1_layout.addWidget(self.camera_feed_1)
        cam_layout.addLayout(cam_1_layout)

        # Camera feed 2 and its logging window
        cam_2_layout = QHBoxLayout()
        self.camera_feed_2 = self.camera_feed(
            cam_height, cam_width, title="Camera Feed 2"
        )
        self.log_2 = self.camera_log(
            log_height, log_width, title="Camera 2 Logging"
        )

        cam_2_layout.addWidget(self.camera_feed_2)
        cam_2_layout.addWidget(self.log_2)
        cam_layout.addLayout(cam_2_layout)

        return cam_layout

    def camera_feed(self, cam_height, cam_width, title="Camera Feed"):
        camera_feed = QLabel(self)
        camera_feed.height = cam_height
        camera_feed.width = cam_width
        camera_feed.setFixedSize(camera_feed.width,camera_feed.height)
        camera_feed.setText(title)  # Placeholder text
        camera_feed.setStyleSheet("background-color: lightgray;")
        return camera_feed

    def camera_log(self, log_height, log_width, title="Camera logging window"):
        camera_log = QTextEdit(self)
        camera_log.setPlaceholderText(title)
        camera_log.setReadOnly(True)
        camera_log.height = log_height
        camera_log.width = log_width
        camera_log.setFixedSize(camera_log.width, camera_log.height)
        camera_log.setStyleSheet("background-color: black; color: white;")
        return camera_log

    def vibration_layout(self, screen_height, screen_width):
        vib_layout = QHBoxLayout()

        vib_height, vib_width = int(screen_height * (2 / 9)), int(screen_width / 4.5)
        self.vibx_graph = self.vibration_graph(vib_height, vib_width)
        self.viby_graph = self.vibration_graph(vib_height, vib_width)
        self.vibz_graph = self.vibration_graph(vib_height, vib_width)

        # Update the plot data
        # Add the plots to the vertical layout
        vib_layout.addWidget(self.vibx_graph)
        vib_layout.addWidget(self.viby_graph)
        vib_layout.addWidget(self.vibz_graph)
        return vib_layout

    def vibration_graph(self, vib_height, vib_width):
        vib_graph = QLabel(self)
        vib_graph.height = vib_height
        vib_graph.width = vib_width
        vib_graph.setFixedSize(vib_graph.width, vib_graph.height)
        return vib_graph

    def vibration_anomaly_layout(self, screen_height, screen_width):
        # Create a vertical layout for the anomaly plots (far right)
        anomaly_plot_layout = QVBoxLayout()

        vib_height, vib_width = int(screen_height * (2 / 9)), int(screen_width / 4.5)
        # print(f"Vib graph height: {vib_height}")
        # print(f"Vib graph width: {vib_width}")
        self.vibx_anomaly_graph = self.vibration_graph(vib_height, vib_width)

        # Add the anomaly plots to the vertical layout
        anomaly_plot_layout.addWidget(self.vibx_anomaly_graph)

        return anomaly_plot_layout

    def apply_model(self):
        """Handle the Apply Model button action."""
        # selected_model = self.model_dropdown.currentData()
        if self.radio_button1.isChecked():
            selected_model = Model[self.radio_button1.text()]
        else:
            selected_model = Model[self.radio_button2.text()]

        if self.model== selected_model:
            print("no change")
        else:
            self.model = selected_model
            if self.camera_processes.is_alive():
                self.terminate_event.set()
                self.camera_processes.join()
                self.terminate_event = multiprocessing.Event()
                self.camera_processes = multiprocessing.Process(
                    target=update_imx500_shm,
                    args=(
                        self.model,
                        self.terminate_event,
                        self.camera_one_shm.name,
                        self.camera_two_shm.name,
                        self.camera_shape,
                        self.camera_one_lock,
                        self.camera_two_lock,
                    ),
                )
            self.camera_processes.start()

    def exit_application(self):

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
        self.camera_one_shm.close()  # Detach from the shared memory
        self.camera_one_shm.unlink()  # Deallocate the shared memory
        self.adxl359_temp_shm.close()
        self.adxl359_temp_shm.unlink()

        self.camera_two_shm.close()  # Detach from the shared memory
        self.camera_two_shm.unlink()  # Deallocate the shared memory

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
        vibx_graph, viby_graph, vibz_graph, vib_anom_graph = plot_tuple
        self.display_image(self.vibx_graph, vibx_graph)
        self.display_image(self.viby_graph, viby_graph)
        self.display_image(self.vibz_graph, vibz_graph)
        self.display_image(self.vibx_anomaly_graph, vib_anom_graph)
        # self.temperature_label.setText(f"Temperature: {temp[0]:.2f} °C")

    def display_image(self, label, image_pixmap):
        label.setPixmap(image_pixmap.scaled(label.width,label.height))
