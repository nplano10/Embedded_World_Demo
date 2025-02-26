import numpy as np
from PyQt5.QtCore import QTimer, QThread, Qt
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
    # QButtonGroup,
    QRadioButton,
    QMessageBox,
    QFrame,
)
from multiprocessing import shared_memory
import multiprocessing
from x_adxl_process import update_adxl359_vib_data_shm
from x_imx500_process import Model, update_imx500_shm
from x_imx500_gui_thread import CameraThread, CameraShm
from x_adxl_gui_thread import Adxl359Thread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camera_shape = (480, 640, 3)
        self.anom_log_shape = (100, 3)
        self.det_log_shape = (100, 3)
        self.adxl359_sample_rate = 1000
        self.adxl359_sample_length = 1000

        self.model = Model.NOMODEL

        # Set up the window to match the screen size
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = int(screen.height()*0.90)
        self.setGeometry(0, 0, screen_width, screen_height)

        # ============================
        # Set up the central widget and the main layout
        # ============================

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: black;")
        self.setCentralWidget(central_widget)

        # Create the main layout (vertical)
        main_layout = QVBoxLayout()

        # Add the camera layout (camera feeds and logging windows) to top row
        top_layout = self.create_camera_layout(screen_height, screen_width)
        main_layout.addLayout(top_layout)

        # Create the bottom row (buttons, temperature, vib plots)
        bottom_layout = QHBoxLayout()

        # Add the buttons, apply model controls, and temperature layout to the bottom-left of the layout
        button_layout = self.create_button_layout(screen_height, screen_width)
        bottom_layout.addLayout(button_layout)
        bottom_layout.addStretch(1)

        # Add the plot layout to the bottom row (right side)
        vib_layout = self.create_vibration_layout(screen_height, screen_width)
        bottom_layout.addLayout(vib_layout)
        bottom_layout.addStretch(1) 

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)
        # Set the central widget layout
        central_widget.setLayout(main_layout)

        # ============================
        # Set up threads, shared memory for updating GUI contents
        # ============================

        # Set up the QTimer to update the images at 25 fps seconds
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.start_camera_thread)
        self.camera_timer.start(int(1000 / 25))  # Update 25 times every 1000ms

        self.adxl359_timer = QTimer(self)
        self.adxl359_timer.timeout.connect(self.start_sensor_thread)
        self.adxl359_timer.start(int(1000))  # Update every 1000ms (1 second)

        self.adxl359_lock = multiprocessing.Lock()
        vib_h, vib_w = self.vibx_graph.height, self.vibx_graph.width
        self.adxl359_vib_data_shape = (vib_h, vib_w, 3, 4)  # sync with layout size
        self.adxl359_vib_data_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.adxl359_vib_data_shape) * np.uint8().itemsize
        )
        self.adxl359_temp_shm = shared_memory.SharedMemory(
            create=True, size=np.float16().itemsize
        )

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

        self.det_camera_shm = CameraShm(self.camera_shape, self.det_log_shape)
        self.anom_camera_shm = CameraShm(self.camera_shape, self.anom_log_shape)
        self.camera_thread = CameraThread(
            self,
            self.det_camera_shm,
            self.anom_camera_shm,
        )

        # Connect the thread signals to slots in the main window
        self.camera_thread.camera_feed_signal.connect(self.update_camera_feed)
        self.camera_thread.log_feed_signal.connect(self.update_camera_log)
        self.terminate_event = multiprocessing.Event()
        self.camera_processes = multiprocessing.Process(
            target=update_imx500_shm,
            args=(
                self.model,
                self.terminate_event,
                self.det_camera_shm,
                self.anom_camera_shm,
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

    def create_button_layout(self, screen_height, screen_width):
        height_buttons = int(screen_height * (1 / 32))
        width_buttons = int(screen_width / 16)

        # ============================
        # Set up buttons
        # ============================

        button_start = QPushButton("Start", self)
        button_start.setFixedSize(width_buttons, height_buttons)
        button_start.setStyleSheet(
            "background-color: darkgray; border: 1px solid lightgray; color: white;"
        )

        button_stop = QPushButton("Stop", self)
        button_stop.setFixedSize(width_buttons, height_buttons)
        button_stop.setStyleSheet(
            "background-color: darkgray; border: 1px solid lightgray; color: white;"
        )
        button_dispense = QPushButton("Dispense", self)
        button_dispense.setFixedSize(width_buttons, height_buttons)
        button_dispense.setStyleSheet(
            "background-color: darkgray; border: 1px solid lightgray; color: white;"
        )
        # Create button layout and add buttons to it
        button_layout = QVBoxLayout()
        button_layout.addWidget(button_start, alignment=Qt.AlignCenter)
        button_layout.addWidget(button_stop, alignment=Qt.AlignCenter)
        button_layout.addWidget(button_dispense, alignment=Qt.AlignCenter)

        # ============================
        # Set up model selection
        # ============================
        # Group the model selection widgets together
        group_frame = QFrame()
        group_frame.setFrameShape(QFrame.StyledPanel)  # Border style
        group_frame.setFrameShadow(QFrame.Raised)     # Shadow effect (optional)

        # Create Apply Model button and dropdown
        apply_model_button = QPushButton("Apply Model", self)
        apply_model_button.clicked.connect(
            self.apply_model
        )  # Connect to a function for applying model
        apply_model_button.setFixedSize(width_buttons, height_buttons)
        apply_model_button.setStyleSheet(
            "background-color: lightblue; border: 1px solid lightgray;"
        )

        # Create radio buttons for model selection
        models = list(Model)
        self.radio_button1 = QRadioButton(models[0].name)
        self.radio_button2 = QRadioButton(models[1].name)
        self.radio_button1.setStyleSheet("color: white;")
        self.radio_button2.setStyleSheet("color: white;")
        if models[0] == self.model:
            self.radio_button1.setChecked(True)
        else:
            self.radio_button2.setChecked(True)

        # self.model_selection = QButtonGroup()  # create exclusive selection
        # self.model_selection.addButton(self.radio_button1)
        # self.model_selection.addButton(self.radio_button2)

        apply_model_layout = QVBoxLayout(group_frame)
        apply_model_layout.addWidget(self.radio_button1)
        apply_model_layout.addWidget(self.radio_button2)
        apply_model_layout.addWidget(apply_model_button)

        # button_layout.addLayout(apply_model_layout)
        button_layout.addWidget(group_frame)

        # # Create temperature label
        # self.temperature_label = QLabel(self)
        # self.temperature_label.setStyleSheet("font-size: 18px;")

        # # Add buttons, apply model controls, and temperature label to the bottom-left layout
        # button_layout.addWidget(self.temperature_label)

        return button_layout

    def create_camera_layout(self, screen_height, screen_width):

        cam_layout = QHBoxLayout()

        # Sizing camera feed. Maintain camera's aspect ratio
        h = self.camera_shape[0]
        w = self.camera_shape[1]
        scale = h / w
        cam_width = int(screen_width * 1.45 / 4)
        cam_height = int(scale * cam_width)

        # Sizing for log plot
        log_height = cam_height
        log_width = int(screen_width / 8)

        # Camera feed 1 and its logging window
        cam_1_layout = QHBoxLayout()
        self.camera_feed_1 = self.create_camera_feed(
            cam_height, cam_width, title="Camera Feed 1"
        )
        self.camera_log_1 = self.create_camera_log(
            log_height, log_width, title="Camera 1 Logging"
        )

        cam_1_layout.addWidget(self.camera_log_1)
        cam_1_layout.addWidget(self.camera_feed_1)
        cam_layout.addLayout(cam_1_layout)

        # Camera feed 2 and its logging window
        cam_2_layout = QHBoxLayout()
        self.camera_feed_2 = self.create_camera_feed(
            cam_height, cam_width, title="Camera Feed 2"
        )
        self.camera_log_2 = self.create_camera_log(
            log_height, log_width, title="Camera 2 Logging"
        )

        cam_2_layout.addWidget(self.camera_feed_2)
        cam_2_layout.addWidget(self.camera_log_2)
        cam_layout.addLayout(cam_2_layout)

        return cam_layout

    def create_camera_feed(self, cam_height, cam_width, title="Camera Feed"):
        camera_feed = QLabel(self)
        camera_feed.height = cam_height
        camera_feed.width = cam_width
        camera_feed.setFixedSize(camera_feed.width,camera_feed.height)
        camera_feed.setText(title)  # Placeholder text
        camera_feed.setStyleSheet("background-color: lightgray;")
        return camera_feed

    def create_camera_log(self, log_height, log_width, title="Camera logging window"):
        camera_log = QTextEdit(self)
        camera_log.setPlaceholderText(title)
        camera_log.setReadOnly(True)
        camera_log.height = log_height
        camera_log.width = log_width
        camera_log.setFixedSize(camera_log.width, camera_log.height)
        camera_log.setStyleSheet("background-color: black; color: white;")
        return camera_log

    def create_vibration_layout(self, screen_height, screen_width):
        vib_layout = QHBoxLayout()

        vib_height, vib_width = int(screen_height * (2 / 9)), int(screen_width / 4.5)
        # print(f"Vib graph height: {vib_height}")
        # print(f"Vib graph width: {vib_width}")

        self.vibx_graph = self.create_vibration_graph(vib_height, vib_width)
        self.viby_graph = self.create_vibration_graph(vib_height, vib_width)
        self.vibz_graph = self.create_vibration_graph(vib_height, vib_width)
        self.vibx_anomaly_graph = self.create_vibration_graph(vib_height, vib_width)

        # Add the plots to the horizontal layout
        vib_layout.addWidget(self.vibx_graph)
        vib_layout.addWidget(self.viby_graph)
        vib_layout.addWidget(self.vibz_graph)
        vib_layout.addWidget(self.vibx_anomaly_graph)
        return vib_layout

    def create_vibration_graph(self, vib_height, vib_width):
        vib_graph = QLabel(self)
        vib_graph.height = vib_height
        vib_graph.width = vib_width
        vib_graph.setFixedSize(vib_graph.width, vib_graph.height)
        return vib_graph

    def apply_model(self):
        """Handle the Apply Model button action."""
        if self.radio_button1.isChecked():
            selected_model = Model[self.radio_button1.text()]
        else:
            selected_model = Model[self.radio_button2.text()]

        if self.model== selected_model:
            print("no change")
        else:
            self.model = selected_model
            if self.camera_processes.is_alive():
                # Terminate existing process
                self.terminate_event.set()
                self.camera_processes.join()

            # Start new process
            self.terminate_event = multiprocessing.Event()
            self.camera_processes = multiprocessing.Process(
                target=update_imx500_shm,
                args=(
                    self.model,
                    self.terminate_event,
                    self.det_camera_shm,
                    self.anom_camera_shm,
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
        self.det_camera_shm.im_shm.close()  # Detach from the shared memory
        self.det_camera_shm.im_shm.unlink()  # Deallocate the shared memory
        self.det_camera_shm.alg_shm.close()
        self.det_camera_shm.alg_shm.unlink()

        self.adxl359_temp_shm.close()
        self.adxl359_temp_shm.unlink()

        self.anom_camera_shm.im_shm.close()  # Detach from the shared memory
        self.anom_camera_shm.im_shm.unlink()  # Deallocate the shared memory
        self.anom_camera_shm.alg_shm.close()
        self.anom_camera_shm.alg_shm.unlink()

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
        det_camera, anom_camera = camera_data_tuple
        self.display_image(self.camera_feed_1, det_camera)
        self.display_image(self.camera_feed_2, anom_camera)

    def update_camera_log(self, camera_log_data_tuple):
        det_log, anom_log = camera_log_data_tuple
        if det_log:
            self.print_log(self.camera_log_1, det_log, overwrite=True)
        if anom_log:
            self.print_log(self.camera_log_2, anom_log)

    def update_adxl359_feed(self,plot_tuple):
        vibx_graph, viby_graph, vibz_graph, vib_anom_graph = plot_tuple
        self.display_image(self.vibx_graph, vibx_graph)
        self.display_image(self.viby_graph, viby_graph)
        self.display_image(self.vibz_graph, vibz_graph)
        self.display_image(self.vibx_anomaly_graph, vib_anom_graph)
        # self.temperature_label.setText(f"Temperature: {temp[0]:.2f} °C")

    def display_image(self, label: QLabel, image_pixmap):
        label.setPixmap(image_pixmap.scaled(label.width,label.height))

    def print_log(self, textbox: QTextEdit, log_info, overwrite=False):

        if overwrite:
            textbox.setText(log_info)
        else:
            textbox.append(log_info)
