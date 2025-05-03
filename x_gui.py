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
from adxl359.x_adxl_process import update_adxl359_vib_data_shm
# Import the new unified camera system class
from imx500.x_imx500 import IMX500CameraSystem, Model
from adxl359.x_adxl_gui_thread import Adxl359Thread
from picamera2.previews.qt import QGlPicamera2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camera_shape = (480, 640, 3)
        self.anom_log_shape = (100, 3)
        self.det_log_shape = (100, 3)
        self.adxl359_sample_rate = 1000
        self.adxl359_sample_length = 1000

        # Keep track of created shared memory objects to ensure cleanup
        self.shared_memory_objects = []

        # Initialize with TRAINED model
        self.model = Model.TRAINED
        
        # Initialize the IMX500 camera system - replaces the detector creation
        self.camera_system = IMX500CameraSystem(parent=self, model_type=self.model)
        
        # Extract necessary components for backward compatibility
        self.detector = self.camera_system.detector
        self.anomaly_detector = self.camera_system.anomaly_detector
        if self.model == Model.TRAINED:
            self.bbox_queue = self.camera_system.bbox_queue
            self.results_queue = self.camera_system.results_queue
            self.det_camera_shm = self.camera_system.det_camera_shm
            self.anom_camera_shm = self.camera_system.anom_camera_shm
        else:
            self.bbox_queue = None
            self.results_queue = None
            self.det_camera_shm = None
            self.anom_camera_shm = None

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
        # Set up the camera previews and callbacks
        # ============================
        
        # Stop any existing previews
        if self.model == Model.TRAINED:
            self.detector.picam2.stop_preview()
            self.anomaly_detector.picam2.stop_preview()
            self.detector.picam2.pre_callback = lambda req: self.detector.draw_detections(req, self.bbox_queue, self.results_queue)
            self.anomaly_detector.picam2.pre_callback = lambda req: self.anomaly_detector.process_frame(req, self.bbox_queue, self.results_queue)
        else:
            self.detector.picam2.stop_preview()
            self.anomaly_detector.picam2.stop_preview()

        # ============================
        # Set up sensors, shared memory for updating vibration data
        # ============================

        # Set up the QTimer to update the images at 25 fps seconds
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.start_camera_thread)
        self.camera_timer.start(int(1000 / 20))  # Update 25 times every 1000ms

        self.adxl359_timer = QTimer(self)
        self.adxl359_timer.timeout.connect(self.start_sensor_thread)
        self.adxl359_timer.start(int(1000))  # Update every 1000ms (1 second)

        self.adxl359_lock = multiprocessing.Lock()
        vib_h, vib_w = self.vibx_graph.height, self.vibx_graph.width
        self.adxl359_vib_data_shape = (vib_h, vib_w, 3, 4)  # sync with layout size
        
        # Create shared memory objects with proper tracking
        try:
            self.adxl359_vib_data_shm = shared_memory.SharedMemory(
                create=True, size=np.prod(self.adxl359_vib_data_shape) * np.uint8().itemsize
            )
            self.shared_memory_objects.append(('adxl359_vib_data_shm', self.adxl359_vib_data_shm))
            
            self.adxl359_temp_shm = shared_memory.SharedMemory(
                create=True, size=np.float16().itemsize
            )
            self.shared_memory_objects.append(('adxl359_temp_shm', self.adxl359_temp_shm))
        except Exception as e:
            print(f"Error creating shared memory: {e}")
            self.exit_application()

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

        # Start camera previews
        self.start_camera_previews()

        # Use the camera system's thread for logging
        self.camera_thread = self.camera_system
        self.camera_thread.log_feed_signal.connect(self.update_camera_log)

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
        # # Create button layout and add buttons to it
        button_layout = QVBoxLayout()

        # ============================
        # Set up model selection
        # ============================
        # Group the model selection widgets together
        group_frame = QFrame()
        group_frame.setFrameShape(QFrame.StyledPanel)  # Border style
        group_frame.setFrameShadow(QFrame.Raised)     # Shadow effect (optional)

        # Create model label to display current model
        self.model_label = QLabel(self)
        self.model_label.setStyleSheet("font-size: 13px; color: lightblue")
        self.model_label.setText(f"<b><i>Current: {self.model.name}</i></b>")

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
        apply_model_layout.addWidget(self.model_label)
        apply_model_layout.addWidget(self.radio_button1)
        apply_model_layout.addWidget(self.radio_button2)
        apply_model_layout.addWidget(apply_model_button)

        # button_layout.addLayout(apply_model_layout)
        button_layout.addStretch(1)
        button_layout.addWidget(group_frame)
        button_layout.addStretch(1)

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
        
        # QGlPicamera2 for camera 1
        self.camera_feed_1 = QWidget()
        self.camera_feed_1.height = cam_height
        self.camera_feed_1.width = cam_width
        self.camera_feed_1.setFixedSize(self.camera_feed_1.width, self.camera_feed_1.height)
        self.camera_feed_1.setStyleSheet("background-color: lightgray;")
        
        self.camera_log_1 = self.create_camera_log(
            log_height, log_width, title="Camera 1 Logging"
        )

        cam_1_layout.addWidget(self.camera_log_1)
        cam_1_layout.addWidget(self.camera_feed_1)
        cam_layout.addLayout(cam_1_layout)

        # Camera feed 2 and its logging window
        cam_2_layout = QHBoxLayout()
        
        # QGlPicamera2 for camera 2
        self.camera_feed_2 = QWidget()
        self.camera_feed_2.height = cam_height
        self.camera_feed_2.width = cam_width
        self.camera_feed_2.setFixedSize(self.camera_feed_2.width, self.camera_feed_2.height)
        self.camera_feed_2.setStyleSheet("background-color: lightgray;")
        
        self.camera_log_2 = self.create_camera_log(
            log_height, log_width, title="Camera 2 Logging"
        )

        cam_2_layout.addWidget(self.camera_feed_2)
        cam_2_layout.addWidget(self.camera_log_2)
        cam_layout.addLayout(cam_2_layout)

        return cam_layout

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
    
    def clear_layout(self, widget: QWidget):
        layout = widget.layout()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                child = item.widget()
                if child is not None:
                    child.setParent(None)
            # Optionally remove the layout itself
            QWidget().setLayout(layout)

    def apply_model(self):
        """Handle the Apply Model button action."""
        import time, gc

        selected_model = Model[self.radio_button1.text()] if self.radio_button1.isChecked() else Model[self.radio_button2.text()]
        if self.model == selected_model:
            print("no change")
            return

        print("Switching model from", self.model.name, "to", selected_model.name)

        sender = self.sender()
        if sender:
            sender.setEnabled(False)

        # Remove previews safely
        for preview_attr, cam_feed in [('preview1', self.camera_feed_1), ('preview2', self.camera_feed_2)]:
            try:
                preview = getattr(self, preview_attr, None)
                if preview:
                    layout = cam_feed.layout()
                    if layout:
                        layout.removeWidget(preview)
                    preview.setParent(None)
                    preview.deleteLater()
                    setattr(self, preview_attr, None)
                print(f"removing {preview_attr} and  {cam_feed}")
            except Exception as e:
                print(f"Error removing {preview_attr}: {e}")

        # Clean up camera system - IMPORTANT: do this BEFORE creating a new one
        if hasattr(self, 'camera_system'):
            try:
                # Stop the thread if it's running
                if self.camera_system.isRunning():
                    self.camera_system.quit()
                    self.camera_system.wait()
                    
                # Stop camera timers
                if hasattr(self, 'camera_timer'):
                    self.camera_timer.stop()
                    
                # Make sure to stop and close the camera devices
                if hasattr(self.camera_system, 'detector') and self.camera_system.detector:
                    print("Stopping detector camera...")
                    try:
                        self.camera_system.detector.picam2.stop()
                        self.camera_system.detector.picam2.close()
                    except Exception as e:
                        print(f"Error stopping detector: {e}")
                        
                if hasattr(self.camera_system, 'anomaly_detector') and self.camera_system.anomaly_detector:
                    print("Stopping anomaly detector camera...")
                    try:
                        self.camera_system.anomaly_detector.picam2.stop()
                        self.camera_system.anomaly_detector.picam2.close()
                    except Exception as e:
                        print(f"Error stopping anomaly detector: {e}")
                    
                # Clean up camera system resources
                self.camera_system.cleanup()
                
                # Remove signal connections
                try:
                    self.camera_system.log_feed_signal.disconnect()
                except Exception as e:
                    print(f"Error disconnecting signals: {e}")
                    
                # Delete the camera system
                del self.camera_system
                self.camera_system = None
            except Exception as e:
                print(f"Error stopping camera system: {e}")

        # Give time for camera hardware release
        print("Waiting for camera resources to be released...")
        time.sleep(20)
        gc.collect()

        # Set new model
        self.model = selected_model
        print("Creating new camera system instance...")

        # Create new camera system with selected model
        try:
            self.camera_system = IMX500CameraSystem(parent=self, model_type=self.model)
            
            # Update internal references for backward compatibility
            self.detector = self.camera_system.detector
            self.anomaly_detector = self.camera_system.anomaly_detector
            if self.model == Model.TRAINED:
                self.bbox_queue = self.camera_system.bbox_queue
                self.results_queue = self.camera_system.results_queue
                self.det_camera_shm = self.camera_system.det_camera_shm
                self.anom_camera_shm = self.camera_system.anom_camera_shm
                self.detector.picam2.stop_preview()
                self.anomaly_detector.picam2.stop_preview()
                self.detector.picam2.pre_callback = lambda req: self.detector.draw_detections(req, self.bbox_queue, self.results_queue)
                self.anomaly_detector.picam2.pre_callback = lambda req: self.anomaly_detector.process_frame(req, self.bbox_queue, self.results_queue)
            else:
                self.bbox_queue = None
                self.results_queue = None
                self.det_camera_shm = None
                self.anom_camera_shm = None
                self.detector.picam2.stop_preview()
                self.anomaly_detector.picam2.stop_preview()
        except Exception as e:
            print(f"Error creating camera system instance: {e}")
            # Optionally, try to recover by setting back to the previous model
            # or just continue with a disabled camera system

        # Use the camera system's thread for logging
        self.camera_thread = self.camera_system
        self.camera_thread.log_feed_signal.connect(self.update_camera_log)

        self.camera_log_1.clear()
        self.camera_log_2.clear()
        self.model_label.setText(f"<b><i>Current: {self.model.name}</i></b>")

        # Restart camera timer
        if hasattr(self, 'camera_timer'):
            self.camera_timer.start(int(1000 / 20))

        print("Starting new camera previews...")
        self.start_camera_previews()

        if hasattr(self, 'log_timer'):
            self.log_timer.start(200)
        if sender:
            sender.setEnabled(True)

    def start_camera_previews(self):
        """Set up camera previews using QGlPicamera2."""
        if not self.detector or not self.anomaly_detector:
            return
        
        # Camera 1
        camera1_layout = QVBoxLayout()
        self.clear_layout(self.camera_feed_1)
        self.camera_feed_1.setLayout(camera1_layout)
        self.preview1 = QGlPicamera2(self.detector.picam2, width=self.camera_feed_1.width, height=self.camera_feed_1.height)
        camera1_layout.addWidget(self.preview1)

        # Camera 2
        camera2_layout = QVBoxLayout()
        self.clear_layout(self.camera_feed_2)
        self.camera_feed_2.setLayout(camera2_layout)
        self.preview2 = QGlPicamera2(self.anomaly_detector.picam2, width=self.camera_feed_2.width, height=self.camera_feed_2.height)
        camera2_layout.addWidget(self.preview2)

    def update_camera_log(self, camera_log_data_tuple):
        det_log, anom_log = camera_log_data_tuple
        if det_log:
            self.print_log(self.camera_log_1, det_log, overwrite=True)
        if anom_log:
            self.print_log(self.camera_log_2, anom_log)

    def exit_application(self):
        print("Cleaning threads")
        
        # Clean up camera system
        if hasattr(self, 'camera_system'):
            try:
                self.camera_system.cleanup()
            except Exception as e:
                print(f"Error cleaning up camera system: {e}")
                
        if hasattr(self, 'adxl359_thread') and self.adxl359_thread.isRunning():
            self.adxl359_thread.quit()
            self.adxl359_thread.wait()
        print("Done Cleaning threads")

        if hasattr(self, 'sensor_processes') and self.sensor_processes.is_alive():
            print("Terminating sensor process as it is still running...")
            self.sensor_processes.terminate()  # Forcefully terminate the process
            # Wait for process to fully terminate to avoid resource leaks
            self.sensor_processes.join(timeout=5)
            if self.sensor_processes.is_alive():
                print("Force killing sensor process...")
                self.sensor_processes.kill()
                self.sensor_processes.join(timeout=1)

        print("Cleaning mem")
        # Systematically clean up all shared memory objects
        for name, shm_obj in self.shared_memory_objects:
            try:
                print(f"Cleaning shared memory object: {name}")
                shm_obj.close()
                shm_obj.unlink()
            except Exception as e:
                print(f"Error cleaning shared memory object {name}: {e}")
        
        # Clear the list to avoid double-free issues
        self.shared_memory_objects.clear()
        
        print("Done Cleaning mem")

        print("exiting")
        QApplication.exit()

    def start_camera_thread(self):
        self.camera_thread.start()
        self.camera_thread.setPriority(QThread.TimeCriticalPriority)

    def start_sensor_thread(self):
        self.adxl359_thread.start()
        self.adxl359_thread.setPriority(QThread.LowPriority)

    def update_adxl359_feed(self, plot_tuple):
        vibx_graph, viby_graph, vibz_graph, vib_anom_graph = plot_tuple
        self.display_image(self.vibx_graph, vibx_graph)
        self.display_image(self.viby_graph, viby_graph)
        self.display_image(self.vibz_graph, vibz_graph)
        self.display_image(self.vibx_anomaly_graph, vib_anom_graph)
        # self.temperature_label.setText(f"Temperature: {temp[0]:.2f} °C")

    def display_image(self, label: QLabel, image_pixmap):
        label.setPixmap(image_pixmap.scaled(label.width, label.height))

    def print_log(self, textbox: QTextEdit, log_info, overwrite=False):
        if overwrite:
            textbox.setText(log_info)
        else:
            textbox.append(log_info)