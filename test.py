import sys
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDesktopWidget, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QComboBox, QSpacerItem, QSizePolicy, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

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
        self.update_camera_feed(self.camera_feed_1)
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
        self.update_camera_feed(self.camera_feed_2)
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

        # Add a spacer item between the top and bottom sections for more spacing
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)

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

        # Create a horizontal layout for Apply Model button and dropdown
        apply_model_layout = QHBoxLayout()
        apply_model_layout.addWidget(apply_model_button)
        apply_model_layout.addWidget(self.model_dropdown)

        # Create temperature label
        self.temperature_label = QLabel(self)
        self.temperature_label.setText(f"Temperature: {self.generate_temperature():.2f} °C")
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

        # Resize the plots
        plot_width = screen_width // 2  # Half the width of the screen
        plot_height = plot_width * 2 // 3  # Aspect ratio of 3:2 (height:width)
        self.plot1.resize(plot_width, plot_height)
        self.plot2.resize(plot_width, plot_height)
        self.plot3.resize(plot_width, plot_height)

        # Update the plot data
        self.update_plot_data(self.plot1, 'X')
        self.update_plot_data(self.plot2, 'Y')
        self.update_plot_data(self.plot3, 'Z')

        # Add the plots to the vertical layout
        plot_layout.addWidget(self.plot1)
        plot_layout.addWidget(self.plot2)
        plot_layout.addWidget(self.plot3)

        # Add the plot layout to the bottom row (left side)
        bottom_layout.addLayout(plot_layout)

        # Create a vertical layout for the anomaly plots (far right)
        anomaly_plot_layout = QVBoxLayout()

        # Create the three anomaly plots
        self.anomaly_plot1 = pg.PlotWidget(title="Anomaly Plot 1")
        self.anomaly_plot2 = pg.PlotWidget(title="Anomaly Plot 2")
        self.anomaly_plot3 = pg.PlotWidget(title="Anomaly Plot 3")

        # Resize the anomaly plots
        self.anomaly_plot1.resize(plot_width, plot_height)
        self.anomaly_plot2.resize(plot_width, plot_height)
        self.anomaly_plot3.resize(plot_width, plot_height)

        # Update the anomaly plot data
        self.update_plot_data(self.anomaly_plot1, 'Anomaly 1')
        self.update_plot_data(self.anomaly_plot2, 'Anomaly 2')
        self.update_plot_data(self.anomaly_plot3, 'Anomaly 3')

        # Add the anomaly plots to the vertical layout
        anomaly_plot_layout.addWidget(self.anomaly_plot1)
        anomaly_plot_layout.addWidget(self.anomaly_plot2)
        anomaly_plot_layout.addWidget(self.anomaly_plot3)

        # Add the anomaly plot layout to the bottom row (right side)
        bottom_layout.addLayout(anomaly_plot_layout)

        # Add the bottom layout to the main layout
        main_layout.addLayout(bottom_layout)

        # Set the central widget layout
        central_widget.setLayout(main_layout)

        # Initialize timer for data collection
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.collect_sample)

    def exit_application(self):
        """Close the application."""
        QApplication.quit()

    def update_plot_data(self, plot_widget, axis):
        """Generate and plot fake data for each axis or anomaly."""
        # Fake data: Sine wave or random values
        x = np.linspace(0, 10, 100)  # X values (time)
        
        if axis == 'X':
            y = np.sin(x) + 0.1 * np.random.randn(100)  # Y values for X axis (sinusoidal with noise)
        elif axis == 'Y':
            y = np.cos(x) + 0.1 * np.random.randn(100)  # Y values for Y axis (cosine with noise)
        elif axis == 'Z':
            y = np.tan(x) + 0.1 * np.random.randn(100)  # Y values for Z axis (tangent with noise)
        elif 'Anomaly' in axis:
            y = np.random.normal(loc=0, scale=1, size=100)  # Random data for anomaly plots

        plot_widget.plot(x, y, pen='g')  # Plot the data (green line)

    def update_camera_feed(self, label):
        """Generate and display fake camera feed data."""
        # Fake image data: Generate random noise for the camera feed
        width, height = 640, 480
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # Random RGB data
        image = QImage(data, width, height, 3 * width, QImage.Format_RGB888)  # Convert to QImage
        pixmap = QPixmap.fromImage(image)  # Convert QImage to QPixmap
        label.setPixmap(pixmap)  # Set the QPixmap on the QLabel

    def generate_temperature(self):
        """Generate a fake temperature reading."""
        # For now, generate a random temperature between 20 and 30°C
        return np.random.uniform(20.0, 30.0)

    def apply_model(self):
        """Handle the Apply Model button action."""
        selected_model = self.model_dropdown.currentText()
        print(f"Applied {selected_model}")
        # You can add logic here to apply the selected model

    def collect_data(self):
        """Collect data at the specified sample rate."""
        try:
            sample_rate = float(self.sample_rate_input.text())  # Get sample rate from input
            self.data_label = self.data_label_input.text()  # Get data label from input
            data_type = self.data_type_dropdown.currentText()  # Get the selected data type

            if sample_rate > 0:
                self.timer.start(1000 / sample_rate)  # Start the timer with the interval based on sample rate
                print(f"Collecting {data_type} data with label: {self.data_label} at {sample_rate} Hz")
            else:
                print("Please enter a valid sample rate.")
        except ValueError:
            print("Please enter a valid number for sample rate.")

    def collect_sample(self):
        """Simulate collecting a data sample."""
        # Simulate a data sample collection (for example, from sensor data)
        print(f"Collecting sample for: {self.data_label}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
