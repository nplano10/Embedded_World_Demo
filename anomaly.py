import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
import pyqtgraph as pg

class AnomalyPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Time Series and Anomaly Scores Visualization')
        self.setGeometry(100, 100, 1200, 800)

        # Create the main QWidget to hold the layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a QHBoxLayout to arrange time series on the left and anomaly scores on the right
        main_layout = QHBoxLayout(central_widget)

        # Create VBoxLayouts for time series and anomaly score sections
        self.time_series_layout = QVBoxLayout()
        self.anomaly_scores_layout = QVBoxLayout()

        # Add the layouts to the main layout
        main_layout.addLayout(self.time_series_layout)
        main_layout.addLayout(self.anomaly_scores_layout)

        # Example data (1000 samples)
        self.time = np.arange(1000)
        self.data1 = np.sin(self.time / 10) + np.random.normal(0, 0.5, 1000)  # Time Series 1
        self.data2 = np.cos(self.time / 10) + np.random.normal(0, 0.5, 1000)  # Time Series 2
        self.data3 = np.sin(self.time / 20) + np.random.normal(0, 0.5, 1000)  # Time Series 3

        # Simulated anomaly scores (e.g., global anomaly score for each point)
        self.anomaly_score1 = np.random.rand(1000)  # Random anomaly scores for Time Series 1
        self.anomaly_score2 = np.random.rand(1000)  # Random anomaly scores for Time Series 2
        self.anomaly_score3 = np.random.rand(1000)  # Random anomaly scores for Time Series 3

        # Define anomaly thresholds
        self.threshold1 = 0.8
        self.threshold2 = 0.8
        self.threshold3 = 0.8

        # Plot the time series and anomaly scores
        self.plot_time_series()
        self.plot_anomaly_scores()

    def plot_time_series(self):
        # Create individual time series plots and add them to the layout
        time_series_plot1 = pg.PlotWidget(title="Time Series 1")
        time_series_plot2 = pg.PlotWidget(title="Time Series 2")
        time_series_plot3 = pg.PlotWidget(title="Time Series 3")

        # Add them to the time series layout
        self.time_series_layout.addWidget(time_series_plot1)
        self.time_series_layout.addWidget(time_series_plot2)
        self.time_series_layout.addWidget(time_series_plot3)

        # Plot time series data
        time_series_plot1.plot(self.time, self.data1, pen='b', name="Time Series 1")
        time_series_plot2.plot(self.time, self.data2, pen='g', name="Time Series 2")
        time_series_plot3.plot(self.time, self.data3, pen='r', name="Time Series 3")

    def plot_anomaly_scores(self):
        # Create individual anomaly score plots and add them to the layout
        anomaly_score_plot1 = pg.PlotWidget(title="Anomaly Score 1")
        anomaly_score_plot2 = pg.PlotWidget(title="Anomaly Score 2")
        anomaly_score_plot3 = pg.PlotWidget(title="Anomaly Score 3")

        # Add them to the anomaly scores layout
        self.anomaly_scores_layout.addWidget(anomaly_score_plot1)
        self.anomaly_scores_layout.addWidget(anomaly_score_plot2)
        self.anomaly_scores_layout.addWidget(anomaly_score_plot3)

        # Plot anomaly scores
        anomaly_score_plot1.plot(self.time, self.anomaly_score1, pen='orange', name="Anomaly Score 1")
        anomaly_score_plot2.plot(self.time, self.anomaly_score2, pen='purple', name="Anomaly Score 2")
        anomaly_score_plot3.plot(self.time, self.anomaly_score3, pen='pink', name="Anomaly Score 3")

        # Highlight anomalies (above the threshold)
        anomaly_score_plot1.scatterPlot(self.time[self.anomaly_score1 > self.threshold1], 
                                        self.anomaly_score1[self.anomaly_score1 > self.threshold1], 
                                        pen=None, symbol='o', symbolBrush='r', symbolSize=6)
        anomaly_score_plot2.scatterPlot(self.time[self.anomaly_score2 > self.threshold2], 
                                        self.anomaly_score2[self.anomaly_score2 > self.threshold2], 
                                        pen=None, symbol='o', symbolBrush='r', symbolSize=6)
        anomaly_score_plot3.scatterPlot(self.time[self.anomaly_score3 > self.threshold3], 
                                        self.anomaly_score3[self.anomaly_score3 > self.threshold3], 
                                        pen=None, symbol='o', symbolBrush='r', symbolSize=6)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnomalyPlotWindow()
    window.show()
    sys.exit(app.exec_())
