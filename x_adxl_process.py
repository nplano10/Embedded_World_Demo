import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
from multiprocessing import shared_memory
from  adxl359  import ADXL359
import time
import pyqtgraph as pg


def pixmap_to_numpy(pixmap):
    # Convert QPixmap to QImage
    image = pixmap.toImage()

    # Ensure the image is in a format compatible with raw data access
    image = image.convertToFormat(QImage.Format_RGB888)

    # Extract the width and height
    width = image.width()
    height = image.height()
    # print(f"Width: {width}, Height: {height}")

    # Extract the raw pixel data
    ptr = image.bits()
    ptr.setsize(image.byteCount())  # Ensure the byte size is correct

    # Calculate bytes per row (includes padding)
    bytes_per_line = image.bytesPerLine()

    # Create a raw numpy array from the pixel data (including padding)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))

    # Remove the padding from each row
    # We only want width * 3 bytes per row (since the image is in RGB format)
    arr = arr[:, :width * 3].reshape((height, width, 3))

    return arr


def get_anomaly_scores(vibx, viby, vibz):
    # vib_score = np.random.rand(1)[0]
    # print(f"Min, Max vibx: {np.min(vibx)}, {np.max(vibx)}")
    # print(f"Min, Max viby: {np.min(viby)}, {np.max(viby)}")
    # print(f"Min, Max vibz: {np.min(vibz)}, {np.max(vibz)}")
    # vib_score = np.sqrt(np.square(vibx) + np.square(viby) + np.square(vibz))
    # vib_score = np.abs(0.006 - np.sqrt(np.square(viby).mean()))  # compute RMSE
    vib_score = np.sqrt(np.square(viby).mean())-.004  # compute RMSE
    #print(vib_score)
    vib_score = vib_score * 60.0
    vib_score = min(vib_score, 1.0)
    vib_score = max(vib_score, 0.0)
    return vib_score


def update_anomaly_score_arrays(vib_anomaly_scores, vibx_data, viby_data, vibz_data):

    vib_score = get_anomaly_scores(vibx_data, viby_data, vibz_data)

    # Shift elements to the left by 1
    vib_anomaly_scores = np.roll(vib_anomaly_scores, -1)
    vib_anomaly_scores[-1] = vib_score
    return vib_anomaly_scores


def update_adxl359_vib_data_shm(
    adxl359_vib_data_shm_name,
    adxl359_temp_shm_name,
    adxl359_vib_data_shape,
    adxl359_lock,
):
    vib_ylim = [-0.1, 0.1]
    anomaly_threshold = 0.3
    anomaly_history = 20

    h = adxl359_vib_data_shape[0]
    w = adxl359_vib_data_shape[1]

    adxl359 = ADXL359()  # Adjust according to your actual initialization code
    adxl359._initialize()

    plot1 = pg.PlotWidget(title="Vibration X Axis")
    plot2 = pg.PlotWidget(title="Vibration Y Axis")
    plot3 = pg.PlotWidget(title="Vibration Z Axis")
    for plot in [plot1, plot2, plot3]:
        # plot.setLabel('left', 'Amplitude')  # Y-axis label
        plot.setLabel("bottom", "Time [ms]")  # X-axis label

    plot1.setFixedSize(w, h)
    plot2.setFixedSize(w, h)
    plot3.setFixedSize(w, h)

    plot1.setYRange(*vib_ylim)
    plot2.setYRange(*vib_ylim)
    plot3.setYRange(*vib_ylim)

    plot1_item = plot1.plot(np.arange(1000),np.zeros(1000).astype(np.float16), pen='b')
    plot2_item = plot2.plot(np.arange(1000),np.zeros(1000).astype(np.float16), pen='g')
    plot3_item = plot3.plot(np.arange(1000),np.zeros(1000).astype(np.float16), pen='r')

    # Create the anomaly plot
    anomaly_score_plot = pg.PlotWidget(title="Anomaly Score")
    anomaly_score_plot.setFixedSize(w, h)
    anomaly_score_plot.setYRange(0, 1)
    anomaly_score_plot.setLabel("bottom", "Time [s]")

    vib_anomaly_scores = np.zeros(anomaly_history)
    index = np.arange(anomaly_history)

    pen = pg.mkPen(color="orange", style=Qt.DashLine)
    _ = anomaly_score_plot.plot(
        [index[0], index[-1]],
        [anomaly_threshold, anomaly_threshold],
        pen=pen,
    )
    anomaly_score_all_item = anomaly_score_plot.plot(
        index,
        vib_anomaly_scores,
        pen="lightgray"
    )
    anomaly_score_anom_item = anomaly_score_plot.scatterPlot(
        index,
        vib_anomaly_scores,
        symbol="o",
        pen="red",
        brush="pink",
        name="Anomaly Score",
    )
    anomaly_score_norm_item = anomaly_score_plot.scatterPlot(
        index,
        vib_anomaly_scores,
        symbol="o",
        pen="blue",
        brush="lightblue",
        name="Anomaly Score",
    )

    # existing_shm = shared_memory.SharedMemory(name=adxl359_vib_data_shm.name)
    # shm_array = np.ndarray((1,), dtype=np.float16, buffer=adxl359_temp_shm.buf)

    existing_adxl359_vib_data_shm = shared_memory.SharedMemory(name=adxl359_vib_data_shm_name)
    # existing_adxl359_temp_shm = shared_memory.SharedMemory(name=adxl359_temp_shm_name)
    shared_adxl359_vib_data = np.ndarray(adxl359_vib_data_shape, dtype=np.int8, buffer=existing_adxl359_vib_data_shm.buf)
    # shared_adxl359_temp_data = np.ndarray((1,), dtype=np.float16, buffer=existing_adxl359_temp_shm.buf)

    while True:
        try:
            x_data,y_data,z_data,temp_data = adxl359.collect_data() # Example method from adxl359 object
            plot1_item.setData(np.linspace(0, 1000, 1000).tolist(), x_data)
            plot2_item.setData(np.linspace(0, 1000, 1000).tolist(), y_data)
            plot3_item.setData(np.linspace(0, 1000, 1000).tolist(), z_data)

            # temperature_label.setText(f"Temperature: {temp:.2f} °C")

            vib_anomaly_scores = update_anomaly_score_arrays(
                vib_anomaly_scores, x_data, y_data, z_data
            )
            # vib_anomaly_scores_good = vib_anomaly_scores.copy()
            # vib_anomaly_scores_bad = vib_anomaly_scores.copy()

            bad_pts = vib_anomaly_scores >= anomaly_threshold

            anomaly_score_all_item.setData(index, vib_anomaly_scores)
            anomaly_score_norm_item.setData(index[~bad_pts], vib_anomaly_scores[~bad_pts])
            anomaly_score_anom_item.setData(index[bad_pts], vib_anomaly_scores[bad_pts])

            with adxl359_lock:                                        
                shared_adxl359_vib_data[:, :, :, 0] = pixmap_to_numpy(plot1.grab())
                shared_adxl359_vib_data[:, :, :, 1] = pixmap_to_numpy(plot2.grab())
                shared_adxl359_vib_data[:, :, :, 2] = pixmap_to_numpy(plot3.grab())
                shared_adxl359_vib_data[:, :, :, 3] = pixmap_to_numpy(
                    anomaly_score_plot.grab()
            )
        except:
            print("data failed to capture")

        

            # shared_adxl359_temp_data[0] = temp_data[0]
