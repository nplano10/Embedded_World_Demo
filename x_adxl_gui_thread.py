import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from multiprocessing import shared_memory
import copy


class Adxl359Thread(QThread):
    # Define a signal to send data to the main thread
    adxl359_plot_signal = pyqtSignal(tuple)

    def __init__(
        self,
        parent,
        adxl359_vib_data_shm_name,
        adxl359_temp_shm_name,
        adxl359_vib_data_shape,
        adxl359_lock,
    ):
        super().__init__(parent,)  # Call the parent constructor
        # Initialize any other variables as needed
        self.previous_data = None
        self.adxl359_vib_data_shm_name = adxl359_vib_data_shm_name
        self.adxl359_temp_shm_name = adxl359_temp_shm_name
        self.adxl359_lock = adxl359_lock
        self.adxl359_vib_data_shape = adxl359_vib_data_shape

    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def run(self):
        # Update data from shared memory

        with self.adxl359_lock:
            # Create a NumPy array from the shared memory buffer

            existing_adxl359_vib_data_shm = shared_memory.SharedMemory(
                name=self.adxl359_vib_data_shm_name
            )
            # existing_adxl359_temp_shm = shared_memory.SharedMemory(
            #     name=self.adxl359_temp_shm_name
            # )
            shared_adxl359_vib_data = np.ndarray(
                self.adxl359_vib_data_shape,
                dtype=np.int8,
                buffer=existing_adxl359_vib_data_shm.buf,
            )
            # shared_adxl359_temp_data = np.ndarray(
            #     (1,), dtype=np.float16, buffer=existing_adxl359_temp_shm.buf
            # )
            adxl359_vib_data = copy.deepcopy(shared_adxl359_vib_data)
            # adxl359_temp_data = copy.deepcopy(shared_adxl359_temp_data)

        # Emit the signal only if the data has changed
        if self.previous_data is None or not np.array_equal(adxl359_vib_data, self.previous_data):
            self.previous_data = adxl359_vib_data # Update with the new data

            self.adxl359_plot_signal.emit(
                (
                    self.numpy_arrray_to_pixmap(adxl359_vib_data[:, :, :, 0]),
                    self.numpy_arrray_to_pixmap(adxl359_vib_data[:, :, :, 1]),
                    self.numpy_arrray_to_pixmap(adxl359_vib_data[:, :, :, 2]),
                    self.numpy_arrray_to_pixmap(adxl359_vib_data[:, :, :, 3]),
                    # adxl359_temp_data,
                )
            )
