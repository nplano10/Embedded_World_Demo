import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from multiprocessing import shared_memory
import copy
from x_imx500_process import DET_LABEL
from x_utils import CameraShm


COLORS = [
    "red",
    "royalblue"
]

class CameraThread(QThread):
    # Define a signal to send data to the main thread
    camera_feed_signal = pyqtSignal(tuple)
    log_feed_signal = pyqtSignal(tuple)

    def __init__(
        self,
        parent,
        det_camera_shm: CameraShm,
        anom_camera_shm: CameraShm,
        det_fn = DET_LABEL,
    ):
        super().__init__(parent)  # Make sure to call the base class's constructor
        self.previous_camera_det = None
        self.previous_camera_anom = None
        self.det_camera_shm = det_camera_shm
        self.anom_camera_shm = anom_camera_shm
        self.det_labels = self.parse_det_labels(det_fn)

    def numpy_arrray_to_pixmap(self,numpy_array):
        height, width, _ = numpy_array.shape
        q_image = QImage(numpy_array.tobytes(), width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def run(self):

        # Attach to the shared memory for camera
        camera_image_det = self.readCameraImageShm(self.det_camera_shm)
        camera_image_anom = self.readCameraImageShm(self.anom_camera_shm)

        # Check if either camera has new data
        if (self.previous_camera_det is None or not np.array_equal(camera_image_det, self.previous_camera_det)) or \
           (self.previous_camera_anom is None or not np.array_equal(camera_image_anom, self.previous_camera_anom)):
            # Update previous images with current ones
            self.previous_camera_det = camera_image_det
            self.previous_camera_anom = camera_image_anom
            # Emit the signal with the new data
            self.camera_feed_signal.emit((self.numpy_arrray_to_pixmap(camera_image_det), self.numpy_arrray_to_pixmap(camera_image_anom)))

        # Process output from algorithms for log information
        det_results = self.readAlgResults(self.det_camera_shm)
        anom_results = self.readAlgResults(self.anom_camera_shm)

        if det_results.shape[0] > 0 or anom_results.shape[0] > 0:
            det_log = self.det_results_to_string(det_results)
            anom_log = self.anom_results_to_string(anom_results)
            self.log_feed_signal.emit((det_log, anom_log))

    @staticmethod
    def readCameraImageShm(camera_shm: CameraShm):
        with camera_shm.im_lock:
            shm = shared_memory.SharedMemory(name=camera_shm.im_shm.name)
            image = np.ndarray(
                camera_shm.im_shape, dtype=np.uint8, buffer=shm.buf
            )
            camera_image = copy.deepcopy(image)

        return camera_image

    @staticmethod
    def readAlgResults(camera_shm: CameraShm):
        with camera_shm.alg_lock:
            shm = shared_memory.SharedMemory(name=camera_shm.alg_shm.name)
            results = np.ndarray(
                camera_shm.alg_shape, dtype=np.float16, buffer=shm.buf
            )
            alg_results = copy.deepcopy(results)

            # clear the shm
            results[:] = 0.

        # crop the bottom part of the array
        to_remove = np.all(alg_results == 0, axis=1)
        alg_results = alg_results[~to_remove, :]

        return alg_results

    def det_results_to_string(self, det_results):

        output_str = ""
        # sort the rows by tracking_id (column 0)
        det_results = det_results[det_results[:, 0].argsort()]

        for row in det_results:

            # buffer is tracking_id, category, confidence
            cat = int(row[1])
            class_name = self.det_labels[cat]
            curr_str = f"ID {int(row[0])}: <b style='color: {COLORS[cat]};'>{class_name}</b>, conf = {row[2]:.3f}<br>"

            output_str += curr_str

        return output_str

    def anom_results_to_string(self, anom_results):

        output_str = ""
        # ignore the first row -- this holds the index for appending
        for row in anom_results[1:,:]:
            # buffer is bbox_id, category (is_anomaly), confidence
            cat = int(row[1])
            color = "red" if cat else "limegreen"
            class_name = "anomaly" if cat else "normal"
            curr_str = f"ID {int(row[0])}: <b style='color: {color};'>{class_name}</b>, conf = {row[2]:.3f}<br>"

            output_str += curr_str

        # remove extra <br>
        return output_str.rstrip("<br>")

    @staticmethod
    def parse_det_labels(label_fn):
        # create a list containing class per category
        with open(label_fn, 'r') as f:
            labels = f.read().splitlines()

        return labels
