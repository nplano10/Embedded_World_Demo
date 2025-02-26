import numpy as np
from multiprocessing import shared_memory
from picamera2 import Picamera2
import argparse
from multiprocessing import Process, Queue
from sony_code.imx500_object_detection_SORT import IMX500Detector, Detection
from sony_code.imx500_anomaly_detection import IMX500AnomalyDetector
from sony_code.imx500_object_detection_demo import IMX500ObjectDetector
from sony_code.imx500_no_model import IMX500NoModel
import sony_code.imx500_object_detection_demo as ob_det
import time
from enum import Enum
from multiprocessing import shared_memory
from x_utils import CameraShm

DET_MODEL = "sony_code/Models/detection-imx500-newlight/network.rpk"
DET_LABEL = "sony_code/Models/detection-imx500-newlight/labels.txt"
ANOM_MODEL = "sony_code/Models/anomaly-imx500-newlight/network.rpk"


class Model(Enum):
    NOMODEL = 1
    TRAINED = 2


def anomaly_process(
    event,
    bbox_queue,
    results_queue,
    args,
    camera_shm: CameraShm,
):

    detector = IMX500AnomalyDetector(args)
    detector.picam2.pre_callback = lambda req: detector.process_frame(
        req, bbox_queue, results_queue, camera_shm)
    print("Done loading model")

    # Attach to shared memory block
    im_shm = shared_memory.SharedMemory(name=camera_shm.im_shm.name)
    image = np.ndarray(camera_shm.im_shape, dtype=np.uint8, buffer=im_shm.buf)

    while not event.is_set():
        time.sleep(1 / args.fps)  # TODO: running at 20 fps or 30?
        with camera_shm.im_lock:
            image[:] = detector.picam2.capture_array().astype(np.uint8)[:, :, :3]


def detection_process(event, bbox_queue, results_queue, args, camera_shm: CameraShm):

    detector = IMX500Detector(args)
    detector.picam2.pre_callback = lambda req: detector.draw_detections(req, results_queue)

    # Attach to shared memory block
    im_shm = shared_memory.SharedMemory(name=camera_shm.im_shm.name)
    image = np.ndarray(
        camera_shm.im_shape, dtype=np.uint8, buffer=im_shm.buf
    )

    alg_shm = shared_memory.SharedMemory(name=camera_shm.alg_shm.name)
    results = np.ndarray(camera_shm.alg_shape, dtype=np.float16, buffer=alg_shm.buf)

    while not event.is_set():
        time.sleep(1/args.fps)

        # Postprocess the results
        metadata = detector.picam2.capture_metadata()
        detector.last_results = detector.parse_detections(
            metadata, args.iou, args.max_detections, args.threshold
        )
        detector.update_bbox_queue(bbox_queue)

        # Update image in shared memory
        with camera_shm.im_lock:
            image[:] = detector.picam2.capture_array().astype(np.uint8)[:, :, :3]

        # Update detection results in shared memory
        with camera_shm.alg_lock:
            for ind, detection in enumerate(detector.last_results):

                curr_results = [
                    detection.tracking_id,
                    detection.category,
                    detection.conf
                ]

                results[ind, :] = np.array(curr_results, dtype=np.float16)


def no_model_process(
    event,
    camera_one_shm: CameraShm,
    camera_two_shm: CameraShm,
):

    imx500_1 = IMX500NoModel(0)
    imx500_2 = IMX500NoModel(1)
    one_shm = shared_memory.SharedMemory(name=camera_one_shm.im_shm.name)
    one_image = np.ndarray(camera_one_shm.im_shape, dtype=np.uint8, buffer=one_shm.buf)
    two_shm = shared_memory.SharedMemory(name=camera_two_shm.im_shm.name)
    two_image = np.ndarray(camera_two_shm.im_shape, dtype=np.uint8, buffer=two_shm.buf)

    while not event.is_set():
        time.sleep(1/30)  # TODO: running at 20 fps or 30?
        with camera_one_shm.im_lock:
            one_image[:] = imx500_1.picam2.capture_array().astype(np.uint8)[:, :, :3]
        with camera_two_shm.im_lock:
            two_image[:] =imx500_2.picam2.capture_array().astype(np.uint8)[:, :, :3]


# def obj_detection_process(
#     event,
#     mode,
#     camera_one_shm: CameraShm,
#     camera_two_shm: CameraShm,
# ):

#     one_shm = shared_memory.SharedMemory(name=camera_one_shm.shm.name)
#     one_image = np.ndarray(camera_one_shm.shape, dtype=np.uint8, buffer=one_shm.buf)
#     two_shm = shared_memory.SharedMemory(name=camera_two_shm.shm.name)
#     two_image = np.ndarray(camera_two_shm.shape, dtype=np.uint8, buffer=two_shm.buf)

#     if(mode == 0):
#         camera1 = IMX500ObjectDetector(ob_det.sony_args(),1)
#         camera2 = IMX500ObjectDetector(ob_det.sony_args(),0)
#     if(mode ==1 ):
#         camera1 = IMX500ObjectDetector(ob_det.sue_args(),1)
#         camera2 = IMX500ObjectDetector(ob_det.sue_args(),0)
#     camera1.picam2.pre_callback = camera1.draw_detections
#     camera2.picam2.pre_callback = camera2.draw_detections
#     while not event.is_set():
#         time.sleep(1/30)  # TODO: running at 20 fps or 30?
#         with camera_one_shm.lock:  # Ensure exclusive access to the shared memory
#             meta_data = camera1.picam2.capture_metadata()
#             camera1.last_results = camera1.parse_detections(meta_data)
#             one_image[:] = camera1.picam2.capture_array().astype(np.uint8)[:, :, :3]
#         with camera_two_shm.lock:
#             camera2.last_results = camera2.parse_detections(camera2.picam2.capture_metadata())
#             two_image[:] =camera2.picam2.capture_array().astype(np.uint8)[:, :, :3]


def update_imx500_shm(
    selected_model,
    event,
    det_camera_shm: CameraShm,
    anom_camera_shm: CameraShm,
):

    CAMERA_DISTANCE_MM = 40  # Physical distance between cameras in mm
    CAMERA_DISTANCE_PIXELS = -160  # Distance in pixels
    PIXELS_PER_MM = CAMERA_DISTANCE_PIXELS / CAMERA_DISTANCE_MM
    DETECTION_REGION = [20, 0, 600, 410]  # x, y, w, h

    bbox_queue = Queue(maxsize=50)  # Queue for passing bounding boxes
    results_queue = Queue()  # Queue for receiving classification results

    pill_detection_args = argparse.Namespace(
        model=DET_MODEL,
        labels=DET_LABEL,
        camera_index=0,
        fps=25,
        max_disappeared=20,
        iou=0.65,
        threshold=0.5,
        max_detections=10,
        pixels_per_mm=PIXELS_PER_MM,
        detection_region=DETECTION_REGION
    )
    anomaly_detection_args = argparse.Namespace(
        model=ANOM_MODEL,
        camera_index=1,
        fps=20,
        image_threshold=0.435,
        pixel_threshold=0.30,
        constant_offset_in_pixel=CAMERA_DISTANCE_PIXELS,
        roi_box_size=90,
        pixels_per_mm=PIXELS_PER_MM
    )

    pill_detection_proc = Process(
        target=detection_process,
        args=(
            event,
            bbox_queue,
            results_queue,
            pill_detection_args,
            det_camera_shm,
        ),
    )
    anomaly_detection_proc = Process(
        target=anomaly_process,
        args=(
            event,
            bbox_queue,
            results_queue,
            anomaly_detection_args,
            anom_camera_shm,
        ),
    )

    if selected_model == Model.TRAINED:
        pill_detection_proc.start()
        anomaly_detection_proc.start()
        pill_detection_proc.join()
        anomaly_detection_proc.join()

    if selected_model == Model.NOMODEL:
        no_model_process(event, det_camera_shm, anom_camera_shm)
