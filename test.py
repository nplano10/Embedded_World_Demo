#!/usr/bin/python3

# Example of setting controls using the "direct" attribute method.

import time
import numpy as np
import cv2
from picamera2 import Picamera2, Preview
from picamera2.controls import Controls
from PIL import Image

picam2 = Picamera2()
picam2.start()

picam2.set_controls({
    'ExposureTime': 10000,  # Example control (in microseconds)
    'Saturation': 16.0,      # Example control
})
time.sleep(1)

frame = picam2.capture_array()
frame = Image.fromarray(frame)
frame.save("1.png", "PNG")
time.sleep(1)

picam2.set_controls({
    'ExposureTime': 10000,  # Example control (in microseconds)
    'Saturation': 1.0,      # Example control
})
time.sleep(1)
frame = picam2.capture_array()
frame = Image.fromarray(frame)
frame.save("2.png", "PNG")
time.sleep(2)

picam2.set_controls({
    'ExposureTime': 10000,  # Example control (in microseconds)
    'Saturation': 7.0,      # Example control
})
time.sleep(1)
frame = picam2.capture_array()
frame = Image.fromarray(frame)
frame.save("3.png", "PNG")
time.sleep(2)
