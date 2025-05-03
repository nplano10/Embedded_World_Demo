from picamera2 import Picamera2
from picamera2.devices import IMX500
import json


class IMX500NoModel:
    def __init__(self, camera_index):
        self.picam2 = Picamera2(camera_index)
        self._setup_camera()

    def _setup_camera(self) -> None:
        self.set_camera_config("imx500/camera_settings.json")
        config = self.picam2.create_preview_configuration(controls={"FrameRate": 30}, buffer_count=12)
        self.picam2.start(config, show_preview=False)

    def set_camera_config(self,json_file):
        with open(json_file, 'r') as file:
            config = json.load(file)
        self.picam2.set_controls(config["controls"])

    

