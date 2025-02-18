from picamera2 import Picamera2
from picamera2.devices import IMX500
import json


class IMX500NoModel:
    def __init__(self, camera_index):
        self.camera_path = self._select_camera(camera_index)
        self.imx500 = IMX500(network_file = '',camera_id=self.camera_path)
        self.picam2 = Picamera2(self.imx500.camera_num)
        self._setup_camera()

    def _select_camera(self, camera_index: int) -> str:
        cameras = [
            "/base/axi/pcie@120000/rp1/i2c@88000/imx500@1a",
            "/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a",
        ]
        if camera_index < 0 or camera_index >= len(cameras):
            raise ValueError(f"Invalid camera index: {camera_index}. Available cameras: {len(cameras)}")
        return cameras[camera_index]

    def _setup_camera(self) -> None:

        self.picam2.video_configuration.controls.FrameRate = 30.0
        self.picam2.video_configuration.size = (640, 480)
        self.picam2.start("video")
        self.set_camera_config("camera_settings.json")

    def set_camera_config(self,json_file):
            with open(json_file, 'r') as file:
                config = json.load(file)
            self.picam2.set_controls(config["controls"])

