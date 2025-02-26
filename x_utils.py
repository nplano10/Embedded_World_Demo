from multiprocessing import shared_memory, Lock
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraShm:
    im_shape: list[int]
    alg_shape: list[int]

    def __post_init__(self):
        # Create shared memory and lock for camera image
        self.im_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.im_shape) * np.uint8().itemsize
        )
        self.im_lock = Lock()

        # Create shared memory and lock for algorithm output
        self.alg_shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.im_shape) * np.float16().itemsize
        )
        self.alg_lock = Lock()
