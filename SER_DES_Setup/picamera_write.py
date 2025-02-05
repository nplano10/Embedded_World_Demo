import time
import sys
sys.path.append('../..')            # Path to the gmsli2c module
import gmsli2c

# GM24 = 0x4E
HS97 = 0x80
HS96 = 0x98
rpi4 = gmsli2c.rpii2c(4)
cam = gmsli2c.GMSLi2c(rpi4)

try:
    cam.regWrite(HS97, 0x00, 0x82)
except:
    HS97 = 0x82
