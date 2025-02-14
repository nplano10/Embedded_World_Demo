import time
import sys
sys.path.append('../..')            # Path to the gmsli2c module
import gmsli2c

HS96 = 0x98
HS97 = 0x80
rpi4 = gmsli2c.rpii2c(6)
cam = gmsli2c.GMSLi2c(rpi4)
cam.LOG = True  # Enable logging

#cam.regWrite(HS96, 0x10, 0x31)
#time.sleep(0.2)
#val=cam.regRead(HS96,0x10)
#print(hex(val))


#cam.regWrite(HS97, 0x10, 0x91)
#time.sleep(0.2)

cam.parseFile('MAX96717+MAX96716A_tunnel.cpp')

# MAX96717 Setup
# cam.regWrite(HS97, 0x110, 0x6C)     # PCLK_DET bypass
# cam.regWrite(HS97, 0x308, 0xE4)     # PCLK_DET bypass
# cam.regWrite(HS97, 0x330, 0x40)     # Enable non-continuous clock

# cam.regWrite(HS97, 0x2bf, 0xA0)     # Output is push-pull
# cam.regWrite(HS97, 0x2be, 0x00)     # Drive output low
# time.sleep(0.1)
# cam.regWrite(HS97, 0x2be, 0x10)     # Keep MFP0 high to enable Picamera2
# time.sleep(0.1)


# print('Serializer dev_id = ' + hex(cam.regRead(HS97, 0xD)))
# print('Deserializer dev_id = ' + hex(cam.regRead(HS96, 0xD)))
