import time
import sys
sys.path.append('../..')            # Path to the gmsli2c module
import gmsli2c




HS96 = 0x98
HS97 = 0x84
rpi4 = gmsli2c.rpii2c(4)
cam = gmsli2c.GMSLi2c(rpi4)
cam.LOG = True  # Enable logging

cam.regWrite(HS96, 0x10, 0x80)
time.sleep(0.2)


cam.regWrite(HS97, 0x10, 0x80)
time.sleep(0.2)



#cam.parseFile('MAX96717+MAX96716A_tunnel.cpp')
cam.parseFile('test.cpp')

val = cam.regRead(HS97, 0x4D)
print(val)
time.sleep(0.2)


val = cam.regRead(HS97, 0x4C)
print(val)
time.sleep(0.2)







HS96 = 0x98
HS97 = 0x84
rpi4 = gmsli2c.rpii2c(6)
cam = gmsli2c.GMSLi2c(rpi4)
cam.LOG = True  # Enable logging

cam.regWrite(HS96, 0x10, 0x80)
time.sleep(0.2)


cam.regWrite(HS97, 0x10, 0x80)
time.sleep(0.2)



#cam.parseFile('MAX96717+MAX96716A_tunnel.cpp')
cam.parseFile('test.cpp')

val = cam.regRead(HS97, 0x4D)
print(val)
time.sleep(0.2)


val = cam.regRead(HS97, 0x4C)
print(val)
time.sleep(0.2)



