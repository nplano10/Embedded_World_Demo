
import smbus
import time
from time import sleep








class ADXL359:
    def __init__(self, bus_num=1, device_addr=0x1D,serial_communication_protocol='I2C'):
        """
        Initialize the I2C bus and the ADXL359 sensor.
        
        :param bus_num: I2C bus number (default is 1 for Raspberry Pi)
        :param device_addr: I2C address of ADXL359 (default is 0x1D)
        """
        self.serial_communication_protocol = serial_communication_protocol
        self.bus = smbus.SMBus(bus_num)
        self.addr = device_addr

        self.g_min = -10
        self.g_max =10
        self.min_raw_acc =-32768
        self.max_raw_acc = 32767

        self.temp_min =-40
        self.tem_max = 125 

        # ADXL359 register addresses (based on the datasheet)
        self.sample_rate= 1000 
        self.DEVID_AD = 0x00
        self.POWER_CTL = 0x2D
        self.FILTER = 0x28
        self.FILTER_OP = 0x12
        self.OP_MODE=0x00
        self.TEMP_AND_ACC_START_ADDRESS =0x06
        self.X_DATA = 0x08
        self.Y_DATA = 0x0B
        self.Z_DATA = 0x0E
        self.RESET = 0x2F
        self.RESET_CODE = 0x52
        self.STATUS = 0x04

    def _write_register(self, reg, value):
        """Write a byte to a register."""
        self.bus.write_byte_data(self.addr, reg, value)

        
    def _read_register(self, reg):
        """Read a byte from a register."""
        return self.bus.read_byte_data(self.addr, reg)

    
    def _read_registers(self, start_register, length):
        """Read multiple bytes from a register."""
        return self.bus.read_i2c_block_data(self.addr, start_register, length)
    def _initialize(self):
        """
        Initialize the ADXL359 sensor (e.g., power on, data format setup).
        This function can be expanded based on your specific needs.
        """
        self.reset()
        self.set_filter(self.FILTER_OP)  
        self.set_op_mode(self.OP_MODE)
        

    def get_data_ready(self):
        status = self._read_registers(self.STATUS, 1)
        return status[0] & 0x01
    
    
    def get_device_id(self):
        """
        Read the device ID from the DEVID_AD register.
        """
        return self._read_register(self.DEVID_AD)


    def set_filter(self, filter_op):
        """
        Set the filter operation mode of the ADXL359 sensor.

        """
        self._write_register(self.FILTER, filter_op)
        time.sleep(0.5) 
    
    def set_op_mode(self, op_mode):
        """
        Set the operation mode of the ADXL359 sensor.
        """
        self._write_register(self.POWER_CTL, op_mode)
        time.sleep(0.5) 

    def reset(self):
        """
        Reset the ADXL359 sensor by writing to the power control register.
        """
        self._write_register(self.RESET,self.RESET_CODE)
        time.sleep(0.5)  # Delay for reset



    def read_raw_temp_and_acceleration_data(self):
        """
        Read the acceleration data from all three axes.
        
        :return: Tuple of (x, y, z) acceleration in g.
        """
        raw_data = self._read_registers(self.TEMP_AND_ACC_START_ADDRESS, 11)

        return raw_data
    

    def raw_acc_to_int16(self,raw):
        return np.int16(np.uint16(raw))

    
    def raw_temp_to_int8(self,raw):
        return np.int8(np.uint8(raw))

    def acceleration_bytes_to_g(self,bytes):
    
        g_value = (self.raw_acc_to_int16((bytes[0]<<8)|bytes[1])-self.min_raw_acc)/(self.max_raw_acc-self.min_raw_acc)*(self.g_max-self.g_min)+self.g_min
        return g_value


        
    def temp_bytes_to_celcius(self,bytes):
        temp = ((((bytes[0]<<8)|bytes[1])-1885)/-9.05)+25

        return temp

    
    def raw_data_to_arrays(self,raw_data):
        x_data = []
        y_data = []
        z_data = []
        temp_data = []
        for i in range(int(len(raw_data)/11)):
            start_index=i*11
            x_data.append(self.acceleration_bytes_to_g(raw_data[start_index+2:start_index+4]))
            y_data.append(self.acceleration_bytes_to_g(raw_data[start_index+5:start_index+7]))
            z_data.append(self.acceleration_bytes_to_g(raw_data[start_index+8:start_index+10]))
            temp_data.append(self.temp_bytes_to_celcius(raw_data[start_index:start_index+2]))
        return x_data,y_data,z_data,temp_data
            
    def collect_data(self, num_samples=1000):
        """
        Collect a number of samples from the ADXL359 sensor. Using polling method.
        
        :param num_samples: Number of samples to collect.
        :return: List of (x, y, z) acceleration data.
        """
        raw_data_list =[]       
        num_sampled =0
        while num_sampled < (num_samples):
            data_ready =self.get_data_ready()
            if(data_ready==1):
                raw_data_list = raw_data_list+self.read_raw_temp_and_acceleration_data()
                num_sampled = num_sampled+1

        
    
        return self.raw_data_to_arrays(raw_data_list)



import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    adxl359 = ADXL359()
    adxl359._initialize()
    time.sleep(3)
    while(1):
        # x_data,y_data,z_data,temp_data =adxl359.collect_data()
        # id = adxl359.get_device_id()
        # id =adxl359.read_raw_temp_and_acceleration_data()
        
        print(id)
        bytes = adxl359.read_raw_temp_and_acceleration_data()
        print(bytes)
        ret = adxl359.temp_bytes_to_celcius(bytes)
        
        print(ret)
        time.sleep(0.3)



    # x_data,y_data,z_data,temp_data =adxl359.collect_data()
    # plt.plot(x_data)
    # plt.show()
    # plt.plot(y_data)
    # plt.show()
    # plt.plot(z_data)
    # plt.show()
    # plt.plot(temp_data)
    # plt.show()


 


 
