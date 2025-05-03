#!/bin/bash

sudo dtoverlay imx500-pi5, cam0=1
sudo dtoverlay imx500-pi5
cd SER_DES_Setup
python setup_gmsl.py
