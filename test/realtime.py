#!/usr/bin/env python3
#-*- coding:utf-8 -*-

# First import the library
import pyrealsense2 as rs

# import parameters
from parameters import *
from process import process 


pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

process(pipeline)