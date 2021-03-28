#!/usr/bin/env python3
#-*- coding:utf-8 -*-

# First import the library
import pyrealsense2 as rs
import argparse
import os.path
import datetime
from process import process

# import parameters
from parameters import * 

realtime = True

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, dest="input")
parser.add_argument("-r", "--record", dest="record", action="store_true")
args = parser.parse_args()

if args.input:
    if args.record:
        print("Cannot record .bag files")
        exit()
    realtime = False
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

try:
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    if realtime is False:
        rs.config.enable_device_from_file(config, args.input)
    else :
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        if args.record:
            now = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S.bag")
            config.enable_record_to_file(now)
        
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    process(pipeline)

finally:
    pass
