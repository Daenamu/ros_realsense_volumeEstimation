#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import pyrealsense2 as rs
import datetime
from parameters import *

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

now = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S.bag")
config.enable_record_to_file(now)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
finally:
    pipeline.stop()