#!/usr/bin/env python3
#-*- coding:utf-8 -*-

# First import the library
import pyrealsense2 as rs
import numpy
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# import parameters
from parameters import * 

def visualize_histogram(minimum, maximum, append, frame):
    bins = numpy.arange(minimum, maximum, append)
    weight = frame
    hist, bins = numpy.histogram(weight, bins)
    plt.hist(weight, bins)
    plt.xlabel('distance(mm)')
    plt.show()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # Get image
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_image = numpy.asanyarray(depth_frame.get_data())
    color_image = numpy.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = numpy.hstack((resized_color_image, depth_colormap))
    else:
        images = numpy.hstack((color_image, depth_colormap))


    # Averaging distances of each pixels
    print("Calculating average of distance of each pixels...")

    avg_frame = numpy.zeros((HEIGHT, WIDTH))
    num_of_samples = 0
    
    while num_of_samples < AVG_FRAME:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        num_of_samples += 1

        for y in range(HEIGHT):
            for x in range(WIDTH):
                dist = depth.get_distance(x, y)
                avg_frame[y][x] += dist

    # Averaging
    avg_frame = avg_frame / AVG_FRAME * 1000
    print(f"The camera is facing an object {avg_frame[HEIGHT//2][WIDTH//2]} mm away.\n")

    # Get threshold
    print("Getting threshold...")
    hist, bins = numpy.histogram(avg_frame.flatten(), numpy.arange(0, 1000, 5)) # 5mm 단위로 최빈값 구함
    min_bin = numpy.argmax(hist)
    min_thes = (bins[min_bin]+bins[min_bin+1]) / 2

    print(f"threshold: {min_thes} mm\n")
    

    # filtering
    avg_frame = numpy.subtract(avg_frame, min_thes)

    # show +- 0.5m range from threshold
    visualize_histogram(-500, 500, 1, avg_frame.flatten())

    # convert to points
    point_array = {'x': [], 'y': []}

    for y in range(HEIGHT):
        for x in range(WIDTH):
            if abs(avg_frame[y][x]) > LOW_CUT_DEPTH and abs(avg_frame[y][x]) < HIGH_CUT_DEPTH:
                point_array['x'].append(x)
                point_array['y'].append(y)

    plt.scatter(point_array['x'], point_array['y'], s=0.05)
    plt.show()


    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(0)

finally:
    pipeline.stop()
