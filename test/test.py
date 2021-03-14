#!/usr/bin/env python3
#-*- coding:utf-8 -*-

print("Python compile is completed.")


# First import the library
import pyrealsense2 as rs
import numpy
import cv2

# import parameters
from parameters import * 

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

avg_frame = [0]*WIDTH*HEIGHT
min = 10000
max = 0
num_of_samples = 0

try:
    # Averaging distances of each pixels
    while num_of_samples < AVG_FRAME:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        num_of_samples += 1


        for y in range(HEIGHT):
            for x in range(WIDTH):
                dist = depth.get_distance(x, y)
                avg_frame[x*HEIGHT + y] += dist
                if min > dist:
                    min = dist
                if max < dist:
                    max = dist

    # print last image
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = numpy.asanyarray(depth_frame.get_data())
    color_image = numpy.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = numpy.hstack((resized_color_image, depth_colormap))
    else:
        images = numpy.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)


    dist_range = max - min
    
    for y in range(HEIGHT):
        for x in range(WIDTH):
            avg_frame[x*HEIGHT + y] /= AVG_FRAME

    print(f"\nThe camera is facing an object {avg_frame[WIDTH*HEIGHT//2 + HEIGHT//2]} meters away.")
    print(f"min dist : {min}")
    print(f"max dist : {max}\n")

    # Get threshold
    append = 1
    while append > THES_ACC:

        print(f"(Getting threshold) current acc: {append} meters")
        for thes in numpy.arange(min, max, append):
            var = 0
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    error = avg_frame[x*HEIGHT + y] - thes
                    var += error*error
                
            var = var / WIDTH*HEIGHT

            if thes == min or min_var > var:
                min_var = var
                min_thes = thes

        min = min_thes - append
        if min < 0:
            min = 0

        max = min_thes + append
        append /= 10

    print(f"\nthreshold: {min_thes} meters")

    # filtering
    for y in range(HEIGHT):
        for x in range(WIDTH):
            avg_frame[x*HEIGHT + y] -= min_thes
            if avg_frame[x*HEIGHT + y] < IGN_DEPTH:
                avg_frame[x*HEIGHT + y] = 0

    # convert to image
    img = numpy.zeros((HEIGHT, WIDTH))

    for y in range(HEIGHT):
        for x in range(WIDTH):
            img[y][x] = avg_frame[x*HEIGHT + y] * 256 // dist_range
    
    cv2.imshow('sample', img)
    cv2.waitKey(0)
    


finally:
    pipeline.stop()