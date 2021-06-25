#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from parameters import *

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


# rs.config.enable_device_from_file(config, "/home/daeuk/catkin_ws/src/realsense_depth/bag/2021-04-01-03-03-20.bag")
# config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

"""
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
"""

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

captures = []
count = 0

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Get distance from each pixels
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))


        # Visualization and Capture
        cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera View', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            print("quit...")
            break

        if key & 0xFF == ord('y'):
            captures.append([color_image, depth_image, depth_colormap])
            print(f"capture{count} has been saved")
            count += 1

    
    # Save captures
    with open('captures2.p', 'wb') as file:
        pickle.dump(captures, file)

finally:
    # Stop streaming
    pipeline.stop()

