#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from parameters import *
import math

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

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will apply low / high cut filter
LOW_CUT_DEPTH = LOW_CUT_DEPTH / 100 / depth_scale
HIGH_CUT_DEPTH = HIGH_CUT_DEPTH / 100 / depth_scale


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

captures = []
count = 0

def calculate_distance(depth_frame, intrin, x1, y1, x2, y2):
    udist = depth_frame.get_distance(x1, y1)
    vdist = depth_frame.get_distance(x2, y2)
    # print(udist, vdist)

    point1 = rs.rs2_deproject_pixel_to_point(intrin, [x1, y1], udist)
    point2 = rs.rs2_deproject_pixel_to_point(intrin, [x2, y2], vdist)
    # print(point1, point2)

    dist = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)) # Ignore z point

    return dist

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


        # Get pixels per 1 meter
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics 

        if CAMERA_MODE == 'h': # horizontal
            distance_entire = calculate_distance(aligned_depth_frame, color_intrin, WIDTH_SLICE, HEIGHT//2, WIDTH-WIDTH_SLICE, HEIGHT//2)
            pixel_by_one_meter = (WIDTH-WIDTH_SLICE*2) / distance_entire
        
        else: # vertical
            distance_entire = calculate_distance(aligned_depth_frame, color_intrin, WIDTH//2, HEIGHT_SLICE, WIDTH//2, HEIGHT-HEIGHT_SLICE)
            pixel_by_one_meter = (HEIGHT-HEIGHT_SLICE*2) / distance_entire

        pixel_by_one_meter = int(pixel_by_one_meter)
        # print(distance_entire)
        # print(f"pixel_by_one_meter is {pixel_by_one_meter} ")


        # Get distance from each pixels
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # print(depth_image_temp.shape)
        # print(f"The camera is facing on object {depth_image_temp[240][320] * depth_scale} m")


        # Slice out edge pixels
        depth_image = depth_image[HEIGHT_SLICE:HEIGHT-HEIGHT_SLICE, WIDTH_SLICE:WIDTH-WIDTH_SLICE]
        color_image = color_image[HEIGHT_SLICE:HEIGHT-HEIGHT_SLICE, WIDTH_SLICE:WIDTH-WIDTH_SLICE,:]
        # print(depth_image.shape, color_image.shape)


        # Get threshold
        hist, bins = np.histogram(depth_image.flatten(), np.arange(0.3 / depth_scale, 1 / depth_scale, 0.01 / depth_scale)) # 1cm 단위로 최빈값 구함
        min_bin = np.argmax(hist)
        min_thes = (bins[min_bin]+bins[min_bin+1]) / 2
        # print(f"threshold: {min_thes * depth_scale * 100} cm")
        depth_image = depth_image - min_thes


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
            captures.append([color_image, depth_image])
            print(f"capture{count} has been saved")
            count += 1

    
    # Save captures
    with open('captures.p', 'wb') as file:
        pickle.dump(captures, file)
        pickle.dump(depth_scale, file)
        pickle.dump(pixel_by_one_meter, file)
finally:
    # Stop streaming
    pipeline.stop()

