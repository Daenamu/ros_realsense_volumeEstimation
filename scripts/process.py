#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import math
import pcl
from parameters import *



with open('captures.p', 'rb') as file:
    captures = pickle.load(file)
    depth_scale = pickle.load(file)
    distance_entire = pickle.load(file)


pixel_by_one_meter = (WIDTH-WIDTH_SLICE*2) / distance_entire
pixel_by_one_meter = int(pixel_by_one_meter)

moved_pixel = int(INTERVAL * pixel_by_one_meter)

color_image = None
depth_image = None
for capture in captures:
    if color_image is None:
        color_image = np.array(capture[0])
        depth_image = np.array(capture[1])
        continue

    color_paste = np.array(capture[0])
    depth_paste = np.array(capture[1])

    color_paste = color_paste[:, WIDTH-WIDTH_SLICE*2-moved_pixel:, :]
    depth_paste = depth_paste[:, WIDTH-WIDTH_SLICE*2-moved_pixel:]

    color_image = np.hstack((color_image, color_paste))
    depth_image = np.hstack((depth_image, depth_paste))


if color_image is not None:
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    image = np.hstack((color_image, depth_colormap))
    cv2.namedWindow('Panorama View', cv2.WINDOW_NORMAL)
    cv2.imshow('Panorama View', image)
    key = cv2.waitKey(0)


# filtering
distances = depth_image.astype(np.float64) * depth_scale

distances = np.where(((distances < LOW_CUT_DEPTH) &\
        (distances > -LOW_CUT_DEPTH)), 0, distances)
# distances = np.where(((distances < HIGH_CUT_DEPTH) &\
#        (distances > -HIGH_CUT_DEPTH)), distances, float('nan'))


# rectangle detection
dst = cv2.inRange(color_image, (80, 80, 80), (255, 255, 255))
dst = cv2.medianBlur(dst, 5)    # medianBlur

contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find contours
value = list()
for cnt in contours:
    value.append(cnt)
cnt = sorted(value, key=lambda x: len(x))[-1]   # find most numerous contour

epsilon = 0.04 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)   # use approx

cv2.drawContours(color_image, [approx], 0, (0, 255, 0), 1)

x_coord = sorted(approx, key=lambda x: x[0][0])   # find min and max coordinates
y_coord = sorted(approx, key=lambda x: x[0][1])

bias = 5
x_min = x_coord[0][0][0] + bias
x_max = x_coord[-1][0][0] - bias
y_min = y_coord[0][0][1] + bias
y_max = y_coord[-1][0][1] - bias

cv2.circle(color_image, (x_min, y_min), 3, (0, 255, 0), -1)   # draw circle at edges
cv2.circle(color_image, (x_min, y_max), 3, (0, 255, 0), -1)
cv2.circle(color_image, (x_max, y_min), 3, (0, 255, 0), -1)
cv2.circle(color_image, (x_max, y_max), 3, (0, 255, 0), -1)

# extract grid data
x_unit = (x_max - x_min) / 180
y_unit = (y_max - y_min) / 60

# visualization
point_array = np.zeros((60*180, 3), dtype=np.float32) # for pcl
for i in range(60):
    for j in range(180):
        x = int(x_min+x_unit*j)
        y = int(y_min+y_unit*i)
        dist = distances[y][x]

        point_array[i*180+j][0] = i
        point_array[i*180+j][1] = j
        point_array[i*180+j][2] = -dist * 100 # cm unit

        if dist != 0 and not math.isnan(dist):
            if dist < 0:
                cv2.circle(color_image, (x, y), 1, (0, 0, 255), 1)    
            else:
                cv2.circle(color_image, (x, y), 1, (255, 0, 0), 1)

pc = pcl.PointCloud(point_array)
pcl.save(pc, 'pc2pcd.pcd')

image = np.hstack((color_image, depth_colormap))
cv2.namedWindow('Panorama View', cv2.WINDOW_NORMAL)
cv2.imshow('Panorama View', image)
key = cv2.waitKey(0)
cv2.imwrite('Image.jpg', image)
