#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import math
import pcl
from parameters import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import time

def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_

    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*', 'v', '<', '>', '.', '1', '2', '3', '4']
    isNoise=False
    plt.figure(figsize=(12, 4))

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)

        plt.scatter(x=label_cluster['x'], y=label_cluster['y'], s=20,\
                    edgecolor='k', marker=markers[label % len(markers)], label=cluster_legend, linewidth=0)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label % len(markers)], linewidth=0)
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                        edgecolor='k', marker='$%d$' % label, linewidth=0)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'

    plt.legend(loc=legend_loc)
    plt.gca().invert_yaxis()
    

    # plt.show()
    plt.savefig('DBSCAN.png')


with open('captures.p', 'rb') as file:
    captures = pickle.load(file)
    depth_scale = pickle.load(file)
    pixel_by_one_meter = pickle.load(file)


moved_pixel = int(INTERVAL * pixel_by_one_meter)
print(pixel_by_one_meter)

color_image = None
depth_image = None

start = time.time()
for capture in captures:
    if color_image is None:
        color_image = np.array(capture[0])
        depth_image = np.array(capture[1])
        continue

    color_paste = np.array(capture[0])
    depth_paste = np.array(capture[1])

    if CAMERA_MODE == 'h':
        color_paste = color_paste[:, WIDTH-WIDTH_SLICE*2-moved_pixel:, :]
        depth_paste = depth_paste[:, WIDTH-WIDTH_SLICE*2-moved_pixel:]

        color_image = np.hstack((color_image, color_paste))
        depth_image = np.hstack((depth_image, depth_paste))
    else:
        color_paste = color_paste[HEIGHT-HEIGHT_SLICE*2-moved_pixel:, :, :]
        depth_paste = depth_paste[HEIGHT-HEIGHT_SLICE*2-moved_pixel:, :]
        # color_paste = color_paste[:moved_pixel, :, :]
        # depth_paste = depth_paste[:moved_pixel, :]

        color_image = np.vstack((color_image, color_paste))
        depth_image = np.vstack((depth_image, depth_paste))



if color_image is not None:
    if CAMERA_MODE == 'v':
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    image = np.hstack((color_image, depth_colormap))
    # cv2.namedWindow('Panorama View', cv2.WINDOW_NORMAL)
    # cv2.imshow('Panorama View', image)
    # key = cv2.waitKey(0)

image = np.hstack((color_image, depth_colormap))
# cv2.namedWindow('Panorama View', cv2.WINDOW_NORMAL)
# cv2.imshow('Panorama View', image)
# cv2.waitKey(0)
cv2.imwrite('Image.jpg', image)

print("Image stitching :", round(time.time() - start, 4), "sec")


# filtering
distances = depth_image.astype(np.float64) * depth_scale

distances = np.where(((distances < LOW_CUT_DEPTH) &\
        (distances > -LOW_CUT_DEPTH)), 0, distances)
distances = np.where(((distances < HIGH_CUT_DEPTH) &\
        (distances > -HIGH_CUT_DEPTH)), distances, float('nan'))

# rectangle detection
start = time.time()

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

print("Rectangle detection :", round(time.time() - start, 4), "sec")

# extract grid data
x_cnt = int(BOX_WIDTH * DENSITY)
y_cnt = int(BOX_HEIGHT * DENSITY)

x_unit = (x_max - x_min) / x_cnt
y_unit = (y_max - y_min) / y_cnt
z_unit = pixel_by_one_meter / x_unit

# visualization
start = time.time()

x_index = np.tile(np.arange(x_cnt), y_cnt)
y_index= np.repeat(np.arange(y_cnt), x_cnt)
dist_temp = distances[(y_index*y_unit+y_min).astype(np.int64), (x_index*x_unit+x_min).astype(np.int64)]
dist = (-dist_temp.flatten() * z_unit)
pointcloud = np.zeros((x_cnt*y_cnt, 3), dtype=np.float32)
pointcloud[:, 0] = x_index
pointcloud[:, 1] = y_index
pointcloud[:, 2] = dist

point_coord = np.where(dist != 0)
point_x = x_index[point_coord]
point_y = y_index[point_coord]
point_array = {'x': list(point_x), 'y': list(point_y)}


"""
pointcloud = np.zeros((x_cnt*y_cnt, 3), dtype=np.float32) # for pcl
point_array = {'x': [], 'y': []} # for DBSCAN
for i in range(y_cnt):
    for j in range(x_cnt):
        x = int(x_min+x_unit*j)
        y = int(y_min+y_unit*i)
        dist = distances[y][x]

        pointcloud[i*x_cnt+j][0] = i
        pointcloud[i*x_cnt+j][1] = j
        pointcloud[i*x_cnt+j][2] = -dist * z_unit

        if dist != 0 and not math.isnan(dist):
            if dist < 0:
                cv2.circle(color_image, (x, y), 1, (0, 0, 255), 1)    
            else:
                cv2.circle(color_image, (x, y), 1, (255, 0, 0), 1)
            
            point_array['x'].append(x)
            point_array['y'].append(y)
"""

pc = pcl.PointCloud(pointcloud)
pcl.save(pc, 'pc2pcd.pcd')

print("Make pointcloud :", round(time.time() - start, 4), "sec")


# DBSCAN    
start = time.time()

df = pd.DataFrame(point_array)
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
dbscan_labels = dbscan.fit_predict(df)
df['dbscan_cluster'] = dbscan_labels
visualize_cluster_plot(dbscan, df, 'dbscan_cluster', iscenter=False)

print("DBSCAN :", round(time.time() - start, 4), "sec")


# Volume estimating
start = time.time()

distance_values = list(distances[(point_y*y_unit+y_min).astype(np.int64), (point_x*x_unit+x_min).astype(np.int64)])
df['distance'] = distance_values


volumes = {'key': [], 'x_width': [], 'y_width': [], 'height': [], 'volume': []}
grouped = df.groupby('dbscan_cluster')
xmin = ymin = 1000
xmax = ymax = 0
height = 0
for key, group in grouped:  # 그룹별로 부피 계산
    xmin = group['x'].min()
    ymin = group['y'].min()
    xmax = group['x'].max()
    ymax = group['y'].max()

    if xmin==xmax or ymin==ymax:
        continue

    x_width = (xmax - xmin) / pixel_by_one_meter # measure rectangle box
    y_width = (ymax - ymin) / pixel_by_one_meter
    height = group['distance'].sum() / ((xmax-xmin) * (ymax-ymin))

    volume = x_width*100*y_width*100*height*100 # convert to cm unit

    volumes['key'].append(key)
    volumes['x_width'].append(x_width)
    volumes['y_width'].append(y_width)
    volumes['height'].append(height)
    volumes['volume'].append(volume)

volume_data = pd.DataFrame(volumes)
# print(volume_data)

print("Volume estimating :", round(time.time() - start, 4), "sec")
