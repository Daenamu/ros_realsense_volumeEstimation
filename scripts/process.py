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

def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_

    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*', 'v', '<', '>', '.', '1', '2', '3', '4']
    isNoise=False

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

color_image = None
depth_image = None

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


# filtering
distances = depth_image.astype(np.float64) * depth_scale

distances = np.where(((distances < LOW_CUT_DEPTH) &\
        (distances > -LOW_CUT_DEPTH)), 0, distances)
distances = np.where(((distances < HIGH_CUT_DEPTH) &\
        (distances > -HIGH_CUT_DEPTH)), distances, float('nan'))


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
z_unit = pixel_by_one_meter / x_unit

# visualization
pointcloud = np.zeros((60*180, 3), dtype=np.float32) # for pcl
point_array = {'x': [], 'y': []} # for DBSCAN
for i in range(60):
    for j in range(180):
        x = int(x_min+x_unit*j)
        y = int(y_min+y_unit*i)
        dist = distances[y][x]

        pointcloud[i*180+j][0] = i
        pointcloud[i*180+j][1] = j
        pointcloud[i*180+j][2] = -dist * z_unit

        if dist != 0 and not math.isnan(dist):
            if dist < 0:
                cv2.circle(color_image, (x, y), 1, (0, 0, 255), 1)    
            else:
                cv2.circle(color_image, (x, y), 1, (255, 0, 0), 1)
            
            point_array['x'].append(x)
            point_array['y'].append(y)

pc = pcl.PointCloud(pointcloud)
pcl.save(pc, 'pc2pcd.pcd')

image = np.hstack((color_image, depth_colormap))
# cv2.namedWindow('Panorama View', cv2.WINDOW_NORMAL)
# cv2.imshow('Panorama View', image)
# cv2.waitKey(0)
cv2.imwrite('Image.jpg', image)

# DBSCAN    
df = pd.DataFrame(point_array)
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
dbscan_labels = dbscan.fit_predict(df)
df['dbscan_cluster'] = dbscan_labels
visualize_cluster_plot(dbscan, df, 'dbscan_cluster', iscenter=False)


# Volume estimating
distance_values = []
for i in range(df.shape[0]):    # 각 point에 distance정보 추가
    distance_values.append(distances[df.loc[i, 'y']][df.loc[i, 'x']])
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
print(volume_data)