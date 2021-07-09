#!/usr/bin/env python3

import open3d as o3d
import pickle
import copy
import math
import numpy as np
from parameters import *

def convert_RCS_to_ACS(x, y, z, pcd):
    return copy.deepcopy(pcd).translate((x, y, z))


def divide_by_grid(x, y, z, pcd):
    divided_data = [[[] for row in range(n_WIDTH)] for col in range(n_HEIGHT)]

    converted_pcd = convert_RCS_to_ACS(x, y, z, pcd)

    for i, k in enumerate(converted_pcd.points):
        x = math.floor(k[0]*100)
        y = math.floor(k[1]*100)

        if x < 0 or y < 0 or x > n_WIDTH - 1 or y > n_HEIGHT - 1:
            continue

        divided_data[y][x].append(i)

    return divided_data

def get_distance_from_plane(pcd, model):
    distance_pcd = copy.deepcopy(pcd)
    
    a = model[0]
    b = model[1]
    c = model[2]
    d = model[3]

    for point in distance_pcd.points:
        z = -a/c*point[0]-b/c*point[1]-d/c
        dist = point[2] - z
        point[2] = dist

    return distance_pcd

def get_volume_of_one_grid(pcd, ind):
    volume = 0

    n = len(ind)
    avg_depth = 0

    inlier = pcd.select_by_index(ind)

    for point in inlier.points:
        avg_depth += point[2]

    avg_depth = avg_depth * 1000 / n # mm unit

    volume = avg_depth * SAMPLING_UNIT * SAMPLING_UNIT * 1000 * 1000

    return volume

def get_volume_of_anomaly(x, y, z, pcd, model):
    volume_data = np.zeros((n_HEIGHT, n_WIDTH))
    divided_data = divide_by_grid(x, y, z, pcd)

    distance_pcd = get_distance_from_plane(pcd, model)

    for i in range(n_HEIGHT):
        for j in range(n_WIDTH):
            if not divided_data[i][j]: # empty list
                volume_data[i][j] = 0
            else: 
                volume_data[i][j] = get_volume_of_one_grid(distance_pcd, divided_data[i][j])

    return volume_data

def print_volume_data(data, n=20, entire=False):
    count = 0
    for j in range(n_WIDTH):
        for i in range(n_HEIGHT):
            if data[i][j] != 0:
                print((j, i), data[i][j], end='mm^3\n')
                count += 1
            if entire==False and count == n:
                return

def get_volume_of_cluster(data):
    volume = 0
    x = 0
    y = 0
    count = 0

    for j in range(n_WIDTH):
        for i in range(n_HEIGHT):
            if data[i][j] != 0:
                volume += data[i][j]
                count += 1
                x += j
                y += i
    
    if count != 0:
        x /= count
        y /= count
    return (int(x), int(y)), volume


hill_cloud = o3d.io.read_point_cloud('no/hill_cloud.pcd') # only hill outlier
hole_cloud = o3d.io.read_point_cloud('no/hole_cloud.pcd') # only hole outlier
# inlier_cloud = o3d.io.read_point_cloud('no/inlier_cloud.pcd') # entire inlier
# outlier_cloud = o3d.io.read_point_cloud('no/outlier_cloud.pcd') # entier outlier
# test5_crop = o3d.io.read_point_cloud('no/test5_crop_no.pcd')

# o3d.visualization.draw_geometries([hill_cloud])
# o3d.visualization.draw_geometries([hole_cloud])
# o3d.visualization.draw_geometries([inlier_cloud])
# o3d.visualization.draw_geometries([outlier_cloud])
# o3d.visualization.draw_geometries([test5_crop])


with open('no/hill_labels.pickle', 'rb') as f:
    hill_labels = pickle.load(f)
   
with open('no/hole_labels.pickle', 'rb') as f:
    hole_labels = pickle.load(f)

with open('no/plane_model.pickle', 'rb') as f:
    plane_model = pickle.load(f)

"""
# get origin coordinates temporarily
min_x=1000
min_y=1000
for k in test5_crop.points:
    if min_x > k[0]:
        min_x=k[0]
    if min_y > k[1]:
        min_y=k[1]
print(min_x, min_y)
"""

n_WIDTH = int(BOX_WIDTH / GRID_UNIT)
n_HEIGHT = int(BOX_HEIGHT / GRID_UNIT)

# clusturing 
for i in range(hill_labels.max() + 1):
    index = np.where(hill_labels == i)[0]
    temp_cloud = hill_cloud.select_by_index(list(index))

    volume_data = get_volume_of_anomaly(0.87, 0.26, 0, temp_cloud, plane_model)

    centroid, volume = get_volume_of_cluster(volume_data)

    print(i, centroid, volume, end="mm^3\n")


"""
# not clusturing

hill_volume_data = get_volume_of_anomaly(0.87, 0.26, 0, hill_cloud, plane_model)
hole_volume_data = get_volume_of_anomaly(0.87, 0.26, 0, hole_cloud, plane_model)

print_volume_data(hill_volume_data, entire=True)
print_volume_data(hole_volume_data, entire=True)
"""