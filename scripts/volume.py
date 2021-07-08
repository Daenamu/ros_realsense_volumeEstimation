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
    divided_data = [[[] for row in range(120)] for col in range(60)]

    converted_pcd = convert_RCS_to_ACS(x, y, z, pcd)

    for i, k in enumerate(converted_pcd.points):
        x = math.floor(k[0]*100)
        y = math.floor(k[1]*100)

        if x < 0 or y < 0 or x > 119 or y > 59:
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

    volume = avg_depth * 10 * 10

    return volume

def get_volume_of_anomaly(x, y, z, pcd, model):
    volume_data = np.zeros((60, 120))
    divided_data = divide_by_grid(x, y, z, pcd)

    distance_pcd = get_distance_from_plane(pcd, model)

    for i in range(60):
        for j in range(120):
            if not divided_data[i][j]: # empty list
                volume_data[i][j] = 0
            else: 
                volume_data[i][j] = get_volume_of_one_grid(distance_pcd, divided_data[i][j])

    return volume_data

def print_volume_data(data):
    count = 0
    for i in range(60):
        for j in range(120):
            if data[i][j] != 0:
                print((i, j), data[i][j], end='mm^3\n')
                count += 1
            if count == 20:
                return


hill_cloud = o3d.io.read_point_cloud('data/hill_cloud.pcd') # only hill outlier
hole_cloud = o3d.io.read_point_cloud('data/hole_cloud.pcd') # only hole outlier
inlier_cloud = o3d.io.read_point_cloud('data/inlier_cloud.pcd') # entire inlier
outlier_cloud = o3d.io.read_point_cloud('data/outlier_cloud.pcd') # entier outlier
test5_crop = o3d.io.read_point_cloud('data/test5_crop.pcd')

"""
o3d.visualization.draw_geometries([hill_cloud])
o3d.visualization.draw_geometries([hole_cloud])
o3d.visualization.draw_geometries([inlier_cloud])
o3d.visualization.draw_geometries([outlier_cloud])
o3d.visualization.draw_geometries([test5_crop])
"""

with open('data/hill_labels.pickle', 'rb') as f:
    hill_labels = pickle.load(f)

with open('data/hole_labels.pickle', 'rb') as f:
    hole_labels = pickle.load(f)

with open('data/plane_model.pickle', 'rb') as f:
    plane_model = pickle.load(f)

"""
distance_pcd = get_distance_from_plane(test5_crop, plane_model)
o3d.visualization.draw_geometries([distance_pcd])
"""


hill_volume_data = get_volume_of_anomaly(0.6, 0.3, 0, hill_cloud, plane_model)
hole_volume_data = get_volume_of_anomaly(0.6, 0.3, 0, hole_cloud, plane_model)

print_volume_data(hill_volume_data)
print_volume_data(hole_volume_data)