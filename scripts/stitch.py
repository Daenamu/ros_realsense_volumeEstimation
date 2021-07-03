#!/usr/bin/env python3

import open3d as o3d
import os
from parameters import *
import copy
import math


def degree_to_radian(degree):
    return degree * math.pi / 180

def get_coordinates_to_move_for_stitching(width, height, FOV_horizontal, FOV_vertical, depth, length_of_overlap=0.1):
    # distance * tan(FOV_horizontal / 2) * 2 >= height
    min_distance = height / 2 / math.tan(degree_to_radian(FOV_horizontal / 2))

    min_distance = math.ceil(min_distance * 10) / 10 # round up to second decimal place 

    height_of_image = min_distance * math.tan(degree_to_radian(FOV_vertical / 2)) * 2

    # h + (h - length_of_overlap) * n >= width
    number_of_captures = (width - height_of_image) / (height_of_image - length_of_overlap)
    number_of_captures = math.ceil(number_of_captures) + 1 # for safe stitching, add one more.

    interval = width / (number_of_captures + 1)
    
    coordinates = []
    y_coord = height / 2
    z_coord = depth - min_distance
    for i in range(number_of_captures):
        coord = ((interval * (i+1)), y_coord, z_coord)
        coordinates.append(coord)

    return interval, coordinates

def stitching_pointcloud(pcds, interval):
    combined_pcd = o3d.geometry.PointCloud()
    count = 0
    for point_id in range(len(pcds)):
        pcd_tx = copy.deepcopy(pcds[point_id]).translate((0, interval * count, 0))
        combined_pcd += pcd_tx
        count += 1

    return combined_pcd


def load_point_clouds():
    pcds = []

    filepaths = []
    for file in os.scandir('captures/'):
        filepaths.append(file.path)
        filepaths.sort()

    for filepath in filepaths:
        pcd = o3d.io.read_point_cloud(filepath)
        pcds.append(pcd)

    return pcds

"""
interval, coordinates = get_coordinates_to_move_for_stitching(1.2, 0.6, 90, 65, 0.8)
print(interval, coordinates)
"""

pcds = load_point_clouds()
pcd_combined = stitching_pointcloud(pcds, INTERVAL)
o3d.io.write_point_cloud("registration.pcd", pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])
