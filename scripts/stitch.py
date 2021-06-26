#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
from parameters import *
import copy

def load_point_clouds():
    pcds = []

    filepaths = []
    for file in os.scandir('captures/'):
        filepaths.append(file.path)
        filepaths.sort()

    count = 0
    for filepath in filepaths:
        pcd = o3d.io.read_point_cloud(filepath)
        pcd_tx = copy.deepcopy(pcd).translate((0, INTERVAL * count, 0))
        pcds.append(pcd_tx)
        count += 1 

    return pcds

count = 0
pcds = load_point_clouds()

pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcd_combined += pcds[point_id]

o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])