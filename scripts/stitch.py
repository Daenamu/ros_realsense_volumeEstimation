import numpy as np
import pickle
from parameters import *
import open3d as o3d
import matplotlib.pyplot as plt

with open('captures2.p', 'rb') as file:
    captures = pickle.load(file)

pointclouds = []

for capture in captures:
    color_image = o3d.geometry.Image(np.array(capture[0]))
    depth_image = o3d.geometry.Image(np.array(capture[1], dtype='float32'))
    depth_colormap_image = o3d.geometry.Image(np.array(capture[2]))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_colormap_image)

    pcd = o3d.geometry.Pointcloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pointclouds.append(pcd)

