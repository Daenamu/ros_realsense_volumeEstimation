#!/usr/bin/env python3
#-*- coding:utf-8 -*-

# First import the library
import pyrealsense2 as rs
import numpy
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# import parameters
from parameters import * 

def visualize_histogram(minimum, maximum, append, frame):
    bins = numpy.arange(minimum, maximum, append)
    weight = frame
    hist, bins = numpy.histogram(weight, bins)
    plt.hist(weight, bins)
    plt.xlabel('distance(mm)')
    plt.show()


def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = numpy.unique(dataframe[label_name].values)
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

    plt.show()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # Get image
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_image = numpy.asanyarray(depth_frame.get_data())
    color_image = numpy.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = numpy.hstack((resized_color_image, depth_colormap))
    else:
        images = numpy.hstack((color_image, depth_colormap))


    # Averaging distances of each pixels
    print("Calculating average of distance of each pixels...")

    avg_frame = numpy.zeros((HEIGHT, WIDTH))
    num_of_samples = 0
    
    while num_of_samples < AVG_FRAME:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        num_of_samples += 1

        for y in range(HEIGHT):
            for x in range(WIDTH):
                dist = depth.get_distance(x, y)
                avg_frame[y][x] += dist

    # Averaging and cutting outside pixels
    avg_frame = avg_frame / AVG_FRAME * 1000
    avg_frame = avg_frame[(HEIGHT-FIX_HEIGHT)//2 : (HEIGHT+FIX_HEIGHT)//2, (WIDTH-FIX_WIDTH)//2 : (WIDTH+FIX_WIDTH)//2]
    print(f"The camera is facing an object {avg_frame[FIX_HEIGHT//2][FIX_WIDTH//2]} mm away.\n")

    # Get threshold
    print("Getting threshold...")
    hist, bins = numpy.histogram(avg_frame.flatten(), numpy.arange(0, 1000, 5)) # 5mm 단위로 최빈값 구함
    min_bin = numpy.argmax(hist)
    min_thes = (bins[min_bin]+bins[min_bin+1]) / 2

    print(f"threshold: {min_thes} mm\n")
    

    # filtering
    avg_frame = numpy.subtract(avg_frame, min_thes)

    # show +- 0.5m range from threshold
    # visualize_histogram(-500, 500, 1, avg_frame.flatten())

    # convert to points
    point_array = {'x': [], 'y': []}

    for y in range(FIX_HEIGHT):
        for x in range(FIX_WIDTH):
            if abs(avg_frame[y][x]) > LOW_CUT_DEPTH and abs(avg_frame[y][x]) < HIGH_CUT_DEPTH:
                point_array['x'].append(x)
                point_array['y'].append(y)

    # plt.scatter(point_array['x'], point_array['y'])
    # plt.show()

    
    # DBSCAN    
    df = pd.DataFrame(point_array)
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(df)
    df['dbscan_cluster'] = dbscan_labels

    """
    pca = PCA(n_components=2, random_state=0)
    pca_transformed = pca.fit_transform(df)
    df['ftr1'] = pca_transformed[:, 0]
    df['ftr2'] = pca_transformed[:, 1]
    """
    
    # volume estimation
    print("Volume estimating...")

    distance_values = []
    for i in range(df.shape[0]):    # 각 point에 distance정보 추가
        distance_values.append(avg_frame[df.loc[i, 'y']][df.loc[i, 'x']])
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
        
        min_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [ymin, xmin], avg_frame[ymin][xmin] / 1000)
        max_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [ymax, xmax], avg_frame[ymax][xmax] / 1000)

        x_width = abs(max_point[0] - min_point[0]) * 1000
        y_width = abs(max_point[1] - min_point[1]) * 1000
        height = group['distance'].sum() / ((xmax-xmin) * (ymax-ymin))

        volume = x_width*y_width * height
        
        volumes['key'].append(key)
        volumes['x_width'].append(x_width)
        volumes['y_width'].append(y_width)
        volumes['height'].append(height)
        volumes['volume'].append(volume)
    
    volume_data = pd.DataFrame(volumes)
    print(volume_data)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(0)
    visualize_cluster_plot(dbscan, df, 'dbscan_cluster', iscenter=False)

finally:
    pipeline.stop()
