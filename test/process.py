import pyrealsense2 as rs
import numpy
import cv2
from parameters import * 

def process(pipeline):
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = numpy.asanyarray(depth_frame.get_data())
            color_image = numpy.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_BONE)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape


            # Get distances
            distances = numpy.zeros((HEIGHT, WIDTH))
            
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    dist = depth_frame.get_distance(x, y)
                    distances[y][x] = dist*100

            print(f"The camera is facing an object {distances[HEIGHT//2][WIDTH//2]} cm away")


            # Get threshold
            hist, bins = numpy.histogram(distances.flatten(), numpy.arange(30, 100, 0.5)) # 0.5cm 단위로 최빈값 구함
            min_bin = numpy.argmax(hist)
            min_thes = (bins[min_bin]+bins[min_bin+1]) / 2

            print(f"threshold: {min_thes} cm\n")
            

            # extract points
            distances = numpy.subtract(distances, min_thes)

            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if abs(distances[y][x]) > LOW_CUT_DEPTH and abs(distances[y][x]) < HIGH_CUT_DEPTH:
                        if distances[y][x] < 0:     
                            depth_colormap[y][x][2] = 255
                        else:
                            depth_colormap[y][x][0] = 255
                        pass
                        
            
            # bounding rect
            

            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = numpy.hstack((resized_color_image, depth_colormap))
            else:
                images = numpy.hstack((color_image, depth_colormap))

            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('image', images)
            cv2.waitKey(1)
    finally:
        pipeline.stop()