import pyrealsense2 as rs
import numpy
import cv2
import math
from parameters import * 

def head(point, n):
    for i in range(n):
        print(point[i])
    print(".\n.\n.")
    for i in range(n, 0, -1):
        print(point[-1-i])
    print("\n")


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

            print(f"threshold: {min_thes} cm")
            

            # extract points
            distances = numpy.subtract(distances, min_thes)

            distances = numpy.where(((distances < LOW_CUT_DEPTH) &\
                 (distances > -LOW_CUT_DEPTH)), 0, distances)

            distances = numpy.where(((distances < HIGH_CUT_DEPTH) &\
                 (distances > -HIGH_CUT_DEPTH)), distances, float('nan'))

            rows, cols = numpy.where(distances != 0)
            coordinates = list(zip(rows, cols))

            for coord in coordinates:
                if distances[coord[0]][coord[1]] < 0:     
                    depth_colormap[coord[0]][coord[1]][2] = 255
                elif math.isnan(distances[coord[0]][coord[1]]):
                    depth_colormap[coord[0]][coord[1]] = (0, 0, 0)
                else:
                    depth_colormap[coord[0]][coord[1]][0] = 255


            # image trim 하기
            dst = cv2.inRange(color_image, (80, 80, 80), (255, 255, 255))
            dst = cv2.medianBlur(dst, 5)
            
            contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            value = list()
            for cnt in contours:
                value.append(cnt)
            cnt = sorted(value, key=lambda x: len(x))[-1]
            
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            cv2.drawContours(color_image, [approx], 0, (0, 255, 0), 1)

            x_coord = sorted(approx, key=lambda x: x[0][0])
            y_coord = sorted(approx, key=lambda x: x[0][1])

            x_min = x_coord[0][0][0]
            x_max = x_coord[-1][0][0]
            y_min = y_coord[0][0][1]
            y_max = y_coord[-1][0][1]


            # point 추출
            x_unit = (x_max - x_min) // 180
            y_unit = (y_max - y_min) // 60

            point = list()
            for i in range(60):
                for j in range(180):
                    if distances[y_min+y_unit*i][x_min+x_unit*j] != 0 and not math.isnan(distances[y_min+y_unit*i][x_min+x_unit*j]):
                        point.append((i, j, distances[y_min+y_unit*i][x_min+x_unit*j]))

            if len(point) > 10:
                head(point, 5)

            images = numpy.hstack((color_image, depth_colormap))

            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('image', images)

            cv2.waitKey(1)
    finally:
        pipeline.stop()