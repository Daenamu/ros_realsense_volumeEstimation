import pyrealsense2 as rs
import numpy
import cv2
import math
from parameters import * 

def process(pipeline):
    try:
        while True:
            # Get image
            frames = pipeline.wait_for_frames()

            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            color_image = numpy.asanyarray(color_frame.get_data())

            # Get distances
            distances = numpy.zeros((HEIGHT, WIDTH))
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    dist = depth_frame.get_distance(x, y)
                    distances[y][x] = dist*100
            # print(f"The camera is facing an object {distances[HEIGHT//2][WIDTH//2]} cm away")


            # Get threshold
            hist, bins = numpy.histogram(distances.flatten(), numpy.arange(30, 100, 0.5)) # 0.5cm 단위로 최빈값 구함
            min_bin = numpy.argmax(hist)
            min_thes = (bins[min_bin]+bins[min_bin+1]) / 2
            # print(f"threshold: {min_thes} cm")
            

            # filtering
            distances = numpy.subtract(distances, min_thes)

            distances = numpy.where(((distances < LOW_CUT_DEPTH) &\
                 (distances > -LOW_CUT_DEPTH)), 0, distances)

            distances = numpy.where(((distances < HIGH_CUT_DEPTH) &\
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

            x_min = x_coord[0][0][0]
            x_max = x_coord[-1][0][0]
            y_min = y_coord[0][0][1]
            y_max = y_coord[-1][0][1]

            cv2.circle(color_image, (x_min, y_min), 3, (0, 255, 0), -1)   # draw circle at edges
            cv2.circle(color_image, (x_min, y_max), 3, (0, 255, 0), -1)
            cv2.circle(color_image, (x_max, y_min), 3, (0, 255, 0), -1)
            cv2.circle(color_image, (x_max, y_max), 3, (0, 255, 0), -1)


            # extract grid data
            x_unit = (x_max - x_min) / 180
            y_unit = (y_max - y_min) / 60

            point = list()
            for i in range(60):
                for j in range(180):
                    x = int(x_min+x_unit*j)
                    y = int(y_min+y_unit*i)
                    dist = distances[y][x]

                    if dist != 0 and not math.isnan(dist):
                        point.append((i, j, dist))

                        if dist < 0:
                            cv2.circle(color_image, (x, y), 1, (0, 0, 255), 1)
                        else:
                            cv2.circle(color_image, (x, y), 1, (255, 0, 0), 1)

            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('image', color_image)
            cv2.waitKey(1)
    finally:
        pipeline.stop()