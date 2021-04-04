#!/usr/bin/env python3

import pyrealsense2 as rs
import rospy
from realsense_depth.msg import depth_msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import pcl
import numpy
import cv2
import math

WIDTH           =    640
HEIGHT          =    480
AVG_FRAME       =    30                # Defines the number of sample frames                                                        
LOW_CUT_DEPTH   =    1.5               # low cut under 1.5cm error   
HIGH_CUT_DEPTH  =    10                # high cut over 10cm error

"""
def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = numpy.array(numpy.hstack([points, colors]), dtype=numpy.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg 
"""

def talker():
    rospy.init_node('realsense_depth', anonymous = True)

    pub = rospy.Publisher('realsense_depth', depth_msg, queue_size=10)
    img_pub = rospy.Publisher('realsense_depth_img', Image, queue_size=10)
    # point_pub = rospy.Publisher('realsense_depth_pointcloud', PointCloud2, queue_size=10)

    msg = depth_msg()
    bridge = CvBridge()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, "/home/daeuk/catkin_ws/src/realsense_depth/bag/2021-04-01-03-03-20.bag")
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

    pipeline.start(config)

    try:  
        while not rospy.is_shutdown():       
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
            print(f"threshold: {min_thes} cm")
            

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

            msg.len = 0
            msg.depths.clear()

            point_array = numpy.zeros((60*180, 3), dtype=numpy.float32)
            # color_array = numpy.zeros((60*180, 3), dtype=numpy.float32)
            for i in range(60):
                for j in range(180):
                    x = int(x_min+x_unit*j)
                    y = int(y_min+y_unit*i)
                    dist = distances[y][x]

                    point_array[i*180+j][0] = i
                    point_array[i*180+j][1] = j
                    point_array[i*180+j][2] = dist
                    """
                    color_array[i*180+j][0] = 255
                    color_array[i*180+j][1] = 255
                    color_array[i*180+j][2] = 255
                    """

                    if dist != 0 and not math.isnan(dist):
                        point = Point()
                        point.x = i
                        point.y = j
                        point.z = dist

                        msg.depths.append(point)
                        msg.len += 1

                        if dist < 0:
                            cv2.circle(color_image, (x, y), 1, (0, 0, 255), 1)
                            """
                            color_array[i*180+j][0] = 0
                            color_array[i*180+j][1] = 0
                            color_array[i*180+j][2] = 255
                            """
                        else:
                            cv2.circle(color_image, (x, y), 1, (255, 0, 0), 1)
                            """
                            color_array[i*180+j][0] = 255
                            color_array[i*180+j][1] = 0
                            color_array[i*180+j][2] = 0
                            """

            # point_msg = xyzrgb_array_to_pointcloud2(point_array, color_array)

            # point_pub.publish(point_msg)
            img_pub.publish(bridge.cv2_to_imgmsg(color_image, "bgr8"))
            pub.publish(msg)

        pc = pcl.PointCloud(point_array)
        pcl.save(pc, 'pc2pcd.pcd')

    finally:
        pipeline.stop()

if __name__=='__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
