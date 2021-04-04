cmake_minimum_required(VERSION 3.0.2)
project(realsense_depth)

find_package(catkin REQUIRED COMPONENTS 
  message_generation 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  rospy 
  cv_bridge
) 

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_message_files(FILES
  depth_msg.msg
)

generate_messages(
  DEPENDENCIES 
  std_msgs 
  geometry_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES realsense_depth realsense2 opencv
  CATKIN_DEPENDS std_msgs sensor_msgs geometry_msgs cv_bridge rospy
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIRS}
)


