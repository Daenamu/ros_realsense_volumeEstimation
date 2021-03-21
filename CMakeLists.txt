cmake_minimum_required(VERSION 3.0.2)
project(realsense_depth)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs geometry_msgs roscpp) 

find_package(realsense2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_message_files(
  FILES
  center_depth_msg.msg 
  depth_msg.msg
  point_msg.msg
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES realsense_depth realsense2
  CATKIN_DEPENDS std_msgs sensor_msgs geometry_msgs roscpp
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(realsense_center_depth src/realsense_center_depth.cpp)
add_dependencies(realsense_center_depth ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(realsense_center_depth
  ${catkin_LIBRARIES} ${realsense2_LIBRARY}
)

add_executable(realsense_depth src/realsense_depth.cpp)
add_dependencies(realsense_depth ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(realsense_depth
  ${catkin_LIBRARIES} ${realsense2_LIBRARY}
)

add_executable(realsense_extract_point src/realsense_extract_point.cpp)
add_dependencies(realsense_extract_point ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(realsense_extract_point
  ${catkin_LIBRARIES} ${realsense2_LIBRARY}
)

