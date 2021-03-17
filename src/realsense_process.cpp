#include "ros/ros.h"
#include "realsense_depth/depth_msg.h"

#define AVG_FRAME       30 
#define HZ              30
#define WIDTH           640              
#define HEIGHT          480
#define THES_ACC        0.00001
#define REALSENSE_MIN   0.3
#define REALSENSE_MAX   1
#define LOW_CUT_DEPTH   5                // low cut under 5mm error   
#define HIGH_CUT_DEPTH  500               // high cut over 50cm error

#include <cmath>
#include <vector>

struct Point {
    int x;
    int y;
    float distance;
};

float frame[WIDTH][HEIGHT] = {0};

void msgCallback(const realsense_depth::depth_msg::ConstPtr& msg) {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            frame[i][j] += msg->data[i*HEIGHT + j];
        }
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "realsense_process_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("realsense_depth_msg", 100, msgCallback);
    ros::Rate loop_rate(HZ);

    int i = 0;
    while (ros::ok() && i < AVG_FRAME) {
        i++;
        ros::spinOnce();
        loop_rate.sleep();        
    }

    // Averaging
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            frame[i][j] /= AVG_FRAME;
        }
    }

    // Threshold

    float append = 1, var, error, min_var, min_thes; 
    float min = REALSENSE_MIN, max = REALSENSE_MAX;

    while (append > THES_ACC) {
        for (float thes = min; thes < max; thes += append) {
            var = 0;
            for (int i = 0; i < WIDTH; i++) {
                for (int j = 0; j < HEIGHT; j++) {
                    if (frame[i][j] < REALSENSE_MIN || frame[i][j] > REALSENSE_MAX) {   // Noise filtering
                        continue;
                    }
                    error = frame[i][j] - thes;
                    var += error*error;
                }
            }
            var = var / WIDTH*HEIGHT;

            if (thes == min || min_var > var) {
                min_var = var;
                min_thes = thes;
            }
        }
        min = min_thes - append;
        if (min < 0) {
            min = 0;
        }
        max = min_thes + append;
        append /= 10;
    }

    printf("The camera is facing an object %f meters away\n", frame[WIDTH/2][HEIGHT/2]);
    printf("Threshold: %f meters\n", min_thes);


    // Filtering low or high values

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            frame[i][j] -= min_thes;
            if (frame[i][j] < LOW_CUT_DEPTH) {      // low cut filtering
                frame[i][j] = 0;
            }
            else if (frame[i][j] > HIGH_CUT_DEPTH) {     // high cut filtering
                frame[i][j] = NAN;
            }
            else {

            }
        }
    }

    std::vector<Point> point_array;

    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            if (frame[i][j] != 0 && !isnan(frame[i][j])) {
                point_array.push_back({i, j, frame[i][j]});
            }        
        }
    }
    for (auto iter = point_array.begin(); iter != point_array.end(); iter++) {
        printf("(%d, %d, %f)\n", iter->x, iter->y, iter->distance);
    }
    return 0;
}