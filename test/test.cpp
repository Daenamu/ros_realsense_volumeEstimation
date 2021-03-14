#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_option.h>
#include <librealsense2/h/rs_frame.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_Z16    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           640               // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          0                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
#define AVG_FRAME       30                // Defines the number of sample frames                                  //
#define THES_ACC        0.00001           // Defines the accuracy of threshold                                    //
#define IGN_DEPTH       0.002             // Defines how long depth to ignore                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/* Function calls to librealsense may raise errors of type rs_error*/
void check_error(rs2_error* e)
{
    if (e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs2_get_failed_function(e), rs2_get_failed_args(e));
        printf("    %s\n", rs2_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}

void print_device_info(rs2_device* dev)
{
    rs2_error* e = 0;
    printf("\nUsing device 0, an %s\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_NAME, &e));
    check_error(e);
    printf("    Serial number: %s\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER, &e));
    check_error(e);
    printf("    Firmware version: %s\n\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION, &e));
    check_error(e);
}


int main(int argc, char **argv)
{
    rs2_error* e = 0;

    // Create a context object. This object owns the handles to all connected realsense devices.
    // The returned object should be released with rs2_delete_context(...)
    rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
    check_error(e);

    /* Get a list of all the connected devices. */
    // The returned object should be released with rs2_delete_device_list(...)
    rs2_device_list* device_list = rs2_query_devices(ctx, &e);
    check_error(e);

    int dev_count = rs2_get_device_count(device_list, &e);
    check_error(e);
    printf("There are %d connected RealSense devices.\n", dev_count);
    if (0 == dev_count)
        return EXIT_FAILURE;

    // Get the first connected device
    // The returned object should be released with rs2_delete_device(...)
    rs2_device* dev = rs2_create_device(device_list, 0, &e);
    check_error(e);

    print_device_info(dev);

    // Create a pipeline to configure, start and stop camera streaming
    // The returned object should be released with rs2_delete_pipeline(...)
    rs2_pipeline* pipeline =  rs2_create_pipeline(ctx, &e);
    check_error(e);

    // Create a config instance, used to specify hardware configuration
    // The retunred object should be released with rs2_delete_config(...)
    rs2_config* config = rs2_create_config(&e);
    check_error(e);

    // Request a specific configuration
    rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
    check_error(e);

    // Start the pipeline streaming
    // The retunred object should be released with rs2_delete_pipeline_profile(...)
    rs2_pipeline_profile* pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
    if (e)
    {
        printf("The connected device doesn't support depth streaming!\n");
        exit(EXIT_FAILURE);
    }

    rs2_stream_profile_list* stream_profile_list = rs2_pipeline_profile_get_streams(pipeline_profile, &e);
    if (e)
    {
        printf("Failed to create stream profile list!\n");
        exit(EXIT_FAILURE);
    }

    rs2_stream_profile* stream_profile = (rs2_stream_profile*)rs2_get_stream_profile(stream_profile_list, 0, &e);
    if (e)
    {
        printf("Failed to create stream profile!\n");
        exit(EXIT_FAILURE);
    }

    int width; int height;
    rs2_get_video_stream_resolution(stream_profile, &width, &height, &e);
    if (e)
    {
        printf("Failed to get video stream resolution data!\n");
        exit(EXIT_FAILURE);
    }

    float** depth_array = new float*[width];
    for (int i = 0; i < width; i++) {
        depth_array[i] = new float[height];
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            depth_array[i][j] = 0;
        }
    }


    // 1. 30프레임으로 깊이 평균 내기
    float min = 100000, max = 0;
    int frame_count = 0;
    while (frame_count < AVG_FRAME) 
    {
        // This call waits until a new composite_frame is available
        // composite_frame holds a set of frames. It is used to prevent frame drops
        // The returned object should be released with rs2_release_frame(...)
        rs2_frame* frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, &e);
        check_error(e);

        // Returns the number of frames embedded within the composite frame
        int num_of_frames = rs2_embedded_frames_count(frames, &e);
        check_error(e);
        frame_count += num_of_frames;

        for (int i = 0; i < num_of_frames; ++i)
        {
            // The retunred object should be released with rs2_release_frame(...)
            rs2_frame* frame = rs2_extract_frame(frames, i, &e);
            check_error(e);

            // Check if the given frame can be extended to depth frame interface
            // Accept only depth frames and skip other frames
            if (0 == rs2_is_frame_extendable_to(frame, RS2_EXTENSION_DEPTH_FRAME, &e))
                continue;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float dist = rs2_depth_frame_get_distance(frame, w, h, &e);
                    check_error(e);

                    depth_array[w][h] += dist;
                    if (min > dist) {
                        min = dist;
                    }
                    if (max < dist) {
                        max = dist;
                    }
                }
            }
            rs2_release_frame(frame);
        }

        rs2_release_frame(frames);
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            depth_array[i][j] /= AVG_FRAME;
        }
    }
    printf("The camera is facing an object %.3f meters away.\n", depth_array[width/2][height/2]);
    printf("min dist : %.3f\n", min);
    printf("max dist : %.3f\n\n", max);

    // 2. 쓰레숄드 구하기

    float append = 1, var, error, min_var, min_thes;   // 초기에 1m 단위로 최소 분산 찾음. 점점 줄여 나간다.

    while (append > THES_ACC) {
        for (float thes = min; thes < max; thes += append) {
            var = 0;
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    error = depth_array[i][j] - thes;
                    var += error*error;
                }
            }
            var = var / width*height;

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

    printf("쓰레숄드: %f meters\n", min_thes);

    // filtering
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            depth_array[i][j] -= min_thes;
            if (depth_array[i][j] < IGN_DEPTH) {
                depth_array[i][j] = 0;
            }
        }
    }

    // convert to bw image
    uint8_t *buffer = new uint8_t[width*height];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            buffer[height*i + j] = (depth_array[i][j] == 0) ? 0 : 255;
        }
    }

    // Stop the pipeline streaming
    rs2_pipeline_stop(pipeline, &e);
    check_error(e);

    // Release resources
    rs2_delete_pipeline_profile(pipeline_profile);
    rs2_delete_config(config);
    rs2_delete_pipeline(pipeline);
    rs2_delete_device(dev);
    rs2_delete_device_list(device_list);
    rs2_delete_context(ctx);

    for (int i = 0; i < width; i++) {
        delete [] depth_array[i];
    }
    delete [] depth_array;

    return EXIT_SUCCESS;
}