# yolov5_trt
A implementation of yolov5 s/m//l/x with TensorRT API, it supports serialization/deserialization+image detection/video detection and recording.

This implementation bases on https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5  but made improvement, such as get_rect() and add code for video detection and recording.

Usage:

# build out binary:
mkdir build && cd build

cmake ..

make

# serialize model to trt engine file
./yolov5 -s [.wts] [.engine] [s/m/l/x or c gd gw] 
# deserialize out model from trt engine file and do image detection and show result
./yolov5 -d [.engine] [sample_image_dir] 
# deserialize out model from trt engine file and do video detection and do recording
./yolov5 -v [.engine] video_file 

