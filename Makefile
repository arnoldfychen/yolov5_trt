################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include \
	 -I/usr/include/opencv4

LIBS:=  -lnvinfer_plugin -lnvinfer -lnvparsers \
	-L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs \
	-lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h) $(wildcard *.hpp)
SRCFILES:= calibrator.cpp \
           yolov5.cpp     \
           yololayer.cu
TARGET_LIB:= libnvdsinfer_Yolov5.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

ifdef BUILD_DEBUG
        CFLAGS+= -DJZYQ_DEBUG -O0 -g
else
        CFLAGS+= -O3
endif

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB) $(TARGET_OBJS)
