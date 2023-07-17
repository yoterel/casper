#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "stdafx.h"
// #include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DynaFlash.h"
#ifdef PYTHON_BINDINGS_BUILD
    #include <nanobind/ndarray.h>
    #include <nanobind/nanobind.h>
    namespace nb = nanobind;
#endif
class DynaFlashProjector
{
public:
    DynaFlashProjector(const int width, const int height) :
     width(width),
     height(height),
     white_image(width, height, CV_8UC3, cv::Scalar(255, 255, 255)){};
    ~DynaFlashProjector(){
        gracefully_close();
    };
    bool init();
    void show(const cv::Mat frame);
    void show_buffer(const uint8_t* buffer);
    void show();
    bool is_initialized(){return initialized;};
    void gracefully_close();
    int get_width(){return width;};
    int get_height(){return height;};
    #ifdef PYTHON_BINDINGS_BUILD
    void project(nb::ndarray<uint8_t, nb::shape<nb::any, nb::any, 3>,
                            nb::c_contig, nb::device::cpu> data){
                                show_buffer(data.data());
                            };
    #endif
private:
    int width;
    int height;
    void print_version();
    void print_led_values();
    void set_led_values();
    bool initialized = false;
    int board_index=0;
    float frame_rate=946.0f; //max: 946.0f
    int bit_depth=8;
    int alloc_frame_buffer=16;
    // int allloc_src_frame_buffer=2000;
    ILLUMINANCE_MODE mode = HIGH_MODE;
    int frame_mode = FRAME_MODE_RGB;
    int frame_size = FRAME_BUF_SIZE_24BIT;
    // char *pFrameData = NULL;
    // char *pFrameData[2] = { NULL };
    char *pBuf = NULL;
	unsigned long nFrameCnt = 0;
    unsigned long nGetFrameCnt = 0;
    CDynaFlash *pDynaFlash = NULL;
    DYNAFLASH_STATUS stDynaFlashStatus;
    cv::Mat white_image;
};

#ifdef PYTHON_BINDINGS_BUILD
    NB_MODULE(dynaflash, m) {
        nb::class_<DynaFlashProjector>(m, "projector")
            .def(nb::init<const int, const int>(), nb::arg("width"), nb::arg("height"), "a class to control a dynaflash projector")
            .def("init", &DynaFlashProjector::init, "initializes the projector")
            .def("is_initialized", &DynaFlashProjector::is_initialized, "returns true if the projector is initialized")
            .def("kill", &DynaFlashProjector::gracefully_close, "frees the internal projector resources")
            .def("project_white", nb::overload_cast<>(&DynaFlashProjector::show), "projects a white image")
            .def("project", &DynaFlashProjector::project, "projects an arbitrary buffer of size width*height*3");
    }
#endif
