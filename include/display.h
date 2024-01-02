#ifndef DYNAFLASH_H
#define DYNAFLASH_H
// #define NOMINMAX
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>
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
#define DYNA_FRAME_HEIGHT 768
#define DYNA_FRAME_WIDTH 1024

class DynaFlashProjector
{
public:
    DynaFlashProjector(bool flip_ver = false, bool flip_hor = false) : white_image(DYNA_FRAME_WIDTH, DYNA_FRAME_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255)),
                                                                       m_flip_ver(flip_ver),
                                                                       m_flip_hor(flip_hor){};
    ~DynaFlashProjector()
    {
        gracefully_close();
    };
    bool init();
    void show(const cv::Mat frame);
    void show_buffer(const uint8_t *buffer);
    void show();
    void kill() { gracefully_close(); };
    bool is_initialized() { return initialized; };
    void gracefully_close();
    int get_width() { return DYNA_FRAME_WIDTH; };
    int get_height() { return DYNA_FRAME_HEIGHT; };
#ifdef PYTHON_BINDINGS_BUILD
    void project(nb::ndarray<uint8_t, nb::shape<DYNA_FRAME_HEIGHT, DYNA_FRAME_WIDTH, 3>,
                             nb::c_contig, nb::device::cpu>
                     data,
                 bool bgr = true)
    {
        if (!bgr)
        {
            cv::Mat data_mat(DYNA_FRAME_HEIGHT, DYNA_FRAME_WIDTH, CV_8UC3, data.data());
            cv::cvtColor(data_mat, data_mat, cv::COLOR_RGB2BGR);
            show_buffer(data_mat.data);
        }
        else
        {
            show_buffer(data.data());
        }
    };
#endif
private:
    void print_version();
    void print_led_values();
    void set_led_values();
    bool initialized = false;
    bool m_flip_ver = false;
    bool m_flip_hor = false;
    int board_index = 0;
    float frame_rate = 946.0f; // max: 946.0f
    int bit_depth = 8;
    int alloc_frame_buffer = 16;
    // int allloc_src_frame_buffer=2000;
    ILLUMINANCE_MODE ilum_mode = HIGH_MODE;
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
NB_MODULE(dynaflash, m)
{
    nb::class_<DynaFlashProjector>(m, "projector")
        .def(nb::init<bool, bool>(), nb::arg("flip_ver") = false, nb::arg("flip_hor") = false, "a class to control a dynaflash projector")
        .def("init", &DynaFlashProjector::init, "initializes the projector")
        .def("is_initialized", &DynaFlashProjector::is_initialized, "returns true if the projector is initialized")
        .def("kill", &DynaFlashProjector::kill, "frees the internal projector resources")
        .def("project_white", nb::overload_cast<>(&DynaFlashProjector::show), "projects a white image")
        .def("project", &DynaFlashProjector::project, nb::arg("data"), nb::arg("bgr") = true, "projects an arbitrary numpy array of type uint8, with shape (height, width, 3), bgr");
}
#endif
#endif DYNAFLASH_H