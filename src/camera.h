#pragma once
#include <iostream>
#include <pylon/PylonIncludes.h>
// #include <pylon/PylonGUI.h>
#include "queue.h"
#include "opencv2/opencv.hpp"
#ifdef PYTHON_BINDINGS_BUILD
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#endif
// Namespace for using pylon objects.
using namespace Pylon;

class BaslerCamera
{
public:
    BaslerCamera(){};
    ~BaslerCamera(){};  /* PylonTerminate(); */ 
    void init(blocking_queue<CPylonImage>& camera_queue, bool& close_signal, uint32_t& height, uint32_t& width);
    void acquire();
    #ifdef PYTHON_BINDINGS_BUILD
    void init_single();
    nb::ndarray<nb::numpy, const uint8_t> capture_single();
    void kill();
    #endif
private:
    CInstantCamera camera;
    bool is_open = false;    
};

#ifdef PYTHON_BINDINGS_BUILD
    NB_MODULE(basler, m) {
        nb::class_<BaslerCamera>(m, "camera")
            .def(nb::init<>(), "a class to control a basler camera")
            .def("init", &BaslerCamera::init_single, "initializes the camera for single image captures")
            .def("capture", &BaslerCamera::capture_single, "returns a numpy array of the captured image h,w,3 uint8")
            .def("kill", &BaslerCamera::kill, "kills the camera"); 
    }
#endif