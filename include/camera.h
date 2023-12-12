#ifndef BASLER_CAMERA_H
#define BASLER_CAMERA_H

#include <iostream>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
// #include <pylon/PylonGUI.h>
#include "queue.h"
#include "readerwritercircularbuffer.h"
#include "opencv2/opencv.hpp"
#ifdef PYTHON_BINDINGS_BUILD
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#endif
// Namespace for using pylon objects.
using namespace Pylon;
using namespace Basler_UniversalCameraParams;

class BaslerCamera
{
public:
    BaslerCamera(){};
    ~BaslerCamera()
    {
        kill();
    }; /* PylonTerminate(); */
    bool init(moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> &camera_queue, bool &close_signal,
              uint32_t height, uint32_t width,
              float exposureTime = 1850.0f, bool hardwareTrigger = false);
    void acquire();
    void kill();
    void balance_white();
    double get_exposure_time();
    void set_exposure_time(double exposure_time);
#ifdef PYTHON_BINDINGS_BUILD
    void init_single(float exposure_time = 1850.0);
    nb::ndarray<nb::numpy, const uint8_t> capture_single();
#endif
private:
    CBaslerUniversalInstantCamera camera;
    bool is_open = false;
};

#ifdef PYTHON_BINDINGS_BUILD
NB_MODULE(basler, m)
{
    nb::class_<BaslerCamera>(m, "camera")
        .def(nb::init<>(), "a class to control a basler camera")
        .def("init", &BaslerCamera::init_single, nb::arg("exposure_time") = 1850.0, "initializes the camera for single image captures")
        .def("balance_white", &BaslerCamera::balance_white, "balances the white level of the camera once")
        .def("capture", &BaslerCamera::capture_single, "returns a numpy array of the captured image (h,w,3) uint8")
        .def("kill", &BaslerCamera::kill, "kills the camera");
}
#endif

#endif // BASLER_CAMERA_H