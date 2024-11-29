#ifndef BASLER_CAMERA_H
#define BASLER_CAMERA_H

#include <iostream>
#ifdef USE_PYLON
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#endif
// #include <pylon/PylonGUI.h>
#include "timer.h"
#include "readerwritercircularbuffer.h"
#include "opencv2/opencv.hpp"
#ifdef PYTHON_BINDINGS_BUILD
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#endif
#ifdef USE_PYLON
// Namespace for using pylon objects.
using namespace Pylon;
using namespace Basler_UniversalCameraParams;

class BaslerCamera
{
public:
    BaslerCamera() {};
    ~BaslerCamera()
    {
        kill();
    }; /* PylonTerminate(); */
    void init(moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> &camera_queue, bool &close_signal,
              uint32_t height, uint32_t width,
              float exposureTime = 1850.0f, bool hardwareTrigger = false);
    void init_poll(uint32_t height, uint32_t width, float exposureTime = 1850.0f);
    void acquire();
    void kill();
    void balance_white();
    double get_exposure_time();
    void set_exposure_time(double exposure_time);
    void init_single(float exposure_time = 1850.0);
    double getAvgEnqueueTimeAndReset();
    bool capture_single_image(CGrabResultPtr &ptrGrabResult);
    CGrabResultPtr capture_single_image_slow();
#ifdef PYTHON_BINDINGS_BUILD
    nb::ndarray<nb::numpy, const uint8_t> capture_single();
#endif
private:
    CBaslerUniversalInstantCamera camera;
    bool is_open = false;
    bool is_pylon_init = false;
    Timer enqueue_timer;
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
#else
class CGrabResultPtr
{
public:
    CGrabResultPtr() {};
    CGrabResultPtr *operator->()
    {
        return this;
    }
    int GetHeight() { return 0; };
    int GetWidth() { return 0; };
    uint8_t *GetBuffer() { return nullptr; };
    ~CGrabResultPtr() {};
};
class BaslerCamera
{
public:
    BaslerCamera() {};
    ~BaslerCamera() {};
    void init(moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> &camera_queue, bool &close_signal,
              uint32_t height, uint32_t width,
              float exposureTime = 1850.0f, bool hardwareTrigger = false) {};
    void init_poll(uint32_t height, uint32_t width, float exposureTime = 1850.0f) {};
    void acquire() {};
    void kill() {};
    void balance_white() {};
    double get_exposure_time() { return 0.0; };
    void set_exposure_time(double exposure_time) {};
    void init_single(float exposure_time = 1850.0) {};
    double getAvgEnqueueTimeAndReset() { return 0.0; };
    bool capture_single_image(CGrabResultPtr &ptrGrabResult) { return false; };
    CGrabResultPtr capture_single_image_slow() { return CGrabResultPtr(); };
};
#endif // USE_PYLON

#endif // BASLER_CAMERA_H