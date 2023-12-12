#ifndef LEAP_H
#define LEAP_H

#include <chrono>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
// #include <glm/glm.hpp>
// #define WINDOWS_LEAN_AND_MEAN
// #define NOMINMAX
// #include <windows.h>
// #pragma warning(disable : 4996)
// #include <process.h>
#include "LeapC.h"
// #define LockMutex EnterCriticalSection
// #define UnlockMutex LeaveCriticalSection

#ifdef PYTHON_BINDINGS_BUILD
// #include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
#endif

enum class LEAP_STATUS
{
    LEAP_NEWFRAME,
    LEAP_NONEWFRAME,
    LEAP_FAILED,
};

class LeapConnect
{
public:
    LeapConnect(bool pollMode = true, bool with_images = false);
    ~LeapConnect();
    void OpenConnection(void);
    void kill(void);
    LEAP_CONNECTION *getConnectionHandle(void) { return &connectionHandle; };
    LEAP_DEVICE_INFO *GetDeviceProperties(void);
    void setDevice(const LEAP_DEVICE_INFO *deviceProps);
    int64_t LeapGetTime() { return LeapGetNow(); };
    void setFrame(const LEAP_TRACKING_EVENT *frame);
    LEAP_TRACKING_EVENT *getFrame();
    void setImage(const LEAP_IMAGE_EVENT *imageEvent);
    void setPollMode(bool pollMode);
    void setImageMode(bool imageMode);
    bool getImage(std::vector<uint8_t> &image1, std::vector<uint8_t> &image2, uint32_t &width, uint32_t &height);
    bool getDistortion(std::vector<float> &dist1, std::vector<float> &dist2, uint32_t &width, uint32_t &height);
    // std::vector<float> getFrame();
    std::vector<float> getIndexTip();
    // glm::vec3 triangulate(const glm::vec2 &leap1,
    //                       const glm::vec2 &leap2,
    //                       const int leap_width, const int leap_height);
    // std::vector<glm::vec3> triangulate(const std::vector<glm::vec2> &leap1,
    //                                    const std::vector<glm::vec2> &leap2,
    //                                    const int leap_width, const int leap_height);
    bool IsConnected = false;

private:
    void CloseConnection(void);
    void deepCopyTrackingEvent(LEAP_TRACKING_EVENT *dst, const LEAP_TRACKING_EVENT *src);
    // Internal state
    volatile bool _isRunning = false;
    LEAP_CONNECTION connectionHandle = NULL;
    LEAP_TRACKING_EVENT *lastFrame = NULL;
    LEAP_DEVICE_INFO *lastDevice = NULL;
    LEAP_CLOCK_REBASER m_clockSynchronizer;
    int64_t m_targetFrameTime = 0;
    bool m_poll = false;
    uint64_t m_imageFrameID = 0;
    void *m_imageBuffer = NULL;
    void *distortion_buffer_left = NULL;
    void *distortion_buffer_right = NULL;
    uint64_t m_currentDistortionId = 0;
    uint64_t m_imageSize = 0;
    bool m_imageReady = false;
    bool m_textureChanged = false;
    uint32_t m_imageWidth = 0;
    uint32_t m_imageHeight = 0;
    // Callback function pointers
    //  struct Callbacks ConnectionCallbacks;

    // Threading variables
    std::thread pollingThread;
    std::mutex m_mutex;
    // CRITICAL_SECTION dataLock;
    void serviceMessageLoop();
    void handleConnectionEvent(const LEAP_CONNECTION_EVENT *connection_event);
    void handleConnectionLostEvent(const LEAP_CONNECTION_LOST_EVENT *connection_lost_event);
    void handleDeviceEvent(const LEAP_DEVICE_EVENT *device_event);
    void handlePolicyEvent(const LEAP_POLICY_EVENT *policy_event);
    void handleConfigChangeEvent(const LEAP_CONFIG_CHANGE_EVENT *config_change_event);
    void handleConfigResponseEvent(const LEAP_CONFIG_RESPONSE_EVENT *config_response_event);
    void handleTrackingEvent(const LEAP_TRACKING_EVENT *tracking_event);
    void handleTrackingModeEvent(const LEAP_TRACKING_MODE_EVENT *tracking_mode_event);
    void handlePointMappingChangeEvent(const LEAP_POINT_MAPPING_CHANGE_EVENT *point_mapping_change_event);
    void handleImageEvent(const LEAP_IMAGE_EVENT *imageEvent);
    const char *ResultString(eLeapRS r);
};

#ifdef PYTHON_BINDINGS_BUILD
NB_MODULE(leap, m)
{
    nb::class_<LeapConnect>(m, "device")
        .def(nb::init<bool, bool>(), nb::arg("polling_mode") = true, nb::arg("with_images") = false, "a class controlling a leap motion (ultraleap) device")
        .def("kill", &LeapConnect::kill, "frees the resources associated with the device")
        .def("get_index_tip", &LeapConnect::getIndexTip, "get location of tip of index finger (xyz)");
    // .def("get_joints", &LeapConnect::getFrame, "get a list of joint locations (xyz)")
    // .def("get_time", &LeapConnect::LeapGetTime, "get device time in microseconds")
    // .def("rebase", &LeapConnect::rebase, "get device time in microseconds")
    // .def("get_images", &LeapConnect::get_images, "returns raw images from sensors");
}
#endif

#endif