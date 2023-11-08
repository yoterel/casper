#ifndef LEAP_H
#define LEAP_H

#include <chrono>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4996)
#include <process.h>
#include "LeapC.h"
#define LockMutex EnterCriticalSection
#define UnlockMutex LeaveCriticalSection

#ifdef PYTHON_BINDINGS_BUILD
// #include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
#endif

class LeapConnect
{
public:
    LeapConnect()
    {
        OpenConnection();
        while (!IsConnected)
        {
            std::cout << "Leap: waiting for connection..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        LeapSetPolicyFlags(connectionHandle,
                           0, 0);
        // LeapSetPolicyFlags(connectionHandle,
        //                    eLeapPolicyFlag_Images & eLeapPolicyFlag_MapPoints, 0);
        // LeapSetPolicyFlags(connectionHandle,
        //                    eLeapPolicyFlag_BackgroundFrames & eLeapPolicyFlag_Images, 0);
        LeapSetTrackingMode(connectionHandle, eLeapTrackingMode_ScreenTop); // eLeapTrackingMode_Desktop, eLeapTrackingMode_HMD, eLeapTrackingMode_ScreenTop
        // LeapRequestConfigValue();
        // LeapSaveConfigValue();
    };
    LeapConnect(bool pollMode) : LeapConnect()
    {
        m_poll = pollMode;
    };
    ~LeapConnect()
    {
        kill();
    }
    void OpenConnection(void);
    void kill(void);
    LEAP_CONNECTION *getConnectionHandle(void) { return &connectionHandle; };
    LEAP_DEVICE_INFO *GetDeviceProperties(void);
    void setDevice(const LEAP_DEVICE_INFO *deviceProps);
    int64_t LeapGetTime() { return LeapGetNow(); };
    void setFrame(const LEAP_TRACKING_EVENT *frame);
    LEAP_TRACKING_EVENT *getFrame();
    // std::vector<float> getFrame();
    std::vector<float> getIndexTip();
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
    // Callback function pointers
    //  struct Callbacks ConnectionCallbacks;

    // Threading variables
    std::thread pollingThread;
    CRITICAL_SECTION dataLock;
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
        .def(nb::init<bool>(), nb::arg("polling_mode") = true, "a class controlling a leap motion (ultraleap) device")
        .def("kill", &LeapConnect::kill, "frees the resources associated with the device")
        .def("get_index_tip", &LeapConnect::getIndexTip, "get location of tip of index finger (xyz)");
    // .def("get_joints", &LeapConnect::getFrame, "get a list of joint locations (xyz)")
    // .def("get_time", &LeapConnect::LeapGetTime, "get device time in microseconds")
    // .def("rebase", &LeapConnect::rebase, "get device time in microseconds")
    // .def("get_images", &LeapConnect::get_images, "returns raw images from sensors");
}
#endif

#endif