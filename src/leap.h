#pragma once
#include <chrono>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <Windows.h>
#include <process.h>
#include "LeapC.h"
#define LockMutex EnterCriticalSection
#define UnlockMutex LeaveCriticalSection

class LeapConnect
{
public:
    LeapConnect(){
        OpenConnection();
        while(!IsConnected){
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        LeapSetPolicyFlags(connectionHandle,
                           eLeapPolicyFlag_BackgroundFrames & eLeapPolicyFlag_Images, 0);
        // LeapRequestConfigValue();
        // LeapSaveConfigValue();
    };
    ~LeapConnect() {
        kill();
    }
    void OpenConnection(void);
    void kill(void);
    LEAP_CONNECTION* getConnectionHandle(void){return &connectionHandle;};
    // LEAP_TRACKING_EVENT* GetFrame(void); //Used in polling example
    LEAP_DEVICE_INFO* GetDeviceProperties(void); //Used in polling example
    void setDevice(const LEAP_DEVICE_INFO *deviceProps);
    bool IsConnected = false;
private:
    void CloseConnection(void);
    //Internal state
    volatile bool _isRunning = false;
    LEAP_CONNECTION connectionHandle = NULL;
    LEAP_TRACKING_EVENT *lastFrame = NULL;
    LEAP_DEVICE_INFO *lastDevice = NULL;

    //Callback function pointers
    // struct Callbacks ConnectionCallbacks;

    //Threading variables
    std::thread pollingThread;
    CRITICAL_SECTION dataLock;
    void serviceMessageLoop();
    void handleConnectionEvent(const LEAP_CONNECTION_EVENT *connection_event);
    void handleConnectionLostEvent(const LEAP_CONNECTION_LOST_EVENT *connection_lost_event);
    void handlePolicyEvent(const LEAP_POLICY_EVENT *policy_event);
    void handleConfigChangeEvent(const LEAP_CONFIG_CHANGE_EVENT *config_change_event);
    void handleConfigResponseEvent(const LEAP_CONFIG_RESPONSE_EVENT *config_response_event);
    void handleDeviceEvent(const LEAP_DEVICE_EVENT *device_event);
    void handleImageEvent(const LEAP_IMAGE_EVENT *imageEvent);
    const char* ResultString(eLeapRS r);
};