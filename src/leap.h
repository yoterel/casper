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
    };
    void OpenConnection(void);
    void CloseConnection(void);
    void DestroyConnection(void);
    LEAP_CONNECTION* getConnectionHandle(void){return &connectionHandle;};
    // LEAP_TRACKING_EVENT* GetFrame(void); //Used in polling example
    LEAP_DEVICE_INFO* GetDeviceProperties(void); //Used in polling example
    void setDevice(const LEAP_DEVICE_INFO *deviceProps);
    bool IsConnected = false;
private:
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
    void handleDeviceEvent(const LEAP_DEVICE_EVENT *device_event);
    const char* ResultString(eLeapRS r);
};