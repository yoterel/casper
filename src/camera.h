#pragma once
#include <iostream>
// Include files to use the pylon API.
#include <pylon/PylonIncludes.h>
#include <pylon/PylonGUI.h>
#include "ConfigurationEventPrinter.h"
#include "queue.h"
#include "opencv2/opencv.hpp"
// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
// using namespace std;

class BaslerCamera
{
public:
    BaslerCamera(blocking_queue<CPylonImage>& camera_queue, bool& close_signal, uint32_t& height, uint32_t& width);
    ~BaslerCamera();
    void acquire();
private:
    CInstantCamera camera;
    bool is_open = false;
};