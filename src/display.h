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

class DynaFlashProjector
{
public:
    DynaFlashProjector(int width, int height) : width(width), height(height){};
    ~DynaFlashProjector(){
        gracefully_close();
    };
    bool init();
    void print_led_values();
    void set_led_values();
    void show(cv::Mat frame);
    void gracefully_close();
    void print_version();
    bool is_initialized(){return initialized;};
    int width;
    int height;
private:
    bool initialized = false;
    int board_index=0;
    float frame_rate=946.0f; //max: 946.0f
    int bit_depth=8;
    int alloc_frame_buffer=2;
    // int allloc_src_frame_buffer=2000;
    int high_mode = HIGH_MODE;
    int frame_mode = FRAME_MODE_RGB;
    int frame_size = FRAME_BUF_SIZE_24BIT;
    CDynaFlash *pDynaFlash = NULL;
    char *pFrameData = NULL;
    // char *pFrameData[2] = { NULL };
    char *pBuf = NULL;
	unsigned long nFrameCnt = 0;
    unsigned long nGetFrameCnt = 0;
    DYNAFLASH_STATUS stDynaFlashStatus;
};