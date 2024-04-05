#ifndef DYNAFLASH_H
#define DYNAFLASH_H
#include "stdafx.h"
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "readerwritercircularbuffer.h"
#include <thread>
#include "DynaFlash.h"
#ifdef PYTHON_BINDINGS_BUILD
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image.h"
#include "stb_image_write.h"
#define DYNA_FRAME_HEIGHT 768
#define DYNA_FRAME_WIDTH 1024

class Display
{
public:
    Display(int height, int width, int channels, bool verbose) : m_height(height),
                                                                 m_width(width),
                                                                 m_channels(channels),
                                                                 m_verbose(verbose){};
    int get_width() { return m_width; };
    int get_height() { return m_height; };
    virtual std::size_t get_queue_size() { return 0; };
    virtual bool init() { return true; };
    virtual uint8_t *get_buffer() { return nullptr; };
    virtual void show(){};
    virtual void show(cv::Mat frame){};
    virtual void show_buffer(uint8_t *buffer){};
    virtual void kill(){};

protected:
    int m_height;
    int m_width;
    int m_channels;
    bool m_verbose;
};

class SaveToDisk : public Display
{
public:
    SaveToDisk(std::string dst, int height, int width, int channels = 3, bool verbose = false) : Display(height, width, channels, verbose),
                                                                                                 m_copy_queue(5000),
                                                                                                 m_work_queue(5000),
                                                                                                 m_dst(dst),
                                                                                                 white_image(DYNA_FRAME_WIDTH, DYNA_FRAME_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255)){};
    bool init();
    uint8_t *get_buffer();
    void show();
    void show(cv::Mat frame);
    void show_buffer(uint8_t *buffer);
    void show_buffer_internal(uint8_t *buffer, bool free_buffer = true);
    void kill();
    void setDestination(std::string dst) { m_dst = dst; };
    std::size_t get_queue_size() { return m_work_queue.size_approx(); };
    int frame_counter = 0;
    // void setSaveToDisk(bool save_to_disk) { m_save_to_disk = save_to_disk; };

private:
    std::string m_dst;
    moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> m_copy_queue;
    moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> m_work_queue;
    std::thread m_copy_thread;
    std::thread m_work_thread;
    bool m_close_signal = false;
    cv::Mat white_image;
    bool m_initialized = false;
    // bool m_save_to_disk = false;
};

class DynaFlashProjector : public Display
{
public:
    DynaFlashProjector(bool flip_ver = false, bool flip_hor = false, int queue_length = 20, bool verbose = false) : Display(DYNA_FRAME_HEIGHT, DYNA_FRAME_WIDTH, 3, verbose),
                                                                                                                    m_projector_queue(queue_length),
                                                                                                                    white_image(DYNA_FRAME_WIDTH, DYNA_FRAME_HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255)),
                                                                                                                    m_flip_ver(flip_ver),
                                                                                                                    m_flip_hor(flip_hor){};
    ~DynaFlashProjector()
    {
        gracefully_close();
    };
    bool init();
    uint8_t *get_buffer();
    void show();
    void show(cv::Mat frame);
    void show_buffer(uint8_t *buffer);
    void show_buffer_internal(uint8_t *buffer);
    void kill() { gracefully_close(); };
    bool is_initialized() { return initialized; };
    void gracefully_close();
    std::size_t get_queue_size() { return m_projector_queue.size_approx(); };

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
    bool m_close_signal = false;
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
    moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> m_projector_queue;
    std::thread m_projector_thread;
};

#ifdef PYTHON_BINDINGS_BUILD
NB_MODULE(dynaflash, m)
{
    nb::class_<DynaFlashProjector>(m, "projector")
        .def(nb::init<bool, bool, bool>(), nb::arg("flip_ver") = false,
             nb::arg("flip_hor") = false,
             nb::arg("verbose") = false,
             "a class to control a dynaflash projector")
        .def("init", &DynaFlashProjector::init, "initializes the projector")
        .def("is_initialized", &DynaFlashProjector::is_initialized, "returns true if the projector is initialized")
        .def("kill", &DynaFlashProjector::kill, "frees the internal projector resources")
        .def("project_white", nb::overload_cast<>(&DynaFlashProjector::show), "projects a white image")
        .def("project", &DynaFlashProjector::project, nb::arg("data"), nb::arg("bgr") = true, "projects an arbitrary numpy array of type uint8, with shape (height, width, 3), bgr");
}
#endif
#endif DYNAFLASH_H