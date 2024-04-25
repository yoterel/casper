#ifndef DIFFUSE_H
#define DIFFUSE_H

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "json.hpp"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#define _DEBUG
#else
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#endif

using json = nlohmann::json;
// API: http://127.0.0.1:7860//docs/
// python API: https://github.com/mix1009/sdwebuiapi/blob/main/webuiapi/webuiapi.py#L175
// official docs: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

class Client
{
public:
    std::vector<uint8_t> decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV = false);
    std::string encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels = 3);
};

class StableDiffusionClient
{
public:
    static void print_backend_config();
    static std::vector<uint8_t> txt2img(const std::string prompt,
                                        int &width_out, int &height_out,
                                        int seed = -1,
                                        int width_in = 512, int height_in = 512,
                                        bool OpenCVDecode = false);
    static std::vector<uint8_t> img2img(const std::string prompt,
                                        int &width_out, int &height_out,
                                        const std::vector<uint8_t> &img,
                                        const std::vector<uint8_t> &mask,
                                        int seed = -1,
                                        int width_in = 512, int height_in = 512, int channels_in = 3,
                                        int width_request = 512, int height_request = 512,
                                        bool inputIsPNGEncoded = false, bool OpenCVDecode = false,
                                        int mask_mode = 2);
    static std::vector<uint8_t> img2img(const std::string prompt,
                                        int &width_out, int &height_out,
                                        cv::Mat img,
                                        cv::Mat mask,
                                        int seed = -1,
                                        int width_request = 512, int height_request = 512, bool OpenCVDecode = true,
                                        int mask_mode = 2);
    static std::vector<uint8_t> decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV = false);
    static std::vector<uint8_t> encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels = 3);

private:
    StableDiffusionClient(){};
};

class ChatGPTClient : public Client
{
public:
    ChatGPTClient(bool pyinit = false);
    ~ChatGPTClient();
    std::string get_animal(const std::vector<uint8_t> &raw_data,
                           const int width, const int height, const int channels);

private:
    bool m_pyinit;
    PyObject *py_chatgpt;
    PyObject *py_chatgpt_client;
};

class ControlNetClient : public Client
{
public:
    ControlNetClient(bool pyinit = false);
    bool inference(const std::vector<uint8_t> &raw_data,
                   std::vector<uint8_t> &out_data,
                   int preset_payload_num,
                   int width, int height, int channels,
                   int seed = -1,
                   std::string animal = "",
                   bool fit_to_view = true,
                   int extra_pad_size = 50);

private:
    void changeModel(const std::string &modelName);
    bool txt2img(const json &payload, json &response);
    std::vector<uint8_t> fit_mask_to_view(const std::vector<uint8_t> &mask,
                                          int sd_width, int sd_height,
                                          int orig_width, int orig_height,
                                          cv::Rect &rect, int extra_pad);
    std::vector<uint8_t> fit_sd_to_view(const std::vector<uint8_t> &sd,
                                        int sd_width, int sd_height,
                                        int orig_width, int orig_height,
                                        cv::Rect &rect, int extra_pad);
    std::string url = "http://127.0.0.1:7860";
    std::string modelName;
    ChatGPTClient chatGPTClient;
};

class ControlNetPayload
{
public:
    ControlNetPayload() = default;

    ControlNetPayload(std::string model, std::string prompt, int steps, float cfg_scale, int width, int height,
                      std::string sampler_name, std::string controlnet_module, float controlnet_weight,
                      float controlnet_guidance_end, float enlarge_ratio);
    json getPayload(const std::string &encoded_image, const std::string &animal, int seed);
    static std::string getControlNetModel(const std::string &model, const std::string &controlnet_module);
    static ControlNetPayload get_preset_payload(int preset_num);

    std::string model;
    std::string prompt;
    int steps;
    float cfg_scale;
    int width;
    int height;
    std::string sampler_name;
    std::string controlnet_module;
    std::string controlnet_model;
    float controlnet_weight;
    float controlnet_guidance_end;
    float enlarge_ratio;
};

#endif /* DIFFUSE_H */
