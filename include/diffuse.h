#ifndef DIFFUSE_H
#define DIFFUSE_H

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "json.hpp"
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

class ControlNetClient : public Client
{
public:
    std::vector<uint8_t> inference(int preset_payload_num,
                                   const std::vector<uint8_t> &raw_data,
                                   int width, int height, int channels,
                                   std::string animal = "", int retry = 1);

private:
    void changeModel(const std::string &modelName);
    bool txt2img(const json &payload, json &response);
    std::vector<uint8_t> enlarge_mask(const std::vector<uint8_t> &mask, int width, int height, float enlarge_ration = 0.8);
    std::string url = "http://127.0.0.1:7860";
    std::string modelName;
};

class ControlNetPayload
{
public:
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

    ControlNetPayload() = default;

    ControlNetPayload(std::string model, std::string prompt, int steps, float cfg_scale, int width, int height,
                      std::string sampler_name, std::string controlnet_module, float controlnet_weight,
                      float controlnet_guidance_end, float enlarge_ratio)
        : model(model), prompt(prompt), steps(steps), cfg_scale(cfg_scale), width(width), height(height),
          sampler_name(sampler_name), controlnet_module(controlnet_module), controlnet_weight(controlnet_weight),
          controlnet_guidance_end(controlnet_guidance_end), enlarge_ratio(enlarge_ratio)
    {
        this->controlnet_model = getControlNetModel(model, controlnet_module);
    }

    json getPayload(const std::string &encoded_image, const std::string &animal);
    static std::string getControlNetModel(const std::string &model, const std::string &controlnet_module);
    static ControlNetPayload get_preset_payload(int preset_num);
};

class ChatGPTClient : public Client
{
public:
    json send_request(const std::vector<uint8_t> &raw_data,
                      const int width, const int height,
                      const int channels);

    json decode_response(const json &response);
};

#endif /* DIFFUSE_H */
