#ifndef DIFFUSE_H
#define DIFFUSE_H

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
// API: http://127.0.0.1:7860//docs/
// python API: https://github.com/mix1009/sdwebuiapi/blob/main/webuiapi/webuiapi.py#L175
// official docs: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

class Diffuse
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
                                        std::vector<uint8_t> img,
                                        std::vector<uint8_t> mask,
                                        int seed = -1,
                                        int width_in = 512, int height_in = 512, int channels_in = 3,
                                        int width_request = 512, int height_request = 512, bool inputIsPNGEncoded = false, bool OpenCVDecode = false);
    static std::vector<uint8_t> img2img(const std::string prompt,
                                        int &width_out, int &height_out,
                                        cv::Mat img,
                                        cv::Mat mask,
                                        int seed = -1,
                                        int width_request = 512, int height_request = 512, bool OpenCVDecode = true);
    static std::vector<uint8_t> decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV = false);
    static std::vector<uint8_t> encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels = 3);

private:
    Diffuse(){};
    // std::string base64_decode(const std::string& input);
    // bool is_base64(unsigned char c) {return (isalnum(c) || (c == '+') || (c == '/'));};
    // std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
};

#endif /* DIFFUSE_H */
