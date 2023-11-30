#ifndef DIFFUSE_H
#define DIFFUSE_H

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "opencv2/opencv.hpp"

// API: http://127.0.0.1:7860//docs/
// python API: https://github.com/mix1009/sdwebuiapi/blob/main/webuiapi/webuiapi.py#L175
// official docs: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

class Diffuse
{
public:
    Diffuse();
    void print_backend_config();
    void txt2img(const std::string prompt);
    void img2img(const std::string prompt, cv::Mat img);

private:
    // std::string base64_decode(const std::string& input);
    // bool is_base64(unsigned char c) {return (isalnum(c) || (c == '+') || (c == '/'));};
    // std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
};

#endif /* DIFFUSE_H */
