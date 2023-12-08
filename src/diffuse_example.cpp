#include "diffuse.h"
#include <stdio.h>

using namespace std::chrono;

int main(int argc, char *argv[])
{
    Diffuse diffuseObject = Diffuse();
    // txt2img
    int requested_width = 512;
    int requested_height = 512;
    int txt2img_width, txt2img_height;
    std::vector<uint8_t> txt2img_data = diffuseObject.txt2img("A snow squirrel",
                                                              txt2img_width, txt2img_height,
                                                              1003,
                                                              requested_width, requested_height, false);
    // cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
    cv::Mat txt2img_result = cv::Mat(txt2img_height, txt2img_width, CV_8UC3, txt2img_data.data()).clone();
    cv::cvtColor(txt2img_result, txt2img_result, cv::COLOR_RGB2BGR);
    cv::imwrite("output_txt2img.png", txt2img_result);
    // img2img
    // cv::Mat input_img = cv::imread("output_txt2img.png");
    // std::vector<uint8_t> buffer(input_img.data, input_img.data + input_img.total() * input_img.elemSize());
    // cv::resize(input_img, input_img, cv::Size(512, 512));
    // cv::Mat mask(txt2img_height, txt2img_width, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat gray, mask;
    cv::cvtColor(txt2img_result, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 127, 255, cv::THRESH_BINARY);
    cv::imwrite("output_mask.png", mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    std::vector<uint8_t> mask_buffer(mask.data, mask.data + mask.total() * mask.elemSize());
    // std::vector<uint8_t> mask_buffer;
    int img2img_width, img2img_height;
    std::vector<uint8_t> img2img_data = diffuseObject.img2img("A glowing (red:1.5) fire squirrel",
                                                              img2img_width, img2img_height,
                                                              txt2img_data, mask_buffer, 333,
                                                              txt2img_width, txt2img_height, false, false);
    cv::Mat img2img_result = cv::Mat(img2img_height, img2img_width, CV_8UC3, img2img_data.data());
    cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
    cv::imwrite("output_img2img.png", img2img_result);
    return 0;
}