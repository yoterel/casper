#include "diffuse.h"
#include <stdio.h>

using namespace std::chrono;

int main(int argc, char *argv[])
{
    Diffuse diffuseObject = Diffuse();
    // txt2img
    int requested_width = 720;
    int requested_height = 540;
    int txt2img_width, txt2img_height;
    std::vector<uint8_t> txt2img_data = diffuseObject.txt2img("cute squirrel",
                                                              txt2img_width, txt2img_height,
                                                              1003,
                                                              requested_width, requested_height, true);
    // cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
    cv::Mat txt2img_result = cv::Mat(txt2img_height, txt2img_width, CV_8UC3, txt2img_data.data());
    cv::imwrite("output_txt2img.png", txt2img_result);
    // img2img
    // cv::Mat input_img = cv::imread("output_txt2img.png");
    // std::vector<uint8_t> buffer(input_img.data, input_img.data + input_img.total() * input_img.elemSize());
    // cv::resize(input_img, input_img, cv::Size(512, 512));
    // cv::Mat input_mask(512, 512, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    int img2img_width, img2img_height;
    std::vector<uint8_t> img2img_data = diffuseObject.img2img("a golden fish",
                                                              img2img_width, img2img_height,
                                                              txt2img_data, txt2img_data, 333,
                                                              txt2img_width, txt2img_height, false, true);
    cv::Mat img2img_result = cv::Mat(img2img_height, img2img_width, CV_8UC3, img2img_data.data());
    cv::imwrite("output_img2img.png", img2img_result);
    return 0;
}