#include "diffuse.h"
#include <stdio.h>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std::chrono;
namespace fs = std::filesystem;

void testStableDiffusion();
void testControlNet();

int main(int argc, char *argv[])
{
    // testStableDiffusion();
    testControlNet();
    return 0;
}

void testControlNet()
{
    fs::path item_dir = "../../resource/images";
    fs::path mask_path = item_dir / "mask_dual.png";
    // Load input image
    cv::Mat mask = cv::imread(mask_path.string(), cv::IMREAD_UNCHANGED);
    cv::cvtColor(mask, mask, cv::COLOR_RGBA2GRAY);
    std::vector<uint8_t> mask_buffer(mask.begin<uint8_t>(), mask.end<uint8_t>());
    // Run inference
    ControlNetClient control_net_client = ControlNetClient();
    int preset_id = 1;
    std::string animal = "butterfly";
    std::vector<uint8_t> result_buffer = control_net_client.inference(preset_id,
                                                                      mask_buffer,
                                                                      mask.cols, mask.rows, mask.channels(),
                                                                      animal, 1);

    // Save result
    if (result_buffer.size() > 0)
    {
        cv::Mat result_image = cv::Mat(512, 512, CV_8UC3, result_buffer.data());
        cv::cvtColor(result_image, result_image, cv::COLOR_RGB2BGR);
        cv::imwrite((item_dir / "result.png").string(), result_image);
    }
}

void testStableDiffusion()
{
    // txt2img
    int requested_width = 512;
    int requested_height = 512;
    int txt2img_width, txt2img_height;
    std::vector<uint8_t> txt2img_data = StableDiffusionClient::txt2img("A snow squirrel",
                                                                       txt2img_width, txt2img_height,
                                                                       1003,
                                                                       requested_width, requested_height, false);
    cv::Mat txt2img_result = cv::Mat(txt2img_height, txt2img_width, CV_8UC3, txt2img_data.data()).clone();
    cv::cvtColor(txt2img_result, txt2img_result, cv::COLOR_RGB2BGR);
    cv::imwrite("output_txt2img.png", txt2img_result);

    // img2img
    int img2img_width, img2img_height;
    cv::Mat img2img_result;
    std::vector<uint8_t> img2img_data;
    /* cv::mat */
    // cv::Mat input_img = cv::imread("camera_image.png");
    // cv::Mat input_mask = cv::imread("camera_mask.png");
    // img2img_data = Diffuse::img2img("a human hand with its palm in front, with a colorful parrot tattoo, super realistic",
    //                                 img2img_width, img2img_height,
    //                                 input_img, input_mask, 13,
    //                                 512, 512, false);
    // img2img_result = cv::Mat(img2img_height, img2img_width, CV_8UC3, img2img_data.data());
    // cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
    // cv::imwrite("output_img2img_test.png", img2img_result);
    /* raw buffers */
    cv::Mat gray, mask;
    cv::cvtColor(txt2img_result, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 127, 255, cv::THRESH_BINARY);
    cv::imwrite("output_mask.png", mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    std::vector<uint8_t> mask_buffer(mask.data, mask.data + mask.total() * mask.elemSize());
    // std::vector<uint8_t> mask_buffer;
    img2img_data = StableDiffusionClient::img2img("A glowing red fire squirrel",
                                                  img2img_width, img2img_height,
                                                  txt2img_data, mask_buffer, 333,
                                                  txt2img_width, txt2img_height, 3, 512, 512, false, false);
    img2img_result = cv::Mat(img2img_height, img2img_width, CV_8UC3, img2img_data.data());
    cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
    cv::imwrite("output_img2img.png", img2img_result);
}