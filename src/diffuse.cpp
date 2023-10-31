#include "diffuse.h"
#include "HTTPRequest.h"
#include "json.hpp"
#include "base64.h"
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace std::chrono;
using json = nlohmann::json;

Diffuse::Diffuse()
{
    // try
    // {
    //     http::Request request{"http://127.0.0.1:7860"};
    //     const std::string body = "{\"foo\": 1, \"bar\": \"baz\"}";
    //     const auto response = request.send("POST", body, {
    //         {"Content-Type", "application/json"}
    //     });
    //     std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    // }
    // catch (const std::exception& e)
    // {
    //     std::cerr << "Request failed, error: " << e.what() << '\n';
    // }
    // Py_Initialize();
    // PyRun_SimpleString("from time import time,ctime\n"
    //                    "print('Today is',ctime(time()))\n");
    // Py_Finalize();
}

void Diffuse::print_backend_config()
{
    try
    {
        // you can pass http::InternetProtocol::V6 to Request to make an IPv6 request
        http::Request request{"http://127.0.0.1:7860/config/"};

        // send a get request
        const auto response = request.send("GET");
        json j = json::parse(response.body);
        std::cout << j << std::endl;
        // std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    }
    catch (const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << '\n';
    }
}

void Diffuse::txt2img(const std::string prompt)
{
    // std::cout << "using prompt: " << prompt << std::endl;
    json j2 = {
        {"prompt", prompt.c_str()},
        {"negative_prompt", ""},
        {"styles", {""}},
        {"seed", -1},
        {"subseed", -1},
        {"subseed_strength", 0},
        {"seed_resize_from_h", -1},
        {"seed_resize_from_w", -1},
        {"sampler_name", "Euler a"},
        {"batch_size", 1},
        {"n_iter", 1},
        {"steps", 20},
        {"cfg_scale", 7},
        {"width", 512},
        {"height", 512},
        {"restore_faces", false},
        {"tiling", false},
        {"do_not_save_samples", false},
        {"do_not_save_grid", false},
        {"eta", 0},
        {"denoising_strength", 0},
        {"s_min_uncond", 0},
        {"s_churn", 0},
        {"s_tmax", 0},
        {"s_tmin", 0},
        {"s_noise", 0},
        {"override_settings", json::object()},
        {"override_settings_restore_afterwards", true},
        {"refiner_checkpoint", ""},
        {"refiner_switch_at", 0},
        {"disable_extra_networks", false},
        {"comments", json::object()},
        {"enable_hr", false},
        {"firstphase_width", 0},
        {"firstphase_height", 0},
        {"hr_scale", 2},
        {"hr_upscaler", ""},
        {"hr_second_pass_steps", 0},
        {"hr_resize_x", 0},
        {"hr_resize_y", 0},
        {"hr_checkpoint_name", ""},
        {"hr_sampler_name", ""},
        {"hr_prompt", ""},
        {"hr_negative_prompt", ""},
        {"sampler_index", ""},
        {"script_name", ""},
        {"script_args", json::array()},
        {"send_images", true},
        {"save_images", false},
        {"alwayson_scripts", json::object()}};
    // std::cout << j2 << std::endl;
    try
    {
        http::Request request{"http://127.0.0.1:7860/sdapi/v1/txt2img"};
        const std::string body = j2.dump();
        const auto response = request.send("POST", body, {{"Content-Type", "application/json"}});
        auto start = high_resolution_clock::now();
        json j = json::parse(response.body);
        // std::cout << std::setw(4) << j << std::endl;
        // std::cout << j["images"] << std::endl;
        // std::cout <<  std::string(j["images"][0]) << std::endl;
        std::string decoded = base64_decode(std::string(j["images"][0]));
        std::vector<char> data(decoded.begin(), decoded.end());
        cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
        cv::imwrite("output.png", img);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "parse time:" << duration.count() << std::endl;
        // std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    }
    catch (const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << '\n';
    }
}

// std::string Diffuse::base64_decode(const std::string &encoded_string)
// {
//     int in_len = encoded_string.size();
//     int i = 0;
//     int j = 0;
//     int in_ = 0;
//     unsigned char char_array_4[4], char_array_3[3];
//     std::string ret;

//     while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
//     {
//         char_array_4[i++] = encoded_string[in_];
//         in_++;
//         if (i == 4)
//         {
//             for (i = 0; i < 4; i++)
//                 char_array_4[i] = base64_chars.find(char_array_4[i]);

//             char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
//             char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
//             char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

//             for (i = 0; (i < 3); i++)
//                 ret += char_array_3[i];
//             i = 0;
//         }
//     }

//     if (i)
//     {
//         for (j = i; j < 4; j++)
//             char_array_4[j] = 0;

//         for (j = 0; j < 4; j++)
//             char_array_4[j] = base64_chars.find(char_array_4[j]);

//         char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
//         char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
//         char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

//         for (j = 0; (j < i - 1); j++)
//             ret += char_array_3[j];
//     }
//     return ret;
// }