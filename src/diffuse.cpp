#include "diffuse.h"
#include "HTTPRequest.h"
#include "json.hpp"
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
    catch (const std::exception& e)
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
        {"styles", {
            ""
        }},
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
        {"alwayson_scripts", json::object()}
    };
    // std::cout << j2 << std::endl;
    try
    {
        http::Request request{"http://127.0.0.1:7860/sdapi/v1/txt2img"};
        const std::string body = j2.dump();
        const auto response = request.send("POST", body, {
            {"Content-Type", "application/json"}
        });
        json j = json::parse(response.body);
        std::cout << j << std::endl;
        // std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    }
    catch (const std::exception& e)
    {
        std::cerr << "Request failed, error: " << e.what() << '\n';
    }
    return true;
}