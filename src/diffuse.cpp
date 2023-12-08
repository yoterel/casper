#include "diffuse.h"
#include "HTTPRequest.h"
#include "json.hpp"
#include "base64.h"
#include <chrono>
#include "timer.h"
#include "stb_image.h"
#include "stb_image_write.h"
using json = nlohmann::json;

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

std::vector<uint8_t> Diffuse::txt2img(const std::string prompt,
                                      int &width_out, int &height_out,
                                      int seed,
                                      int width_in, int height_in,
                                      bool OpenCVDecode) // if OpenCVDecode output channels will be BGR
{
    // std::cout << "using prompt: " << prompt << std::endl;
    json j2 = {
        {"prompt", prompt.c_str()},
        {"negative_prompt", ""},
        {"styles", {}}, // json::array() also works, but not json::object()
        {"seed", seed},
        {"subseed", -1},
        {"subseed_strength", 0},
        {"seed_resize_from_h", -1},
        {"seed_resize_from_w", -1},
        {"sampler_name", "Euler a"},
        {"batch_size", 1},
        {"n_iter", 1},
        {"steps", 20},
        {"cfg_scale", 7},
        {"width", width_in},
        {"height", height_in},
        {"restore_faces", false},
        {"tiling", false},
        {"do_not_save_samples", false},
        {"do_not_save_grid", false},
        {"eta", 0},
        {"denoising_strength", 0.7},
        {"s_min_uncond", 0},
        {"s_churn", 0},
        {"s_tmax", 0},
        {"s_tmin", 0},
        {"s_noise", 1},
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
        {"hr_upscaler", "Latent"},
        {"hr_second_pass_steps", 0},
        {"hr_resize_x", 0},
        {"hr_resize_y", 0},
        {"hr_checkpoint_name", ""},
        {"hr_sampler_name", ""},
        {"hr_prompt", ""},
        {"hr_negative_prompt", ""},
        {"sampler_index", "Euler a"},
        {"script_name", ""},
        {"script_args", json::array()}, // nothing but json::array() works
        {"send_images", true},
        {"save_images", false},
        {"alwayson_scripts", json::object()},
    }; // json::array() also works, but not {}
    // std::cout << j2 << std::endl;
    try
    {
        http::Request request{"http://127.0.0.1:7860/sdapi/v1/txt2img"};
        const std::string body = j2.dump();
        const auto response = request.send("POST", body, {{"Content-Type", "application/json"}});
        json j = json::parse(response.body);
        // std::cout << std::setw(4) << j << std::endl;
        // std::cout << j["images"] << std::endl;
        // std::cout <<  std::string(j["images"][0]) << std::endl;
        std::string decoded_string = base64_decode(std::string(j["images"][0]));
        std::vector<uint8_t> data = decode_png(decoded_string, width_out, height_out, OpenCVDecode);
        // std::vector<uint8_t> data(decoded.begin(), decoded.end());
        return data;
        // std::cout << std::string{response.body.begin(), response.body.end()} << '\n'; // print the result
    }
    catch (const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << '\n';
        exit(1);
    }
}

std::vector<uint8_t> Diffuse::img2img(const std::string prompt,
                                      int &width_out, int &height_out,
                                      std::vector<uint8_t> img,
                                      std::vector<uint8_t> mask,
                                      int seed,
                                      int width_in, int height_in, int channels_in,
                                      int width_request, int height_request,
                                      bool inputIsPNGEncoded, bool OpenCVDecode) // if OpenCVDecode output channels will be BGR
{
    std::string encoded_buffer, encoded_mask_buffer;
    if (inputIsPNGEncoded)
    {
        encoded_buffer = std::string("data:image/png;base64,") + base64_encode(std::string(img.begin(), img.end()));
        if (mask.size() > 0)
            encoded_mask_buffer = std::string("data:image/png;base64,") + base64_encode(std::string(mask.begin(), mask.end()));
    }
    else
    {
        std::vector<uint8_t> p = encode_png(img, width_in, height_in, channels_in);
        encoded_buffer = std::string("data:image/png;base64,") + base64_encode(std::string(p.begin(), p.end()));
        if (mask.size() > 0)
        {
            std::vector<uint8_t> p2 = encode_png(mask, width_in, height_in, channels_in);
            encoded_mask_buffer = std::string("data:image/png;base64,") + base64_encode(std::string(p2.begin(), p2.end()));
        }
    }
    json j2 = {
        {"prompt", prompt.c_str()},
        {"negative_prompt", ""},
        {"styles", {}},
        {"seed", seed},
        {"subseed", -1},
        {"subseed_strength", 0},
        {"seed_resize_from_h", 0},
        {"seed_resize_from_w", 0},
        {"sampler_name", "Euler a"},
        {"batch_size", 1},
        {"n_iter", 1},
        {"steps", 20},
        {"cfg_scale", 7},
        {"width", width_request},
        {"height", height_request},
        {"restore_faces", false},
        {"tiling", false},
        {"do_not_save_samples", false},
        {"do_not_save_grid", false},
        {"eta", 1.0}, // todo what is this?
        {"denoising_strength", 0.75},
        // {"s_min_uncond", 0},
        {"s_churn", 0},
        {"s_tmax", 0},
        {"s_tmin", 0},
        {"s_noise", 1},
        {"override_settings", json::object()},
        {"override_settings_restore_afterwards", true},
        // {"refiner_checkpoint", ""},
        // {"refiner_switch_at", 0},
        // {"disable_extra_networks", false},
        // {"comments", json::object()},
        {"init_images", {encoded_buffer.c_str()}},
        {"resize_mode", 0},
        {"image_cfg_scale", 1.5},
        // {"mask_blur_x", 4},
        // {"mask_blur_y", 4},
        {"mask_blur", 4},
        {"inpainting_fill", 2},      // masked content - 0: fill, 1: original, 2: latent noise, 3: latent nothing
        {"inpaint_full_res", false}, // inpaint all, or only inpaint mask
        {"inpaint_full_res_padding", 0},
        {"inpainting_mask_invert", 0},
        {"initial_noise_multiplier", 1},
        {"sampler_index", "Euler a"},
        {"include_init_images", false},
        // {"script_name", ""},
        {"script_args", json::array()}, // or {}, or json::object()
        {"send_images", true},
        {"save_images", false},
        {"alwayson_scripts", json::object()},
    };
    if (mask.size() > 0)
    {
        // j2["latent_mask"] = encoded_mask_buffer.c_str();
        j2["mask"] = encoded_mask_buffer.c_str();
    }
    // std::cout << j2 << std::endl;
    try
    {
        http::Request request{"http://127.0.0.1:7860/sdapi/v1/img2img"};
        const std::string body = j2.dump();
        const auto response = request.send("POST", body, {{"Content-Type", "application/json"}});
        json j = json::parse(response.body);
        std::string decoded_string = base64_decode(std::string(j["images"][0]));
        std::vector<uint8_t> data = decode_png(decoded_string, width_out, height_out, OpenCVDecode);
        return data;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Request failed, error: " << e.what() << '\n';
        exit(1);
    }
}

std::vector<uint8_t> Diffuse::img2img(const std::string prompt,
                                      int &width_out, int &height_out,
                                      cv::Mat img,
                                      cv::Mat mask,
                                      int seed,
                                      int width_request, int height_request,
                                      bool OpenCVDecode) // if OpenCVDecode output channels will be BGR
{
    std::vector<uint8_t> img_buffer, mask_buffer;
    cv::imencode(".png", img, img_buffer);
    cv::imencode(".png", mask, mask_buffer);
    return img2img(prompt,
                   width_out, height_out,
                   img_buffer,
                   mask_buffer, seed,
                   img.cols, img.rows, img.channels(),
                   width_request, height_request,
                   true, OpenCVDecode);
}

std::vector<uint8_t> Diffuse::decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV)
{
    std::vector<uint8_t> data;
    if (useOpenCV)
    {
        std::vector<uint8_t> decoded_vec(png_data.begin(), png_data.end());
        cv::Mat tmp = cv::imdecode(decoded_vec, cv::IMREAD_UNCHANGED);
        // cv::imwrite("tmp.png", tmp1);
        uint8_t *input = (uint8_t *)(tmp.data);
        data = std::vector<uint8_t>(input, input + tmp.total() * tmp.elemSize());
        width = tmp.cols;
        height = tmp.rows;
    }
    else
    {
        int response_bpp;
        uint8_t *tmp = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(png_data.c_str()),
                                             png_data.size(), &width, &height, &response_bpp, NULL);
        // cv::Mat img2img_mat = cv::imdecode(img2img_data, cv::IMREAD_UNCHANGED); // todo which decoding is faster?
        data = std::vector<uint8_t>(tmp, tmp + width * height * response_bpp);
        stbi_image_free(tmp);
        // cv::Mat test2(response_height, response_width, CV_8UC3, data.data());
        // cv::cvtColor(test2, test2, cv::COLOR_RGB2BGR);
        // cv::imwrite("test2.png", test2);
    }
    return data;
}

std::vector<uint8_t> Diffuse::encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels)
{
    int out_len;
    uint8_t *png = stbi_write_png_to_mem(raw_data.data(), channels * width, width, height, channels, &out_len);
    std::vector<uint8_t> data(png, png + out_len);
    return data;
}