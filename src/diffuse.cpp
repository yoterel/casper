#include "diffuse.h"
#include "HTTPRequest.h"
#include "json.hpp"
#include "base64.h"
#include "timer.h"
#include <chrono>
#include <filesystem>
#include "stb_image.h"
#include "stb_image_write.h"
#include "httplib.h"
using json = nlohmann::json;
namespace fs = std::filesystem;

void StableDiffusionClient::print_backend_config()
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

std::vector<uint8_t> StableDiffusionClient::txt2img(const std::string prompt,
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

std::vector<uint8_t> StableDiffusionClient::img2img(const std::string prompt,
                                                    int &width_out, int &height_out,
                                                    const std::vector<uint8_t> &img,
                                                    const std::vector<uint8_t> &mask,
                                                    int seed,
                                                    int width_in, int height_in, int channels_in,
                                                    int width_request, int height_request,
                                                    bool inputIsPNGEncoded, bool OpenCVDecode,
                                                    int mask_mode) // if OpenCVDecode output channels will be BGR
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
        {"inpainting_fill", mask_mode}, // masked content - 0: fill, 1: original, 2: latent noise, 3: latent nothing
        {"inpaint_full_res", false},    // inpaint all, or only inpaint mask
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
    http::Request request{"http://127.0.0.1:7860/sdapi/v1/img2img"};
    const std::string body = j2.dump();
    const auto response = request.send("POST", body, {{"Content-Type", "application/json"}}, std::chrono::milliseconds(5000));
    json j = json::parse(response.body);
    std::string decoded_string = base64_decode(std::string(j["images"][0]));
    std::vector<uint8_t> data = decode_png(decoded_string, width_out, height_out, OpenCVDecode);
    return data;
}

std::vector<uint8_t> StableDiffusionClient::img2img(const std::string prompt,
                                                    int &width_out, int &height_out,
                                                    cv::Mat img,
                                                    cv::Mat mask,
                                                    int seed,
                                                    int width_request, int height_request,
                                                    bool OpenCVDecode, int mask_mode) // if OpenCVDecode output channels will be BGR
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
                   true, OpenCVDecode, mask_mode);
}

std::vector<uint8_t> StableDiffusionClient::decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV)
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
        stbi_set_flip_vertically_on_load(0);
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

std::vector<uint8_t> StableDiffusionClient::encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels)
{
    int out_len;
    stbi_flip_vertically_on_write(false);
    uint8_t *png = stbi_write_png_to_mem(raw_data.data(), channels * width, width, height, channels, &out_len);
    std::vector<uint8_t> data(png, png + out_len);
    return data;
}

ControlNetClient::ControlNetClient(bool pyinit) : chatGPTClient(pyinit){};

ControlNetPayload ControlNetPayload::get_preset_payload(int preset_num)
{
    std::unordered_map<int, ControlNetPayload> presets = {
        {0, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 20, 7, 512, 512, "DPM++ 2M", "canny", 1.5,
                              0.3, 0.8)},
        {1, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 1.5,
                              0.3, 0.8)},
        {2, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 50, 7, 512, 512, "DPM++ 2M", "canny", 1.5,
                              0.3, 0.8)},
        {3, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 20, 7, 512, 512, "DPM++ 2M", "canny", 1.8,
                              0.3, 0.8)},
        {4, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 1.8,
                              0.3, 0.8)},
        {5, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 2,
                              0.3, 0.8)},
        {6, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 50, 7, 512, 512, "DPM++ 2M", "canny", 2,
                              0.3, 0.8)},
        {7, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 20, 7, 512, 512, "DPM++ 2M", "canny", 1.8,
                              0.5, 0.8)},
        {8, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 1.8,
                              0.5, 0.8)},
        {9, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 20, 7, 512, 512, "DPM++ 2M", "canny", 2,
                              0.5, 0.8)},
        {10, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 2,
                               0.5, 0.8)},
        {11, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 1.5,
                               1, 0.8)},
        {12, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 30, 7, 512, 512, "DPM++ 2M", "canny", 1.8,
                               1, 0.8)},
        {13, ControlNetPayload("dreamshaper_8.safetensors", "a cute ", 50, 7, 512, 512, "DPM++ 2M", "canny", 1.5,
                               1, 0.8)},
        {14, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 5, 3, 512,
                               512, "DPM++ SDE", "canny", 1.5, 0.3, 0.8)},
        {15, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 5, 3, 512,
                               512, "DPM++ SDE", "canny", 1.8, 1, 0.8)},
        {16, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 3, 512,
                               512, "DPM++ SDE", "canny", 1.5, 0.3, 0.8)},
        {17, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 3, 512,
                               512, "DPM++ SDE", "canny", 1.8, 0.3, 0.8)},
        {18, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 1.5,
                               512, 512, "DPM++ SDE", "canny", 2, 0.3, 0.8)},
        {19, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 3, 512,
                               512, "DPM++ SDE", "canny", 1.8, 0.5, 0.8)},
        {20, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 1.5,
                               512, 512, "DPM++ SDE", "canny", 2, 0.5, 0.8)},
        {21, ControlNetPayload("RealVisXL_V3.0_Turbo.safetensors", "National Geographic Wildlife photo of ", 7, 1.5,
                               512, 512, "DPM++ SDE", "canny", 1.5, 1, 0.8)},
        {22, ControlNetPayload("RealVisXL_V3.0.safetensors", "National Geographic Wildlife photo of ", 20, 7, 512, 512,
                               "DPM++ 2M", "canny", 2, 1, 0.8)},
    };

    if (presets.find(preset_num) == presets.end())
    {
        std::cout << "Preset not found, using default preset 0" << std::endl;
        preset_num = 0;
    }

    return presets[preset_num];
}

ControlNetPayload::ControlNetPayload(std::string model, std::string prompt, int steps, float cfg_scale, int width, int height,
                                     std::string sampler_name, std::string controlnet_module, float controlnet_weight,
                                     float controlnet_guidance_end, float enlarge_ratio)
    : model(model), prompt(prompt), steps(steps), cfg_scale(cfg_scale), width(width), height(height),
      sampler_name(sampler_name), controlnet_module(controlnet_module), controlnet_weight(controlnet_weight),
      controlnet_guidance_end(controlnet_guidance_end), enlarge_ratio(enlarge_ratio)
{
    this->controlnet_model = getControlNetModel(model, controlnet_module);
}

json ControlNetPayload::getPayload(const std::string &encoded_image, const std::string &animal, int seed)
{
    json payload = {{"prompt", prompt + animal},
                    {"negative_prompt", "deformed, disfigured, underexposed, overexposed, blurred"},
                    {"batch_size", 1},
                    {"steps", steps},
                    {"cfg_scale", cfg_scale},
                    {"width", width},
                    {"height", height},
                    {"seed", seed},
                    {"subseed", -1},
                    {"sampler_name", sampler_name},
                    {"scheduler", "Karras"},
                    {"alwayson_scripts",
                     {{"ControlNet",
                       {{"args",
                         {{{"enabled", true},
                           {"module", controlnet_module},
                           {"model", controlnet_model},
                           {"weight", controlnet_weight},
                           {"processor_res", 512},
                           {"threshold_a", 100},
                           {"threshold_b", 200},
                           {"image", {{"image", encoded_image.c_str()}}},
                           {"guidance_end", controlnet_guidance_end},
                           {"control_mode", "ControlNet is more important"}}}}}}}}};

    return payload;
}

std::string ControlNetPayload::getControlNetModel(const std::string &model, const std::string &controlnet_module)
{
    static std::unordered_map<std::string, std::unordered_map<std::string, std::string>> mapping = {
        {"RealVisXL_V3.0_Turbo.safetensors",
         {
             {"canny", "diffusers_xl_canny_mid [112a778d]"},
             {"depth", "diffusers_xl_depth_mid [39c49e13]"},
         }},
        {"RealVisXL_V3.0.safetensors",
         {
             {"canny", "control_v11p_sd15_canny [d14c016b]"},
             {"depth", "control_v11f1p_sd15_depth [cfd03158]"},
         }},
        {"meinamix_meinaV11.safetensors",
         {
             {"canny", "control_v11p_sd15_canny [d14c016b]"},
             {"depth", "control_v11f1p_sd15_depth [cfd03158]"},
         }},
        {"dreamshaper_8.safetensors",
         {
             {"canny", "control_v11p_sd15_canny [d14c016b]"},
             {"depth", "control_v11f1p_sd15_depth [cfd03158]"},
         }},
        {"sd_xl_turbo_1.0_fp16.safetensors",
         {
             {"canny", "diffusers_xl_canny_mid [112a778d]"},
             {"depth", "diffusers_xl_depth_mid [39c49e13]"},
         }},
    };

    return mapping[model][controlnet_module];
}

// changes the foundation model to a given model name
void ControlNetClient::changeModel(const std::string &modelName)
{
    if (modelName == this->modelName)
        return;

    http::Request request{url + "/sdapi/v1/options"};
    const std::string body = json{{"sd_model_checkpoint", modelName}}.dump();
    const auto response = request.send("POST", body, {{"Content-Type", "application/json"}});

    if (response.status.code == 200)
    {
        std::cout << "Changed model to " << modelName << std::endl;
    }
    else
    {
        std::cerr << "Failed to change model to " << modelName << std::endl;
    }

    this->modelName = modelName;
}

// sends a http POST request to the txt2img endpoint with the given payload
bool ControlNetClient::txt2img(const json &payload, json &response)
{
    http::Request request{url + "/sdapi/v1/txt2img"};
    const std::string body = payload.dump();
    const auto res = request.send("POST", body, {{"Content-Type", "application/json"}});
    response = json::parse(res.body);
    if (res.status.code == 200)
    {
        return true;
    }
    else
    {
        std::cout << "API Response error! details: " << response << std::endl;
        std::cout << "API Response error code: " << res.status.code << std::endl;
        return false;
    }
}

// runs inference with the given preset payload number and raw data
bool ControlNetClient::inference(const std::vector<uint8_t> &raw_data,
                                 std::vector<uint8_t> &out_data,
                                 int preset_payload_num,
                                 int width, int height, int channels,
                                 int seed,
                                 std::string animal,
                                 bool fit_to_view,
                                 int extra_pad_size,
                                 bool select_top_animal,
                                 bool no_preprompt,
                                 std::string save_path_stem,
                                 std::mutex *py_mutex)
{
    ControlNetPayload payload = ControlNetPayload::get_preset_payload(preset_payload_num);
    if (no_preprompt)
    {
        payload.prompt = "";
    }
    std::vector<uint8_t> resized;
    cv::Rect rect;
    if (fit_to_view) // pad the mask to square and resize to payload size (512x512)
    {
        bool success = fit_mask_to_view(raw_data,
                                        resized,
                                        payload.width,
                                        payload.height,
                                        width, height,
                                        rect, extra_pad_size);
        if (!success)
            return false;
    }
    else // mask will be resized naively to 512x512
    {
        std::vector<uint8_t> raw_vec = raw_data;
        cv::Mat input = cv::Mat(height, width, CV_8UC1, raw_vec.data());
        cv::resize(input, input, cv::Size(payload.width, payload.height));
        resized = std::vector<uint8_t>(input.data, input.data + input.total() * input.elemSize());
    }
    if (!save_path_stem.empty())
    {
        std::string mask_save_path = save_path_stem + "_mask.png";
        cv::Mat cvmask = cv::Mat(payload.width, payload.height, CV_8UC1, resized.data());
        cv::imwrite(mask_save_path, cvmask);
    }
    std::string encoded_image = encode_png(resized, payload.width, payload.height, channels);

    if (animal.empty())
    {
        if (py_mutex != nullptr)
        {
            std::lock_guard<std::mutex> lock(*py_mutex);
            animal = chatGPTClient.get_animal(raw_data, width, height, channels, select_top_animal);
        }
        else
        {
            animal = chatGPTClient.get_animal(raw_data, width, height, channels, select_top_animal);
        }
        if (animal.empty())
        {
            std::cerr << "Failed to get animal from ChatGPT" << std::endl;
            return false;
        }
    }

    auto payload_dict = payload.getPayload(encoded_image, animal, seed);
    std::cout << "prompt: " << payload_dict["prompt"] << std::endl;
    if (!save_path_stem.empty())
    {
        std::string prompt_save_path = save_path_stem + "_prompt.txt";
        std::ofstream myfile;
        myfile.open(prompt_save_path);
        int pad = fit_to_view ? extra_pad_size : -1;
        myfile << payload_dict["prompt"] << ", "
               << "preset: " << preset_payload_num << ", "
               << "seed: " << seed << ", "
               << "pad: " << pad
               << std::endl;
        myfile.close();
    }
    changeModel(payload.model);
    std::vector<uint8_t> result_image;
    try
    {
        json response;
        if (!txt2img(payload_dict, response))
        {
            return false;
        }
        std::string result = base64_decode(std::string(response["images"][0]));
        std::vector decoded = decode_png(result, payload.width, payload.height, false);
        if (!save_path_stem.empty())
        {
            std::string result_save_path = save_path_stem + "_gen.png";
            cv::Mat output = cv::Mat(payload.width, payload.height, CV_8UC3, decoded.data()).clone();
            cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
            cv::imwrite(result_save_path, output);
        }
        if (fit_to_view)
        {
            out_data = fit_sd_to_view(decoded,
                                      payload.width, payload.height,
                                      width, height,
                                      rect, extra_pad_size);
        }
        else // we need to resize back to our original size
        {
            cv::Mat output = cv::Mat(payload.width, payload.height, CV_8UC3, decoded.data());
            cv::resize(output, output, cv::Size(width, height));
            std::vector<uint8_t> result(output.data, output.data + output.total() * output.elemSize());
            // std::vector<uint8_t> result(output.begin<uint8_t>(), output.end<uint8_t>());
            out_data = result;
        }
        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << "Failed to run inference: " << e.what() << std::endl;
        return false;
    }
}

ChatGPTClient::ChatGPTClient(bool pyinit) : m_initialized(false), m_pyinit(pyinit) {}

bool ChatGPTClient::init()
{
    if (m_pyinit)
        Py_Initialize();
    py_chatgpt = PyImport_ImportModule("chatgpt");
    if (!py_chatgpt)
    {
        PyErr_Print();
        std::cerr << "Failed to import chatgpt module" << std::endl;
        return false;
    }
    Py_INCREF(py_chatgpt);
    py_chatgpt_client = PyObject_CallMethod(py_chatgpt, "ChatGPTClient", NULL);
    if (!py_chatgpt_client)
    {
        PyErr_Print();
        std::cerr << "Failed to create ChatGPTClient object" << std::endl;
        return false;
    }
    Py_INCREF(py_chatgpt_client);
    m_initialized = true;
    return true;
}

ChatGPTClient::~ChatGPTClient()
{
    Py_DECREF(py_chatgpt_client);
    Py_DECREF(py_chatgpt);
    if (m_pyinit)
        Py_Finalize();
}

std::string ChatGPTClient::get_animal(const std::vector<uint8_t> &raw_data, const int width, const int height,
                                      const int channels, bool select_top_animal)
{
    std::string response = "";
    if (m_initialized)
    {
        std::string encoded_image = encode_png(raw_data, width, height, channels);
        PyObject *py_encoded_image = PyUnicode_FromString(encoded_image.c_str());
        PyObject *py_response = PyObject_CallMethod(py_chatgpt_client, "send_request", "O,i", py_encoded_image, static_cast<int>(select_top_animal));
        if (!py_response)
        {
            PyErr_Print();
            std::cerr << "Failed to call send_request" << std::endl;
            return {};
        }
        response = PyUnicode_AsUTF8(py_response);
        Py_DECREF(py_response);
        Py_DECREF(py_encoded_image);
    }
    return response;
}

std::vector<uint8_t> Client::decode_png(const std::string &png_data, int &width, int &height, bool useOpenCV)
{
    std::vector<uint8_t> data;
    if (useOpenCV)
    {
        std::vector<uint8_t> decoded_vec(png_data.begin(), png_data.end());
        cv::Mat tmp = cv::imdecode(decoded_vec, cv::IMREAD_UNCHANGED);
        uint8_t *input = (uint8_t *)(tmp.data);
        data = std::vector<uint8_t>(input, input + tmp.total() * tmp.elemSize());
        width = tmp.cols;
        height = tmp.rows;
    }
    else
    {
        int response_bpp;
        stbi_set_flip_vertically_on_load(0);
        uint8_t *tmp = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(png_data.c_str()), png_data.size(),
                                             &width, &height, &response_bpp, NULL);
        data = std::vector<uint8_t>(tmp, tmp + width * height * response_bpp);
        stbi_image_free(tmp);
    }
    return data;
}

std::string Client::encode_png(const std::vector<uint8_t> &raw_data, const int width, const int height, const int channels)
{
    int out_len;
    stbi_flip_vertically_on_write(false);
    uint8_t *png = stbi_write_png_to_mem(raw_data.data(), channels * width, width, height, channels, &out_len);
    std::vector<uint8_t> data(png, png + out_len);
    return std::string("data:image/png;base64,") + base64_encode(std::string(data.begin(), data.end()));
}

bool ControlNetClient::fit_mask_to_view(const std::vector<uint8_t> &mask,
                                        std::vector<uint8_t> &fitted_mask,
                                        int sd_width, int sd_height,
                                        int orig_width, int orig_height,
                                        cv::Rect &rect, int extra_pad)
{
    std::vector<uint8_t> mask_vec = mask; // copy the mask
    cv::Mat mask_mat = cv::Mat(orig_height, orig_width, CV_8UC1, mask_vec.data());
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask_mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2)
         { return c1.size() > c2.size(); });
    switch (contours.size())
    {
    case 0:
    {
        std::cerr << "no contours found in mask" << std::endl;
        return false;
        break;
    }
    case 1:
    {
        rect = cv::boundingRect(contours[0]);
        break;
    }
    default:
        cv::Rect rect1 = cv::boundingRect(contours[0]);
        cv::Rect rect2 = cv::boundingRect(contours[1]);
        rect = rect1 | rect2;
        break;
    }
    // pad to square before resizing
    bool padh = rect.height > rect.width; // should we pad horizontally or vertically?
    int pad = std::max(rect.width, rect.height) - std::min(rect.width, rect.height);
    int pad_parity = pad % 2;
    int pad1 = (pad / 2) + extra_pad;
    int pad2 = (pad / 2) + pad_parity + extra_pad;

    cv::Mat padded;
    cv::copyMakeBorder(mask_mat(rect), padded,
                       padh ? extra_pad : pad1,
                       padh ? extra_pad : pad2,
                       padh ? pad1 : extra_pad,
                       padh ? pad2 : extra_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    if (padded.rows != padded.cols)
    {
        std::cerr << "Padding failed, padded image is not square" << std::endl;
        return false;
    }
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(sd_width, sd_height), cv::INTER_NEAREST);
    std::vector<uint8_t> fitted_mask_buffer(resized.begin<uint8_t>(), resized.end<uint8_t>());
    fitted_mask = fitted_mask_buffer;
    return true;
}

// fits the sd image to the mask size
std::vector<uint8_t> ControlNetClient::fit_sd_to_view(const std::vector<uint8_t> &sd_data,
                                                      int sd_width, int sd_height,
                                                      int orig_width, int orig_height,
                                                      cv::Rect &rect,
                                                      int extra_pad)
{
    std::vector<uint8_t> sd_vec = sd_data; // copy the sd image
    cv::Mat sd_image = cv::Mat(sd_height, sd_width, CV_8UC3, sd_vec.data());
    // compute padding so we can resize back to original size
    if (rect.height != rect.width)
    {
        cv::Mat dummy_image = cv::Mat::zeros(orig_height, orig_width, CV_8UC3);
        // we need to pad to square before resizing
        bool padh = rect.height > rect.width; // should we pad horizontally or vertically?
        int pad = std::max(rect.width, rect.height) - std::min(rect.width, rect.height);
        int pad_parity = pad % 2;
        int pad1 = (pad / 2) + extra_pad;
        int pad2 = (pad / 2) + pad_parity + extra_pad;
        cv::Mat padded;
        cv::copyMakeBorder(dummy_image(rect), padded,
                           padh ? extra_pad : pad1,
                           padh ? extra_pad : pad2,
                           padh ? pad1 : extra_pad,
                           padh ? pad2 : extra_pad,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        if (padded.rows != padded.cols)
        {
            std::cerr << "Padding failed, padded image is not square" << std::endl;
            exit(1);
        }
        cv::Rect myROI(padh ? pad1 : extra_pad, padh ? extra_pad : pad1, rect.width, rect.height);
        cv::resize(sd_image, sd_image, cv::Size(padded.cols, padded.rows), cv::INTER_NEAREST); // resize the square to original size
        sd_image = sd_image(myROI);                                                            // "peel" the padding
    }
    else
    {
        cv::resize(sd_image, sd_image, cv::Size(rect.width, rect.height), cv::INTER_NEAREST); // resize the square to original size
    }

    cv::Mat out_image = cv::Mat::zeros(orig_height, orig_width, CV_8UC3);
    sd_image.copyTo(out_image(rect));
    // std::vector<uint8_t> result(out_image.begin<uint8_t>(), out_image.end<uint8_t>());
    return std::vector<uint8_t>(out_image.data, out_image.data + out_image.total() * out_image.elemSize());
}