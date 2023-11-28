#include <iostream>
#include <thread>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "queue.h"
#include "camera.h"
#include "helpers.h"
#include "display.h"
#include "cnpy.h"
#include "SerialPort.h"
#include "shader.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "leap.h"
#include "text.h"
#include "post_process.h"
#include "point_cloud.h"
#include "image_process.h"
#include "stb_image_write.h"
#include <filesystem>
namespace fs = std::filesystem;

// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void processInput(GLFWwindow *window);
void initGLBuffers(unsigned int *pbo);
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

// global settings
bool cam_color_mode = false;
bool leap_undistort = false;
bool load_calib = false;
bool load_points = false;
int points_to_display = 0;
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_width = 720;
const unsigned int cam_height = 540;
unsigned int n_cam_channels = cam_color_mode ? 4 : 1;
unsigned int cam_buffer_format = cam_color_mode ? GL_RGBA : GL_RED;
float exposure = 10000.0f;
// global state
int cur_window_width = cam_width;
int cur_window_height = cam_height;
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
int CHECKERBOARD[2]{10, 7};
float deltaTime = 0.0f;
glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec2 cur_mouse_pos = glm::vec2(0.5f, 0.5f);
std::vector<glm::vec2> screen_vert = {{0.5f, 0.5f}};
std::vector<glm::vec3> screen_vert_color = {{1.0f, 0.0f, 0.0f}};
std::vector<cv::Point2f> image_points;
std::vector<cv::Point3f> object_points;
std::vector<float> obj_to_save;
std::vector<float> img_to_save;
std::vector<glm::vec2> reprojected_image_points;
std::vector<glm::vec2> proj_verts;
std::vector<glm::vec2> cam_verts;
std::vector<glm::vec2> leap1_verts;
std::vector<glm::vec2> leap2_verts;
float leap_width = 0.0f;
float leap_height = 0.0f;
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
int64_t lastFrameID = 0;
bool space_modifier = false;
bool shift_modifier = false;
bool ctrl_modifier = false;
unsigned int n_bones = 0;
int state_machine = 0;
int n_user_locations = 0;
const int max_user_locations = 30;
bool right_pressed = false;
bool confirm_flag = false;
bool left_pressed = false;
bool up_pressed = false;
bool down_pressed = false;
bool enter_pressed = false;
bool dragging = false;
float screen_z = -10.0f;
bool hand_in_frame = false;
const unsigned int num_texels = proj_width * proj_height;
const unsigned int image_size = num_texels * 3 * sizeof(uint8_t);

int main(int argc, char *argv[])
{
    /* init GLFW */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow *window = glfwCreateWindow(cam_width, cam_height, "augmented_hands", NULL, NULL); // monitors[0], NULL for full screen
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    int secondary_screen_x, secondary_screen_y;
    glfwGetMonitorPos(monitors[1], &secondary_screen_x, &secondary_screen_y);
    glfwSetWindowMonitor(window, NULL, secondary_screen_x + 100, secondary_screen_y + 100, cam_width, cam_height, GLFW_DONT_CARE);
    glfwSetWindowMonitor(window, NULL, secondary_screen_x + 150, secondary_screen_y + 150, cam_width, cam_height, GLFW_DONT_CARE); // really glfw?
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    std::cout << "Succeeded to create GL window." << std::endl;
    std::cout << "  GL Version   : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "  GL Vendor    : " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "  GL Renderer  : " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "  GLSL Version : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << std::endl;

    // glfwSwapInterval(0);                       // do not sync to monitor
    glViewport(0, 0, cam_width, cam_height); // set viewport
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glPointSize(3.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // callback for resizing window
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_TRUE);
    Quad fullScreenQuad(0.0f);
    Timer t_app;
    t_app.start();
    Text text("../../resource/arial.ttf");
    // FBO hands_fbo(proj_width, proj_height);
    // FBO postprocess_fbo(proj_width, proj_height);
    /* setup shaders*/
    Shader leapUndistortShader("../../src/shaders/leap_undistort.vs", "../../src/shaders/leap_undistort.fs");
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    Shader vertexShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    /* more inits */
    double previousAppTime = t_app.getElapsedTimeInSec();
    double previousSecondAppTime = t_app.getElapsedTimeInSec();
    double currentAppTime = t_app.getElapsedTimeInSec();
    double whole = 0.0;
    long frameCount = 0;
    int64_t targetFrameTime = 0;
    uint64_t targetFrameSize = 0;
    bool close_signal = false;
    CGrabResultPtr ptrGrabResult;
    cv::Mat cur_cam_image;
    cv::Mat cur_image_copy;
    std::vector<cv::Point2f> cur_corner_pts;
    // Texture camTexture = Texture();
    // camTexture.init(cam_width, cam_height, n_cam_channels);
    Texture displayTexture = Texture();
    displayTexture.init(cam_width, cam_height, n_cam_channels);
    blocking_queue<CGrabResultPtr> camera_queue;
    // queue_spsc<cv::Mat> camera_queue_cv(50);
    blocking_queue<cv::Mat> camera_queue_cv;
    BaslerCamera camera;
    LeapConnect leap(true, true);
    cv::Mat tvec, rvec;
    if (load_calib)
    {
        try
        {
            cnpy::NpyArray w2c_npy = cnpy::npy_load("../../resource/calibrations/leap_calibration/w2c.npy");
            cv::Mat w2c(4, 4, CV_64FC1, w2c_npy.data<double>());
            cv::Mat rot_mat = w2c(cv::Range(0, 3), cv::Range(0, 3)).clone();
            tvec = w2c(cv::Range(0, 3), cv::Range(3, 4)).clone();
            rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            // cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Rodrigues(rot_mat, rvec);
            // std::cout << "w2c: " << w2c << std::endl;
            // std::cout << "tvec: " << tvec << std::endl;
            // std::cout << "rot_mat: " << rot_mat << std::endl;
            // std::cout << "rvec: " << rvec << std::endl;
        }
        catch (std::runtime_error &e)
        {
            std::cout << e.what() << std::endl;
            exit(1);
        }
    }
    if (load_points)
    {
        try
        {
            cnpy::NpyArray points_2d_npy = cnpy::npy_load("../../resource/calibrations/leap_calibration/2dpoints.npy");
            cnpy::NpyArray points_3d_npy = cnpy::npy_load("../../resource/calibrations/leap_calibration/3dpoints.npy");
            img_to_save = std::vector<float>(points_2d_npy.data<float>(), points_2d_npy.data<float>() + points_2d_npy.num_vals);
            obj_to_save = std::vector<float>(points_3d_npy.data<float>(), points_3d_npy.data<float>() + points_3d_npy.num_vals);
            for (int i = 0; i < img_to_save.size(); i += 2)
            {
                image_points.push_back(cv::Point2f(img_to_save[i], img_to_save[i + 1]));
            }
            for (int i = 0; i < obj_to_save.size(); i += 3)
            {
                object_points.push_back(cv::Point3f(obj_to_save[i], obj_to_save[i + 1], obj_to_save[i + 2]));
            }
        }
        catch (std::runtime_error &e)
        {
            std::cout << e.what() << std::endl;
            exit(1);
        }
        state_machine = 5;
    }
    /* actual thread loops */
    /* image producer (real camera = virtual projector) */
    if (camera.init(camera_queue, close_signal, cam_height, cam_width, exposure))
    {
        /* real producer */
        std::cout << "Using real camera to produce images" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // camera.balance_white();
        camera.acquire();
    }
    else
    {
        std::cout << "camera init failed" << std::endl;
        exit(1);
    }

    /* main loop */
    while (!glfwWindowShouldClose(window))
    {
        /* update / sync clocks */
        currentAppTime = t_app.getElapsedTimeInSec(); // glfwGetTime();
        deltaTime = static_cast<float>(currentAppTime - previousAppTime);
        previousAppTime = currentAppTime;
        frameCount++;

        /* display stats */
        if (currentAppTime - previousSecondAppTime >= 1.0)
        {
            fps = frameCount;
            ms_per_frame = 1000.0f / frameCount;
            // std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
            // std::cout << "total app: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
        }
        /* deal with user input */
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        /* deal with camera input */
        ptrGrabResult = camera_queue.pop();
        cur_cam_image = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer());
        cv::flip(cur_cam_image, cur_cam_image, 1);
        // camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
        switch (state_machine)
        {
        case 0:
        {
            displayTexture.load((uint8_t *)cur_cam_image.data, true, cam_buffer_format);
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipHor", false);
            textureShader.setBool("flipVer", true);
            textureShader.setBool("binary", false);
            textureShader.setBool("isGray", true);
            textureShader.setInt("src", 0);
            displayTexture.bind();
            fullScreenQuad.render();
            if (load_calib)
            {
                if (n_user_locations >= max_user_locations)
                {
                    n_user_locations = 0;
                    state_machine += 3;
                }
            }
            break;
        }
        case 1:
        {
            cur_image_copy = cur_cam_image.clone();
            cur_corner_pts.clear();
            bool success = cv::findChessboardCorners(cur_cam_image, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), cur_corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (success)
            {
                cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
                cv::cornerSubPix(cur_cam_image, cur_corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
                cv::drawChessboardCorners(cur_image_copy, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), cur_corner_pts, success);
                std::reverse(cur_corner_pts.begin(), cur_corner_pts.end());
                cam_verts.clear();
                for (int i = 0; i < max_user_locations; i++)
                {
                    glm::vec2 point = Helpers::ScreenToNDC(glm::vec2(cur_corner_pts[i].x, cur_corner_pts[i].y), cam_width, cam_height, false);
                    // glm::vec2 point = glm::vec2((2.0f * cur_corner_pts[i].x / cam_width) - 1.0f, ((2.0f * cur_corner_pts[i].y / cam_height) - 1.0f));
                    cam_verts.push_back(point);
                }
                state_machine += 1;
                n_user_locations = 0;
            }
            else
            {
                state_machine = 0;
                n_user_locations = 0;
            }
            break;
        }
        case 2:
        {
            displayTexture.load((uint8_t *)cur_image_copy.data, true, cam_buffer_format);
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipHor", false);
            textureShader.setInt("src", 0);
            textureShader.setBool("binary", false);
            textureShader.setBool("isGray", true);
            displayTexture.bind();
            fullScreenQuad.render();
            break;
        }
        case 3:
        {
            std::vector<uint8_t> buffer1, buffer2;
            uint32_t width, height;
            if (!leap.getImage(buffer1, buffer2, width, height))
                exit(1);
            leap_width = width;
            leap_height = height;
            // FBO test_fbo(width, height, 1);
            Texture leapTexture = Texture();
            leapTexture.init(width, height, 1);
            // cv::Mat leap_image = cv::Mat(height, width, CV_8UC1, buffer1.data());
            // cv::Mat leap_image2;
            // std::vector<uint8_t> leap_image_vector(leap_image.begin<uint8_t>(), leap_image.end<uint8_t>());
            leapTexture.load(buffer1, true, cam_buffer_format);
            // leapTexture.load((uint8_t *)leap_image.data, true, cam_buffer_format);
            // cv::resize(leap_image, leap_image, cv::Size(cam_width, cam_height));
            leapTexture.bind();
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipVer", true);
            textureShader.setInt("src", 0);
            textureShader.setBool("isGray", true);
            textureShader.setBool("binary", false);
            fullScreenQuad.render();
            if (n_user_locations >= max_user_locations)
            {
                n_user_locations = 0;
                state_machine += 1;
            }
            break;
        }
        case 4:
        {
            std::vector<uint8_t> buffer1, buffer2;
            uint32_t width, height;
            if (!leap.getImage(buffer1, buffer2, width, height))
                exit(1);
            leap_width = width;
            leap_height = height;
            // FBO test_fbo(width, height, 1);
            Texture leapTexture = Texture();
            leapTexture.init(width, height, 1);
            cv::Mat leap_image = cv::Mat(height, width, CV_8UC1, buffer2.data());
            // cv::Mat leap_image2;
            if (leap_undistort)
            {
                /*
                std::vector<float> dist_coeffs_raw(8);
                std::vector<float> intrinsics_raw(9);
                LeapCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, intrinsics_raw.data());
                cv::Mat intrinsics(3, 3, CV_32F, intrinsics_raw.data());
                // intrinsics.convertTo(intrinsics, CV_64F);
                LeapDistortionCoeffs(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, dist_coeffs_raw.data());
                cv::Mat distortion_coeffs(8, 1, CV_32F, dist_coeffs_raw.data());
                // distortion_coeffs.convertTo(distortion_coeffs, CV_64F);
                // std::cout << intrinsics << std::endl;
                // std::cout << distortion_coeffs << std::endl;
                cv::undistort(leap_image, leap_image2, intrinsics, distortion_coeffs);
                leapTexture.load((uint8_t *)leap_image2.data, true, cam_buffer_format);
                */
                // leapTexture.load((uint8_t *)leap_image.data, true, cam_buffer_format);
                // std::vector<float> dist_buffer1, dist_buffer2;
                // uint32_t dist_width, dist_height;
                // if (!leap.getDistortion(dist_buffer1, dist_buffer2, dist_width, dist_height))
                //     exit(1);
                // Texture distortionTexture = Texture();
                // distortionTexture.init((uint8_t *)dist_buffer2.data(), dist_width, dist_height, 2);
                // test_fbo.bind();
                // leapUndistortShader.use();
                // leapUndistortShader.setInt("src", 0);
                // leapUndistortShader.setInt("distortion_map", 1);
                // leapTexture.bind(GL_TEXTURE0);
                // distortionTexture.bind(GL_TEXTURE1);
                // fullScreenQuad.render();
                // test_fbo.unbind();
                // glViewport(0, 0, cam_width, cam_height);
                // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                // // test_fbo.saveColorToFile("test.png");
                // test_fbo.getTexture()->bind(GL_TEXTURE0);
            }
            // std::vector<uint8_t> leap_image_vector(leap_image.begin<uint8_t>(), leap_image.end<uint8_t>());
            leapTexture.load(buffer2, true, cam_buffer_format);
            // leapTexture.load((uint8_t *)leap_image.data, true, cam_buffer_format);
            // cv::resize(leap_image, leap_image, cv::Size(cam_width, cam_height));
            leapTexture.bind();
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipVer", true);
            textureShader.setInt("src", 0);
            textureShader.setBool("isGray", true);
            textureShader.setBool("binary", false);
            fullScreenQuad.render();
            if (n_user_locations >= max_user_locations)
            {
                state_machine += 1;
                n_user_locations = 0;
            }
            break;
        }
        case 5:
        {
            if (!load_points)
            {
                // extract 3d points from leap1_verts and leap2_verts
                // first get rays from the leap camera corrected for distortion, in 2D camera space
                std::vector<LEAP_VECTOR> leap1_rays_2d, leap2_rays_2d;
                for (int i = 0; i < leap1_verts.size(); i++)
                {
                    glm::vec2 l1_vert = Helpers::NDCtoScreen(leap1_verts[i], leap_width, leap_height, false);
                    LEAP_VECTOR l1_vert_leap = {l1_vert.x, l1_vert.y, 1.0f};
                    glm::vec2 l2_vert = Helpers::NDCtoScreen(leap2_verts[i], leap_width, leap_height, false);
                    LEAP_VECTOR l2_vert_leap = {l2_vert.x, l2_vert.y, 1.0f};
                    // LEAP_VECTOR l1_verts = {leap_width * (leap1_verts[i].x + 1) / 2, leap_height * (leap1_verts[i].y + 1) / 2, 1.0f};
                    // LEAP_VECTOR l2_verts = {leap_width * (leap2_verts[i].x + 1) / 2, leap_height * (leap2_verts[i].y + 1) / 2, 1.0f};
                    LEAP_VECTOR l1_ray = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, l1_vert_leap);
                    leap1_rays_2d.push_back(l1_ray);
                    LEAP_VECTOR l2_ray = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, l2_vert_leap);
                    leap2_rays_2d.push_back(l2_ray);
                }
                // second convert rays to 3D leap space using the extrinsics matrix
                glm::mat4 leap1_extrinsic = glm::mat4(1.0f);
                glm::mat4 leap2_extrinsic = glm::mat4(1.0f);
                LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, glm::value_ptr(leap1_extrinsic));
                LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, glm::value_ptr(leap2_extrinsic));
                /*
                std::vector<cv::Point3f> leap1_ray_dirs_3d, leap2_ray_dirs_3d;
                cv::Point3f leap1_origin = {leap1_extrinsic[3][0], leap1_extrinsic[3][1], leap1_extrinsic[3][2]};
                cv::Point3f leap2_origin = {leap2_extrinsic[3][0], leap2_extrinsic[3][1], leap2_extrinsic[3][2]};
                for (int i = 0; i < leap1_verts.size(); i++)
                {
                    glm::vec4 leap1_ray_3d = glm::inverse(leap1_extrinsic) * glm::vec4(leap1_rays_2d[i].x, leap1_rays_2d[i].y, leap1_rays_2d[i].z, 1.0f);
                    leap1_ray_3d.x /= leap1_ray_3d.z;
                    leap1_ray_3d.y /= leap1_ray_3d.z;
                    leap1_ray_dirs_3d.push_back(cv::Point3f(leap1_ray_3d.x, leap1_ray_3d.y, 1.0f));
                    glm::vec4 leap2_ray_3d = glm::inverse(leap2_extrinsic) * glm::vec4(leap2_rays_2d[i].x, leap2_rays_2d[i].y, leap2_rays_2d[i].z, 1.0f);
                    leap2_ray_3d.x /= leap2_ray_3d.z;
                    leap2_ray_3d.y /= leap2_ray_3d.z;
                    leap2_ray_dirs_3d.push_back(cv::Point3f(leap2_ray_3d.x, leap2_ray_3d.y, 1.0f));
                }
                // triangulate the 3D rays to get the 3D points
                for (int i = 0; i < leap2_ray_dirs_3d.size(); i++)
                {
                    cv::Point3f point = approximate_ray_intersection(leap1_ray_dirs_3d[i], leap1_origin, leap2_ray_dirs_3d[i], leap2_origin, NULL, NULL, NULL);
                    object_points.push_back(point);
                }
                */
                float baseline = leap2_extrinsic[3][0] - leap1_extrinsic[3][0];
                for (int i = 0; i < leap1_rays_2d.size(); i++)
                {
                    // see https://forums.leapmotion.com/t/sdk-2-1-raw-data-get-pixel-position-xyz/1604/12
                    float z = baseline / (leap2_rays_2d[i].x - leap1_rays_2d[i].x);
                    float alty1 = z * -leap2_rays_2d[i].y; // reason for negative is that the y direction is flipped in openGL
                    float alty2 = z * -leap1_rays_2d[i].y; // reason for negative is that the y direction is flipped in openGL
                    float x = z * leap2_rays_2d[i].x - baseline / 2.0f;
                    float altx = z * leap1_rays_2d[i].x + baseline / 2.0f;
                    float y = (alty1 + alty2) / 2.0f;
                    object_points.push_back(cv::Point3f(x, -z, y));
                    obj_to_save.push_back(x);
                    obj_to_save.push_back(-z);
                    obj_to_save.push_back(y);
                }
                for (int i = 0; i < cam_verts.size(); i++)
                {
                    // image_points.push_back(cv::Point2f(proj_width * (proj_verts[i].x + 1) / 2, proj_height * (proj_verts[i].y + 1) / 2));
                    glm::vec2 point = Helpers::NDCtoScreen(cam_verts[i], cam_width, cam_height, false);
                    image_points.push_back(cv::Point2f(point.x, point.y));
                    img_to_save.push_back(point.x);
                    img_to_save.push_back(point.y);
                    // image_points.push_back(cv::Point2f(cam_width * (cam_verts[i].x + 1) / 2, cam_height * (cam_verts[i].y + 1) / 2));
                }
            }
            std::cout << "object_points: " << std::endl;
            for (int i = 0; i < object_points.size(); i++)
            {
                std::cout << object_points[i] << std::endl;
            }
            std::cout << "image_points: " << std::endl;
            for (int i = 0; i < image_points.size(); i++)
            {
                std::cout << image_points[i] << std::endl;
            }
            // use solve pnp to find transformation of projector to leap space
            cnpy::npz_t my_npz;
            try
            {
                my_npz = cnpy::npz_load("../../resource/calibrations/cam_calibration/cam_calibration.npz");
            }
            catch (std::runtime_error &e)
            {
                std::cout << e.what() << std::endl;
                exit(1);
            }
            // glm::mat3 camera_intrinsics = glm::make_mat3(my_npz["cam_intrinsics"].data<double>());
            cv::Mat camera_intrinsics(3, 3, CV_64F, my_npz["cam_intrinsics"].data<double>());
            cv::Mat distortion_coeffs(5, 1, CV_64F, my_npz["cam_distortion"].data<double>());
            // std::cout << "camera_intrinsics: " << camera_intrinsics << std::endl;
            // std::cout << "distortion_coeffs: " << distortion_coeffs << std::endl;
            if (!load_calib)
            {
                // rvec = cv::Mat::zeros(3, 1, CV_64FC1);
                // tvec = cv::Mat::zeros(3, 1, CV_64FC1);
                // construct initial guess with magic numbers
                cv::Mat transform = cv::Mat::zeros(4, 4, CV_64FC1);
                transform.at<double>(0, 0) = -1.0f;
                transform.at<double>(1, 1) = 1.0f;
                transform.at<double>(2, 2) = -1.0f;
                transform.at<double>(3, 3) = 1.0f;
                transform.at<double>(0, 3) = 0.0f;
                transform.at<double>(1, 3) = 300.0f;
                transform.at<double>(2, 3) = 1000.0f;
                // std::cout << "transform: " << transform << std::endl;
                // cv::Mat rotmat = transform(cv::Range(0, 3), cv::Range(0, 3)).clone();
                // std::cout << "rot_mat: " << rotmat << std::endl;
                // tvec = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
                // std::cout << "tvec: " << tvec << std::endl;
                cv::invert(transform, transform);
                // std::cout << "transform_inverse: " << transform << std::endl;
                cv::Mat rotmat_inverse = transform(cv::Range(0, 3), cv::Range(0, 3)).clone();
                // std::cout << "rot_mat_inverse: " << transform << std::endl;
                cv::Rodrigues(rotmat_inverse, rvec);
                // std::cout << "rvec_inverse: " << rvec << std::endl;
                tvec = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
                // std::cout << "tvec_inverse: " << tvec << std::endl;
                cv::solvePnP(object_points, image_points, camera_intrinsics, distortion_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                // std::cout << "rvec: " << rvec << std::endl;
                // std::cout << "tvec: " << tvec << std::endl;
                cv::Mat rot_mat(3, 3, CV_64FC1);
                cv::Rodrigues(rvec, rot_mat);
                // std::cout << "rotmat: " << rot_mat << std::endl;
                cv::Mat w2c(4, 4, CV_64FC1);
                w2c.at<double>(0, 0) = rot_mat.at<double>(0, 0);
                w2c.at<double>(0, 1) = rot_mat.at<double>(0, 1);
                w2c.at<double>(0, 2) = rot_mat.at<double>(0, 2);
                w2c.at<double>(0, 3) = tvec.at<double>(0, 0);
                w2c.at<double>(1, 0) = rot_mat.at<double>(1, 0);
                w2c.at<double>(1, 1) = rot_mat.at<double>(1, 1);
                w2c.at<double>(1, 2) = rot_mat.at<double>(1, 2);
                w2c.at<double>(1, 3) = tvec.at<double>(1, 0);
                w2c.at<double>(2, 0) = rot_mat.at<double>(2, 0);
                w2c.at<double>(2, 1) = rot_mat.at<double>(2, 1);
                w2c.at<double>(2, 2) = rot_mat.at<double>(2, 2);
                w2c.at<double>(2, 3) = tvec.at<double>(2, 0);
                w2c.at<double>(3, 0) = 0.0f;
                w2c.at<double>(3, 1) = 0.0f;
                w2c.at<double>(3, 2) = 0.0f;
                w2c.at<double>(3, 3) = 1.0f;
                std::cout << "w2c: " << w2c << std::endl;
                cv::Mat c2w = w2c.inv();
                std::cout << "c2w: " << c2w << std::endl;
                std::vector<double> w2c_vec(w2c.begin<double>(), w2c.end<double>());
                cnpy::npy_save("../../resource/calibrations/leap_calibration/w2c.npy", w2c_vec.data(), {4, 4}, "w");
                cnpy::npy_save("../../resource/calibrations/leap_calibration/3dpoints.npy", obj_to_save.data(), {max_user_locations, 3}, "w");
                cnpy::npy_save("../../resource/calibrations/leap_calibration/2dpoints.npy", img_to_save.data(), {max_user_locations, 2}, "w");
                // cnpy::npy_save("../../resource/calibrations/leap_calibration/c2w.npy", c2w.data, {4, 4}, "w");
            }
            std::vector<cv::Point2f> reproj_image_points;
            cv::projectPoints(object_points, rvec, tvec, camera_intrinsics, distortion_coeffs, reproj_image_points);
            std::cout << "reprojected: " << std::endl;
            for (int i = 0; i < reproj_image_points.size(); i++)
            {
                std::cout << reproj_image_points[i] << std::endl;
            }
            for (int i = 0; i < reproj_image_points.size(); i++)
            {
                glm::vec2 point = Helpers::ScreenToNDC(glm::vec2(reproj_image_points[i].x, reproj_image_points[i].y), cam_width, cam_height, true);
                // glm::vec2 point = glm::vec2((2.0f * reproj_image_points[i].x / cam_width) - 1.0f, ((2.0f * reproj_image_points[i].y / cam_height) - 1.0f));
                reprojected_image_points.push_back(point);
            }
            state_machine += 1;
            break;
        }
        case 6:
        {
            displayTexture.load((uint8_t *)cur_cam_image.data, true, cam_buffer_format);
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipHor", false);
            textureShader.setBool("flipVer", true);
            textureShader.setBool("binary", false);
            textureShader.setBool("isGray", true);
            textureShader.setInt("src", 0);
            displayTexture.bind();
            fullScreenQuad.render();
            vertexShader.use();
            vertexShader.setMat4("view", glm::mat4(1.0f));
            vertexShader.setMat4("projection", glm::mat4(1.0f));
            vertexShader.setMat4("model", glm::mat4(1.0f));
            PointCloud pointCloud(reprojected_image_points, screen_vert_color);
            pointCloud.render();
            break;
        }
        default:
        {
            break;
        }
        }
        // render user marked points
        {
            screen_vert.clear();
            if (dragging)
            {
                double x;
                double y;
                glfwGetCursorPos(window, &x, &y);
                cur_mouse_pos = Helpers::ScreenToNDC(glm::vec2(x, y), cur_window_width, cur_window_height, false);
                // cur_mouse_pos = glm::vec2((2.0f * x / cam_width) - 1.0f, -1.0f * ((2.0f * y / cam_height) - 1.0f));
            }
            screen_vert.push_back(glm::vec2(cur_mouse_pos.x, -cur_mouse_pos.y));
            if (state_machine < 3)
            {
                if (cam_verts.size() > 0)
                {
                    int limit;
                    if (load_calib)
                        limit = n_user_locations;
                    else
                        limit = points_to_display;
                    for (int i = 0; i < limit; i++)
                    {
                        screen_vert.push_back(glm::vec2(cam_verts[i].x, -cam_verts[i].y));
                    }
                }
            }
            if (state_machine == 3)
            {
                if (leap1_verts.size() > 0)
                {
                    for (int i = 0; i < leap1_verts.size(); i++)
                    {
                        screen_vert.push_back(glm::vec2(leap1_verts[i].x, -leap1_verts[i].y));
                    }
                }
            }
            if (state_machine == 4)
            {
                if (leap2_verts.size() > 0)
                {
                    for (int i = 0; i < leap2_verts.size(); i++)
                    {
                        screen_vert.push_back(glm::vec2(leap2_verts[i].x, -leap2_verts[i].y));
                    }
                }
            }
            vertexShader.use();
            vertexShader.setMat4("view", glm::mat4(1.0f));
            vertexShader.setMat4("projection", glm::mat4(1.0f));
            vertexShader.setMat4("model", glm::mat4(1.0f));
            PointCloud pointCloud(screen_vert, screen_vert_color);
            pointCloud.render();
        }
        // render text on screen
        {
            float text_spacing = 25.0f;
            std::vector<std::string> texts_to_render = {
                std::format("state: {}", state_machine),
                std::format("n_user_locations: {}", n_user_locations),
                std::format("undistort_flag: {}", leap_undistort),
                std::format("points to display: {}", points_to_display),
            };
            for (int i = 0; i < texts_to_render.size(); ++i)
            {
                text.Render(textShader, texts_to_render[i], 25.0f, texts_to_render.size() * text_spacing - text_spacing * i, 0.50f, glm::vec3(1.0f, 1.0f, 1.0f));
            }
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // cleanup
    close_signal = true;
    camera.kill();
    glfwTerminate();
    return 0;
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_RELEASE)
    {
        if (right_pressed)
        {
            if (state_machine < 3)
            {
                if (load_calib)
                {
                    cam_verts.push_back(cur_mouse_pos);
                    n_user_locations += 1;
                }
                else
                {
                    state_machine += 1;
                }
            }
            else
            {
                if (state_machine == 3)
                {
                    leap1_verts.push_back(cur_mouse_pos);
                    n_user_locations += 1;
                }
                if (state_machine == 4)
                {
                    leap2_verts.push_back(cur_mouse_pos);
                    n_user_locations += 1;
                }
            }
        }
        right_pressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_RELEASE)
    {
        if (left_pressed)
        {
            state_machine = 0;
            cam_verts.clear();
            leap1_verts.clear();
            leap2_verts.clear();
            n_user_locations = 0;
        }
        left_pressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_RELEASE)
    {
        if (up_pressed)
        {
            if (points_to_display < max_user_locations)
            {
                points_to_display += 1;
            }
        }
        up_pressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_RELEASE)
    {
        if (down_pressed)
        {
            if (points_to_display > 0)
            {
                points_to_display -= 1;
            }
        }
        down_pressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE)
    {
        if (enter_pressed)
        {
            if (load_calib)
            {
                if (state_machine == 0)
                    state_machine += 3;
                else
                    state_machine += 1;
                n_user_locations = 0;
            }
            // leap_undistort = !leap_undistort;
        }
        enter_pressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        right_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        left_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        up_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        down_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
    {
        enter_pressed = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
    cur_window_width = width;
    cur_window_height = height;
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if (dragging == false)
        {
            dragging = true;
        }
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        dragging = false;
    }
}