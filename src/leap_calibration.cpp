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
#include "display.h"
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
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
cv::Point3d approximate_ray_intersection(const cv::Point3d &v1, const cv::Point3d &q1,
                                         const cv::Point3d &v2, const cv::Point3d &q2,
                                         double *distance, double *out_lambda1, double *out_lambda2);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void processInput(GLFWwindow *window);
void initGLBuffers(unsigned int *pbo);
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

// global settings
bool debug_mode = false;
bool freecam_mode = false;
bool use_cuda = false;
bool simulated_camera = false;
bool use_pbo = false;
bool use_projector = true;
bool use_screen = true;
bool cam_color_mode = false;
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_width = 720;
const unsigned int cam_height = 540;
unsigned int n_cam_channels = cam_color_mode ? 4 : 1;
unsigned int cam_buffer_format = cam_color_mode ? GL_RGBA : GL_RED;
float exposure = 1850.0f;
// global state
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
std::vector<glm::vec2> screen_vert = {{0.5f, 0.5f}};
std::vector<glm::vec3> screen_vert_color = {{1.0f, 0.0f, 0.0f}};
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
bool enter_pressed = false;
bool finish_calibration = false;
bool dragging = false;
float screen_z = -10.0f;
bool hand_in_frame = false;
const unsigned int num_texels = proj_width * proj_height;
const unsigned int image_size = num_texels * 3 * sizeof(uint8_t);
cv::Mat white_image(cam_height, cam_width, CV_8UC4, cv::Scalar(255, 255, 255, 255));

int main(int argc, char *argv[])
{
    /* init GLFW */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow *window = glfwCreateWindow(proj_width, proj_height, "augmented_hands", NULL, NULL); // monitors[0], NULL for full screen
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    int secondary_screen_x, secondary_screen_y;
    glfwGetMonitorPos(monitors[1], &secondary_screen_x, &secondary_screen_y);
    glfwSetWindowMonitor(window, NULL, secondary_screen_x + 100, secondary_screen_y + 100, proj_width, proj_height, GLFW_DONT_CARE);
    glfwSetWindowMonitor(window, NULL, secondary_screen_x + 150, secondary_screen_y + 150, proj_width, proj_height, GLFW_DONT_CARE); // really glfw?
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
    glViewport(0, 0, proj_width, proj_height); // set viewport
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glPointSize(10.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // callback for resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_TRUE);
    Quad fullScreenQuad(0.0f);
    Timer t_app;
    t_app.start();
    Text text("../../resource/arial.ttf");
    FBO hands_fbo(proj_width, proj_height);
    FBO postprocess_fbo(proj_width, proj_height);
    /* setup shaders*/
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
    uint8_t *colorBuffer = new uint8_t[image_size];
    CGrabResultPtr ptrGrabResult;
    Texture camTexture = Texture();
    camTexture.init(cam_width, cam_height, n_cam_channels);
    blocking_queue<CGrabResultPtr> camera_queue;
    // queue_spsc<cv::Mat> camera_queue_cv(50);
    blocking_queue<cv::Mat> camera_queue_cv;
    // blocking_queue<std::vector<uint8_t>> projector_queue;
    blocking_queue<uint8_t *> projector_queue;
    BaslerCamera camera;
    DynaFlashProjector projector(true, false);
    if (use_projector)
    {
        if (!projector.init())
        {
            std::cerr << "Failed to initialize projector\n";
        }
    }
    LeapConnect leap(true, true);
    std::thread producer, consumer;
    /* actual thread loops */
    /* image producer (real camera = virtual projector) */
    if (camera.init(camera_queue, close_signal, cam_height, cam_width, exposure) && !simulated_camera)
    {
        /* real producer */
        std::cout << "Using real camera to produce images" << std::endl;
        projector.show();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // camera.balance_white();
        camera.acquire();
    }
    else
    {
        /* fake producer */
        // see https://ja.docs.baslerweb.com/pylonapi/cpp/sample_code#utility_image for pylon image
        std::cout << "Using fake camera to produce images" << std::endl;
        simulated_camera = true;
        std::string path = "../../resource/hand_capture";
        std::vector<cv::Mat> fake_cam_images;
        // white image
        fake_cam_images.push_back(white_image);
        // images from folder
        // int file_counter = 0;
        // for (const auto &entry : fs::directory_iterator(path))
        // {
        //     std::cout << '\r' << std::format("Loading images: {:04d}", file_counter) << std::flush;
        //     std::string file_path = entry.path().string();
        //     cv::Mat img3 = cv::imread(file_path, cv::IMREAD_UNCHANGED);
        //     cv::Mat img4;
        //     cv::cvtColor(img3, img4, cv::COLOR_BGR2BGRA);
        //     fake_cam_images.push_back(img4);
        //     file_counter++;
        // }
        producer = std::thread([&camera_queue_cv, &close_signal, fake_cam_images]() { //, &projector
            // CPylonImage image = CPylonImage::Create(PixelType_BGRA8packed, cam_width, cam_height);
            Timer t_block;
            int counter = 0;
            t_block.start();
            while (!close_signal)
            {
                camera_queue_cv.push(fake_cam_images[counter]);
                if (counter < fake_cam_images.size() - 1)
                    counter++;
                else
                    counter = 0;
                while (t_block.getElapsedTimeInMilliSec() < 1.9)
                {
                }
                t_block.stop();
                t_block.start();
            }
            std::cout << "Producer finish" << std::endl;
        });
    }
    // image consumer (real projector = virtual camera)
    consumer = std::thread([&projector_queue, &projector, &close_signal]() { //, &projector
        uint8_t *image;
        // std::vector<uint8_t> image;
        // int stride = 3 * proj_width;
        // stride += (stride % 4) ? (4 - stride % 4) : 0;
        bool sucess;
        while (!close_signal)
        {
            sucess = projector_queue.pop_with_timeout(100, image);
            if (sucess)
                projector.show_buffer(image);
            // projector.show_buffer(image.data());
            // stbi_write_png("test.png", proj_width, proj_height, 3, image.data(), stride);
        }
        std::cout << "Consumer finish" << std::endl;
    });
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
        // prevFrame = curFrame.clone();
        if (simulated_camera)
        {
            cv::Mat tmp = camera_queue_cv.pop();
            camTexture.load((uint8_t *)tmp.data, true, cam_buffer_format);
        }
        else
        {
            ptrGrabResult = camera_queue.pop();
            camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
        }

        // render
        if (finish_calibration)
        {
            // extract 3d points from leap1_verts and leap2_verts

            // first get rays from the leap camera corrected for distortion, in 2D camera space
            std::vector<LEAP_VECTOR> leap1_rays_2d, leap2_rays_2d;
            for (int i = 0; i < leap1_verts.size(); i++)
            {
                LEAP_VECTOR l1_verts = {leap_width * (leap1_verts[i].x + 1) / 2, leap_height * (leap1_verts[i].y + 1) / 2, 1.0f};
                LEAP_VECTOR l2_verts = {leap_width * (leap2_verts[i].x + 1) / 2, leap_height * (leap2_verts[i].y + 1) / 2, 1.0f};
                LEAP_VECTOR l1_ray = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, l1_verts);
                leap1_rays_2d.push_back(l1_ray);
                LEAP_VECTOR l2_ray = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, l2_verts);
                leap2_rays_2d.push_back(l2_ray);
            }
            std::vector<cv::Point2f> image_points;
            std::vector<cv::Point3f> object_points;

            // second convert rays to 3D leap space using the extrinsics matrix
            /*
            std::vector<cv::Point3f> leap1_ray_dirs_3d, leap2_ray_dirs_3d;
            glm::mat4 leap1_extrinsic = glm::mat4(1.0f);
            glm::mat4 leap2_extrinsic = glm::mat4(1.0f);
            LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, glm::value_ptr(leap1_extrinsic));
            LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, glm::value_ptr(leap2_extrinsic));
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
            float baseline = 40.0f;
            for (int i = 0; i < leap1_rays_2d.size(); i++)
            {
                // see https://forums.leapmotion.com/t/sdk-2-1-raw-data-get-pixel-position-xyz/1604/12
                float z = baseline / (leap1_rays_2d[i].x - leap2_rays_2d[i].x);
                float y = z * leap1_rays_2d[i].y;
                float x = z * leap1_rays_2d[i].x - baseline / 2.0f;
                object_points.push_back(cv::Point3f(x, -z, y));
            }
            for (int i = 0; i < cam_verts.size(); i++)
            {
                // image_points.push_back(cv::Point2f(proj_width * (proj_verts[i].x + 1) / 2, proj_height * (proj_verts[i].y + 1) / 2));
                image_points.push_back(cv::Point2f(cam_width * (cam_verts[i].x + 1) / 2, cam_height * (cam_verts[i].y + 1) / 2));
            }
            // use solve pnp to find transformation of projector to leap space
            cv::Mat1f camera_matrix = cv::Mat::eye(3, 3, CV_32F);
            cv::Mat1f dist_coeffs = cv::Mat::zeros(8, 1, CV_32F);
            cv::Mat1f rvec, tvec;
            cv::solvePnP(object_points, image_points, camera_matrix, cv::Mat1f(), rvec, tvec);
            cv::Mat1f rot_mat;
            cv::Rodrigues(rvec, rot_mat);
            cv::Mat1f w2p(4, 4, CV_32FC1);
            w2p.at<float>(0, 0) = rot_mat.at<float>(0, 0);
            w2p.at<float>(0, 1) = rot_mat.at<float>(0, 1);
            w2p.at<float>(0, 2) = rot_mat.at<float>(0, 2);
            w2p.at<float>(0, 3) = tvec.at<float>(0, 0);
            w2p.at<float>(1, 0) = rot_mat.at<float>(1, 0);
            w2p.at<float>(1, 1) = rot_mat.at<float>(1, 1);
            w2p.at<float>(1, 2) = rot_mat.at<float>(1, 2);
            w2p.at<float>(1, 3) = tvec.at<float>(1, 0);
            w2p.at<float>(2, 0) = rot_mat.at<float>(2, 0);
            w2p.at<float>(2, 1) = rot_mat.at<float>(2, 1);
            w2p.at<float>(2, 2) = rot_mat.at<float>(2, 2);
            w2p.at<float>(2, 3) = tvec.at<float>(2, 0);
            w2p.at<float>(3, 0) = 0.0f;
            w2p.at<float>(3, 1) = 0.0f;
            w2p.at<float>(3, 2) = 0.0f;
            w2p.at<float>(3, 3) = 1.0f;
            cv::Mat p2w = w2p.inv();
            std::cout << p2w << std::endl;
            glfwSetWindowShouldClose(window, true);
        }
        else
        {
            unsigned int cur_width;
            unsigned int cur_height;
            if (state_machine % 3 == 0)
            {
                // cur_width = proj_width;
                // cur_height = proj_height;
                cur_width = cam_width;
                cur_height = cam_height;
                glfwSetWindowMonitor(window, NULL, secondary_screen_x + 100, secondary_screen_y + 100, cur_width, cur_height, GLFW_DONT_CARE);
                glViewport(0, 0, cur_width, cur_height); // set viewport
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                textureShader.use();
                textureShader.setMat4("view", glm::mat4(1.0f));
                textureShader.setMat4("projection", glm::mat4(1.0f));
                textureShader.setMat4("model", glm::mat4(1.0f));
                textureShader.setBool("flipVer", true);
                textureShader.setInt("src", 0);
                textureShader.setBool("binary", false);
                camTexture.bind();
                fullScreenQuad.render();
            }
            else
            {
                std::vector<uint8_t> buffer1, buffer2;
                uint32_t width, height;
                leap.getImage(buffer1, buffer2, width, height);
                cur_width = width;
                cur_height = height;
                leap_width = width;
                leap_height = height;
                glfwSetWindowMonitor(window, NULL, secondary_screen_x + 100, secondary_screen_y + 100, width, height, GLFW_DONT_CARE);
                glViewport(0, 0, width, height); // set viewport
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                Texture leapTexture = Texture();
                if (state_machine % 3 == 1)
                {
                    leapTexture.init(buffer1.data(), width, height, 1);
                    textureShader.use();
                    textureShader.setMat4("view", glm::mat4(1.0f));
                    textureShader.setMat4("projection", glm::mat4(1.0f));
                    textureShader.setMat4("model", glm::mat4(1.0f));
                    textureShader.setBool("flipVer", true);
                    textureShader.setInt("src", 0);
                    textureShader.setBool("binary", false);
                    textureShader.setBool("isGray", true);
                    leapTexture.bind();
                    fullScreenQuad.render();
                }
                else
                {
                    leapTexture.init(buffer2.data(), width, height, 1);
                    textureShader.use();
                    textureShader.setMat4("view", glm::mat4(1.0f));
                    textureShader.setMat4("projection", glm::mat4(1.0f));
                    textureShader.setMat4("model", glm::mat4(1.0f));
                    textureShader.setBool("flipVer", true);
                    textureShader.setInt("src", 0);
                    textureShader.setBool("binary", false);
                    leapTexture.bind();
                    fullScreenQuad.render();
                }
            }
            if (dragging)
            {
                double x;
                double y;
                glfwGetCursorPos(window, &x, &y);
                glm::vec2 mouse_pos = glm::vec2((2.0f * x / cur_width) - 1.0f, -1.0f * ((2.0f * y / cur_height) - 1.0f));
                screen_vert[0].x = mouse_pos.x;
                screen_vert[0].y = mouse_pos.y;
            }
            vertexShader.use();
            vertexShader.setMat4("view", glm::mat4(1.0f));
            vertexShader.setMat4("projection", glm::mat4(1.0f));
            vertexShader.setMat4("model", glm::mat4(1.0f));
            PointCloud pointCloud(screen_vert, screen_vert_color);
            pointCloud.render();
            // LEAP_STATUS status = getLeapFrame(leap, targetFrameTime, bones_to_world_left, bones_to_world_right, skeleton_vertices, poll_mode, lastFrameID);
            // display projector screen, where mouse can control the projector output
            // display the leap camera images side by side, where mouse can control where to mark
            // https://docs.ultraleap.com/api-reference/tracking-api/group/group___functions.html#_CPPv422LeapPixelToRectilinear15LEAP_CONNECTION20eLeapPerspectiveType11LEAP_VECTOR
            // use LeapPixelToRectilinear to get the 3d ray of the mouse (x2)
            // use LeapExtrinsicCameraMatrix (inverted) to convert rays to leap 3d tracking space
            // use solve pnp to find transformation of projector to leap space
        }
        // send result to projector queue
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
        // std::vector<uint8_t> data(colorBuffer, colorBuffer + image_size);
        projector_queue.push(colorBuffer);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // cleanup
    close_signal = true;
    consumer.join();
    projector.kill();
    camera.kill();
    glfwTerminate();
    delete[] colorBuffer;
    if (simulated_camera)
    {
        producer.join();
    }
    /* setup trigger */
    // char* portName = "\\\\.\\COM4";
    // #define DATA_LENGTH 255
    // SerialPort *arduino = new SerialPort(portName);
    // std::cout << "Arduino is connected: " << arduino->isConnected() << std::endl;
    // const char *sendString = "trigger\n";
    // if (arduino->isConnected()){
    //     bool hasWritten = arduino->writeSerialPort(sendString, DATA_LENGTH);
    //     if (hasWritten) std::cout << "Data Written Successfully" << std::endl;
    //     else std::cerr << "Data was not written" << std::endl;
    // }
    /* end setup trigger */
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS)
    {
        enter_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE)
    {
        if (enter_pressed)
        {
            if (!finish_calibration)
            {
                if (state_machine % 3 == 0)
                {
                    // proj_verts.push_back(screen_vert[0]);
                    cam_verts.push_back(screen_vert[0]);
                }
                else if (state_machine % 3 == 1)
                {
                    leap1_verts.push_back(screen_vert[0]);
                }
                else if (state_machine % 3 == 2)
                {
                    leap2_verts.push_back(screen_vert[0]);
                }
                state_machine += 1;
            }
            if (leap2_verts.size() >= 6)
            {
                finish_calibration = true;
            }
        }
        enter_pressed = false;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
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
// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float x = static_cast<float>(xposIn);
    float y = static_cast<float>(yposIn);
    glm::vec2 mouse_pos = glm::vec2((2.0f * x / proj_width) - 1.0f, -1.0f * ((2.0f * y / proj_height) - 1.0f));
    if (dragging)
    {
        screen_vert[0].x = mouse_pos.x;
        screen_vert[0].y = mouse_pos.y;
    }
}

cv::Point3d approximate_ray_intersection(const cv::Point3d &v1, const cv::Point3d &q1,
                                         const cv::Point3d &v2, const cv::Point3d &q2,
                                         double *distance, double *out_lambda1, double *out_lambda2)
{
    cv::Mat v1mat = cv::Mat(v1);
    cv::Mat v2mat = cv::Mat(v2);

    double v1tv1 = cv::Mat(v1mat.t() * v1mat).at<double>(0, 0);
    double v2tv2 = cv::Mat(v2mat.t() * v2mat).at<double>(0, 0);
    double v1tv2 = cv::Mat(v1mat.t() * v2mat).at<double>(0, 0);
    double v2tv1 = cv::Mat(v2mat.t() * v1mat).at<double>(0, 0);

    // cv::Mat V(2, 2, CV_64FC1);
    // V.at<double>(0,0) = v1tv1;  V.at<double>(0,1) = -v1tv2;
    // V.at<double>(1,0) = -v2tv1; V.at<double>(1,1) = v2tv2;
    // std::cout << " V: "<< V << std::endl;

    cv::Mat Vinv(2, 2, CV_64FC1);
    double detV = v1tv1 * v2tv2 - v1tv2 * v2tv1;
    Vinv.at<double>(0, 0) = v2tv2 / detV;
    Vinv.at<double>(0, 1) = v1tv2 / detV;
    Vinv.at<double>(1, 0) = v2tv1 / detV;
    Vinv.at<double>(1, 1) = v1tv1 / detV;
    // std::cout << " V.inv(): "<< V.inv() << std::endl << " Vinv: " << Vinv << std::endl;

    // cv::Mat Q(2, 1, CV_64FC1);
    // Q.at<double>(0,0) = cv::Mat(v1mat.t()*(cv::Mat(q2-q1))).at<double>(0,0);
    // Q.at<double>(1,0) = cv::Mat(v2mat.t()*(cv::Mat(q1-q2))).at<double>(0,0);
    // std::cout << " Q: "<< Q << std::endl;

    cv::Point3d q2_q1 = q2 - q1;
    double Q1 = v1.x * q2_q1.x + v1.y * q2_q1.y + v1.z * q2_q1.z;
    double Q2 = -(v2.x * q2_q1.x + v2.y * q2_q1.y + v2.z * q2_q1.z);

    // cv::Mat L = V.inv()*Q;
    // cv::Mat L = Vinv*Q;
    // std::cout << " L: "<< L << std::endl;

    double lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) / detV;
    double lambda2 = (v2tv1 * Q1 + v1tv1 * Q2) / detV;
    // std::cout << "lambda1: " << lambda1 << " lambda2: " << lambda2 << std::endl;

    // cv::Mat p1 = L.at<double>(0,0)*v1mat + cv::Mat(q1); //ray1
    // cv::Mat p2 = L.at<double>(1,0)*v2mat + cv::Mat(q2); //ray2
    // cv::Point3d p1 = L.at<double>(0,0)*v1 + q1; //ray1
    // cv::Point3d p2 = L.at<double>(1,0)*v2 + q2; //ray2
    cv::Point3d p1 = lambda1 * v1 + q1; // ray1
    cv::Point3d p2 = lambda2 * v2 + q2; // ray2

    // cv::Point3d p = cv::Point3d(cv::Mat((p1+p2)/2.0));
    cv::Point3d p = 0.5 * (p1 + p2);

    if (distance != NULL)
    {
        *distance = cv::norm(p2 - p1);
    }
    if (out_lambda1)
    {
        *out_lambda1 = lambda1;
    }
    if (out_lambda2)
    {
        *out_lambda2 = lambda2;
    }

    return p;
}