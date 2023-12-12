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
#include "readerwritercircularbuffer.h"
#include "camera.h"
#include "display.h"
#include "shader.h"
#include "GLMhelpers.h"
#include "timer.h"
#include "quad.h"
#include "text.h"
#include "stb_image_write.h"
#include "texture.h"
#include "point_cloud.h"
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
namespace fs = std::filesystem;

// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void processInput(GLFWwindow *window);
void initGLBuffers(unsigned int *pbo);
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

// global settings
bool debug_mode = false;
bool freecam_mode = false;
bool use_cuda = false;
bool producer_is_fake = false;
bool use_pbo = true;
bool use_projector = true;
bool use_screen = true;
bool poll_mode = false;
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
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
int64_t lastFrameID = 0;
bool space_modifier = false;
bool shift_modifier = false;
bool ctrl_modifier = false;
unsigned int n_bones = 0;
float screen_z = -10.0f;
bool hand_in_frame = false;
const unsigned int num_texels = proj_width * proj_height;
const unsigned int image_size = num_texels * 3 * sizeof(uint8_t);
cv::Mat white_image(cam_height, cam_width, CV_8UC4, cv::Scalar(255, 255, 255, 255));
std::vector<glm::vec2> orig_screen_verts = {{-1.0f, 1.0f},
                                            {-1.0f, -1.0f},
                                            {1.0f, -1.0f},
                                            {1.0f, 1.0f}};

std::vector<glm::vec2> screen_verts = {{-1.0f, 1.0f},
                                       {-1.0f, -1.0f},
                                       {1.0f, -1.0f},
                                       {1.0f, 1.0f}};
std::vector<glm::vec3> screen_verts_color = {{1.0f, 0.0f, 0.0f},
                                             {1.0f, 0.0f, 0.0f},
                                             {1.0f, 0.0f, 0.0f},
                                             {1.0f, 0.0f, 0.0f}};
bool dragging = false;
int dragging_vert = 0;
int closest_vert = 0;
float min_dist = 100000.0f;

int main(int argc, char *argv[])
{
    Timer t_cam, t_upload, t_download, t_render, t_copy, t_debug, t_app, t_misc, t_swap;
    t_app.start();
    /* init GLFW */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow *window = glfwCreateWindow(proj_width, proj_height, "augmented_hands", NULL /*monitors[i] for full screen*/, NULL);
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
    // glfwSetWindowPos(window, secondary_screen_x + 100, secondary_screen_y + 100);
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
    glfwSwapInterval(0);                       // do not sync to monitor
    glViewport(0, 0, proj_width, proj_height); // set viewport
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glPointSize(10.0f);
    // glEnable(GL_CULL_FACE);
    // glEnable(GL_BLEND);
    // glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // callback for resizing
    // int a, b;
    // glfwGetFramebufferSize(window, &a, &b);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_TRUE);
    /* setup global GL buffers */
    unsigned int pbo[2] = {0};
    if (use_pbo)
    {
        initGLBuffers(pbo);
    }

    Text text("../../resource/arial.ttf");

    /* setup shaders*/

    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader vertexShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    Quad screen(orig_screen_verts);
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
    Texture camTexture = Texture();
    camTexture.init(cam_width, cam_height, n_cam_channels);
    // uint32_t cam_height = 0;
    // uint32_t cam_width = 0;
    moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> camera_queue(20);
    // queue_spsc<cv::Mat> camera_queue_cv(50);
    // blocking_queue<cv::Mat> camera_queue_cv;
    // blocking_queue<std::vector<uint8_t>> projector_queue;
    moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> projector_queue(20);
    BaslerCamera camera;
    DynaFlashProjector projector(true, false);
    if (use_projector)
    {
        if (!projector.init())
        {
            std::cerr << "Failed to initialize projector\n";
        }
    }
    std::thread producer, consumer;
    // load calibration results if they exist
    glm::mat4 vproj_project;
    glm::mat4 vcam_project;
    std::vector<double> camera_distortion;
    glm::mat4 w2vp;
    glm::mat4 w2vc;
    /* actual thread loops */
    /* image producer (real camera = virtual projector) */
    if (camera.init(camera_queue, close_signal, cam_height, cam_width, exposure) && !producer_is_fake)
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
        exit(1);
        /* fake producer */
        // std::vector<cv::Mat> fake_cam_images;
        // std::cout << "Using fake camera to produce images" << std::endl;
        // producer_is_fake = true;
        // std::string file_path = "../../resource/uv.png";
        // cv::Mat img3 = cv::imread(file_path, cv::IMREAD_UNCHANGED);
        // cv::resize(img3, img3, cv::Size(cam_width, cam_height));
        // cv::Mat img4;
        // cv::cvtColor(img3, img4, cv::COLOR_BGR2BGRA);
        // fake_cam_images.push_back(img4);
        // // cam_height = 540;
        // // cam_width = 720;
        // producer = std::thread([&camera_queue_cv, &close_signal, fake_cam_images]() { //, &projector
        //     // CPylonImage image = CPylonImage::Create(PixelType_BGRA8packed, cam_width, cam_height);
        //     Timer t_block;
        //     int counter = 0;
        //     t_block.start();
        //     while (!close_signal)
        //     {
        //         camera_queue_cv.push(fake_cam_images[counter]);
        //         if (counter < fake_cam_images.size() - 1)
        //             counter++;
        //         else
        //             counter = 0;
        //         while (t_block.getElapsedTimeInMicroSec() < 1000.0)
        //         {
        //         }
        //         t_block.stop();
        //         t_block.start();
        //     }
        //     std::cout << "Producer finish" << std::endl;
        // });
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
            sucess = projector_queue.wait_dequeue_timed(image, std::chrono::milliseconds(100));
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
        t_misc.start();
        currentAppTime = t_app.getElapsedTimeInSec(); // glfwGetTime();
        deltaTime = static_cast<float>(currentAppTime - previousAppTime);
        previousAppTime = currentAppTime;
        frameCount++;
        // stats display
        if (currentAppTime - previousSecondAppTime >= 1.0)
        {
            fps = frameCount;
            ms_per_frame = 1000.0f / frameCount;
            std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
            std::cout << "total app time: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
            std::cout << "t_misc: " << t_misc.averageLapInMilliSec() << std::endl;
            std::cout << "cam time: " << t_cam.averageLapInMilliSec() << std::endl;
            std::cout << "cam upload: " << t_upload.averageLapInMilliSec() << std::endl;
            std::cout << "swap buffers time: " << t_swap.averageLapInMilliSec() << std::endl;
            std::cout << "GPU->CPU time: " << t_download.averageLapInMilliSec() << std::endl;
            std::cout << "debug time: " << t_debug.averageLapInMilliSec() << std::endl;
            std::cout << "cam q1 size: " << camera_queue.size_approx() << std::endl;
            // std::cout << "cam q2 size: " << camera_queue_cv.size() << std::endl;
            std::cout << "proj q size: " << projector_queue.size_approx() << std::endl;
            frameCount = 0;
            previousSecondAppTime = currentAppTime;
            t_misc.reset();
            t_cam.reset();
            t_upload.reset();
            t_swap.reset();
            t_download.reset();
            t_debug.reset();
        }
        // input
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t_misc.stop();
        t_cam.start();
        // retrieve camera image
        CGrabResultPtr ptrGrabResult;
        camera_queue.wait_dequeue(ptrGrabResult);
        t_cam.stop();
        t_upload.start();
        camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
        t_upload.stop();
        t_render.start();
        // render
        {
            std::vector<cv::Point2f> origpts, newpts;
            for (int i = 0; i < 4; ++i)
            {
                origpts.push_back(cv::Point2f(orig_screen_verts[i].x, orig_screen_verts[i].y));
                newpts.push_back(cv::Point2f(screen_verts[i].x, screen_verts[i].y));
            }
            cv::Mat1f hom = cv::getPerspectiveTransform(origpts, newpts, cv::DECOMP_SVD);
            cv::Mat1f perspective = cv::Mat::zeros(4, 4, CV_32F);
            perspective.at<float>(0, 0) = hom.at<float>(0, 0);
            perspective.at<float>(0, 1) = hom.at<float>(0, 1);
            perspective.at<float>(0, 3) = hom.at<float>(0, 2);
            perspective.at<float>(1, 0) = hom.at<float>(1, 0);
            perspective.at<float>(1, 1) = hom.at<float>(1, 1);
            perspective.at<float>(1, 3) = hom.at<float>(1, 2);
            perspective.at<float>(3, 0) = hom.at<float>(2, 0);
            perspective.at<float>(3, 1) = hom.at<float>(2, 1);
            perspective.at<float>(3, 3) = hom.at<float>(2, 2);
            for (int i = 0; i < 4; ++i)
            {
                cv::Vec4f cord = cv::Vec4f(orig_screen_verts[i].x, orig_screen_verts[i].y, 0.0f, 1.0f);
                cv::Mat tmp = perspective * cv::Mat(cord);
                screen_verts[i].x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);
                screen_verts[i].y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);
            }
            // cv::Mat hom4x4 = cv::Mat::eye(4, 4, CV_32FC1);
            // hom.copyTo(hom4x4(cv::Rect(0, 0, 3, 3)));
            GLMHelpers::CV2GLM(perspective, &w2vp);
            textureShader.use();
            textureShader.setMat4("view", w2vp);
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            // textureShader.setBool("flipVer", true);
            textureShader.setInt("src", 0);
            textureShader.setBool("justGreen", true);
            // textureShader.setBool("isGray", true);
            camTexture.bind();
            screen.render();
            PointCloud cloud(screen_verts, screen_verts_color);
            vertexShader.use();
            vertexShader.setMat4("view", glm::mat4(1.0f));
            vertexShader.setMat4("projection", glm::mat4(1.0f));
            vertexShader.setMat4("model", glm::mat4(1.0f));
            cloud.render();
        }
        t_render.stop();
        t_debug.start();
        {
            float text_spacing = 20.0f;
            double x;
            double y;
            glfwGetCursorPos(window, &x, &y);
            glm::vec2 mouse_pos = glm::vec2((2.0f * x / proj_width) - 1.0f, -1.0f * ((2.0f * y / proj_height) - 1.0f));
            std::vector<std::string> texts_to_render = {
                std::format("mouse x, y: {:.02f}, {:.02f}", mouse_pos.x, mouse_pos.y),
                std::format("closest_vert: {}, min_dist vert: {:.03f}", closest_vert, min_dist),
                std::format("screen vert 0: {:.04f}, {:.04f}", screen_verts[0].x, screen_verts[0].y),
                std::format("screen vert 1: {:.04f}, {:.04f}", screen_verts[1].x, screen_verts[1].y),
                std::format("screen vert 2: {:.04f}, {:.04f}", screen_verts[2].x, screen_verts[2].y),
                std::format("screen vert 3: {:.04f}, {:.04f}", screen_verts[3].x, screen_verts[3].y),
                std::format("modifiers : shift: {}, ctrl: {}, space: {}", shift_modifier ? "on" : "off", ctrl_modifier ? "on" : "off", space_modifier ? "on" : "off"),
            };
            for (int i = 0; i < texts_to_render.size(); ++i)
            {
                text.Render(textShader, texts_to_render[i], 25.0f, texts_to_render.size() * text_spacing - text_spacing * i, 0.5f, glm::vec3(1.0f, 1.0f, 1.0f));
            }
        }
        t_debug.stop();
        // send result to projector queue
        glReadBuffer(GL_FRONT);
        if (use_pbo) // something fishy going on here. using pbo collapses program after a while
        {
            t_download.start();
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[frameCount % 2]);
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, 0);
            t_download.stop();

            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[(frameCount + 1) % 2]);
            GLubyte *src = (GLubyte *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            if (src)
            {
                // std::vector<uint8_t> data(src, src + image_size);
                // tmpdata.assign(src, src + image_size);
                // std::copy(src, src + tmpdata.size(), tmpdata.begin());
                memcpy(colorBuffer, src, image_size);
                projector_queue.try_enqueue(colorBuffer);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER); // release pointer to the mapped buffer
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }
        else
        {
            t_download.start();
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
            t_download.stop();
            // std::vector<uint8_t> data(colorBuffer, colorBuffer + image_size);
            projector_queue.try_enqueue(colorBuffer);
        }
        // glCheckError();
        // glCheckError();

        // auto projector_thread = std::thread([&projector, &colorBuffer]() {  //, &projector

        // projector.show_buffer(colorBuffer);
        // });
        // stbi_flip_vertically_on_write(true);
        // int stride = 3 * proj_width;
        // stride += (stride % 4) ? (4 - stride % 4) : 0;
        // stbi_write_png("test.png", proj_width, proj_height, 3, colorBuffer, stride);
        // swap buffers and poll IO events
        t_swap.start();
        glfwSwapBuffers(window);
        glfwPollEvents();
        t_swap.stop();
    }
    // cleanup
    close_signal = true;
    consumer.join();
    projector.kill();
    camera.kill();
    glfwTerminate();
    delete[] colorBuffer;
    if (producer_is_fake)
    {
        producer.join();
    }
    return 0;
}

void initGLBuffers(unsigned int *pbo)
{
    // set up vertex data parameter
    void *data = malloc(image_size);
    // create ping pong pbos
    glGenBuffers(2, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[0]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_size, data, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[1]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_size, data, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    free(data);
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if (dragging == false)
        {
            if (min_dist < 1.0f)
            {
                dragging = true;
                dragging_vert = closest_vert;
            }
        }
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        dragging = false;
    }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float x = static_cast<float>(xposIn);
    float y = static_cast<float>(yposIn);
    glm::vec2 mouse_pos = glm::vec2((2.0f * x / proj_width) - 1.0f, -1.0f * ((2.0f * y / proj_height) - 1.0f));
    float cur_min_dist = 100.0f;
    for (int i = 0; i < screen_verts.size(); i++)
    {
        glm::vec2 v = glm::vec2(screen_verts[i]);

        float dist = glm::distance(v, mouse_pos);
        if (dist < cur_min_dist)
        {
            cur_min_dist = dist;
            closest_vert = i;
        }
    }
    min_dist = cur_min_dist;
    if (dragging)
    {
        screen_verts[dragging_vert].x = mouse_pos.x;
        screen_verts[dragging_vert].y = mouse_pos.y;
    }
}