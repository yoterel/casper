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
#include "utils.h"
#include "image_process.h"
#include "stb_image_write.h"
#include <filesystem>
namespace fs = std::filesystem;

// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void processInput(GLFWwindow *window);
LEAP_STATUS getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime, std::vector<glm::mat4> &bones_to_world_left, std::vector<glm::mat4> &bones_to_world_right, std::vector<glm::vec3> &skeleton_vertices, bool poll_mode, int64_t &lastFrameID);
void setup_skeleton_hand_buffers(unsigned int &VAO, unsigned int &VBO);
void setup_gizmo_buffers(unsigned int &VAO, unsigned int &VBO);
void setup_cube_buffers(unsigned int &VAO, unsigned int &VBO);
void setup_frustrum_buffers(unsigned int &VAO, unsigned int &VBO);
void initGLBuffers(unsigned int *pbo);
bool loadCalibrationResults(glm::mat4 &cam_project, glm::mat4 &proj_project, std::vector<double> &camera_distortion, glm::mat4 &w2vp, glm::mat4 &w2vc);
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

// global settings
bool debug_mode = false;
bool freecam_mode = false;
bool use_cuda = false;
bool producer_is_fake = false;
bool use_pbo = false;
bool use_projector = true;
bool use_screen = true;
bool poll_mode = false;
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_width = 720;
const unsigned int cam_height = 540;
float exposure = 1850.0f;
// global state
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
std::vector<glm::vec2> screen_vert = {{0.5f, 0.5f}};
std::vector<glm::vec3> screen_vert_color = {{1.0f, 0.0f, 0.0f}};
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
int64_t lastFrameID = 0;
bool space_modifier = false;
bool shift_modifier = false;
bool ctrl_modifier = false;
unsigned int n_bones = 0;
int state_machine = 0;
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
    int secondary_screen_x, secondary_screen_y;
    glfwGetMonitorPos(monitors[1], &secondary_screen_x, &secondary_screen_y);
    glfwSetWindowPos(window, secondary_screen_x + 100, secondary_screen_y + 100);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
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
    Texture camTexture = Texture();
    camTexture.init(cam_width, cam_height, 4);
    blocking_queue<CPylonImage> camera_queue;
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
    LeapConnect leap(poll_mode);
    std::thread producer, consumer;
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
        /* fake producer */
        std::cout << "Using fake camera to produce images" << std::endl;
        producer_is_fake = true;
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
                while (t_block.getElapsedTimeInMilliSec() < 2.0)
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
            std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
            std::cout << "total app: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
        }
        /* deal with user input */
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        /* deal with camera input */
        // prevFrame = curFrame.clone();
        uint8_t *buffer;
        if (producer_is_fake)
        {
            // buffer = white_image.data;
            cv::Mat cv_image = camera_queue_cv.pop();
            buffer = cv_image.data;
        }
        else
        {
            CPylonImage pylonImage = camera_queue.pop();
            auto test = pylonImage.GetImageSize();
            buffer = (uint8_t *)pylonImage.GetBuffer();
        }
        // cv::Mat tmp(cam_height, cam_width, CV_8UC4, buffer);
        // curFrame = tmp.clone();
        camTexture.load(buffer, true);

        // render
        {
            if (state_machine == 0)
            {
                textureShader.use();
                textureShader.setMat4("view", glm::mat4(1.0f));
                textureShader.setMat4("projection", glm::mat4(1.0f));
                textureShader.setMat4("model", glm::mat4(1.0f));
                textureShader.setBool("flipVer", true);
                textureShader.setInt("src", 0);
                textureShader.setBool("binary", false);
                camTexture.bind();
                fullScreenQuad.render();
                double x;
                double y;
                glfwGetCursorPos(window, &x, &y);
                glm::vec2 mouse_pos = glm::vec2((2.0f * x / proj_width) - 1.0f, -1.0f * ((2.0f * y / proj_height) - 1.0f));
                vertexShader.use();
                vertexShader.setMat4("view", glm::mat4(1.0f));
                vertexShader.setMat4("projection", glm::mat4(1.0f));
                vertexShader.setMat4("model", glm::mat4(1.0f));
                // pointShader.setVec2("mouse_pos", mouse_pos);
                PointCloud pointCloud(screen_vert, screen_vert_color);
                pointCloud.render();
            }
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
    if (producer_is_fake)
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
    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_RELEASE)
    {
        state_machine += 1;
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

LEAP_STATUS getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &skeleton_vertices,
                         bool poll_mode,
                         int64_t &lastFrameID)
{
    // some defs
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    // magic numbers
    int magic_leap_time_delay = 40000; // us
    float magic_scale_factor = 10.0f;
    float magic_wrist_offset = -65.0f;
    glm::mat4 magic_leap_basis_fix = roty * flip_z * flip_y;
    glm::mat4 chirality = glm::mat4(1.0f);
    // init
    glm::mat4 scalar = glm::scale(glm::mat4(1.0f), glm::vec3(magic_scale_factor));
    uint64_t targetFrameSize = 0;
    LEAP_TRACKING_EVENT *frame = nullptr;
    if (poll_mode)
    {
        frame = leap.getFrame();
        if (frame && (frame->tracking_frame_id > lastFrameID))
        {
            lastFrameID = frame->tracking_frame_id;
            skeleton_vertices.clear();
            bones_to_world_left.clear();
            bones_to_world_right.clear();
        }
        else
        {
            return LEAP_STATUS::LEAP_NONEWFRAME;
        }
    }
    else
    {
        /* code */
        skeleton_vertices.clear();
        bones_to_world_left.clear();
        bones_to_world_right.clear();
        // Get the buffer size needed to hold the tracking data
        eLeapRS retVal = LeapGetFrameSize(*leap.getConnectionHandle(), targetFrameTime + magic_leap_time_delay, &targetFrameSize);
        if (retVal != eLeapRS_Success)
        {
            // std::cout << "ERROR: LeapGetFrameSize() returned " << retVal << std::endl;
            return LEAP_STATUS::LEAP_FAILED;
        }
        // Allocate enough memory
        frame = (LEAP_TRACKING_EVENT *)malloc((size_t)targetFrameSize);
        // Get the frame data
        retVal = LeapInterpolateFrame(*leap.getConnectionHandle(), targetFrameTime + magic_leap_time_delay, frame, targetFrameSize);
        if (retVal != eLeapRS_Success)
        {
            // std::cout << "ERROR: LeapInterpolateFrame() returned " << retVal << std::endl;
            return LEAP_STATUS::LEAP_FAILED;
        }
    }
    // Use the data...
    //  std::cout << "frame id: " << interpolatedFrame->tracking_frame_id << std::endl;
    //  std::cout << "frame delay (us): " << (long long int)LeapGetNow() - interpolatedFrame->info.timestamp << std::endl;
    //  std::cout << "frame hands: " << interpolatedFrame->nHands << std::endl;
    glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
    for (uint32_t h = 0; h < frame->nHands; h++)
    {
        LEAP_HAND *hand = &frame->pHands[h];
        if (debug_vec.x > 0)
            if (hand->type == eLeapHandType_Right)
                chirality = flip_z;
        std::vector<glm::mat4> bones_to_world;
        // palm
        glm::vec3 palm_pos = glm::vec3(hand->palm.position.x,
                                       hand->palm.position.y,
                                       hand->palm.position.z);
        glm::vec3 towards_hand_tips = glm::vec3(hand->palm.direction.x, hand->palm.direction.y, hand->palm.direction.z);
        towards_hand_tips = glm::normalize(towards_hand_tips);
        // we offset the palm to coincide with wrist, as a real hand has a wrist joint that needs to be controlled
        palm_pos = palm_pos + towards_hand_tips * magic_wrist_offset;
        glm::mat4 palm_orientation = glm::toMat4(glm::quat(hand->palm.orientation.w,
                                                           hand->palm.orientation.x,
                                                           hand->palm.orientation.y,
                                                           hand->palm.orientation.z));
        // for some reason using the "basis" from leap rotates and flips the coordinate system of the palm
        // also there is an arbitrary scale factor associated with the 3d mesh
        // so we need to fix those
        palm_orientation = palm_orientation * chirality * magic_leap_basis_fix * scalar;

        bones_to_world.push_back(glm::translate(glm::mat4(1.0f), palm_pos) * palm_orientation);
        // arm
        LEAP_VECTOR arm_j1 = hand->arm.prev_joint;
        LEAP_VECTOR arm_j2 = hand->arm.next_joint;
        skeleton_vertices.push_back(glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
        skeleton_vertices.push_back(red);
        skeleton_vertices.push_back(glm::vec3(arm_j2.x, arm_j2.y, arm_j2.z));
        skeleton_vertices.push_back(red);
        glm::mat4 arm_rot = glm::toMat4(glm::quat(hand->arm.rotation.w,
                                                  hand->arm.rotation.x,
                                                  hand->arm.rotation.y,
                                                  hand->arm.rotation.z));
        // arm_rot = glm::rotate(arm_rot, glm::radians(debug_vec.x), glm::vec3(arm_rot[0][0], arm_rot[0][1], arm_rot[0][2]));
        glm::mat4 arm_translate = glm::translate(glm::mat4(1.0f), glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
        bones_to_world.push_back(arm_translate * arm_rot * chirality * magic_leap_basis_fix * scalar);
        // fingers
        for (uint32_t f = 0; f < 5; f++)
        {
            LEAP_DIGIT finger = hand->digits[f];
            for (uint32_t b = 0; b < 4; b++)
            {
                LEAP_VECTOR joint1 = finger.bones[b].prev_joint;
                LEAP_VECTOR joint2 = finger.bones[b].next_joint;
                skeleton_vertices.push_back(glm::vec3(joint1.x, joint1.y, joint1.z));
                skeleton_vertices.push_back(red);
                skeleton_vertices.push_back(glm::vec3(joint2.x, joint2.y, joint2.z));
                skeleton_vertices.push_back(red);
                glm::mat4 rot = glm::toMat4(glm::quat(finger.bones[b].rotation.w,
                                                      finger.bones[b].rotation.x,
                                                      finger.bones[b].rotation.y,
                                                      finger.bones[b].rotation.z));
                glm::vec3 translate = glm::vec3(joint1.x, joint1.y, joint1.z);
                glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
                bones_to_world.push_back(trans * rot * chirality * magic_leap_basis_fix * scalar);
            }
        }
        if (hand->type == eLeapHandType_Right)
            bones_to_world_right = bones_to_world;
        else
            bones_to_world_left = bones_to_world;
    }
    // Free the allocated buffer when done.
    free(frame);
    return LEAP_STATUS::LEAP_NEWFRAME;
}