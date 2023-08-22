#include <iostream>
#include <thread>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "queue.h"
#include "camera.h"
#include "gl_camera.h"
#include "display.h"
#include "SerialPort.h"
#include "shader.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "leap.h"
#include "text.h"
#include "canvas.h"
#include "utils.h"
#include "cnpy.h"
#include "image_process.h"
#include "stb_image_write.h"
#include <helper_string.h>
// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime, std::vector<glm::mat4> &bones_to_world, std::vector<glm::vec3> &skeleton_vertices, bool debug);
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
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_height = 540;
const unsigned int cam_width = 720;
// global state
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
double lastFrame = 0.0;
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
bool space_pressed_flag = false;
unsigned int n_bones = 0;
glm::mat4 cur_palm_orientation = glm::mat4(1.0f);
bool hand_in_frame = false;
const unsigned int num_texels = proj_width * proj_height;
const unsigned int image_size = num_texels * 3 * sizeof(uint8_t);
// GLCamera gl_camera(glm::vec3(41.64f, 26.92f, -2.48f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // "fixed" camera
GLCamera gl_flycamera;
GLCamera gl_camera;
GLCamera gl_projector;
// GLCamera gl_camera(glm::vec3(0.0f, -20.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)); // "orbit" camera

int main(int argc, char *argv[])
{
    /* parse cmd line options */
    if (checkCmdLineFlag(argc, (const char **)argv, "debug"))
    {
        std::cout << "Debug mode on..." << std::endl;
        debug_mode = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "freecam"))
    {
        std::cout << "Freecam mode on..." << std::endl;
        freecam_mode = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "cuda"))
    {
        std::cout << "Using CUDA..." << std::endl;
        use_cuda = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "fake_cam"))
    {
        std::cout << "Fake camera on..." << std::endl;
        producer_is_fake = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "pbo"))
    {
        std::cout << "Using PBO for async unpacking..." << std::endl;
        use_pbo = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "no_proj"))
    {
        std::cout << "No projector mode is on" << std::endl;
        use_projector = false;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "no_screen"))
    {
        std::cout << "No screen mode is on" << std::endl;
        use_screen = false;
    }
    Timer t0, t1, t2, t3, t4, t5, t6, t7, t_app, t_misc, t_debug1, t_debug2;
    t_app.start();
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
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // callback for resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    /* setup global GL buffers */
    unsigned int skeletonVAO = 0;
    unsigned int skeletonVBO = 0;
    setup_skeleton_hand_buffers(skeletonVAO, skeletonVBO);
    unsigned int gizmoVAO = 0;
    unsigned int gizmoVBO = 0;
    setup_gizmo_buffers(gizmoVAO, gizmoVBO);
    unsigned int cubeVAO = 0;
    unsigned int cubeVBO = 0;
    setup_cube_buffers(cubeVAO, cubeVBO);
    unsigned int frustrumVAO = 0;
    unsigned int frustrumVBO = 0;
    setup_frustrum_buffers(frustrumVAO, frustrumVBO);
    unsigned int pbo[2] = {0};
    if (use_pbo)
    {
        initGLBuffers(pbo);
    }
    // unsigned int circleVAO, circleVBO;
    // setup_circle_buffers(circleVAO, circleVBO);
    SkinnedModel skinnedModel("C:/src/augmented_hands/resource/GenericHand.fbx",
                              "C:/src/augmented_hands/resource/uv.png",
                              //   "C:/src/augmented_hands/resource/wood.jpg",
                              proj_width, proj_height,
                              cam_width, cam_height);
    n_bones = skinnedModel.NumBones();
    Canvas canvas(cam_width, cam_height, proj_width, proj_height, use_cuda);
    glm::vec3 coa = skinnedModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);
    glm::mat4 mm_to_cm = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    glm::mat4 cm_to_mm = glm::inverse(mm_to_cm);
    glm::mat4 timesTwenty = glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 20.0f, 20.0f));
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    Text text("C:/src/augmented_hands/resource/arial.ttf");
    std::array<glm::vec3, 28> frustumCornerVertices{
        {// near
         {-1.0f, 1.0f, -1.0f},
         {-1.0f, -1.0f, -1.0f},
         {-1.0f, -1.0f, -1.0f},
         {1.0f, -1.0f, -1.0f},
         {1.0f, -1.0f, -1.0f},
         {1.0f, 1.0f, -1.0f},
         {1.0f, 1.0f, -1.0f},
         {-1.0f, 1.0f, -1.0f},
         // far
         {-1.0f, -1.0f, 1.0f},
         {1.0f, -1.0f, 1.0f},
         {1.0f, -1.0f, 1.0f},
         {1.0f, 1.0f, 1.0f},
         {1.0f, 1.0f, 1.0f},
         {-1.0f, 1.0f, 1.0f},
         {-1.0f, 1.0f, 1.0f},
         {-1.0f, -1.0f, 1.0f},
         // connect
         {-1.0f, -1.0f, 1.0f},
         {-1.0f, -1.0f, -1.0f},
         {1.0f, -1.0f, 1.0f},
         {1.0f, -1.0f, -1.0f},
         {1.0f, 1.0f, 1.0f},
         {1.0f, 1.0f, -1.0f},
         {-1.0f, 1.0f, 1.0f},
         {-1.0f, 1.0f, -1.0f},
         // hat
         {-1.0f, 1.0f, -1.0f},
         {0.0f, 1.5f, -1.0f},
         {0.0f, 1.5f, -1.0f},
         {1.0f, 1.0f, -1.0f}}};
    /* setup shaders*/
    Shader jfaInitShader("C:/src/augmented_hands/src/shaders/jfa.vs", "C:/src/augmented_hands/src/shaders/jfa_init.fs");
    Shader jfaShader("C:/src/augmented_hands/src/shaders/jfa.vs", "C:/src/augmented_hands/src/shaders/jfa.fs");
    Shader fastTrackerShader("C:/src/augmented_hands/src/shaders/fast_tracker.vs", "C:/src/augmented_hands/src/shaders/fast_tracker.fs");
    Shader debugShader("C:/src/augmented_hands/src/shaders/debug.vs", "C:/src/augmented_hands/src/shaders/debug.fs");
    Shader textureShader("C:/src/augmented_hands/src/shaders/color_by_texture.vs", "C:/src/augmented_hands/src/shaders/color_by_texture.fs");
    Shader lineShader("C:/src/augmented_hands/src/shaders/line_shader.vs", "C:/src/augmented_hands/src/shaders/line_shader.fs");
    Shader canvasShader;
    if (use_cuda)
        canvasShader = Shader("C:/src/augmented_hands/src/shaders/canvas.vs", "C:/src/augmented_hands/src/shaders/canvas_cuda.fs");
    else
        canvasShader = Shader("C:/src/augmented_hands/src/shaders/canvas.vs", "C:/src/augmented_hands/src/shaders/canvas.fs");
    Shader vcolorShader("C:/src/augmented_hands/src/shaders/color_by_vertex.vs", "C:/src/augmented_hands/src/shaders/color_by_vertex.fs");
    SkinningShader skinnedShader("C:/src/augmented_hands/src/shaders/skin_hand.vs", "C:/src/augmented_hands/src/shaders/skin_hand.fs");
    Shader textShader("C:/src/augmented_hands/src/shaders/text.vs", "C:/src/augmented_hands/src/shaders/text.fs");
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    /* more inits */
    NPP_wrapper::printfNPPinfo();
    double previousTime = glfwGetTime();
    double currentFrame = glfwGetTime();
    double whole = 0.0;
    long frameCount = 0;
    int64_t targetFrameTime = 0;
    uint64_t targetFrameSize = 0;
    std::vector<glm::vec3> skeleton_vertices;
    std::vector<glm::mat4> bones_to_world;
    size_t n_skeleton_primitives = 0;
    bool close_signal = false;
    int leap_time_delay = 50000; // us
    uint8_t *colorBuffer = new uint8_t[image_size];
    uint32_t cam_height = 0;
    uint32_t cam_width = 0;
    blocking_queue<CPylonImage> camera_queue;
    // blocking_queue<std::vector<uint8_t>> projector_queue;
    blocking_queue<uint8_t *> projector_queue;
    BaslerCamera camera;
    DynaFlashProjector projector;
    if (use_projector)
    {
        if (!projector.init())
        {
            std::cerr << "Failed to initialize projector\n";
        }
    }
    LeapConnect leap;
    LEAP_CLOCK_REBASER clockSynchronizer;
    LeapCreateClockRebaser(&clockSynchronizer);
    std::thread producer, consumer;
    // load calibration results if they exist
    glm::mat4 vproj_project;
    glm::mat4 vcam_project;
    std::vector<double> camera_distortion;
    glm::mat4 w2vp;
    glm::mat4 w2vc;
    if (loadCalibrationResults(vcam_project, vproj_project, camera_distortion, w2vp, w2vc))
    {
        std::cout << "Using calibration data for camera and projector settings" << std::endl;
        gl_camera = GLCamera(w2vc, vcam_project, Camera_Mode::FIXED_CAMERA);
        gl_projector = GLCamera(w2vp, vproj_project, Camera_Mode::FIXED_CAMERA);
        if (freecam_mode)
            // gl_flycamera = GLCamera(w2vc, vcam_project, Camera_Mode::FREE_CAMERA);
            gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), Camera_Mode::FREE_CAMERA);
        else
            gl_flycamera = GLCamera(w2vc, vcam_project, Camera_Mode::FIXED_CAMERA);
    }
    else
    {
        std::cout << "Using hard-coded values for camera and projector settings" << std::endl;
        gl_camera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), Camera_Mode::FIXED_CAMERA);
        gl_projector = GLCamera(glm::vec3(-4.76f, 18.2f, 38.6f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), Camera_Mode::FIXED_CAMERA);
        if (freecam_mode)
            gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), Camera_Mode::FREE_CAMERA);
        else
            gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), Camera_Mode::FIXED_CAMERA);
    }
    /* actual thread loops */
    /* image producer (real camera = virtual projector) */
    float exposure = 1850.0f;
    if (camera.init(camera_queue, close_signal, cam_height, cam_width, exposure) && !producer_is_fake)
    {
        /* real producer */
        std::cout << "using real camera to produce images" << std::endl;
        projector.show();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        // camera.balance_white();
        camera.acquire();
    }
    else
    {
        /* fake producer */
        std::cout << "using fake camera to produce images" << std::endl;
        producer_is_fake = true;
        cam_height = 540;
        cam_width = 720;
        producer = std::thread([&camera_queue, &close_signal, &cam_height, &cam_width]() { //, &projector
            CPylonImage image = CPylonImage::Create(PixelType_BGRA8packed, cam_width, cam_height);
            Timer t_block;
            t_block.start();
            while (!close_signal)
            {
                camera_queue.push(image);
                while (t_block.getElapsedTimeInMicroSec() < 300.0)
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
        t_misc.start();
        currentFrame = glfwGetTime();
        std::modf(currentFrame, &whole);
        LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
        deltaTime = static_cast<float>(currentFrame - lastFrame);
        lastFrame = currentFrame;
        frameCount++;
        // stats display
        if (currentFrame - previousTime >= 1.0)
        {
            fps = frameCount;
            ms_per_frame = 1000.0f / frameCount;
            double tpbo, ttex, tproc;
            canvas.getTimerValues(tpbo, ttex, tproc);
            std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
            std::cout << "total app time: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
            std::cout << "misc time: " << t_misc.averageLap() << std::endl;
            std::cout << "wait for cam time: " << t0.averageLap() << std::endl;
            std::cout << "leap frame time: " << t1.averageLap() << std::endl;
            std::cout << "skinning time: " << t2.averageLap() << std::endl;
            std::cout << "debug info: " << t_debug1.averageLap() + t_debug2.averageLap() << std::endl;
            std::cout << "canvas pbo time: " << tpbo << std::endl;
            std::cout << "canvas tex transfer time: " << ttex << std::endl;
            std::cout << "canvas process time: " << tproc << std::endl;
            std::cout << "swap buffers time: " << t3.averageLap() << std::endl;
            std::cout << "GPU->CPU time: " << t4.averageLap() << std::endl;
            // std::cout << "project time: " << t4.averageLap() << std::endl;
            std::cout << "cam q size: " << camera_queue.size() << std::endl;
            std::cout << "proj q size: " << projector_queue.size() << std::endl;

            frameCount = 0;
            previousTime = currentFrame;
            t0.reset();
            t1.reset();
            t2.reset();
            t3.reset();
            t4.reset();
            t5.reset();
            t_misc.reset();
            canvas.resetTimers();
            t_debug1.reset();
            t_debug2.reset();
        }
        // input
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t_misc.stop();
        t0.start();
        // retrieve camera image
        CPylonImage pylonImage = camera_queue.pop();
        uint8_t *buffer = (uint8_t *)pylonImage.GetBuffer();
        // uint8_t* output = (uint8_t*)malloc(cam_width * cam_height * sizeof(uint8_t));
        // uint16_t* dist_output = (uint16_t*)malloc(cam_width * cam_height * sizeof(uint16_t));
        // NPP_wrapper::distanceTransform(buffer, output, cam_width, cam_height);
        // cv::Mat cv_image_input(cam_height, cam_width, CV_8UC4, buffer);
        // cv::imwrite("input.png", cv_image_input);
        // cv::Mat cv_image_output(cam_height, cam_width, CV_8UC4, output);
        // cv::imwrite("output.png", cv_image_output);
        // cv::Mat cv_image_output_distance(cam_height, cam_width, CV_16UC1, dist_output);
        // cv_image_output_distance.convertTo(cv_image_output_distance, CV_8U);
        // cv::imwrite("output_distance.png", cv_image_output_distance);
        // double minVal;
        // double maxVal;
        // cv::Point minLoc;
        // cv::Point maxLoc;
        // cv::minMaxLoc( cv_image_output_distance, &minVal, &maxVal, &minLoc, &maxLoc );
        t0.stop();
        t1.start();
        // sync leap clock
        std::modf(glfwGetTime(), &whole);
        LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
        // get leap frame
        getLeapFrame(leap, targetFrameTime, bones_to_world, skeleton_vertices, debug_mode);
        // get view & projection transforms
        glm::mat4 vcam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 vcam_projection_transform = gl_camera.getProjectionMatrix();
        glm::mat4 vproj_view_transform = gl_projector.getViewMatrix();
        glm::mat4 vproj_projection_transform = gl_projector.getProjectionMatrix();
        glm::mat4 flycam_view_transform = gl_flycamera.getViewMatrix();
        glm::mat4 flycam_projection_transform = gl_flycamera.getProjectionMatrix();
        t1.stop();
        // process leap frame
        if (bones_to_world.size() > 0)
        {
            t2.start();
            glm::mat4 LocalToWorld = bones_to_world[0] * rotx * coa_transform;
            if (debug_mode)
            {
                t_debug1.start();
                // draw skeleton vertices
                glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO);
                glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * skeleton_vertices.size(), skeleton_vertices.data(), GL_STATIC_DRAW);
                n_skeleton_primitives = skeleton_vertices.size() / 2;
                vcolorShader.use();
                vcolorShader.setMat4("projection", flycam_projection_transform);
                vcolorShader.setMat4("view", flycam_view_transform);
                // vcolorShader.setMat4("model", glm::mat4(1.0f));
                vcolorShader.setMat4("model", mm_to_cm);
                glBindVertexArray(skeletonVAO);
                glDrawArrays(GL_LINES, 0, static_cast<int>(n_skeleton_primitives));
                // draw circle oriented like hand palm from leap motion
                // glBindVertexArray(circleVAO);
                // vcolorShader.setMat4("model", bones_to_world[0]);
                // glDrawArrays(GL_TRIANGLE_FAN, 0, 52);
                // draw skeleton bones (as gizmos representing their local coordinate system)
                std::vector<glm::mat4> BoneToLocalTransforms;
                skinnedModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                glBindVertexArray(gizmoVAO);
                for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                {
                    // in bind pose
                    vcolorShader.setMat4("model", LocalToWorld * BoneToLocalTransforms[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                for (unsigned int i = 0; i < bones_to_world.size(); i++)
                {
                    // in leap motion pose
                    vcolorShader.setMat4("model", bones_to_world[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                // draw debug info
                // glm::vec4 palm_normal_hom = cur_palm_orientation * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                // glm::vec3 palm_normal(palm_normal_hom);
                // palm_normal = glm::normalize(palm_normal);
                t_debug1.stop();
            }
            // draw skinned mesh
            skinnedShader.use();
            skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
            skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
            skinnedShader.SetProjectorTransform(vproj_projection_transform * vproj_view_transform);
            bool use_FBO = true;
            if (use_FBO)
            {
                skinnedModel.Render(skinnedShader, bones_to_world, LocalToWorld, true, buffer);
                // skinnedModel.m_fbo.saveColorToFile("test1.png");
                // unsigned int slow_tracker_texture = skinnedModel.m_fbo.getTexture();
                // saveImage("test2.png", slow_tracker_texture, proj_width, proj_height, canvasShader);
                t2.stop();
                // canvas.Render(canvasShader, buffer);
                // canvas.Render(jfaInitShader, jfaShader, fastTrackerShader, slow_tracker_texture, buffer, true);
            }
            else
            {
                skinnedModel.Render(skinnedShader, bones_to_world, LocalToWorld, false, buffer);
                t2.stop();
            }
        }
        if (debug_mode)
        {
            t_debug2.start();
            // draws global coordinate system gizmo at origin
            vcolorShader.use();
            vcolorShader.setMat4("projection", flycam_projection_transform);
            vcolorShader.setMat4("view", flycam_view_transform);
            vcolorShader.setMat4("model", glm::mat4(3.0f));
            glBindVertexArray(gizmoVAO);
            glDrawArrays(GL_LINES, 0, 6);
            // draws cube at world origin
            glDisable(GL_CULL_FACE);
            vcolorShader.setMat4("model", glm::mat4(1.0f));
            glBindVertexArray(cubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glEnable(GL_CULL_FACE);
            // draws frustrum of projector (=vcam)
            std::vector<glm::vec3> vprojFrustumVerticesData(28);
            lineShader.use();
            lineShader.setMat4("projection", flycam_projection_transform);
            lineShader.setMat4("view", flycam_view_transform);
            lineShader.setMat4("model", glm::mat4(1.0f));
            lineShader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
            glm::mat4 vprojUnprojectionMat = glm::inverse(vproj_projection_transform * vproj_view_transform);
            for (int i = 0; i < frustumCornerVertices.size(); ++i)
            {
                glm::vec4 unprojected = vprojUnprojectionMat * glm::vec4(frustumCornerVertices[i], 1.0f);
                vprojFrustumVerticesData[i] = glm::vec3(unprojected) / unprojected.w;
            }
            glBindBuffer(GL_ARRAY_BUFFER, frustrumVBO);
            glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * vprojFrustumVerticesData.size(), vprojFrustumVerticesData.data(), GL_STATIC_DRAW);
            glBindVertexArray(frustrumVAO);
            glDrawArrays(GL_LINES, 0, 28);
            // draws frustrum of camera (=vproj)
            std::vector<glm::vec3> vcamFrustumVerticesData(28);
            glm::mat4 vcamUnprojectionMat = glm::inverse(vcam_projection_transform * vcam_view_transform);
            for (int i = 0; i < frustumCornerVertices.size(); ++i)
            {
                glm::vec4 unprojected = vcamUnprojectionMat * glm::vec4(frustumCornerVertices[i], 1.0f);
                vcamFrustumVerticesData[i] = glm::vec3(unprojected) / unprojected.w;
            }
            glBindBuffer(GL_ARRAY_BUFFER, frustrumVBO);
            glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * vcamFrustumVerticesData.size(), vcamFrustumVerticesData.data(), GL_STATIC_DRAW);
            glBindVertexArray(frustrumVAO);
            glDrawArrays(GL_LINES, 0, 28);
            // draw camera input to near plane of vproj frustrum
            std::vector<glm::vec3> vprojNearVerts(4);
            textureShader.use();
            textureShader.setBool("flipVer", false);
            textureShader.setMat4("projection", flycam_projection_transform);
            textureShader.setMat4("view", flycam_view_transform);
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("binary", true);
            vprojNearVerts[0] = vprojFrustumVerticesData[0];
            vprojNearVerts[1] = vprojFrustumVerticesData[2];
            vprojNearVerts[2] = vprojFrustumVerticesData[4];
            vprojNearVerts[3] = vprojFrustumVerticesData[6];
            Quad vcamNearQuad(vprojNearVerts);
            canvas.RenderBuffer(textureShader, buffer, vcamNearQuad);
            // draw projector output to near plane of vcam frustrum
            textureShader.use();
            textureShader.setBool("flipVer", false);
            textureShader.setMat4("projection", flycam_projection_transform);
            textureShader.setMat4("view", flycam_view_transform);
            textureShader.setMat4("model", glm::mat4(1.0f)); // debugShader.setMat4("model", mm_to_cm);
            textureShader.setBool("binary", false);
            if (bones_to_world.size() > 0)
            {
                std::vector<glm::vec3> vcamNearVerts(4);
                vcamNearVerts[0] = vcamFrustumVerticesData[0];
                vcamNearVerts[1] = vcamFrustumVerticesData[2];
                vcamNearVerts[2] = vcamFrustumVerticesData[4];
                vcamNearVerts[3] = vcamFrustumVerticesData[6];
                Quad vprovNearQuad(vcamNearVerts);
                canvas.RenderTexture(textureShader, skinnedModel.m_fbo.getTexture(), vprovNearQuad);
            }
            // draws text
            glm::vec3 cam_pos = gl_flycamera.getPos();
            glm::vec3 cam_front = gl_flycamera.getFront();
            glm::vec3 proj_pos = gl_projector.getPos();

            text.Render(textShader, std::format("ms_per_frame: {:.02f}, fps: {}", ms_per_frame, fps), 25.0f, 125.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("vcamera pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cam_pos.x, cam_pos.y, cam_pos.z, gl_flycamera.Zoom), 25.0f, 100.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("vcamera front: {:.02f}, {:.02f}, {:.02f}", cam_front.x, cam_front.y, cam_front.z), 25.0f, 75.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("vproj pos: {:.02f}, {:.02f}, {:.02f}, proj fov: {:.02f}", proj_pos.x, proj_pos.y, proj_pos.z, gl_projector.Zoom), 25.0f, 50.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("hand visible? {}", bones_to_world.size() > 0 ? "yes" : "no"), 25.0f, 25.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            t_debug2.stop();
            // text.Render(textShader, std::format("bone index: {}, id: {}", displayBoneIndex, skinnedModel.getBoneName(displayBoneIndex)), 25.0f, 50.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
        }

        // send result to projector queue
        glReadBuffer(GL_FRONT);
        if (use_pbo) // something fishy going on here. using pbo collapses program after a while
        {
            t4.start();
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[frameCount % 2]);
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, 0);
            t4.stop();

            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[(frameCount + 1) % 2]);
            GLubyte *src = (GLubyte *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            if (src)
            {
                // std::vector<uint8_t> data(src, src + image_size);
                // tmpdata.assign(src, src + image_size);
                // std::copy(src, src + tmpdata.size(), tmpdata.begin());
                memcpy(colorBuffer, src, image_size);
                projector_queue.push(colorBuffer);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER); // release pointer to the mapped buffer
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }
        else
        {
            t4.start();
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
            t4.stop();
            // std::vector<uint8_t> data(colorBuffer, colorBuffer + image_size);
            projector_queue.push(colorBuffer);
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
        t3.start();
        glfwSwapBuffers(window);
        glfwPollEvents();
        t3.stop();
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

void setup_skeleton_hand_buffers(unsigned int &VAO, unsigned int &VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // colors
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f    // top
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void setup_gizmo_buffers(unsigned int &VAO, unsigned int &VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // colors
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, // Z
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // Z
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
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

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        gl_flycamera.processKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        gl_flycamera.processKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        gl_flycamera.processKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        gl_flycamera.processKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        gl_flycamera.processKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        gl_flycamera.processKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        space_pressed_flag = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE)
    {
        if (space_pressed_flag)
        {
            space_pressed_flag = false;
            displayBoneIndex = (displayBoneIndex + 1) % n_bones;
        }
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

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    gl_flycamera.processMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    gl_flycamera.processMouseScroll(static_cast<float>(yoffset));
}

bool loadCalibrationResults(glm::mat4 &vcam_project,
                            glm::mat4 &vproj_project,
                            std::vector<double> &camera_distortion,
                            glm::mat4 &w2vp, glm::mat4 &w2vc)
{
    // vp = virtual projector
    // vc = virtual camera
    glm::mat4 flipYZ = glm::mat4(1.0f);
    flipYZ[1][1] = -1.0f;
    flipYZ[2][2] = -1.0f;
    cnpy::NpyArray arr;
    cnpy::npz_t my_npz;
    try
    {
        arr = cnpy::npy_load("C:/src/augmented_hands/debug/leap_calibration/w2p.npy");
        my_npz = cnpy::npz_load("C:/src/augmented_hands/debug/calibration/calibration.npz");
    }
    catch (std::runtime_error &e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    w2vc = glm::make_mat4(arr.data<double>());
    glm::mat4 vc2w = glm::inverse(w2vc);

    float ffar = 100.0f;
    float nnear = 1.0f;
    glm::mat3 camera_intrinsics = glm::make_mat3(my_npz["cam_intrinsics"].data<double>());
    float vpfx = camera_intrinsics[0][0];
    float vpfy = camera_intrinsics[1][1];
    float vpcx = camera_intrinsics[0][2];
    float vpcy = camera_intrinsics[1][2];
    vproj_project = glm::mat4(0.0);
    vproj_project[0][0] = 2 * vpfx / cam_width;
    vproj_project[0][2] = (cam_width - 2 * vpcx) / cam_width;
    vproj_project[1][1] = 2 * vpfy / cam_height;
    vproj_project[1][2] = -(cam_height - 2 * vpcy) / cam_height;
    vproj_project[2][2] = -(ffar + nnear) / (ffar - nnear);
    vproj_project[2][3] = -2 * ffar * nnear / (ffar - nnear);
    vproj_project[3][2] = -1.0f;
    vproj_project = glm::transpose(vproj_project);
    glm::mat3 projector_intrinsics = glm::make_mat3(my_npz["proj_intrinsics"].data<double>());
    float vcfx = projector_intrinsics[0][0];
    float vcfy = projector_intrinsics[1][1];
    float vccx = projector_intrinsics[0][2];
    float vccy = projector_intrinsics[1][2];
    vcam_project = glm::mat4(0.0);
    vcam_project[0][0] = 2 * vcfx / proj_width;
    vcam_project[0][2] = (proj_width - 2 * vccx) / proj_width;
    vcam_project[1][1] = 2 * vcfy / proj_height;
    vcam_project[1][2] = -(proj_height - 2 * vccy) / proj_height;

    vcam_project[2][2] = -(ffar + nnear) / (ffar - nnear);
    vcam_project[2][3] = -2 * ffar * nnear / (ffar - nnear);
    vcam_project[3][2] = -1.0f;
    vcam_project = glm::transpose(vcam_project);
    camera_distortion = my_npz["cam_distortion"].as_vec<double>();
    glm::mat4 vp2vc = glm::make_mat4(my_npz["proj_transform"].data<double>()); // this is a real projector to camera transform
    glm::mat4 vp2w = vp2vc * vc2w;                                             // since glm uses column major, we multiply from the left...
    vp2w = flipYZ * vp2w;
    // vp2w[0][3] *= 0.1f;
    // vp2w[1][3] *= 0.1f;
    // vp2w[2][3] *= 0.1f;
    w2vp = glm::inverse(vp2w);
    w2vp = glm::transpose(w2vp);
    vc2w = flipYZ * vc2w;
    // w2vc[0][3] *= 0.1f;
    // w2vc[1][3] *= 0.1f;
    // w2vc[2][3] *= 0.1f;
    w2vc = glm::inverse(vc2w);
    w2vc = glm::transpose(w2vc);
    return true;
}

void getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime, std::vector<glm::mat4> &bones_to_world, std::vector<glm::vec3> &skeleton_vertices, bool debug)
{
    skeleton_vertices.clear();
    bones_to_world.clear();
    uint64_t targetFrameSize = 0;
    int leap_time_delay = 40000; // us
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    glm::mat4 mm_to_cm = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    // glm::mat4 mm_to_cm = glm::mat4(1.0f);
    glm::mat4 cm_to_mm = glm::inverse(mm_to_cm);
    // Get the buffer size needed to hold the tracking data
    if (LeapGetFrameSize(*leap.getConnectionHandle(), targetFrameTime + leap_time_delay, &targetFrameSize) == eLeapRS_Success)
    {
        // Allocate enough memory
        LEAP_TRACKING_EVENT *interpolatedFrame = (LEAP_TRACKING_EVENT *)malloc((size_t)targetFrameSize);
        // Get the frame
        if (LeapInterpolateFrame(*leap.getConnectionHandle(), targetFrameTime + leap_time_delay, interpolatedFrame, targetFrameSize) == eLeapRS_Success)
        {
            // Use the data...
            //  std::cout << "frame id: " << interpolatedFrame->tracking_frame_id << std::endl;
            //  std::cout << "frame delay (us): " << (long long int)LeapGetNow() - interpolatedFrame->info.timestamp << std::endl;
            //  std::cout << "frame hands: " << interpolatedFrame->nHands << std::endl;
            if (debug)
            {
                if (interpolatedFrame->nHands > 0)
                {
                    if (!hand_in_frame)
                    {
                        std::cout << "hand in frame" << std::endl;
                    }
                    hand_in_frame = true;
                }
                else
                {
                    if (hand_in_frame)
                    {
                        std::cout << "no hand in frame" << std::endl;
                    }
                    hand_in_frame = false;
                }
            }
            glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
            for (uint32_t h = 0; h < interpolatedFrame->nHands; h++)
            {
                LEAP_HAND *hand = &interpolatedFrame->pHands[h];
                if (hand->type == eLeapHandType_Right)
                    continue;
                glm::vec3 palm_pos = glm::vec3(hand->palm.position.x,
                                               hand->palm.position.y,
                                               hand->palm.position.z);
                glm::mat4 palm_orientation = glm::toMat4(glm::quat(hand->palm.orientation.w,
                                                                   hand->palm.orientation.x,
                                                                   hand->palm.orientation.y,
                                                                   hand->palm.orientation.z));

                palm_orientation = palm_orientation * flip_z * flip_y;
                cur_palm_orientation = palm_orientation;
                glm::mat4 palm_trans = glm::translate(glm::mat4(1.0f), palm_pos);
                // if (debug)
                // {
                //     bones_to_world.push_back(palm_trans*roty*palm_orientation);
                // }
                // else
                // {
                bones_to_world.push_back(mm_to_cm * palm_trans * palm_orientation * cm_to_mm);
                // }
                LEAP_VECTOR arm_j1 = hand->arm.prev_joint;
                LEAP_VECTOR arm_j2 = hand->arm.next_joint;
                skeleton_vertices.push_back(glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
                skeleton_vertices.push_back(red);
                skeleton_vertices.push_back(glm::vec3(arm_j2.x, arm_j2.y, arm_j2.z));
                skeleton_vertices.push_back(red);
                glm::mat4 rot = glm::toMat4(glm::quat(hand->arm.rotation.w,
                                                      hand->arm.rotation.x,
                                                      hand->arm.rotation.y,
                                                      hand->arm.rotation.z));
                // rot = palm_orientation * rot;
                glm::vec3 translate = glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z);
                glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
                bones_to_world.push_back(mm_to_cm * trans * rot * roty * flip_z * flip_y * cm_to_mm);
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
                        bones_to_world.push_back(mm_to_cm * trans * rot * roty * flip_z * flip_y * cm_to_mm);
                    }
                }
            }
            // Free the allocated buffer when done.
            free(interpolatedFrame);
        }
    }
}
// // create transformation matrices
// glm::mat4 canvas_model_mat = glm::mat4(1.0f);
// glm::mat4 skeleton_model_mat = glm::mat4(1.0f);
// glm::mat4 mesh_model_mat = glm::mat4(1.0f);
// // model_mat = glm::rotate(model_mat, glm::radians(-55.0f), glm::vec3(0.5f, 1.0f, 0.0f));
// mesh_model_mat = glm::scale(mesh_model_mat, glm::vec3(0.5f, 0.5f, 0.5f));
// // glm::mat4 canvas_projection_mat = glm::ortho(0.0f, (float)proj_width, 0.0f, (float)proj_height, 0.1f, 100.0f);
// // canvas_model_mat = glm::scale(canvas_model_mat, glm::vec3(0.75f, 0.75f, 1.0f));  // 2.0f, 2.0f, 2.0f
// glm::mat4 view_mat = gl_camera.GetViewMatrix();
// // glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  10.0f);
// // glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
// // glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
// // view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
// // view_mat = glm::translate(view_mat, glm::vec3(0.0f, 0.0f, -3.0f));
// // glm::mat4 perspective_projection_mat = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
// // glm::mat4 projection_mat = glm::ortho(0.0f, (float)proj_width, 0.0f, (float)proj_height, 0.1f, 100.0f);
// // glm::mat4 projection_mat = glm::frustum(-(float)proj_width*0.5f, (float)proj_width*0.5f, -(float)proj_height*0.5f, (float)proj_height*0.5f, 0.1f, 100.0f);
// // setup shader inputs
// // float bg_thresh = 0.05f;
// // canvasShader.use();
// // canvasShader.setInt("camera_texture", 0);
// // canvasShader.setFloat("threshold", bg_thresh);
// glm::mat4 canvas_projection_mat = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
void setup_frustrum_buffers(unsigned int &VAO, unsigned int &VBO)
{
    float vertices[] = {
        // frame near
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        // frame far
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        // connect frames
        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        // hat
        -1.0f, 1.0f, -1.0f,
        0.0f, 1.5f, -1.0f,
        0.0f, 1.5f, -1.0f,
        1.0f, 1.0f, -1.0f};
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}
void setup_cube_buffers(unsigned int &VAO, unsigned int &VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,

        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f};
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color coord attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO)
// {
//     std::vector<glm::vec3> vertices;
//     // std::vector<glm::vec3> colors;
//     float radius = 0.5f;
//     glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
//     int n_vertices = 50;
//     vertices.push_back(center);
//     vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     for (int i = 0; i <= n_vertices; i++)   {
//         float twicePI = 2*glm::pi<float>();
//         vertices.push_back(glm::vec3(center.x + (radius * cos(i * twicePI / n_vertices)),
//                                      center.y,
//                                      center.z + (radius * sin(i * twicePI / n_vertices))));
//         vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     }
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glBindVertexArray(VAO);

//     auto test = vertices.data();
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, 6*(n_vertices+2) * sizeof(float), vertices.data(), GL_STATIC_DRAW);

//     // position attribute
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
//     // color attribute
//     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
//     glEnableVertexAttribArray(1);
// }