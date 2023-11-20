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
#include "post_process.h"
#include "utils.h"
#include "cnpy.h"
#include "image_process.h"
#include "stb_image_write.h"
#include <helper_string.h>
#include <filesystem>
namespace fs = std::filesystem;

// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
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
bool simulated_camera = false;
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
const unsigned int projected_image_size = num_texels * 3 * sizeof(uint8_t);
cv::Mat white_image(cam_height, cam_width, CV_8UC1, cv::Scalar(255));
cv::Mat curFrame(cam_height, cam_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
cv::Mat prevFrame(cam_height, cam_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
float downscale_factor = 2.0f;
cv::Size down_size = cv::Size(cam_width / downscale_factor, cam_height / downscale_factor);
cv::Mat flow = cv::Mat::zeros(down_size, CV_32FC2);
cv::Mat curFrame_gray, prevFrame_gray;
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
        std::cout << "Debug mode on" << std::endl;
        debug_mode = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "cam_free"))
    {
        std::cout << "Camera free mode on" << std::endl;
        freecam_mode = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "cuda"))
    {
        std::cout << "Using CUDA" << std::endl;
        use_cuda = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "cam_fake"))
    {
        std::cout << "Camera simulator mode on" << std::endl;
        simulated_camera = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "download_pbo"))
    {
        std::cout << "Using PBO for async unpacking" << std::endl;
        use_pbo = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "proj_off"))
    {
        std::cout << "Projector will not be used" << std::endl;
        use_projector = false;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "no_screen"))
    {
        std::cout << "No screen mode is on" << std::endl;
        use_screen = false;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "leap_poll"))
    {
        std::cout << "Leap poll mode is on" << std::endl;
        poll_mode = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "cam_color"))
    {
        std::cout << "Camera in color mode" << std::endl;
        cam_color_mode = true;
    }
    Timer t_camera, t_leap, t_skin, t_swap, t_download, t_warp, t_app, t_misc, t_debug, t_pp, t_debug2;
    t_app.start();
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
    // SkinnedModel cubeModel("C:/src/augmented_hands/resource/cube_test.fbx",
    //                        "C:/src/augmented_hands/resource/uv.png",
    //                        //   "C:/src/augmented_hands/resource/wood.jpg",
    //                        proj_width, proj_height,
    //                        cam_width, cam_height);
    SkinnedModel leftHandModel("../../resource/GenericHand_fixed_weights.fbx",
                               "../../resource/uv.png",
                               //   "C:/src/augmented_hands/resource/wood.jpg",
                               proj_width, proj_height,
                               cam_width, cam_height); // GenericHand.fbx is a left hand model
    SkinnedModel rightHandModel("../../resource/GenericHand_fixed_weights.fbx",
                                "../../resource/uv.png",
                                //   "C:/src/augmented_hands/resource/wood.jpg",
                                proj_width, proj_height,
                                cam_width, cam_height,
                                false);
    // SkinnedModel dinosaur("../../resource/reconst.ply", "", proj_width, proj_height, cam_width, cam_height);
    n_bones = leftHandModel.NumBones();
    // Canvas canvas(cam_width, cam_height, proj_width, proj_height, use_cuda);
    PostProcess postProcess(cam_width, cam_height, proj_width, proj_height);
    Quad fullScreenQuad(0.0f);
    std::vector<glm::vec2> screen_verts = {{-1.0f, 1.0f},
                                           {-1.0f, -1.0f},
                                           {1.0f, -1.0f},
                                           {1.0f, 1.0f}};
    // std::vector<glm::vec2> screen_verts = {{-0.785f, 0.464f},
    //                                        {-0.815f, -0.857f},
    //                                        {0.295f, -0.662f},
    //                                        {0.307f, 0.372f}};
    glm::mat4 c2p_homography = postProcess.findHomography(screen_verts);
    glm::vec3 coa = leftHandModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);

    // glm::mat4 mm_to_cm = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    // glm::mat4 mm_to_cm = glm::mat4(1.0f);
    // glm::mat4 cm_to_mm = glm::inverse(mm_to_cm);
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    Text text("../../resource/arial.ttf");
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
    FBO hands_fbo(proj_width, proj_height);
    FBO postprocess_fbo(proj_width, proj_height);
    FBO c2p_fbo(proj_width, proj_height);
    /* setup shaders*/
    Shader NNShader("../../src/shaders/NN_shader.vs", "../../src/shaders/NN_shader.fs");
    Shader jfaInitShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa_init.fs");
    Shader jfaShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa.fs");
    Shader fastTrackerShader("../../src/shaders/fast_tracker.vs", "../../src/shaders/fast_tracker.fs");
    Shader debugShader("../../src/shaders/debug.vs", "../../src/shaders/debug.fs");
    Shader projectorShader("../../src/shaders/projector_shader.vs", "../../src/shaders/projector_shader.fs");
    Shader projectorOnlyShader("../../src/shaders/projector_only.vs", "../../src/shaders/projector_only.fs");
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader lineShader("../../src/shaders/line_shader.vs", "../../src/shaders/line_shader.fs");
    Shader coordShader("../../src/shaders/coords.vs", "../../src/shaders/coords.fs");
    Shader canvasShader;
    if (use_cuda)
        canvasShader = Shader("../../src/shaders/canvas.vs", "../../src/shaders/canvas_cuda.fs");
    else
        canvasShader = Shader("../../src/shaders/canvas.vs", "../../src/shaders/canvas.fs");
    Shader vcolorShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    SkinningShader skinnedShader("../../src/shaders/skin_hand.vs", "../../src/shaders/skin_hand.fs");
    SkinningShader skinnedShaderSimple("../../src/shaders/skin_hand_simple.vs", "../../src/shaders/skin_hand_simple.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    /* more inits */
    NPP_wrapper::printfNPPinfo();
    double previousAppTime = t_app.getElapsedTimeInSec();
    double previousSecondAppTime = t_app.getElapsedTimeInSec();
    double currentAppTime = t_app.getElapsedTimeInSec();
    double whole = 0.0;
    long frameCount = 0;
    int64_t targetFrameTime = 0;
    uint64_t targetFrameSize = 0;
    std::vector<glm::vec3> skeleton_vertices;
    std::vector<glm::mat4> bones_to_world_left;
    std::vector<glm::mat4> bones_to_world_right;
    size_t n_skeleton_primitives = 0;
    bool close_signal = false;
    int leap_time_delay = 50000; // us
    uint8_t *colorBuffer = new uint8_t[projected_image_size];
    CGrabResultPtr ptrGrabResult;
    Texture camTexture = Texture();
    Texture flowTexture = Texture();
    camTexture.init(cam_width, cam_height, n_cam_channels);
    flowTexture.init(cam_width, cam_height, 2);
    // uint32_t cam_height = 0;
    // uint32_t cam_width = 0;
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
    LeapConnect leap(poll_mode);
    LEAP_CLOCK_REBASER clockSynchronizer;
    LeapCreateClockRebaser(&clockSynchronizer);
    std::thread producer, consumer;
    // load calibration results if they exist
    glm::mat4 vproj_project;
    glm::mat4 vcam_project;
    std::vector<double> camera_distortion;
    glm::mat4 w2vp;
    glm::mat4 w2vc;
    Camera_Mode camera_mode = freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    if (loadCalibrationResults(vcam_project, vproj_project, camera_distortion, w2vp, w2vc))
    {
        std::cout << "Using calibration data for camera and projector settings" << std::endl;
        if (freecam_mode)
        {
            // gl_flycamera = GLCamera(w2vc, vcam_project, Camera_Mode::FREE_CAMERA);
            // gl_flycamera = GLCamera(w2vc, vcam_project, Camera_Mode::FREE_CAMERA, proj_width, proj_height, 10.0f);
            gl_flycamera = GLCamera(glm::vec3(70.0f, -150.0f, 1008.0f),
                                    glm::vec3(0.0f, 0.0f, 0.0f),
                                    glm::vec3(0.0f, -1.0f, 0.0f),
                                    camera_mode,
                                    proj_width,
                                    proj_height,
                                    1500.0f,
                                    100.0f,
                                    true);
            gl_camera = GLCamera(w2vc, vcam_project, camera_mode, proj_width, proj_height, 50.0f, true);
            gl_projector = GLCamera(w2vp, vproj_project, camera_mode, cam_width, cam_height, 50.0f, false);
        }
        else
        {
            gl_camera = GLCamera(w2vc, vcam_project, camera_mode, proj_width, proj_height);
            gl_projector = GLCamera(w2vp, vproj_project, camera_mode, cam_width, cam_height);
            gl_flycamera = GLCamera(w2vc, vcam_project, camera_mode, proj_width, proj_height);
        }
    }
    else
    {
        std::cout << "Using hard-coded values for camera and projector settings" << std::endl;
        gl_camera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                             glm::vec3(0.0f, 0.0f, 0.0f),
                             glm::vec3(0.0f, 1.0f, 0.0f),
                             camera_mode, proj_width, proj_height, 500.0f, 2.0f);
        gl_projector = GLCamera(glm::vec3(-4.76f, 18.2f, 38.6f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, -1.0f, 0.0f),
                                camera_mode, proj_width, proj_height, 500.0f, 2.0f);
        gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                camera_mode, proj_width, proj_height, 1500.0f, 50.0f);
    }
    std::vector<glm::vec3> far_frustrum = {{-1.0f, 1.0f, 1.0f},
                                           {-1.0f, -1.0f, 1.0f},
                                           {1.0f, -1.0f, 1.0f},
                                           {1.0f, 1.0f, 1.0f}};
    std::vector<glm::vec3> vcamFarVerts(4);
    // unproject points
    glm::mat4 vcam_view_transform = gl_camera.getViewMatrix();
    glm::mat4 vcam_projection_transform = gl_camera.getProjectionMatrix();
    glm::mat4 vproj_view_transform = gl_projector.getViewMatrix();
    glm::mat4 vproj_projection_transform = gl_projector.getProjectionMatrix();
    glm::mat4 vcamUnprojectionMat = glm::inverse(vcam_projection_transform * vcam_view_transform);
    glm::mat4 vprojUnprojectionMat = glm::inverse(vproj_projection_transform * vproj_view_transform);
    for (int i = 0; i < far_frustrum.size(); ++i)
    {
        glm::vec4 unprojected = vcamUnprojectionMat * glm::vec4(far_frustrum[i], 1.0f);
        vcamFarVerts[i] = glm::vec3(unprojected) / unprojected.w;
    }
    glm::vec3 normal = glm::triangleNormal(vcamFarVerts[0], vcamFarVerts[1], vcamFarVerts[2]);
    for (int i = 0; i < far_frustrum.size(); ++i)
    {
        vcamFarVerts[i] += 0.1f * normal;
    }
    Quad vcamFarQuad(vcamFarVerts);
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
        t_misc.start();
        currentAppTime = t_app.getElapsedTimeInSec(); // glfwGetTime();
        deltaTime = static_cast<float>(currentAppTime - previousAppTime);
        previousAppTime = currentAppTime;
        if (!poll_mode)
        {
            std::modf(currentAppTime, &whole);
            LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
        }
        frameCount++;

        /* display stats */
        if (currentAppTime - previousSecondAppTime >= 1.0)
        {
            fps = frameCount;
            ms_per_frame = 1000.0f / frameCount;
            std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
            std::cout << "total app: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
            std::cout << "misc: " << t_misc.averageLapInMilliSec() << std::endl;
            std::cout << "cam: " << t_camera.averageLapInMilliSec() << std::endl;
            std::cout << "leap: " << t_leap.averageLapInMilliSec() << std::endl;
            std::cout << "skinning: " << t_skin.averageLapInMilliSec() << std::endl;
            std::cout << "debug info: " << t_debug.averageLapInMilliSec() << std::endl;
            std::cout << "post process: " << t_pp.averageLapInMilliSec() << std::endl;
            std::cout << "warp: " << t_warp.averageLapInMilliSec() << std::endl;
            std::cout << "swap buffers: " << t_swap.averageLapInMilliSec() << std::endl;
            std::cout << "GPU->CPU: " << t_download.averageLapInMilliSec() << std::endl;
            // std::cout << "project time: " << t4.averageLap() << std::endl;
            std::cout << "cam q1 size: " << camera_queue.size() << std::endl;
            std::cout << "cam q2 size: " << camera_queue_cv.size() << std::endl;
            std::cout << "proj q size: " << projector_queue.size() << std::endl;
            frameCount = 0;
            previousSecondAppTime = currentAppTime;
            t_camera.reset();
            t_leap.reset();
            t_skin.reset();
            t_swap.reset();
            t_download.reset();
            t_warp.reset();
            t_misc.reset();
            t_pp.reset();
            t_debug.reset();
        }
        /* deal with user input */
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t_misc.stop();

        /* deal with camera input */
        t_camera.start();
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
        // buffer = (uint8_t *)pylonImage.GetBuffer();
        // curFrame = tmp.clone();

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
        t_camera.stop();

        /* deal with leap input */
        t_leap.start();
        if (!poll_mode)
        {
            // sync leap clock
            std::modf(glfwGetTime(), &whole);
            LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
            // get leap frame
        }
        LEAP_STATUS status = getLeapFrame(leap, targetFrameTime, bones_to_world_left, bones_to_world_right, skeleton_vertices, poll_mode, lastFrameID);
        if (poll_mode && false)
        {
            if (status == LEAP_STATUS::LEAP_NEWFRAME)
            {
                // render frame to texture
                if (bones_to_world_right.size() > 0)
                {
                    /* render skinned mesh to fbo, in camera space*/
                    skinnedShaderSimple.use();
                    skinnedShaderSimple.SetDisplayBoneIndex(displayBoneIndex);
                    skinnedShaderSimple.SetWorldTransform(vproj_projection_transform * vproj_view_transform);
                    skinnedShaderSimple.setInt("src", 0);
                    hands_fbo.bind();
                    glEnable(GL_DEPTH_TEST);
                    rightHandModel.Render(skinnedShaderSimple, bones_to_world_right, rotx);
                    hands_fbo.unbind();
                    glDisable(GL_DEPTH_TEST);
                }
            }
            else
            {
                if (status == LEAP_STATUS::LEAP_NONEWFRAME)
                {
                    // use optical flow to warp texture
                    cv::cvtColor(curFrame, curFrame_gray, cv::COLOR_RGBA2GRAY);
                    cv::resize(curFrame_gray, curFrame_gray, down_size);
                    cv::calcOpticalFlowFarneback(prevFrame_gray, curFrame_gray, flow, 0.5, 5, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
                    prevFrame_gray = curFrame_gray.clone();
                }
            }
        }
        /* camera transforms, todo: use only in debug mode */
        // get view & projection transforms
        glm::mat4 vcam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 vcam_projection_transform = gl_camera.getProjectionMatrix();
        glm::mat4 vproj_view_transform = gl_projector.getViewMatrix();
        glm::mat4 vproj_projection_transform = gl_projector.getProjectionMatrix();
        glm::mat4 flycam_view_transform = gl_flycamera.getViewMatrix();
        glm::mat4 flycam_projection_transform = gl_flycamera.getProjectionMatrix();
        t_leap.stop();
        /* render warped cam image */
        // projectorOnlyShader.use();
        // projectorOnlyShader.setMat4("camTransform", vcam_projection_transform * vcam_view_transform);
        // projectorOnlyShader.setMat4("projTransform", vproj_projection_transform * vproj_view_transform);
        // projectorOnlyShader.setBool("binary", true);
        // projectorOnlyShader.setBool("src", 0);
        // canvas.renderTexture(camTexture.getTexture(), projectorOnlyShader, vcamFarQuad);

        /* skin hand mesh with leap input */
        t_skin.start();
        if (bones_to_world_right.size() > 0)
        {
            /* render skinned mesh to fbo, in camera space*/
            skinnedShaderSimple.use();
            skinnedShaderSimple.SetDisplayBoneIndex(displayBoneIndex);
            skinnedShaderSimple.SetWorldTransform(vproj_projection_transform * vproj_view_transform);
            skinnedShaderSimple.setInt("src", 0);
            hands_fbo.bind(true);
            glEnable(GL_DEPTH_TEST);
            rightHandModel.Render(skinnedShaderSimple, bones_to_world_right, rotx);
            hands_fbo.unbind();
            glDisable(GL_DEPTH_TEST);

            /* render skinned mesh to fbo, in projector space */
            // skinnedShader.use();
            // skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
            // skinnedShader.SetWorldTransform(vcam_projection_transform * vcam_view_transform);
            // skinnedShader.SetProjectorTransform(vproj_projection_transform * vproj_view_transform);
            // skinnedShader.setBool("binary", true);
            // skinnedShader.setInt("src", 0);
            // skinnedShader.setInt("projTexture", 1);
            // skinnedModel.Render(skinnedShader, bones_to_world_right, rotx, camTexture.getTexture(), true);
            /* render another mesh to fbo */
            // projectorShader.use();
            // projectorShader.setBool("flipVer", false);
            // projectorShader.setMat4("camTransform", vcam_projection_transform * vcam_view_transform);
            // projectorShader.setMat4("projTransform", vproj_projection_transform * vproj_view_transform);
            // projectorShader.setBool("binary", true);
            // dinosaur.Render(projectorShader, camTexture.getTexture(), true);
        }
        if (bones_to_world_left.size() > 0)
        {
            /* render skinned mesh to fbo, in camera space*/
            skinnedShaderSimple.use();
            skinnedShaderSimple.SetDisplayBoneIndex(displayBoneIndex);
            skinnedShaderSimple.SetWorldTransform(vproj_projection_transform * vproj_view_transform);
            skinnedShaderSimple.setInt("src", 0);
            hands_fbo.bind(bones_to_world_right.size() == 0);
            glEnable(GL_DEPTH_TEST);
            leftHandModel.Render(skinnedShaderSimple, bones_to_world_left, rotx);
            hands_fbo.unbind();
            glDisable(GL_DEPTH_TEST);
        }
        t_skin.stop();
        /* post process fbo using camera input */
        t_pp.start();
        // if (!debug_mode)
        // {
        // skinnedModel.m_fbo.saveColorToFile("test1.png");
        // canvasShader.use();
        // canvasShader.setBool("binary", false);
        // canvasShader.setBool("flipVer", false);
        // canvasShader.setInt("src", 0);
        // canvas.renderTexture(skinnedModel.m_fbo.getTexture(), canvasShader);
        // projectorOnlyShader.use();
        // projectorOnlyShader.setMat4("camTransform", vcam_projection_transform * vcam_view_transform);
        // projectorOnlyShader.setMat4("projTransform", vproj_projection_transform * vproj_view_transform);
        // projectorOnlyShader.setBool("binary", true);
        // projectorOnlyShader.setBool("src", 0);
        /* render hand with jfa */
        // unsigned int warped_cam = canvas.renderToFBO(camTexture.getTexture(), projectorOnlyShader, vcamFarQuad);
        postProcess.jump_flood(jfaInitShader, jfaShader, NNShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo);
        c2p_fbo.bind();
        textureShader.use();
        textureShader.setMat4("view", glm::mat4(1.0f));
        textureShader.setMat4("projection", c2p_homography);
        textureShader.setMat4("model", glm::mat4(1.0f));
        textureShader.setBool("flipVer", true);
        textureShader.setInt("src", 0);
        textureShader.setBool("binary", false);
        postprocess_fbo.getTexture()->bind();
        fullScreenQuad.render();
        c2p_fbo.unbind();

        // }
        t_pp.stop();

        // saveImage("test2.png", skinnedModel.m_fbo.getTexture(), proj_width, proj_height, canvasShader);

        // canvas.Render(canvasShader, buffer);
        // canvas.Render(jfaInitShader, jfaShader, fastTrackerShader, slow_tracker_texture, buffer, true);
        // }
        // else
        // {
        //     skinnedModel.Render(skinnedShader, bones_to_world, LocalToWorld, false, buffer);
        //     t2.stop();
        // }

        /* debug mode renders*/
        if (!debug_mode)
        {
            textureShader.use();
            textureShader.setBool("flipVer", false);
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("view", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("binary", false);
            textureShader.setInt("src", 0);
            c2p_fbo.getTexture()->bind();
            fullScreenQuad.render();
        }
        else
        {
            t_debug.start();
            // setup some vertices
            std::vector<glm::vec3> near_frustrum = {{-1.0f, 1.0f, -1.0f},
                                                    {-1.0f, -1.0f, -1.0f},
                                                    {1.0f, -1.0f, -1.0f},
                                                    {1.0f, 1.0f, -1.0f}};
            std::vector<glm::vec3> mid_frustrum = {{-1.0f, 1.0f, 0.7f},
                                                   {-1.0f, -1.0f, 0.7f},
                                                   {1.0f, -1.0f, 0.7f},
                                                   {1.0f, 1.0f, 0.7f}};
            std::vector<glm::vec3> far_frustrum = {{-1.0f, 1.0f, 1.0f},
                                                   {-1.0f, -1.0f, 1.0f},
                                                   {1.0f, -1.0f, 1.0f},
                                                   {1.0f, 1.0f, 1.0f}};
            std::vector<glm::vec3> vcamNearVerts(4);
            std::vector<glm::vec3> vcamMidVerts(4);
            std::vector<glm::vec3> vcamFarVerts(4);
            std::vector<glm::vec3> vprojNearVerts(4);
            std::vector<glm::vec3> vprojMidVerts(4);
            // unproject points
            glm::mat4 vcamUnprojectionMat = glm::inverse(vcam_projection_transform * vcam_view_transform);
            glm::mat4 vprojUnprojectionMat = glm::inverse(vproj_projection_transform * vproj_view_transform);
            for (int i = 0; i < mid_frustrum.size(); ++i)
            {
                glm::vec4 unprojected = vcamUnprojectionMat * glm::vec4(mid_frustrum[i], 1.0f);
                vcamMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                unprojected = vcamUnprojectionMat * glm::vec4(near_frustrum[i], 1.0f);
                vcamNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
                unprojected = vcamUnprojectionMat * glm::vec4(far_frustrum[i], 1.0f);
                vcamFarVerts[i] = glm::vec3(unprojected) / unprojected.w;
                unprojected = vprojUnprojectionMat * glm::vec4(mid_frustrum[i], 1.0f);
                vprojMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                unprojected = vprojUnprojectionMat * glm::vec4(near_frustrum[i], 1.0f);
                vprojNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
            }
            Quad vcamNearQuad(vcamNearVerts);
            Quad vcamMidQuad(vcamMidVerts);
            Quad vcamFarQuad(vcamFarVerts);
            Quad vprojNearQuad(vprojNearVerts);
            Quad vprojMidQuad(vprojMidVerts);
            // draws some mesh (lit by camera input)
            {
                /* quad at vcam far plane, shined by vproj (perspective corrected) */
                // projectorOnlyShader.use();
                // projectorOnlyShader.setBool("flipVer", false);
                // projectorOnlyShader.setMat4("camTransform", flycam_projection_transform * flycam_view_transform);
                // projectorOnlyShader.setMat4("projTransform", vproj_projection_transform * vproj_view_transform);
                // projectorOnlyShader.setBool("binary", false);
                // camTexture.bind();
                // projectorOnlyShader.setInt("src", 0);
                // vcamFarQuad.render();

                /* dinosaur */
                // projectorShader.use();
                // projectorShader.setBool("flipVer", false);
                // projectorShader.setMat4("camTransform", flycam_projection_transform * flycam_view_transform);
                // projectorShader.setMat4("projTransform", vproj_projection_transform * vproj_view_transform);
                // projectorShader.setBool("binary", true);
                // dinosaur.Render(projectorShader, camTexture.getTexture(), false);
                // projectorShader.setMat4("camTransform", vcam_projection_transform * vcam_view_transform);
                // dinosaur.Render(projectorShader, camTexture.getTexture(), true);
                // textureShader.use();
                // textureShader.setBool("flipVer", false);
                // textureShader.setMat4("projection", flycam_projection_transform);
                // textureShader.setMat4("view", flycam_view_transform);
                // textureShader.setMat4("model", glm::mat4(1.0f));
                // textureShader.setBool("binary", false);
                // textureShader.setInt("src", 0);
                // glActiveTexture(GL_TEXTURE0);
                // glBindTexture(GL_TEXTURE_2D, dinosaur.m_fbo.getTexture());
                // vcamNearQuad.render();
            }
            // draws global coordinate system gizmo at origin
            {
                vcolorShader.use();
                vcolorShader.setMat4("projection", flycam_projection_transform);
                vcolorShader.setMat4("view", flycam_view_transform);
                vcolorShader.setMat4("model", glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 20.0f, 20.0f)));
                glBindVertexArray(gizmoVAO);
                glDrawArrays(GL_LINES, 0, 6);
            }
            // draws cube at world origin
            {
                vcolorShader.use();
                vcolorShader.setMat4("projection", flycam_projection_transform);
                vcolorShader.setMat4("view", flycam_view_transform);
                vcolorShader.setMat4("model", glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f)));
                glEnable(GL_DEPTH_TEST);
                glDisable(GL_CULL_FACE);
                glBindVertexArray(cubeVAO);
                glDrawArrays(GL_TRIANGLES, 0, 36);
                glEnable(GL_CULL_FACE);
            }
            if (bones_to_world_right.size() > 0)
            {
                // draw skeleton as red lines
                {
                    // glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO);
                    // glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * skeleton_vertices.size(), skeleton_vertices.data(), GL_STATIC_DRAW);
                    // n_skeleton_primitives = skeleton_vertices.size() / 2;
                    // vcolorShader.use();
                    // vcolorShader.setMat4("projection", flycam_projection_transform);
                    // vcolorShader.setMat4("view", flycam_view_transform);
                    // vcolorShader.setMat4("model", glm::mat4(1.0f)); // vcolorShader.setMat4("model", glm::mat4(1.0f));
                    // glBindVertexArray(skeletonVAO);
                    // glDrawArrays(GL_LINES, 0, static_cast<int>(n_skeleton_primitives));
                }
                // draw circle oriented like hand palm from leap motion
                {
                    // vcolorShader.use();
                    // vcolorShader.setMat4("projection", flycam_projection_transform);
                    // vcolorShader.setMat4("view", flycam_view_transform);
                    // vcolorShader.setMat4("model", mm_to_cm); // vcolorShader.setMat4("model", glm::mat4(1.0f));
                    // glBindVertexArray(circleVAO);
                    // vcolorShader.setMat4("model", bones_to_world[0]);
                    // glDrawArrays(GL_TRIANGLE_FAN, 0, 52);
                }
                // draw bones local coordinates as gizmos
                {
                    vcolorShader.use();
                    vcolorShader.setMat4("projection", flycam_projection_transform);
                    vcolorShader.setMat4("view", flycam_view_transform);
                    std::vector<glm::mat4> BoneToLocalTransforms;
                    leftHandModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                    glBindVertexArray(gizmoVAO);
                    // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                    for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                    {
                        // in bind pose
                        vcolorShader.setMat4("model", rotx * BoneToLocalTransforms[i]);
                        glDrawArrays(GL_LINES, 0, 6);
                    }
                    for (unsigned int i = 0; i < bones_to_world_right.size(); i++)
                    {
                        // in leap motion pose
                        vcolorShader.setMat4("model", bones_to_world_right[i]);
                        glDrawArrays(GL_LINES, 0, 6);
                    }
                }
                // draw gizmo for palm orientation
                {
                    vcolorShader.use();
                    vcolorShader.setMat4("projection", flycam_projection_transform);
                    vcolorShader.setMat4("view", flycam_view_transform);
                    vcolorShader.setMat4("model", bones_to_world_right[0]);
                    glBindVertexArray(gizmoVAO);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                // draw skinned mesh in 3D
                {
                    /* without camera texture */
                    skinnedShaderSimple.use();
                    skinnedShaderSimple.SetDisplayBoneIndex(displayBoneIndex);
                    skinnedShaderSimple.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                    skinnedShaderSimple.setInt("src", 0);
                    rightHandModel.Render(skinnedShaderSimple, bones_to_world_right, rotx);
                    /* with camera texture */
                    // skinnedShader.use();
                    // skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                    // skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                    // skinnedShader.SetProjectorTransform(vproj_projection_transform * vproj_view_transform);
                    // skinnedShader.setBool("binary", false);
                    // skinnedShader.setInt("src", 0);
                    // skinnedShader.setInt("projTexture", 1);
                    // skinnedModel.Render(skinnedShader, bones_to_world_right, rotx, camTexture.getTexture(), false, space_modifier);
                }
            }
            if (bones_to_world_left.size() > 0)
            {
                // draw bones local coordinates as gizmos
                {
                    vcolorShader.use();
                    vcolorShader.setMat4("projection", flycam_projection_transform);
                    vcolorShader.setMat4("view", flycam_view_transform);
                    std::vector<glm::mat4> BoneToLocalTransforms;
                    leftHandModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                    glBindVertexArray(gizmoVAO);
                    // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                    for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                    {
                        // in bind pose
                        vcolorShader.setMat4("model", rotx * BoneToLocalTransforms[i]);
                        glDrawArrays(GL_LINES, 0, 6);
                    }
                    for (unsigned int i = 0; i < bones_to_world_left.size(); i++)
                    {
                        // in leap motion pose
                        vcolorShader.setMat4("model", bones_to_world_left[i]);
                        glDrawArrays(GL_LINES, 0, 6);
                    }
                }
                // draw skinned mesh in 3D
                {
                    /* without camera texture */
                    skinnedShaderSimple.use();
                    skinnedShaderSimple.SetDisplayBoneIndex(displayBoneIndex);
                    // glm::mat4 trans = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 20.0f));
                    skinnedShaderSimple.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                    skinnedShaderSimple.setInt("src", 0);
                    leftHandModel.Render(skinnedShaderSimple, bones_to_world_left, rotx);
                    /* with camera texture */
                    // skinnedShader.use();
                    // skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                    // skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                    // skinnedShader.SetProjectorTransform(vproj_projection_transform * vproj_view_transform);
                    // skinnedShader.setBool("binary", true);
                    // skinnedShader.setInt("src", 0);
                    // skinnedShader.setInt("projTexture", 1);
                    // skinnedModel.Render(skinnedShader, bones_to_world_left, rotx, camTexture.getTexture(), false, space_modifier);
                }
            }
            // draws frustrum of camera (=vproj)
            {
                std::vector<glm::vec3> vprojFrustumVerticesData(28);
                lineShader.use();
                lineShader.setMat4("projection", flycam_projection_transform);
                lineShader.setMat4("view", flycam_view_transform);
                lineShader.setMat4("model", glm::mat4(1.0f));
                lineShader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
                glm::mat4 vprojUnprojectionMat = glm::inverse(vproj_projection_transform * vproj_view_transform);
                for (unsigned int i = 0; i < frustumCornerVertices.size(); i++)
                {
                    glm::vec4 unprojected = vprojUnprojectionMat * glm::vec4(frustumCornerVertices[i], 1.0f);
                    vprojFrustumVerticesData[i] = glm::vec3(unprojected) / unprojected.w;
                }
                glBindBuffer(GL_ARRAY_BUFFER, frustrumVBO);
                glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * vprojFrustumVerticesData.size(), vprojFrustumVerticesData.data(), GL_STATIC_DRAW);
                glBindVertexArray(frustrumVAO);
                glDrawArrays(GL_LINES, 0, 28);
            }
            // draws frustrum of projector (=vcam)
            {
                std::vector<glm::vec3> vcamFrustumVerticesData(28);
                lineShader.use();
                lineShader.setMat4("projection", flycam_projection_transform);
                lineShader.setMat4("view", flycam_view_transform);
                lineShader.setMat4("model", glm::mat4(1.0f));
                lineShader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
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
            }
            // draw camera input to near plane of vproj frustrum
            {
                // directly render camera input or any other texture
                textureShader.use();
                textureShader.setBool("flipVer", false);
                textureShader.setMat4("projection", flycam_projection_transform);
                textureShader.setMat4("view", flycam_view_transform);
                textureShader.setMat4("model", glm::mat4(1.0f));
                textureShader.setBool("binary", false);
                textureShader.setInt("src", 0);
                // camTexture.bind();
                // glBindTexture(GL_TEXTURE_2D, resTexture);
                postprocess_fbo.getTexture()->bind();
                // hands_fbo.getTexture()->bind();
                vprojNearQuad.render();
            }
            // draw projector output to near plane of vcam frustrum
            {
                t_warp.start();
                textureShader.use();
                textureShader.setBool("flipVer", false);
                textureShader.setMat4("projection", flycam_projection_transform);
                textureShader.setMat4("view", flycam_view_transform);
                textureShader.setMat4("model", glm::mat4(1.0f)); // debugShader.setMat4("model", mm_to_cm);
                textureShader.setBool("binary", false);
                textureShader.setInt("src", 0);
                c2p_fbo.getTexture()->bind();
                vcamNearQuad.render(); // canvas.renderTexture(skinnedModel.m_fbo.getTexture() /*tex*/, textureShader, vcamNearQuad);
                t_warp.stop();
            }
            // draws text
            {
                float text_spacing = 10.0f;
                glm::vec3 cam_pos = gl_flycamera.getPos();
                glm::vec3 cam_front = gl_flycamera.getFront();
                glm::vec3 proj_pos = gl_projector.getPos();
                std::vector<std::string> texts_to_render = {
                    std::format("debug_vector: {:.02f}, {:.02f}, {:.02f}", debug_vec.x, debug_vec.y, debug_vec.z),
                    std::format("ms_per_frame: {:.02f}, fps: {}", ms_per_frame, fps),
                    std::format("vcamera pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cam_pos.x, cam_pos.y, cam_pos.z, gl_flycamera.Zoom),
                    std::format("vcamera front: {:.02f}, {:.02f}, {:.02f}", cam_front.x, cam_front.y, cam_front.z),
                    std::format("vproj pos: {:.02f}, {:.02f}, {:.02f}, proj fov: {:.02f}", proj_pos.x, proj_pos.y, proj_pos.z, gl_projector.Zoom),
                    std::format("Rhand visible? {}", bones_to_world_right.size() > 0 ? "yes" : "no"),
                    std::format("Lhand visible? {}", bones_to_world_left.size() > 0 ? "yes" : "no"),
                    std::format("modifiers : shift: {}, ctrl: {}, space: {}", shift_modifier ? "on" : "off", ctrl_modifier ? "on" : "off", space_modifier ? "on" : "off")};
                for (int i = 0; i < texts_to_render.size(); ++i)
                {
                    text.Render(textShader, texts_to_render[i], 25.0f, texts_to_render.size() * text_spacing - text_spacing * i, 0.25f, glm::vec3(1.0f, 1.0f, 1.0f));
                }
            }
            t_debug.stop();
        }

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
                memcpy(colorBuffer, src, projected_image_size);
                projector_queue.push(colorBuffer);
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
    void *data = malloc(projected_image_size);
    // create ping pong pbos
    glGenBuffers(2, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[0]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, projected_image_size, data, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[1]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, projected_image_size, data, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    free(data);
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    bool mod = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        mod = true;
        shift_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            gl_camera.processKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            gl_camera.processKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            gl_camera.processKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            gl_camera.processKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            gl_camera.processKeyboard(UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            gl_camera.processKeyboard(DOWN, deltaTime);
    }
    else
    {
        shift_modifier = false;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        mod = true;
        ctrl_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            gl_projector.processKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            gl_projector.processKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            gl_projector.processKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            gl_projector.processKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            gl_projector.processKeyboard(UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            gl_projector.processKeyboard(DOWN, deltaTime);
    }
    else
    {
        ctrl_modifier = false;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        mod = true;
        space_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            debug_vec.x += deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            debug_vec.x -= deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            debug_vec.y += deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            debug_vec.y -= deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            debug_vec.z += deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            debug_vec.z -= deltaTime * 10.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE)
    {
        if (space_modifier)
        {
            space_modifier = false;
            displayBoneIndex = (displayBoneIndex + 1) % n_bones;
        }
    }
    if (!mod)
    {
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
        if (glfwGetKey(window, GLFW_KEY_KP_0) == GLFW_PRESS)
        {
            gl_flycamera.setViewMatrix(gl_camera.getViewMatrix());
            gl_flycamera.setProjectionMatrix(gl_camera.getProjectionMatrix());
        }
        if (glfwGetKey(window, GLFW_KEY_KP_1) == GLFW_PRESS)
        {
            gl_flycamera.setViewMatrix(gl_projector.getViewMatrix());
            gl_flycamera.setProjectionMatrix(gl_projector.getProjectionMatrix());
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
    if (shift_modifier)
    {
        gl_camera.processMouseMovement(xoffset, yoffset);
    }
    else
    {
        if (ctrl_modifier)
        {
            gl_projector.processMouseMovement(xoffset, yoffset);
        }
        else
        {
            gl_flycamera.processMouseMovement(xoffset, yoffset);
        }
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (shift_modifier)
    {
        gl_camera.processMouseScroll(static_cast<float>(yoffset));
    }
    else
    {
        if (ctrl_modifier)
            gl_projector.processMouseScroll(static_cast<float>(yoffset));
        else
            gl_flycamera.processMouseScroll(static_cast<float>(yoffset));
    }
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

    float ffar = 1500.0f;
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
    glm::mat4 vp2vc = glm::make_mat4(my_npz["proj_transform"].data<double>()); // this is the camera to projector transform
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
    glm::mat4 flip_x = glm::mat4(1.0f);
    flip_x[0][0] = -1.0f;
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    // magic numbers
    int magic_leap_time_delay = 40000; // us
    float magic_scale_factor = 10.0f;
    float magic_wrist_offset = -65.0f;
    float magic_arm_forward_offset = -120.0f;
    glm::mat4 magic_leap_basis_fix = roty * flip_z * flip_y;
    glm::mat4 chirality = glm::mat4(1.0f);
    // init
    glm::mat4 scalar = glm::scale(glm::mat4(1.0f), glm::vec3(magic_scale_factor));
    uint64_t targetFrameSize = 0;
    LEAP_TRACKING_EVENT *frame = nullptr;
    if (poll_mode)
    {
        frame = leap.getFrame();
        if (frame != NULL && (frame->tracking_frame_id > lastFrameID))
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
        // if (debug_vec.x > 0)
        if (hand->type == eLeapHandType_Right)
            chirality = flip_x;
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
        // translate arm joint in the local x direction to shorten the arm
        glm::vec3 xforward = glm::normalize(glm::vec3(arm_rot[2][0], arm_rot[2][1], arm_rot[2][2])); // 3rd column of rotation matrix is local x
        xforward *= magic_arm_forward_offset;
        arm_translate = glm::translate(arm_translate, xforward);
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
    if (poll_mode)
        free(frame->pHands);
    free(frame);
    return LEAP_STATUS::LEAP_NEWFRAME;
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