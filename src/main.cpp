#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "readerwritercircularbuffer.h"
#include "camera.h"
#include "gl_camera.h"
#include "display.h"
#include "SerialPort.h"
#include "shader.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "point_cloud.h"
#include "leap.h"
#include "text.h"
#include "post_process.h"
#include "utils.h"
#include "cnpy.h"
#include "image_process.h"
#include "stb_image_write.h"
#include <helper_string.h>
#include <filesystem>
#include "helpers.h"
#include "imgui.h"
#include "imgui_stdlib.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "diffuse.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
namespace fs = std::filesystem;

// forward declarations
void openIMGUIFrame();
void create_virtual_cameras(GLCamera &gl_flycamera, GLCamera &gl_projector, GLCamera &gl_camera);
glm::vec3 triangulate(LeapConnect &leap, const glm::vec2 &leap1, const glm::vec2 &leap2);
bool extract_centroid(cv::Mat binary_image, glm::vec2 &centeroid);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);
LEAP_STATUS getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &skeleton_vertices, bool leap_poll_mode, int64_t &lastFrameID);
void initGLBuffers(unsigned int *pbo);
bool loadLeapCalibrationResults(glm::mat4 &cam_project, glm::mat4 &proj_project,
                                std::vector<double> &camera_distortion,
                                glm::mat4 &w2c_auto,
                                glm::mat4 &w2c_user,
                                std::vector<glm::vec2> &points_2d,
                                std::vector<glm::vec3> &points_3d);
bool loadCoaxialCalibrationResults(std::vector<glm::vec2> &cur_screen_verts);
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

enum class TextureMode
{
    ORIGINAL = 0,
    FROM_FILE = 1,
    PROJECTIVE = 2,
    BAKED = 3,
};
enum class SDMode
{
    PROMPT = 0,
    ANIMAL = 1,
    GESTURE = 2,
};
enum class MaterialMode
{
    DIFFUSE = 0,
    GGX = 1,
    GLASS = 2,
};
enum class PostProcessMode
{
    NONE = 0,
    CAM_FEED = 1,
    MASK = 2,
    JUMP_FLOOD = 3,
    OF = 4,
};
enum class LeapCalibrationSettings
{
    AUTO = 0,
    USER = 1,
};
enum class LeapCalibrationStateMachine
{
    IDLE = 0,
    COLLECT = 1,
    CALIBRATE = 2,
    SHOW = 3,
};
enum class CalibrationMode
{
    OFF = 0,
    COAXIAL = 1,
    LEAP = 2,
};
// global state
bool debug_mode = false;
bool bakeRequest = false;
bool freecam_mode = false;
bool use_cuda = false;
bool simulated_camera = false;
bool use_pbo = true;
bool use_projector = false;
bool use_screen = true;
bool leap_poll_mode = false;
bool cam_color_mode = false;
bool ready_to_collect = false;
int leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::COLLECT);
int use_leap_calib_results = static_cast<int>(LeapCalibrationSettings::USER);
int calib_mode = static_cast<int>(CalibrationMode::OFF);
std::string testFile("../../resource/uv.png");
std::string bakeFile("../../resource/baked.png");
std::string diffuseTextureFile("../../resource/uv.png");
std::string sd_prompt("A natural skinned human hand with a colorful dragon tattoo, photorealistic skin");
// std::string uvUnwrapFile("../../resource/UVUnwrapFile.png");
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_width = 720;
const unsigned int cam_height = 540;
uint32_t leap_width = 640;
uint32_t leap_height = 240;
unsigned int n_cam_channels = cam_color_mode ? 4 : 1;
unsigned int cam_buffer_format = cam_color_mode ? GL_RGBA : GL_RED;
float exposure = 1850.0f; // 1850.0f;
int leap_calib_n_points = 1000;
// global state
int postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD);
int sd_mode = static_cast<int>(SDMode::PROMPT);
int texture_mode = static_cast<int>(TextureMode::ORIGINAL);
int material_mode = static_cast<int>(MaterialMode::DIFFUSE);
int sd_mask_mode = 2;
bool useCoaxialCalib = true;
bool showCamera = true;
bool showProjector = true;
bool saveIntermed = false;
Texture *dynamicTexture = nullptr;
Texture *bakedTexture = nullptr;
int magic_leap_time_delay = 40000; // us
float magic_leap_scale_factor = 10.0f;
float magic_wrist_offset = -65.0f;
float magic_arm_forward_offset = -170.0f;
int diffuse_seed = -1;
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
bool dragging = false;
int dragging_vert = 0;
int closest_vert = 0;
float min_dist = 100000.0f;
float deltaTime = 0.0f;
float masking_threshold = 0.035f;
glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
int64_t lastFrameID = 0;
bool space_modifier = false;
bool shift_modifier = false;
bool ctrl_modifier = false;
bool activateGUI = false;
bool tab_pressed = false;
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
// cv::Mat curCamImage;
// std::vector<uint8_t> curCamBuf;
cv::Mat curCamThresholded;
std::vector<std::string> animals{
    "a fish",
    "an elephant",
    "a giraffe",
    "a tiger",
    "a lion",
    "a cat",
    "a dog",
    "a horse",
    "a cow",
    "a sheep",
    "a pig",
    "a rabbit",
    "a squirrel",
    "a monkey",
    "a gorilla",
    "a panda"};
std::vector<glm::vec3> points_3d;
std::vector<glm::vec2> points_2d, points_2d_inliners;
std::vector<glm::vec2> points_2d_reprojected, points_2d_inliers_reprojected;
int pnp_iters = 500;
float pnp_rep_error = 2.0f;
float pnp_confidence = 0.95f;
bool showInliersOnly = true;
bool showReprojections = true;
std::vector<glm::vec3> screen_verts_color_red = {{1.0f, 0.0f, 0.0f}};
std::vector<glm::vec3> screen_verts_color_blue = {{0.0f, 0.0f, 1.0f}};
std::vector<glm::vec2> screen_verts = {{-1.0f, 1.0f},
                                       {-1.0f, -1.0f},
                                       {1.0f, -1.0f},
                                       {1.0f, 1.0f}};
std::vector<glm::vec2> cur_screen_verts = {{-1.0f, 1.0f},
                                           {-1.0f, -1.0f},
                                           {1.0f, -1.0f},
                                           {1.0f, 1.0f}};
// GLCamera gl_projector(glm::vec3(41.64f, 26.92f, -2.48f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // "fixed" camera
GLCamera gl_flycamera;
GLCamera gl_projector;
GLCamera gl_camera;
glm::mat4 w2c_auto, w2c_user;
glm::mat4 proj_project;
glm::mat4 cam_project;
std::vector<double> camera_distortion;
glm::mat4 c2p_homography;
// GLCamera gl_projector(glm::vec3(0.0f, -20.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)); // "orbit" camera
FBO hands_fbo(proj_width, proj_height, 4, false);
FBO bake_fbo(1024, 1024, 4, false);
FBO postprocess_fbo(proj_width, proj_height, 4, false);
FBO c2p_fbo(proj_width, proj_height, 4, false);
LeapConnect leap(leap_poll_mode);
DynaFlashProjector projector(true, false);
BaslerCamera camera;

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
    if (checkCmdLineFlag(argc, (const char **)argv, "proj_on"))
    {
        std::cout << "Projector will be used" << std::endl;
        use_projector = true;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "no_screen"))
    {
        std::cout << "No screen mode is on" << std::endl;
        use_screen = false;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "leap_poll"))
    {
        std::cout << "Leap poll mode is on" << std::endl;
        leap_poll_mode = true;
        leap.setPollMode(leap_poll_mode);
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
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE); // disable resizing
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
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glPointSize(10.0f);
    glEnable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // callback for resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    /* Setup Dear ImGui context */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
    /* Setup Platform/Renderer backends */
    ImGui_ImplGlfw_InitForOpenGL(window, true); // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();
    /* setup global GL buffers */
    unsigned int skeletonVAO = 0;
    unsigned int skeletonVBO = 0;
    Helpers::setupSkeletonBuffers(skeletonVAO, skeletonVBO);
    unsigned int gizmoVAO = 0;
    unsigned int gizmoVBO = 0;
    Helpers::setupGizmoBuffers(gizmoVAO, gizmoVBO);
    unsigned int cubeVAO = 0;
    unsigned int cubeVBO = 0;
    Helpers::setupCubeBuffers(cubeVAO, cubeVBO);
    unsigned int tcubeVAO = 0;
    unsigned int tcubeVBO1 = 0;
    unsigned int tcubeVBO2 = 0;
    Helpers::setupCubeTexturedBuffers(tcubeVAO, tcubeVBO1, tcubeVBO2);
    unsigned int frustrumVAO = 0;
    unsigned int frustrumVBO = 0;
    Helpers::setupFrustrumBuffers(frustrumVAO, frustrumVBO);
    unsigned int pbo[2] = {0};
    initGLBuffers(pbo);
    hands_fbo.init();
    bake_fbo.init();
    postprocess_fbo.init();
    c2p_fbo.init();
    SkinnedModel leftHandModel("../../resource/GenericHand_fixed_weights.fbx",
                               "../../resource/uv.png", // uv.png
                               proj_width, proj_height,
                               cam_width, cam_height); // GenericHand.fbx is a left hand model
    SkinnedModel rightHandModel("../../resource/GenericHand_fixed_weights.fbx",
                                "../../resource/uv.png", // uv.png
                                proj_width, proj_height,
                                cam_width, cam_height,
                                false);
    dynamicTexture = new Texture(testFile.c_str(), GL_TEXTURE_2D);
    dynamicTexture->init();
    const fs::path user_path{bakeFile};
    if (fs::exists(user_path))
        bakedTexture = new Texture(bakeFile.c_str(), GL_TEXTURE_2D);
    else
        bakedTexture = new Texture(testFile.c_str(), GL_TEXTURE_2D);
    bakedTexture->init();
    // SkinnedModel dinosaur("../../resource/reconst.ply", "", proj_width, proj_height, cam_width, cam_height);
    n_bones = leftHandModel.NumBones();
    PostProcess postProcess(cam_width, cam_height, proj_width, proj_height);
    Quad fullScreenQuad(0.0f);
    glm::vec3 coa = leftHandModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    Text text("../../resource/arial.ttf");
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
    Shader NNShader("../../src/shaders/NN_shader.vs", "../../src/shaders/NN_shader.fs");
    Shader maskShader("../../src/shaders/mask.vs", "../../src/shaders/mask.fs");
    Shader jfaInitShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa_init.fs");
    Shader jfaShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa.fs");
    Shader fastTrackerShader("../../src/shaders/fast_tracker.vs", "../../src/shaders/fast_tracker.fs");
    Shader debugShader("../../src/shaders/debug.vs", "../../src/shaders/debug.fs");
    Shader projectorShader("../../src/shaders/projector_shader.vs", "../../src/shaders/projector_shader.fs");
    Shader projectorOnlyShader("../../src/shaders/projector_only.vs", "../../src/shaders/projector_only.fs");
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader lineShader("../../src/shaders/line_shader.vs", "../../src/shaders/line_shader.fs");
    Shader coordShader("../../src/shaders/coords.vs", "../../src/shaders/coords.fs");
    Shader vcolorShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    SkinningShader skinnedShader("../../src/shaders/skin_hand_simple.vs", "../../src/shaders/skin_hand_simple.fs");
    Shader bakeSimple("../../src/shaders/bake_proj_simple.vs", "../../src/shaders/bake_proj_simple.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    DirectionalLight dirLight(glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, 1.0f, glm::vec3(0.0f, -1.0f, -1.0f));
    /* more inits */
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
    uint8_t *colorBuffer = new uint8_t[projected_image_size];
    CGrabResultPtr ptrGrabResult;
    Texture camTexture = Texture();
    cv::Mat camImage, camImageOrig;
    Texture flowTexture = Texture();
    Texture displayTexture = Texture();
    displayTexture.init(cam_width, cam_height, n_cam_channels);
    camTexture.init(cam_width, cam_height, n_cam_channels);
    flowTexture.init(cam_width, cam_height, 2);
    // uint32_t cam_height = 0;
    // uint32_t cam_width = 0;
    // blocking_queue<CGrabResultPtr> camera_queue;
    moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> camera_queue(20);
    // queue_spsc<cv::Mat> camera_queue_cv(50);
    // blocking_queue<cv::Mat> camera_queue_cv;
    // blocking_queue<std::vector<uint8_t>> projector_queue;
    // blocking_queue<uint8_t *> projector_queue;
    moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> projector_queue(20);
    if (use_projector)
    {
        if (!projector.init())
        {
            std::cerr << "Failed to initialize projector\n";
            use_projector = false;
        }
    }
    LEAP_CLOCK_REBASER clockSynchronizer;
    LeapCreateClockRebaser(&clockSynchronizer);
    std::thread producer, consumer;
    // load calibration results if they exist
    Camera_Mode camera_mode = freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    if (loadLeapCalibrationResults(proj_project, cam_project, camera_distortion, w2c_auto, w2c_user, points_2d, points_3d))
    {
        create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
    }
    else
    {
        std::cout << "Using hard-coded values for camera and projector settings" << std::endl;
        gl_projector = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                camera_mode, proj_width, proj_height, 500.0f, 2.0f);
        gl_camera = GLCamera(glm::vec3(-4.76f, 18.2f, 38.6f),
                             glm::vec3(0.0f, 0.0f, 0.0f),
                             glm::vec3(0.0f, -1.0f, 0.0f),
                             camera_mode, proj_width, proj_height, 500.0f, 2.0f);
        gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                camera_mode, proj_width, proj_height, 1500.0f, 50.0f);
    }
    loadCoaxialCalibrationResults(cur_screen_verts);
    c2p_homography = PostProcess::findHomography(cur_screen_verts);
    /* actual thread loops */
    /* image producer (real camera = virtual projector) */
    if (camera.init(camera_queue, close_signal, cam_height, cam_width, exposure) && !simulated_camera)
    {
        /* real producer */
        std::cout << "Using real camera to produce images" << std::endl;
        projector.show();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        camera.balance_white();
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
        //         while (t_block.getElapsedTimeInMilliSec() < 1.9)
        //         {
        //         }
        //         t_block.stop();
        //         t_block.start();
        //     }
        //     std::cout << "Producer finish" << std::endl;
        // });
    }
    // image consumer (real projector = virtual camera)
    consumer = std::thread([&projector_queue, &close_signal]() { //, &projector
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
        /* update / sync clocks */
        t_misc.start();
        currentAppTime = t_app.getElapsedTimeInSec(); // glfwGetTime();
        deltaTime = static_cast<float>(currentAppTime - previousAppTime);
        previousAppTime = currentAppTime;
        if (!leap_poll_mode)
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
            std::cout << "debug: " << t_debug.averageLapInMilliSec() << std::endl;
            std::cout << "post process: " << t_pp.averageLapInMilliSec() << std::endl;
            std::cout << "warp: " << t_warp.averageLapInMilliSec() << std::endl;
            std::cout << "swap buffers: " << t_swap.averageLapInMilliSec() << std::endl;
            std::cout << "GPU->CPU: " << t_download.averageLapInMilliSec() << std::endl;
            // std::cout << "project time: " << t4.averageLap() << std::endl;
            std::cout << "cam q1 size: " << camera_queue.size_approx() << std::endl;
            // std::cout << "cam q2 size: " << camera_queue_cv.size() << std::endl;
            std::cout << "proj q size: " << projector_queue.size_approx() << std::endl;
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
        glfwPollEvents();
        process_input(window);
        if (activateGUI)
        {
            openIMGUIFrame(); // create imgui frame
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else
        {
            switch (calib_mode)
            {
            case static_cast<int>(CalibrationMode::COAXIAL):
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                break;
            }
            default:
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                break;
            }
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t_misc.stop();

        /* deal with camera input */
        t_camera.start();
        // prevFrame = curFrame.clone();
        // if (simulated_camera)
        // {
        //     cv::Mat tmp = camera_queue_cv.pop();
        //     camTexture.load((uint8_t *)tmp.data, true, cam_buffer_format);
        // }
        // else
        // {
        // std::cout << "before: " << camera_queue.size_approx() << std::endl;
        camera_queue.wait_dequeue_latest(ptrGrabResult);
        // camera_queue.wait_dequeue(ptrGrabResult);
        // std::cout << "after: " << camera_queue.size_approx() << std::endl;
        // curCamImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
        // curCamBuf = std::vector<uint8_t>((uint8_t *)ptrGrabResult->GetBuffer(), (uint8_t *)ptrGrabResult->GetBuffer() + ptrGrabResult->GetImageSize());

        if (calib_mode == static_cast<int>(CalibrationMode::LEAP))
        {
            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
            cv::flip(camImageOrig, camImage, 1);
        }
        else
        {
            camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
        }
        // }
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
        if (calib_mode != static_cast<int>(CalibrationMode::LEAP))
        {
            if (!leap_poll_mode)
            {
                // sync leap clock
                std::modf(glfwGetTime(), &whole);
                LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
                // get leap frame
            }
            LEAP_STATUS status = getLeapFrame(leap, targetFrameTime, bones_to_world_left, bones_to_world_right, skeleton_vertices, leap_poll_mode, lastFrameID);
            // bones_to_world_left.clear();
            // bones_to_world_right.clear();
            // skeleton_vertices.clear();
        }
        /* camera transforms, todo: use only in debug mode */
        // get view & projection transforms
        glm::mat4 proj_view_transform = gl_projector.getViewMatrix();
        glm::mat4 proj_projection_transform = gl_projector.getProjectionMatrix();
        glm::mat4 cam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 cam_projection_transform = gl_camera.getProjectionMatrix();
        glm::mat4 flycam_view_transform = gl_flycamera.getViewMatrix();
        glm::mat4 flycam_projection_transform = gl_flycamera.getProjectionMatrix();
        t_leap.stop();
        /* render warped cam image */
        // projectorOnlyShader.use();
        // projectorOnlyShader.setMat4("camTransform", proj_projection_transform * proj_view_transform);
        // projectorOnlyShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
        // projectorOnlyShader.setBool("binary", true);
        // projectorOnlyShader.setBool("src", 0);
        // canvas.renderTexture(camTexture.getTexture(), projectorOnlyShader, projFarQuad);
        switch (calib_mode)
        {
        case static_cast<int>(CalibrationMode::OFF):
        {
            /* skin hand mesh with leap input */
            t_skin.start();
            if (bones_to_world_right.size() > 0)
            {
                /* render skinned mesh to fbo, in camera space*/
                skinnedShader.use();
                skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform);
                skinnedShader.setBool("bake", false);
                skinnedShader.setBool("flipTexVertically", false);
                skinnedShader.setInt("src", 0);
                if (material_mode == static_cast<int>(MaterialMode::GGX))
                {
                    dirLight.calcLocalDirection(glm::mat4(1.0f));
                    skinnedShader.SetDirectionalLight(dirLight);
                    skinnedShader.setBool("useGGX", true);
                }
                hands_fbo.bind(true);
                glEnable(GL_DEPTH_TEST);
                switch (texture_mode)
                {
                case static_cast<int>(TextureMode::ORIGINAL):
                    skinnedShader.setBool("useProjector", false);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, nullptr);
                    break;
                case static_cast<int>(TextureMode::BAKED):
                    skinnedShader.setBool("useProjector", false);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, bake_fbo.getTexture());
                    break;
                case static_cast<int>(TextureMode::PROJECTIVE):
                    skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                    skinnedShader.setBool("useProjector", true);
                    skinnedShader.setBool("flipTexVertically", true);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, dynamicTexture);
                    break;
                default:
                    skinnedShader.setBool("useProjector", false);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, nullptr);
                    break;
                }
                hands_fbo.unbind();
                glDisable(GL_DEPTH_TEST);
                if (bakeRequest)
                {
                    // download camera image to cpu
                    bake_fbo.bind(true);
                    textureShader.use();
                    textureShader.setMat4("view", glm::mat4(1.0f));
                    textureShader.setMat4("projection", glm::mat4(1.0f));
                    textureShader.setMat4("model", glm::mat4(1.0f));
                    textureShader.setBool("isGray", true);
                    textureShader.setBool("flipVer", false);
                    textureShader.setBool("flipHor", true);
                    textureShader.setInt("src", 0);
                    textureShader.setBool("binary", false);
                    camTexture.bind();
                    fullScreenQuad.render();
                    bake_fbo.unbind();
                    if (saveIntermed)
                        bake_fbo.saveColorToFile("../../resource/camera_image.png", false);
                    std::vector<uint8_t> buf = bake_fbo.getBuffer(1);
                    // cv::Mat test = cv::Mat(1024, 1024, CV_8UC1, buf.data());
                    // cv::imwrite("../../resource/camera_image1.png", test);
                    // download camera image thresholded to cpu (eww)
                    bake_fbo.bind(true);
                    textureShader.use();
                    textureShader.setMat4("view", glm::mat4(1.0f));
                    textureShader.setMat4("projection", glm::mat4(1.0f));
                    textureShader.setMat4("model", glm::mat4(1.0f));
                    textureShader.setBool("isGray", true);
                    textureShader.setBool("flipVer", false);
                    textureShader.setBool("flipHor", true);
                    textureShader.setFloat("threshold", masking_threshold);
                    textureShader.setInt("src", 0);
                    textureShader.setBool("binary", true);
                    camTexture.bind();
                    fullScreenQuad.render();
                    bake_fbo.unbind();
                    if (saveIntermed)
                        bake_fbo.saveColorToFile("../../resource/camera_mask.png", false);
                    std::vector<uint8_t> buf_mask = bake_fbo.getBuffer(1);
                    // send camera image to stable diffusion
                    int outwidth, outheight;
                    // cv::threshold(curCamImage, curCamThresholded, 10, 255, cv::THRESH_BINARY);
                    // cv::imwrite("../../resource/input_image.png", curCamImage);
                    // cv::imwrite("../../resource/input_mask.png", curCamThresholded);
                    // std::vector<uint8_t> tmp;
                    std::string myprompt;
                    std::vector<std::string> random_animal;
                    switch (sd_mode)
                    {
                    case static_cast<int>(SDMode::PROMPT):
                        myprompt = sd_prompt;
                        break;
                    case static_cast<int>(SDMode::ANIMAL):
                        std::sample(animals.begin(),
                                    animals.end(),
                                    std::back_inserter(random_animal),
                                    1,
                                    std::mt19937{std::random_device{}()});
                        myprompt = random_animal[0];
                        break;
                    default:
                        myprompt = sd_prompt;
                        break;
                    }
                    try
                    {
                        std::vector<uint8_t> img2img_data = Diffuse::img2img(myprompt.c_str(),
                                                                             outwidth, outheight,
                                                                             buf, buf_mask, diffuse_seed,
                                                                             1024, 1024, 1,
                                                                             512, 512, false, false, sd_mask_mode);
                        if (saveIntermed)
                        {
                            cv::Mat img2img_result = cv::Mat(outheight, outwidth, CV_8UC3, img2img_data.data()).clone();
                            cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
                            cv::imwrite("../../resource/sd_result.png", img2img_result);
                        }
                        if (dynamicTexture != nullptr)
                            delete dynamicTexture;
                        dynamicTexture = new Texture(GL_TEXTURE_2D);
                        dynamicTexture->init(outwidth, outheight, 3);
                        dynamicTexture->load(img2img_data.data(), true, GL_RGB);
                        bake_fbo.bind(true);
                        /* hand */
                        glDisable(GL_CULL_FACE);
                        glEnable(GL_DEPTH_TEST);
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform);
                        skinnedShader.setBool("useProjector", true);
                        skinnedShader.setBool("bake", true);
                        skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                        skinnedShader.setBool("flipTexVertically", true);
                        skinnedShader.setInt("src", 0);
                        // dynamicTexture->bind();
                        rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, dynamicTexture);
                        /* debug points */
                        // vcolorShader.use();
                        // vcolorShader.setMat4("view", glm::mat4(1.0f));
                        // vcolorShader.setMat4("projection", glm::mat4(1.0f));
                        // vcolorShader.setMat4("model", glm::mat4(1.0f));
                        // std::vector<glm::vec2> points;
                        // rightHandModel.getUnrolledTexCoords(points);
                        // Helpers::UV2NDC(points);
                        // std::vector<glm::vec3> screen_vert_color = {{1.0f, 0.0f, 0.0f}};
                        // PointCloud cloud(points, screen_vert_color);
                        // cloud.render();
                        bake_fbo.unbind();
                        glDisable(GL_DEPTH_TEST);
                        glEnable(GL_CULL_FACE);
                        // glDisable(GL_DEPTH_TEST);
                        bake_fbo.saveColorToFile(bakeFile);
                        // if (bakedTexture != nullptr)
                        // {
                        //     delete bakedTexture;
                        //     bakedTexture = nullptr;
                        // }
                        // bakedTexture = new Texture(bakeFile.c_str(), GL_TEXTURE_2D);
                        // bakedTexture->init();
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    bakeRequest = false;
                }
            }
            if (bones_to_world_left.size() > 0)
            {
                /* render skinned mesh to fbo, in camera space*/
                skinnedShader.use();
                skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform);
                skinnedShader.setBool("useProjector", false);
                skinnedShader.setBool("bake", false);
                skinnedShader.setInt("src", 0);
                hands_fbo.bind(bones_to_world_right.size() == 0);
                glEnable(GL_DEPTH_TEST);
                leftHandModel.Render(skinnedShader, bones_to_world_left, rotx);
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
            // projectorOnlyShader.setMat4("camTransform", proj_projection_transform * proj_view_transform);
            // projectorOnlyShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
            // projectorOnlyShader.setBool("binary", true);
            // projectorOnlyShader.setBool("src", 0);
            /* render hand with jfa */
            // unsigned int warped_cam = canvas.renderToFBO(camTexture.getTexture(), projectorOnlyShader, projFarQuad);
            switch (postprocess_mode)
            {
            case static_cast<int>(PostProcessMode::NONE):
            {
                // bind fbo
                postprocess_fbo.bind();
                // bind texture
                hands_fbo.getTexture()->bind();
                // render
                textureShader.use();
                textureShader.setInt("src", 0);
                textureShader.setBool("flipVer", false);
                textureShader.setBool("flipHor", false);
                textureShader.setMat4("projection", glm::mat4(1.0f));
                textureShader.setBool("isGray", false);
                textureShader.setMat4("view", glm::mat4(1.0f));
                textureShader.setMat4("model", glm::mat4(1.0f));
                fullScreenQuad.render(false, false, true);
                // unbind fbo
                postprocess_fbo.unbind();
                break;
            }
            case static_cast<int>(PostProcessMode::CAM_FEED):
            {
                postprocess_fbo.bind();
                camTexture.bind();
                textureShader.use();
                textureShader.setInt("src", 0);
                textureShader.setBool("flipVer", true);
                textureShader.setBool("flipHor", true);
                textureShader.setMat4("projection", glm::mat4(1.0f));
                textureShader.setBool("isGray", true);
                textureShader.setBool("binary", false);
                textureShader.setMat4("view", glm::mat4(1.0f));
                textureShader.setMat4("model", glm::mat4(1.0f));
                textureShader.setFloat("threshold", masking_threshold);
                fullScreenQuad.render();
                postprocess_fbo.unbind();
                break;
            }
            case static_cast<int>(PostProcessMode::MASK):
            {
                postProcess.mask(maskShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
                break;
            }
            case static_cast<int>(PostProcessMode::JUMP_FLOOD):
            {
                postProcess.jump_flood(jfaInitShader, jfaShader, NNShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
                break;
            }
            case static_cast<int>(PostProcessMode::OF):
            {
                break;
            }
            default:
                break;
            }
            c2p_fbo.bind();
            textureShader.use();
            textureShader.setMat4("view", glm::mat4(1.0f));
            if (useCoaxialCalib)
                textureShader.setMat4("projection", c2p_homography);
            else
                textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipVer", false);
            textureShader.setBool("flipHor", false);
            textureShader.setBool("binary", false);
            textureShader.setBool("isGray", false);
            textureShader.setInt("src", 0);
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
                textureShader.setBool("isGray", false);
                textureShader.setInt("src", 0);
                c2p_fbo.getTexture()->bind();
                fullScreenQuad.render();
            }
            break;
        }
        case static_cast<int>(CalibrationMode::COAXIAL):
        {
            std::vector<cv::Point2f> origpts, newpts;
            for (int i = 0; i < 4; ++i)
            {
                origpts.push_back(cv::Point2f(screen_verts[i].x, screen_verts[i].y));
                newpts.push_back(cv::Point2f(cur_screen_verts[i].x, cur_screen_verts[i].y));
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
                cv::Vec4f cord = cv::Vec4f(screen_verts[i].x, screen_verts[i].y, 0.0f, 1.0f);
                cv::Mat tmp = perspective * cv::Mat(cord);
                cur_screen_verts[i].x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);
                cur_screen_verts[i].y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);
            }
            glm::mat4 viewMatrix;
            GLMHelpers::CV2GLM(perspective, &viewMatrix);
            textureShader.use();
            textureShader.setMat4("view", viewMatrix);
            textureShader.setMat4("projection", glm::mat4(1.0f));
            textureShader.setMat4("model", glm::mat4(1.0f));
            textureShader.setBool("flipVer", true);
            textureShader.setBool("flipHor", true);
            textureShader.setInt("src", 0);
            // textureShader.setBool("justGreen", true);
            textureShader.setBool("isGray", true);
            camTexture.bind();
            fullScreenQuad.render();
            PointCloud cloud(cur_screen_verts, screen_verts_color_red);
            vcolorShader.use();
            vcolorShader.setMat4("view", glm::mat4(1.0f));
            vcolorShader.setMat4("projection", glm::mat4(1.0f));
            vcolorShader.setMat4("model", glm::mat4(1.0f));
            cloud.render();
            break;
        }
        case static_cast<int>(CalibrationMode::LEAP):
        {
            switch (leap_calibration_state)
            {
            case static_cast<int>(LeapCalibrationStateMachine::IDLE):
            {
                break;
            }
            case static_cast<int>(LeapCalibrationStateMachine::COLLECT):
            {
                cv::Mat thr;
                cv::threshold(camImage, thr, 250, 255, cv::THRESH_BINARY);
                glm::vec2 center, center_leap1, center_leap2;
                if (extract_centroid(thr, center))
                {
                    displayTexture.load((uint8_t *)thr.data, true, cam_buffer_format);
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
                    vcolorShader.use();
                    vcolorShader.setMat4("view", glm::mat4(1.0f));
                    vcolorShader.setMat4("projection", glm::mat4(1.0f));
                    vcolorShader.setMat4("model", glm::mat4(1.0f));
                    glm::vec2 center_NDC = Helpers::ScreenToNDC(center, cam_width, cam_height, true);
                    std::vector<glm::vec2> test = {center_NDC};
                    PointCloud pointCloud(test, screen_verts_color_red);
                    pointCloud.render();
                    cv::Mat leap1_thr, leap2_thr;
                    std::vector<uint8_t> buffer1, buffer2;
                    uint32_t ignore1, ignore2;
                    if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                    {
                        // Texture leapTexture1 = Texture();
                        // leapTexture1.init(leap_width, leap_height, 1);
                        // leapTexture1.load(buffer1, true, cam_buffer_format);
                        // Texture leapTexture2 = Texture();
                        // leapTexture2.init(leap_width, leap_height, 1);
                        // leapTexture2.load(buffer2, true, cam_buffer_format);
                        cv::Mat leap1(leap_height, leap_width, CV_8UC1, buffer1.data());
                        cv::Mat leap2(leap_height, leap_width, CV_8UC1, buffer2.data());
                        cv::threshold(leap1, leap1_thr, 250, 255, cv::THRESH_BINARY);
                        cv::threshold(leap2, leap2_thr, 250, 255, cv::THRESH_BINARY);
                        if (extract_centroid(leap1_thr, center_leap1) && extract_centroid(leap2_thr, center_leap2))
                        {
                            glm::vec2 center_NDC_leap1 = Helpers::ScreenToNDC(center_leap1, leap_width, leap_height, true);
                            glm::vec2 center_NDC_leap2 = Helpers::ScreenToNDC(center_leap2, leap_width, leap_height, true);
                            glm::vec3 cur_3d_point = triangulate(leap, center_NDC_leap1, center_NDC_leap2);
                            glm::vec2 cur_2d_point = Helpers::NDCtoScreen(center_NDC, cam_width, cam_height, true);
                            if (ready_to_collect)
                            {
                                points_3d.push_back(cur_3d_point);
                                points_2d.push_back(cur_2d_point);
                                if (points_2d.size() >= leap_calib_n_points)
                                {
                                    ready_to_collect = false;
                                    // leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::IDLE);
                                }
                            }
                            // std::cout << "leap1 2d:" << center_NDC_leap1.x << " " << center_NDC_leap1.y << std::endl;
                            // std::cout << "leap2 2d:" << center_NDC_leap2.x << " " << center_NDC_leap2.y << std::endl;
                            // std::cout << point_3d.x << " " << point_3d.y << " " << point_3d.z << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "Failed to get leap image" << std::endl;
                    }
                }
                break;
            }
            case static_cast<int>(LeapCalibrationStateMachine::CALIBRATE):
            {
                std::vector<cv::Point2f> points_2d_cv;
                std::vector<cv::Point3f> points_3d_cv;
                for (int i = 0; i < points_2d.size(); i++)
                {
                    points_2d_cv.push_back(cv::Point2f(points_2d[i].x, points_2d[i].y));
                    points_3d_cv.push_back(cv::Point3f(points_3d[i].x, points_3d[i].y, points_3d[i].z));
                }
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
                // initial guess
                cv::Mat transform = cv::Mat::zeros(4, 4, CV_64FC1);
                transform.at<double>(0, 0) = -1.0f;
                transform.at<double>(1, 2) = 1.0f;
                transform.at<double>(2, 1) = 1.0f;
                transform.at<double>(0, 3) = -50.0f;
                transform.at<double>(1, 3) = 100.0f;
                transform.at<double>(2, 3) = 300.0f;
                transform.at<double>(3, 3) = 1.0f;
                // std::cout << "transform: " << transform << std::endl;
                // cv::Mat rotmat = transform(cv::Range(0, 3), cv::Range(0, 3)).clone();
                // std::cout << "rot_mat: " << rotmat << std::endl;
                // tvec = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
                // std::cout << "tvec: " << tvec << std::endl;
                std::cout << "initial c2w guess: " << transform << std::endl;
                cv::invert(transform, transform);
                // std::cout << "transform_inverse: " << transform << std::endl;
                cv::Mat rotmat_inverse = transform(cv::Range(0, 3), cv::Range(0, 3)).clone();
                // std::cout << "rot_mat_inverse: " << transform << std::endl;
                cv::Mat tvec, rvec;
                cv::Rodrigues(rotmat_inverse, rvec);
                // std::cout << "rvec_inverse: " << rvec << std::endl;
                tvec = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
                // cv::solvePnP(points_3d_cv, points_2d_cv, camera_intrinsics, distortion_coeffs, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
                cv::Mat inliers;
                cv::solvePnPRansac(points_3d_cv, points_2d_cv, camera_intrinsics, distortion_coeffs, rvec, tvec, true,
                                   pnp_iters, pnp_rep_error, pnp_confidence, inliers, cv::SOLVEPNP_ITERATIVE);
                std::vector<cv::Point2f> points_2d_inliers_cv;
                std::vector<cv::Point3f> points_3d_inliers_cv;
                for (int inliers_index = 0; inliers_index < inliers.rows; ++inliers_index)
                {
                    int n = inliers.at<int>(inliers_index);          // i-inlier
                    points_2d_inliers_cv.push_back(points_2d_cv[n]); // add i-inlier to list
                    points_3d_inliers_cv.push_back(points_3d_cv[n]); // add i-inlier to list
                }
                std::vector<cv::Point2f> reprojected_cv, reprojected_inliers_cv;
                cv::projectPoints(points_3d_cv, rvec, tvec, camera_intrinsics, distortion_coeffs, reprojected_cv);
                cv::projectPoints(points_3d_inliers_cv, rvec, tvec, camera_intrinsics, distortion_coeffs, reprojected_inliers_cv);
                points_2d_reprojected = Helpers::opencv2glm(reprojected_cv);
                points_2d_inliers_reprojected = Helpers::opencv2glm(reprojected_inliers_cv);
                points_2d_inliners = Helpers::opencv2glm(points_2d_inliers_cv);
                float mse = Helpers::MSE(points_2d, points_2d_reprojected);
                float mse_inliers = Helpers::MSE(points_2d_inliners, points_2d_inliers_reprojected);
                std::cout << "avg reprojection error: " << mse << std::endl;
                std::cout << "avg reprojection error (inliers): " << mse_inliers << std::endl;
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
                std::cout << "w2c (row major, openCV): " << w2c << std::endl;
                cv::Mat c2w = w2c.inv();
                std::cout << "c2w (row major, openCV): " << c2w << std::endl;
                std::vector<double> w2c_vec(w2c.begin<double>(), w2c.end<double>());
                // convert row major, opencv matrix to column major open gl matrix
                glm::mat4 flipYZ = glm::mat4(1.0f);
                flipYZ[1][1] = -1.0f;
                flipYZ[2][2] = -1.0f;
                w2c_auto = glm::make_mat4(w2c_vec.data());
                glm::mat4 c2w_auto = glm::inverse(w2c_auto);
                c2w_auto = flipYZ * c2w_auto; // flip y and z columns (corresponding to camera directions)
                w2c_auto = glm::inverse(c2w_auto);
                w2c_auto = glm::transpose(w2c_auto);
                use_leap_calib_results = static_cast<int>(LeapCalibrationSettings::AUTO);
                create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
                leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::SHOW);
                showReprojections = true;
                break;
            }
            case static_cast<int>(LeapCalibrationStateMachine::SHOW):
            {
                vcolorShader.use();
                vcolorShader.setMat4("view", glm::mat4(1.0f));
                vcolorShader.setMat4("projection", glm::mat4(1.0f));
                vcolorShader.setMat4("model", glm::mat4(1.0f));
                // todo: move this logic to LeapCalibrationStateMachine::CALIBRATE
                std::vector<glm::vec2> NDCs;
                std::vector<glm::vec2> NDCs_reprojected;
                if (showInliersOnly)
                {
                    NDCs = Helpers::ScreenToNDC(points_2d_inliners, cam_width, cam_height, true);
                    NDCs_reprojected = Helpers::ScreenToNDC(points_2d_inliers_reprojected, cam_width, cam_height, true);
                }
                else
                {
                    NDCs = Helpers::ScreenToNDC(points_2d, cam_width, cam_height, true);
                    NDCs_reprojected = Helpers::ScreenToNDC(points_2d_reprojected, cam_width, cam_height, true);
                }
                PointCloud pointCloud1(NDCs, screen_verts_color_red);
                pointCloud1.render();
                PointCloud pointCloud2(NDCs_reprojected, screen_verts_color_blue);
                pointCloud2.render();
            }
            default:
            {
                break;
            }
            }
            break;
        }
        default:
        {
            break;
        }
        }

        if (debug_mode && calib_mode == static_cast<int>(CalibrationMode::OFF))
        {
            t_debug.start();
            // draws some mesh (lit by camera input)
            {
                /* quad at vcam far plane, shined by vproj (perspective corrected) */
                // projectorOnlyShader.use();
                // projectorOnlyShader.setBool("flipVer", false);
                // projectorOnlyShader.setMat4("camTransform", flycam_projection_transform * flycam_view_transform);
                // projectorOnlyShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                // projectorOnlyShader.setBool("binary", false);
                // camTexture.bind();
                // projectorOnlyShader.setInt("src", 0);
                // projFarQuad.render();

                /* dinosaur */
                // projectorShader.use();
                // projectorShader.setBool("flipVer", false);
                // projectorShader.setMat4("camTransform", flycam_projection_transform * flycam_view_transform);
                // projectorShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                // projectorShader.setBool("binary", true);
                // dinosaur.Render(projectorShader, camTexture.getTexture(), false);
                // projectorShader.setMat4("camTransform", proj_projection_transform * proj_view_transform);
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
                // projNearQuad.render();
            }
            // draws global coordinate system gizmo at origin
            {
                // vcolorShader.use();
                // vcolorShader.setMat4("projection", flycam_projection_transform);
                // vcolorShader.setMat4("view", flycam_view_transform);
                // vcolorShader.setMat4("model", glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 20.0f, 20.0f)));
                // glBindVertexArray(gizmoVAO);
                // glDrawArrays(GL_LINES, 0, 6);
            }
            // draws cube at world origin
            {
                /* regular rgb cube */
                // vcolorShader.use();
                // vcolorShader.setMat4("projection", flycam_projection_transform);
                // vcolorShader.setMat4("view", flycam_view_transform);
                // vcolorShader.setMat4("model", glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f)));
                // glEnable(GL_DEPTH_TEST);
                // glDisable(GL_CULL_FACE);
                // glBindVertexArray(cubeVAO);
                // glDrawArrays(GL_TRIANGLES, 0, 36);
                // glEnable(GL_CULL_FACE);
                /* bake projective texture */
                // if (baked)
                // {
                //     textureShader.use();
                //     textureShader.setBool("flipVer", false);
                //     textureShader.setBool("flipHor", false);
                //     textureShader.setMat4("projection", flycam_projection_transform);
                //     textureShader.setMat4("view", flycam_view_transform);
                //     textureShader.setMat4("model", glm::mat4(1.0f));
                //     textureShader.setBool("binary", false);
                //     textureShader.setInt("src", 0);
                //     bake_fbo.getTexture()->bind();
                // }
                // else
                // {
                //     projectorOnlyShader.use();
                //     projectorOnlyShader.setBool("flipVer", false);
                //     // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                //     projectorOnlyShader.setMat4("camTransform", flycam_projection_transform * flycam_view_transform);
                //     projectorOnlyShader.setMat4("projTransform", proj_projection_transform * proj_view_transform);
                //     projectorOnlyShader.setBool("binary", false);
                //     projectorOnlyShader.setInt("src", 0);
                //     dynamicTexture->bind();
                // }
                // glEnable(GL_DEPTH_TEST);
                // glBindVertexArray(tcubeVAO);
                // glDrawArrays(GL_TRIANGLES, 0, 36);
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
                    // vcolorShader.use();
                    // vcolorShader.setMat4("projection", flycam_projection_transform);
                    // vcolorShader.setMat4("view", flycam_view_transform);
                    // std::vector<glm::mat4> BoneToLocalTransforms;
                    // leftHandModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                    // glBindVertexArray(gizmoVAO);
                    // // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                    // for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                    // {
                    //     // in bind pose
                    //     vcolorShader.setMat4("model", rotx * BoneToLocalTransforms[i]);
                    //     glDrawArrays(GL_LINES, 0, 6);
                    // }
                    // for (unsigned int i = 0; i < bones_to_world_right.size(); i++)
                    // {
                    //     // in leap motion pose
                    //     vcolorShader.setMat4("model", bones_to_world_right[i]);
                    //     glDrawArrays(GL_LINES, 0, 6);
                    // }
                }
                // draw gizmo for palm orientation
                {
                    // vcolorShader.use();
                    // vcolorShader.setMat4("projection", flycam_projection_transform);
                    // vcolorShader.setMat4("view", flycam_view_transform);
                    // vcolorShader.setMat4("model", bones_to_world_right[0]);
                    // glBindVertexArray(gizmoVAO);
                    // glDrawArrays(GL_LINES, 0, 6);
                }
                // draw skinned mesh in 3D
                {
                    switch (texture_mode)
                    {
                    case static_cast<int>(TextureMode::ORIGINAL):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setBool("useProjector", false);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setInt("src", 0);
                        rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, nullptr);
                        break;
                    }
                    case static_cast<int>(TextureMode::FROM_FILE):
                    {
                        break;
                    }
                    case static_cast<int>(TextureMode::PROJECTIVE):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                        skinnedShader.setBool("useProjector", true);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setBool("flipTexVertically", false);
                        skinnedShader.setInt("src", 0);
                        rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, dynamicTexture);
                        break;
                    }
                    case static_cast<int>(TextureMode::BAKED):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setBool("useProjector", false);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setBool("flipTexVertically", false);
                        skinnedShader.setInt("src", 0);
                        rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, bake_fbo.getTexture());
                        break;
                    }
                    default:
                        break;
                    }
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
                    switch (texture_mode)
                    {
                    case static_cast<int>(TextureMode::ORIGINAL):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setBool("useProjector", false);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setInt("src", 0);
                        leftHandModel.Render(skinnedShader, bones_to_world_right, rotx);
                        break;
                    }
                    case static_cast<int>(TextureMode::FROM_FILE):
                    {
                        break;
                    }
                    case static_cast<int>(TextureMode::PROJECTIVE):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                        skinnedShader.setBool("useProjector", true);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setBool("flipTexVertically", false);
                        skinnedShader.setInt("src", 0);
                        leftHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, dynamicTexture);
                        break;
                    }
                    case static_cast<int>(TextureMode::BAKED):
                    {
                        skinnedShader.use();
                        skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
                        skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform);
                        skinnedShader.setBool("useProjector", false);
                        skinnedShader.setBool("bake", false);
                        skinnedShader.setBool("flipTexVertically", false);
                        skinnedShader.setInt("src", 0);
                        leftHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, bake_fbo.getTexture());
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            // draws frustrum of camera (=vproj)
            {
                if (showCamera)
                {
                    std::vector<glm::vec3> vprojFrustumVerticesData(28);
                    lineShader.use();
                    lineShader.setMat4("projection", flycam_projection_transform);
                    lineShader.setMat4("view", flycam_view_transform);
                    lineShader.setMat4("model", glm::mat4(1.0f));
                    lineShader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
                    glm::mat4 camUnprojectionMat = glm::inverse(cam_projection_transform * cam_view_transform);
                    for (unsigned int i = 0; i < frustumCornerVertices.size(); i++)
                    {
                        glm::vec4 unprojected = camUnprojectionMat * glm::vec4(frustumCornerVertices[i], 1.0f);
                        vprojFrustumVerticesData[i] = glm::vec3(unprojected) / unprojected.w;
                    }
                    glBindBuffer(GL_ARRAY_BUFFER, frustrumVBO);
                    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * vprojFrustumVerticesData.size(), vprojFrustumVerticesData.data(), GL_STATIC_DRAW);
                    glBindVertexArray(frustrumVAO);
                    glDrawArrays(GL_LINES, 0, 28);
                }
            }
            // draws frustrum of projector (=vcam)
            {
                if (showProjector)
                {
                    std::vector<glm::vec3> vcamFrustumVerticesData(28);
                    lineShader.use();
                    lineShader.setMat4("projection", flycam_projection_transform);
                    lineShader.setMat4("view", flycam_view_transform);
                    lineShader.setMat4("model", glm::mat4(1.0f));
                    lineShader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
                    glm::mat4 projUnprojectionMat = glm::inverse(proj_projection_transform * proj_view_transform);
                    for (int i = 0; i < frustumCornerVertices.size(); ++i)
                    {
                        glm::vec4 unprojected = projUnprojectionMat * glm::vec4(frustumCornerVertices[i], 1.0f);
                        vcamFrustumVerticesData[i] = glm::vec3(unprojected) / unprojected.w;
                    }
                    glBindBuffer(GL_ARRAY_BUFFER, frustrumVBO);
                    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * vcamFrustumVerticesData.size(), vcamFrustumVerticesData.data(), GL_STATIC_DRAW);
                    glBindVertexArray(frustrumVAO);
                    glDrawArrays(GL_LINES, 0, 28);
                }
            }
            // draw post process results to near plane of camera
            {
                if (showCamera)
                {

                    std::vector<glm::vec3> camNearVerts(4);
                    std::vector<glm::vec3> camMidVerts(4);
                    // unproject points
                    glm::mat4 camUnprojectionMat = glm::inverse(cam_projection_transform * cam_view_transform);
                    for (int i = 0; i < mid_frustrum.size(); i++)
                    {
                        glm::vec4 unprojected = camUnprojectionMat * glm::vec4(mid_frustrum[i], 1.0f);
                        camMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                        unprojected = camUnprojectionMat * glm::vec4(near_frustrum[i], 1.0f);
                        camNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
                    }
                    Quad camNearQuad(camNearVerts);
                    // Quad camMidQuad(camMidVerts);
                    // directly render camera input or any other texture
                    textureShader.use();
                    textureShader.setBool("flipVer", false);
                    textureShader.setBool("flipHor", false);
                    textureShader.setMat4("projection", flycam_projection_transform);
                    textureShader.setMat4("view", flycam_view_transform);
                    textureShader.setMat4("model", glm::mat4(1.0f));
                    textureShader.setBool("binary", false);
                    textureShader.setInt("src", 0);
                    // camTexture.bind();
                    // glBindTexture(GL_TEXTURE_2D, resTexture);
                    postprocess_fbo.getTexture()->bind();
                    // hands_fbo.getTexture()->bind();
                    camNearQuad.render();
                }
            }
            // draw warped output to near plane of projector
            {
                if (showProjector)
                {
                    t_warp.start();
                    std::vector<glm::vec3> projNearVerts(4);
                    // std::vector<glm::vec3> projMidVerts(4);
                    std::vector<glm::vec3> projFarVerts(4);
                    glm::mat4 projUnprojectionMat = glm::inverse(proj_projection_transform * proj_view_transform);
                    for (int i = 0; i < mid_frustrum.size(); i++)
                    {
                        // glm::vec4 unprojected = projUnprojectionMat * glm::vec4(mid_frustrum[i], 1.0f);
                        // projMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                        glm::vec4 unprojected = projUnprojectionMat * glm::vec4(near_frustrum[i], 1.0f);
                        projNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
                        unprojected = projUnprojectionMat * glm::vec4(far_frustrum[i], 1.0f);
                        projFarVerts[i] = glm::vec3(unprojected) / unprojected.w;
                    }
                    Quad projNearQuad(projNearVerts);
                    // Quad projMidQuad(projMidVerts);
                    Quad projFarQuad(projFarVerts);

                    textureShader.use();
                    textureShader.setBool("flipVer", false);
                    textureShader.setBool("flipHor", false);
                    textureShader.setMat4("projection", flycam_projection_transform);
                    textureShader.setMat4("view", flycam_view_transform);
                    textureShader.setMat4("model", glm::mat4(1.0f)); // debugShader.setMat4("model", mm_to_cm);
                    textureShader.setBool("binary", false);
                    textureShader.setInt("src", 0);
                    c2p_fbo.getTexture()->bind();
                    projNearQuad.render(); // canvas.renderTexture(skinnedModel.m_fbo.getTexture() /*tex*/, textureShader, projNearQuad);
                    t_warp.stop();
                }
            }
            // draws debug text
            {
                float text_spacing = 10.0f;
                glm::vec3 cur_cam_pos = gl_flycamera.getPos();
                glm::vec3 cur_cam_front = gl_flycamera.getFront();
                glm::vec3 cam_pos = gl_camera.getPos();
                std::vector<std::string> texts_to_render = {
                    std::format("debug_vector: {:.02f}, {:.02f}, {:.02f}", debug_vec.x, debug_vec.y, debug_vec.z),
                    std::format("ms_per_frame: {:.02f}, fps: {}", ms_per_frame, fps),
                    std::format("cur_camera pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cur_cam_pos.x, cur_cam_pos.y, cur_cam_pos.z, gl_flycamera.Zoom),
                    std::format("cur_camera front: {:.02f}, {:.02f}, {:.02f}", cur_cam_front.x, cur_cam_front.y, cur_cam_front.z),
                    std::format("cam pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cam_pos.x, cam_pos.y, cam_pos.z, gl_camera.Zoom),
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
        if (use_projector)
        {
            // send result to projector queue
            glReadBuffer(GL_FRONT);
            if (use_pbo)
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
        if (activateGUI)
        {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
        glfwSwapBuffers(window);
        t_swap.stop();
    }
    // cleanup
    close_signal = true;
    consumer.join();
    if (use_projector)
        projector.kill();
    camera.kill();
    glfwTerminate();
    delete[] colorBuffer;
    if (simulated_camera)
    {
        producer.join();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
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
void process_input(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
    {
        tab_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE)
    {
        if (tab_pressed)
            activateGUI = !activateGUI;
        tab_pressed = false;
    }
    if (activateGUI) // dont allow moving cameras when GUI active
        return;
    bool mod = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        mod = true;
        shift_modifier = true;
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
        shift_modifier = false;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        mod = true;
        ctrl_modifier = true;
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
// IMGUI frame creator
// ---------------------------------------------------------------------------------------------
void openIMGUIFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // ImGui::ShowDemoWindow(); // Show demo window
    // return;
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoNav;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    ImGui::Begin("augmented hands", NULL, window_flags);
    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("General"))
        {
            if (ImGui::Checkbox("Use Projector", &use_projector))
            {
                if (use_projector)
                {
                    if (!projector.init())
                    {
                        std::cerr << "Failed to initialize projector\n";
                        use_projector = false;
                    }
                }
                else
                {
                    projector.kill();
                }
            }
            ImGui::Checkbox("Debug Mode", &debug_mode);
            ImGui::SameLine();
            if (ImGui::Checkbox("Freecam Mode", &freecam_mode))
            {
                create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            ImGui::Checkbox("PBO", &use_pbo);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Calibration"))
        {
            ImGui::Text("Calibration Mode");
            if (ImGui::RadioButton("Off", &calib_mode, 0))
            {
                leap.setImageMode(false);
                exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Coaxial Calibration", &calib_mode, 1))
            {
                debug_mode = false;
                leap.setImageMode(false);
                exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Leap Calibration", &calib_mode, 2))
            {
                projector.kill();
                use_projector = false;
                leap.setImageMode(true);
                // throttle down producer speed to allow smooth display
                // see https://docs.baslerweb.com/pylonapi/cpp/pylon_advanced_topics#grab-strategies
                exposure = 10000.0f;
                camera.set_exposure_time(exposure);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                debug_mode = false;
            }
            switch (calib_mode)
            {
            case static_cast<int>(CalibrationMode::OFF):
            {
                if (ImGui::Button("Save Current Extrinsics"))
                {
                    glm::mat4 w2c = gl_camera.getViewMatrix();
                    const float *pSource = (const float *)glm::value_ptr(w2c);
                    std::vector<float> w2c_vec(pSource, pSource + 16);
                    cnpy::npy_save("../../resource/calibrations/leap_calibration/w2c_user.npy", w2c_vec.data(), {4, 4}, "w");
                    w2c_user = w2c;
                }
                break;
            }
            case static_cast<int>(CalibrationMode::COAXIAL):
            {
                if (ImGui::Button("Save Coaxial Calibration"))
                {
                    cnpy::npy_save("../../resource/calibrations/coaxial_calibration/coax_user.npy", cur_screen_verts.data(), {4, 2}, "w");
                }
                if (ImGui::BeginTable("Cam2Proj Vertices", 2))
                {
                    std::vector<glm::vec2> tmpVerts;
                    if (useCoaxialCalib)
                        tmpVerts = cur_screen_verts;
                    else
                        tmpVerts = screen_verts;
                    for (int row = 0; row < tmpVerts.size(); row++)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("Vert %d", row);
                        ImGui::TableNextColumn();
                        ImGui::Text("%f, %f", tmpVerts[row][0], tmpVerts[row][1]);
                    }
                    ImGui::EndTable();
                }
                break;
            }
            case static_cast<int>(CalibrationMode::LEAP):
            {
                ImGui::SliderInt("Calibration Points to Collect", &leap_calib_n_points, 100, 1000);
                if (ImGui::Checkbox("Ready To Collect", &ready_to_collect))
                {
                    if (ready_to_collect)
                    {
                        points_2d.clear();
                        points_3d.clear();
                        points_2d_inliners.clear();
                        points_2d_reprojected.clear();
                        points_2d_inliers_reprojected.clear();
                        leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::COLLECT);
                    }
                }
                float calib_progress = points_2d.size() / (float)leap_calib_n_points;
                char buf[32];
                sprintf(buf, "%d/%d", (int)(calib_progress * leap_calib_n_points), leap_calib_n_points);
                ImGui::ProgressBar(calib_progress, ImVec2(0.f, 0.f), buf);
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputInt("Iters", &pnp_iters);
                ImGui::SameLine();
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputFloat("Err.", &pnp_rep_error);
                ImGui::SameLine();
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputFloat("Conf.", &pnp_confidence);
                if (ImGui::Button("Calibrate"))
                {
                    if (points_2d.size() >= leap_calib_n_points)
                    {
                        leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::CALIBRATE);
                    }
                }
                if (ImGui::Checkbox("Show Reprojections", &showReprojections))
                {
                    if (points_2d.size() >= leap_calib_n_points)
                    {
                        if (showReprojections)
                            leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::SHOW);
                        else
                            leap_calibration_state = static_cast<int>(LeapCalibrationStateMachine::COLLECT);
                    }
                }
                ImGui::Checkbox("Show only inliers", &showInliersOnly);
                if (ImGui::Button("Save Calibration"))
                {
                    const float *pSource = (const float *)glm::value_ptr(w2c_auto);
                    std::vector<float> w2c_vec(pSource, pSource + 16);
                    cnpy::npy_save("../../resource/calibrations/leap_calibration/w2c.npy", w2c_vec.data(), {4, 4}, "w");
                    std::vector<float> flatten_image_points = Helpers::flatten_glm(points_2d);
                    std::vector<float> flatten_object_points = Helpers::flatten_glm(points_3d);
                    cnpy::npy_save("../../resource/calibrations/leap_calibration/2dpoints.npy", flatten_image_points.data(), {points_2d.size(), 2}, "w");
                    cnpy::npy_save("../../resource/calibrations/leap_calibration/3dpoints.npy", flatten_object_points.data(), {points_3d.size(), 3}, "w");
                }
                break;
            }
            default:
                break;
            }
            ImGui::Text("Calibration Source");
            if (ImGui::RadioButton("Calibration", &use_leap_calib_results, 0))
            {
                GLCamera dummy_camera;
                create_virtual_cameras(dummy_camera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Manual", &use_leap_calib_results, 1))
            {
                GLCamera dummy_camera;
                create_virtual_cameras(dummy_camera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Use Coaxial Calib", &useCoaxialCalib))
            {
                if (useCoaxialCalib)
                    c2p_homography = PostProcess::findHomography(cur_screen_verts);
            }
            ImGui::Text("Cam2World (row major, OpenGL convention)");
            if (ImGui::BeginTable("Cam2World", 4))
            {
                glm::mat4 c2w = glm::transpose(glm::inverse(gl_camera.getViewMatrix()));
                for (int row = 0; row < 4; row++)
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", c2w[row][0]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", c2w[row][1]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", c2w[row][2]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", c2w[row][3]);
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Camera Controls"))
        {
            ImGui::Checkbox("Show Camera", &showCamera);
            ImGui::SameLine();
            ImGui::Checkbox("Show Projector", &showProjector);
            if (ImGui::SliderFloat("Camera Exposure [us]", &exposure, 30.0f, 10000.0f))
            {
                // std::cout << "current exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
                camera.set_exposure_time(exposure);
                // std::cout << "new exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
            }
            if (ImGui::Button("Cam Screen Shot"))
            {
                hands_fbo.saveColorToFile("../../debug/screenshot_hands_fbo.png");
                postprocess_fbo.saveColorToFile("../../debug/screenshot_pp_fbo.png");
                c2p_fbo.saveColorToFile("../../debug/screenshot_c2p_fbo.png");
            }
            ImGui::SameLine();
            if (ImGui::Button("Cam2view"))
            {
                gl_camera.setViewMatrix(gl_flycamera.getViewMatrix());
                gl_projector.setViewMatrix(gl_flycamera.getViewMatrix());
            }
            ImGui::SameLine();
            if (ImGui::Button("View2cam"))
            {
                gl_flycamera.setViewMatrix(gl_camera.getViewMatrix());
            }
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Material"))
        {
            ImGui::Text("Material Type");
            ImGui::RadioButton("Diffuse", &material_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("GGX", &material_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Transparent", &material_mode, 2);

            ImGui::Text("Diffuse Texture Type");
            ImGui::RadioButton("Original", &texture_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("From File", &texture_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Projective", &texture_mode, 2);
            ImGui::SameLine();
            ImGui::RadioButton("Baked", &texture_mode, 3);
            ImGui::InputText("Diffuse Texture File", &diffuseTextureFile, 20);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Diffusion"))
        {
            ImGui::InputInt("Diffuse Seed", &diffuse_seed);
            if (ImGui::Button("Bake StableDiffusion Texture"))
            {
                bakeRequest = true;
            }
            ImGui::SameLine();
            ImGui::Checkbox("Save Intermediate Outputs", &saveIntermed);
            ImGui::InputText("Baked Texture File", &bakeFile);
            ImGui::InputText("Prompt", &sd_prompt);
            ImGui::Text("Stable Diffusion Mode");
            ImGui::SameLine();
            ImGui::RadioButton("Use prompt", &sd_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Random Animal", &sd_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Gesture", &sd_mode, 2);
            ImGui::Text("Mask Mode");
            ImGui::SameLine();
            ImGui::RadioButton("Fill", &sd_mask_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Original", &sd_mask_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Latent Noise", &sd_mask_mode, 2);
            ImGui::SameLine();
            ImGui::RadioButton("Latent Nothing", &sd_mask_mode, 3);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Post Process"))
        {
            ImGui::SliderFloat("Masking Threshold", &masking_threshold, 0.0f, 0.1f);
            ImGui::Text("Post Processing Mode");
            ImGui::SameLine();
            ImGui::RadioButton("None", &postprocess_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Camera Feed", &postprocess_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Mask", &postprocess_mode, 2);
            ImGui::SameLine();
            ImGui::RadioButton("Jump Flood", &postprocess_mode, 3);
            ImGui::SameLine();
            ImGui::RadioButton("OF", &postprocess_mode, 4);

            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Leap Control"))
        {
            if (ImGui::Checkbox("Leap Polling Mode", &leap_poll_mode))
            {
                leap.setPollMode(leap_poll_mode);
            }
            ImGui::SliderInt("Leap Prediction [us]", &magic_leap_time_delay, 1, 100000);
            ImGui::SliderFloat("Leap Bone Scale", &magic_leap_scale_factor, 1.0f, 20.0f);
            ImGui::SliderFloat("Leap Wrist Offset", &magic_wrist_offset, -100.0f, 100.0f);
            ImGui::SliderFloat("Leap Arm Offset", &magic_arm_forward_offset, -300.0f, 200.0f);
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (activateGUI)
        return;
    if (calib_mode == static_cast<int>(CalibrationMode::COAXIAL))
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
    if (activateGUI)
        return;
    switch (calib_mode)
    {
    case static_cast<int>(CalibrationMode::OFF):
    {
        if (shift_modifier)
        {
            gl_projector.processMouseMovement(xoffset, yoffset);
        }
        else
        {
            if (ctrl_modifier)
            {
                gl_camera.processMouseMovement(xoffset, yoffset);
            }
            else
            {
                gl_flycamera.processMouseMovement(xoffset, yoffset);
            }
        }
        break;
    }
    case static_cast<int>(CalibrationMode::COAXIAL):
    {

        // glm::vec2 mouse_pos = glm::vec2((2.0f * xpos / proj_width) - 1.0f, -1.0f * ((2.0f * ypos / proj_height) - 1.0f));
        glm::vec2 mouse_pos = Helpers::ScreenToNDC(glm::vec2(xpos, ypos), proj_width, proj_height, true);
        float cur_min_dist = 100.0f;
        for (int i = 0; i < cur_screen_verts.size(); i++)
        {
            glm::vec2 v = glm::vec2(cur_screen_verts[i]);

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
            cur_screen_verts[dragging_vert].x = mouse_pos.x;
            cur_screen_verts[dragging_vert].y = mouse_pos.y;
        }
        break;
    }
    case static_cast<int>(CalibrationMode::LEAP):
    {
        break;
    }
    default:
        break;
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (activateGUI)
        return;
    if (shift_modifier)
    {
        gl_projector.processMouseScroll(static_cast<float>(yoffset));
    }
    else
    {
        if (ctrl_modifier)
            gl_camera.processMouseScroll(static_cast<float>(yoffset));
        else
            gl_flycamera.processMouseScroll(static_cast<float>(yoffset));
    }
}

bool loadCoaxialCalibrationResults(std::vector<glm::vec2> &cur_screen_verts)
{
    const fs::path coax_path("../../resource/calibrations/coaxial_calibration/coax_user.npy");
    cnpy::NpyArray coax_npy;
    if (fs::exists(coax_path))
    {
        cur_screen_verts.clear();
        coax_npy = cnpy::npy_load(coax_path.string());
        std::vector<float> extract = coax_npy.as_vec<float>();
        for (int i = 0; i < extract.size(); i += 2)
        {
            cur_screen_verts.push_back(glm::vec2(extract[i], extract[i + 1]));
        }
        return true;
    }
    return false;
}

bool loadLeapCalibrationResults(glm::mat4 &proj_project,
                                glm::mat4 &cam_project,
                                std::vector<double> &camera_distortion,
                                glm::mat4 &w2c_auto,
                                glm::mat4 &w2c_user,
                                std::vector<glm::vec2> &points_2d,
                                std::vector<glm::vec3> &points_3d)
{
    // vp = virtual projector
    // vc = virtual camera
    glm::mat4 flipYZ = glm::mat4(1.0f);
    flipYZ[1][1] = -1.0f;
    flipYZ[2][2] = -1.0f;
    cnpy::NpyArray points2d_npy, points3d_npy;
    cnpy::NpyArray w2c_user_npy, w2c_auto_npy;
    cnpy::npz_t projcam_npz;
    cnpy::npz_t cam_npz;
    // bool user_defined = false; // if a user saved extrinsics, they are already in openGL format
    try
    {
        const fs::path user_path{"../../resource/calibrations/leap_calibration/w2c_user.npy"};
        const fs::path auto_path{"../../resource/calibrations/leap_calibration/w2c.npy"};
        const fs::path points2d_path{"../../resource/calibrations/leap_calibration/2dpoints.npy"};
        const fs::path points3d_path{"../../resource/calibrations/leap_calibration/3dpoints.npy"};
        const fs::path cam_calib_path{"../../resource/calibrations/cam_calibration/cam_calibration.npz"};
        const fs::path projcam_calib_path{"../../resource/calibrations/camproj_calibration/calibration.npz"};
        if (!fs::exists(user_path))
            return false;
        if (!fs::exists(auto_path))
            return false;
        if (!fs::exists(points2d_path))
            return false;
        if (!fs::exists(points3d_path))
            return false;
        if (!fs::exists(cam_calib_path))
            return false;
        if (!fs::exists(projcam_calib_path))
            return false;
        w2c_user_npy = cnpy::npy_load(user_path.string());
        w2c_auto_npy = cnpy::npy_load(auto_path.string());
        points2d_npy = cnpy::npy_load(points2d_path.string());
        points3d_npy = cnpy::npy_load(points3d_path.string());
        cam_npz = cnpy::npz_load(cam_calib_path.string());
        projcam_npz = cnpy::npz_load(projcam_calib_path.string());
    }
    catch (std::runtime_error &e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    w2c_auto = glm::make_mat4(w2c_auto_npy.data<float>());
    // glm::mat4 c2w_auto = glm::inverse(w2c_auto);
    // c2w_auto = flipYZ * c2w_auto; // flip y and z columns (corresponding to camera directions)
    // w2c_auto = glm::inverse(c2w_auto);
    // w2c_auto = glm::transpose(w2c_auto);
    w2c_user = glm::make_mat4(w2c_user_npy.data<float>());
    // w2c = glm::inverse(w2c);
    std::vector<float> points_2d_unpacked = points2d_npy.as_vec<float>();
    std::vector<float> points_3d_unpacked = points3d_npy.as_vec<float>();
    for (int i = 0; i < points_2d_unpacked.size(); i += 2)
    {
        points_2d.push_back(glm::vec2(points_2d_unpacked[i], points_2d_unpacked[i + 1]));
    }
    for (int i = 0; i < points_3d_unpacked.size(); i += 3)
    {
        points_3d.push_back(glm::vec3(points_3d_unpacked[i], points_3d_unpacked[i + 1], points_3d_unpacked[i + 2]));
    }
    float ffar = 1500.0f;
    float nnear = 1.0f;
    glm::mat3 camera_intrinsics = glm::make_mat3(cam_npz["cam_intrinsics"].data<double>());
    camera_distortion = cam_npz["cam_distortion"].as_vec<double>();
    float cfx = camera_intrinsics[0][0];
    float cfy = camera_intrinsics[1][1];
    float ccx = camera_intrinsics[0][2];
    float ccy = camera_intrinsics[1][2];
    cam_project = glm::mat4(0.0);
    cam_project[0][0] = 2 * cfx / cam_width;
    cam_project[0][2] = (cam_width - 2 * ccx) / cam_width;
    cam_project[1][1] = 2 * cfy / cam_height;
    cam_project[1][2] = -(cam_height - 2 * ccy) / cam_height;
    cam_project[2][2] = -(ffar + nnear) / (ffar - nnear);
    cam_project[2][3] = -2 * ffar * nnear / (ffar - nnear);
    cam_project[3][2] = -1.0f;
    cam_project = glm::transpose(cam_project);
    // glm::mat3 projector_intrinsics = glm::make_mat3(projcam_npz["proj_intrinsics"].data<double>());
    // float pfx = projector_intrinsics[0][0];
    // float pfy = projector_intrinsics[1][1];
    // float pcx = projector_intrinsics[0][2];
    // float pcy = projector_intrinsics[1][2];
    // proj_project = glm::mat4(0.0);
    // proj_project[0][0] = 2 * pfx / proj_width;
    // proj_project[0][2] = (proj_width - 2 * pcx) / proj_width;
    // proj_project[1][1] = 2 * pfy / proj_height;
    // proj_project[1][2] = -(proj_height - 2 * pcy) / proj_height;
    // proj_project[2][2] = -(ffar + nnear) / (ffar - nnear);
    // proj_project[2][3] = -2 * ffar * nnear / (ffar - nnear);
    // proj_project[3][2] = -1.0f;
    // proj_project = glm::transpose(proj_project);
    proj_project = cam_project;
    // glm::mat4 vp2vc = glm::make_mat4(my_npz["proj_transform"].data<double>()); // this is the camera to projector transform
    // glm::mat4 vp2w = vp2vc * vc2w;                                             // since glm uses column major, we multiply from the left...
    // vp2w = flipYZ * vp2w;
    // vp2w[0][3] *= 0.1f;
    // vp2w[1][3] *= 0.1f;
    // vp2w[2][3] *= 0.1f;
    // w2vp = glm::inverse(vp2w);
    // w2vp = glm::transpose(w2vp);
    // vc2w = flipYZ * vc2w;
    // w2vc[0][3] *= 0.1f;
    // w2vc[1][3] *= 0.1f;
    // w2vc[2][3] *= 0.1f;
    // w2vc = glm::inverse(vc2w);
    // w2vc = glm::transpose(w2vc);
    return true;
}

LEAP_STATUS getLeapFrame(LeapConnect &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &skeleton_vertices,
                         bool leap_poll_mode,
                         int64_t &lastFrameID)
{
    // todo: most likely leaks memory / stack overflow here
    //  some defs
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 flip_x = glm::mat4(1.0f);
    flip_x[0][0] = -1.0f;
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    // magic numbers
    glm::mat4 magic_leap_basis_fix = roty * flip_z * flip_y;
    glm::mat4 chirality = glm::mat4(1.0f);
    // init
    glm::mat4 scalar = glm::scale(glm::mat4(1.0f), glm::vec3(magic_leap_scale_factor));
    uint64_t targetFrameSize = 0;
    LEAP_TRACKING_EVENT *frame = nullptr;
    if (leap_poll_mode)
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
            bones_to_world_right = std::move(bones_to_world);
        else
            bones_to_world_left = std::move(bones_to_world);
    }
    // Free the allocated buffer when done.
    if (leap_poll_mode)
        free(frame->pHands);
    free(frame);
    return LEAP_STATUS::LEAP_NEWFRAME;
}

void create_virtual_cameras(GLCamera &gl_flycamera, GLCamera &gl_projector, GLCamera &gl_camera)
{
    Camera_Mode camera_mode = freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    glm::mat4 w2c;
    if (use_leap_calib_results == static_cast<int>(LeapCalibrationSettings::AUTO))
        w2c = w2c_auto;
    else
        w2c = w2c_user;
    glm::mat4 w2p = w2c;
    // std::cout << "Using calibration data for camera and projector settings" << std::endl;
    if (freecam_mode)
    {
        // gl_flycamera = GLCamera(w2vc, proj_project, Camera_Mode::FREE_CAMERA);
        // gl_flycamera = GLCamera(w2vc, proj_project, Camera_Mode::FREE_CAMERA, proj_width, proj_height, 10.0f);
        gl_flycamera = GLCamera(glm::vec3(20.0f, -160.0f, 190.0f),
                                glm::vec3(-50.0f, 200.0f, -30.0f),
                                glm::vec3(0.0f, 0.0f, -1.0f),
                                camera_mode,
                                proj_width,
                                proj_height,
                                1500.0f,
                                25.0f,
                                true);
        gl_projector = GLCamera(w2p, proj_project, camera_mode, proj_width, proj_height, 25.0f, true);
        gl_camera = GLCamera(w2c, cam_project, camera_mode, cam_width, cam_height, 25.0f, true);
    }
    else
    {
        gl_projector = GLCamera(w2p, proj_project, camera_mode, proj_width, proj_height);
        gl_camera = GLCamera(w2c, cam_project, camera_mode, cam_width, cam_height);
        gl_flycamera = GLCamera(w2c, proj_project, camera_mode, proj_width, proj_height);
    }
}

bool extract_centroid(cv::Mat binary_image, glm::vec2 &centeroid)
{
    // find moments of the image
    cv::Moments m = cv::moments(binary_image, true);
    if ((m.m00 == 0) || (m.m01 == 0) || (m.m10 == 0))
        return false;
    glm::vec2 cur_center(m.m10 / m.m00, m.m01 / m.m00);
    centeroid = cur_center;
    return true;
}

glm::vec3 triangulate(LeapConnect &leap, const glm::vec2 &leap1, const glm::vec2 &leap2)
{
    // leap image plane is x right, and y up like opengl...
    glm::vec2 l1_vert = Helpers::NDCtoScreen(leap1, leap_width, leap_height, false);
    LEAP_VECTOR l1_vert_leap = {l1_vert.x, l1_vert.y, 1.0f};
    glm::vec2 l2_vert = Helpers::NDCtoScreen(leap2, leap_width, leap_height, false);
    LEAP_VECTOR l2_vert_leap = {l2_vert.x, l2_vert.y, 1.0f};
    LEAP_VECTOR leap1_rays_2d = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, l1_vert_leap);
    LEAP_VECTOR leap2_rays_2d = LeapPixelToRectilinear(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, l2_vert_leap);
    // glm::mat4 leap1_extrinsic = glm::mat4(1.0f);
    // glm::mat4 leap2_extrinsic = glm::mat4(1.0f);
    // LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, glm::value_ptr(leap1_extrinsic));
    // LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, glm::value_ptr(leap2_extrinsic));
    float baseline = 40.0f;
    // see https://forums.leapmotion.com/t/sdk-2-1-raw-data-get-pixel-position-xyz/1604/12
    float z = baseline / (leap2_rays_2d.x - leap1_rays_2d.x);
    float alty1 = z * leap2_rays_2d.y;
    float alty2 = z * leap1_rays_2d.y;
    float x = z * leap2_rays_2d.x - baseline / 2.0f;
    // float altx = z * leap1_rays_2d.x + baseline / 2.0f;
    float y = (alty1 + alty2) / 2.0f;
    glm::vec3 point_3d = glm::vec3(x, -z, y);
    return point_3d;
}