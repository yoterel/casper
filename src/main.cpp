#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>
#include "readerwritercircularbuffer.h"
#include "camera.h"
#include "gl_camera.h"
#include "display.h"
#include "shader.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "point_cloud.h"
#include "leapCPP.h"
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
#include "kalman.h"
#include "imgui_stdlib.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "diffuse.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#define _DEBUG
#else
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "grid.h"
#include "moving_least_squares.h"
namespace fs = std::filesystem;

/* forward declarations */
void openIMGUIFrame();
void handleCameraInput(CGrabResultPtr ptrGrabResult);
LEAP_STATUS handleLeapInput(uint64_t frameCounter);
void handlePostProcess(Texture *handTexture, Shader &textureShader, Shader &maskShader, Shader &jfaInitShader, Shader &jfaShader, Shader &NNShader, Shader &uv_NNShader, Shader &vcolorShader);
void handleSkinning(SkinningShader &skinnedShader,
                    Shader &vcolorShader,
                    SkinnedModel &handModel,
                    glm::mat4 cam_view_transform,
                    glm::mat4 cam_projection_transform,
                    bool isRightHand);
void handleBaking(SkinningShader &skinnedShader,
                  Shader &textureShader,
                  SkinnedModel &leftHandModel,
                  SkinnedModel &rightHandModel,
                  glm::mat4 cam_view_transform,
                  glm::mat4 cam_projection_transform,
                  bool isRightHand);
void handleMLS(Shader &gridShader);
void handleDebugMode(SkinningShader &skinnedShader,
                     Shader &textureShader,
                     Shader &vcolorShader,
                     Shader &textShader,
                     Shader &lineShader,
                     SkinnedModel &rightHandModel,
                     SkinnedModel &leftHandModel,
                     Text &text);
std::vector<glm::vec2> mp_predict(cv::Mat image, int timestamp);
void create_virtual_cameras(GLCamera &gl_flycamera, GLCamera &gl_projector, GLCamera &gl_camera);
glm::vec3 triangulate(LeapCPP &leap, const glm::vec2 &leap1, const glm::vec2 &leap2);
bool extract_centroid(cv::Mat binary_image, glm::vec2 &centeroid);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);
LEAP_STATUS getLeapFrame(LeapCPP &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &joints_left,
                         std::vector<glm::vec3> &joints_right,
                         bool leap_poll_mode, int64_t &lastFrameID);
LEAP_STATUS getLeapFramePreRecorded(std::vector<glm::mat4> &bones_to_world_left,
                                    std::vector<glm::mat4> &bones_to_world_right,
                                    std::vector<glm::vec3> &joints_left,
                                    std::vector<glm::vec3> &joints_right);
bool loadPrerecordedSession();
bool loadVideo();
void initGLBuffers(unsigned int *pbo);
bool loadLeapCalibrationResults(glm::mat4 &cam_project, glm::mat4 &proj_project,
                                glm::mat4 &w2c_auto,
                                glm::mat4 &w2c_user,
                                std::vector<glm::vec2> &points_2d,
                                std::vector<glm::vec3> &points_3d,
                                cv::Mat &undistort_map1, cv::Mat &undistort_map2,
                                cv::Mat &camera_intrinsics_cv,
                                cv::Mat &camera_distortion_cv);
bool loadCoaxialCalibrationResults(std::vector<glm::vec2> &cur_screen_verts);
void set_texture_shader(Shader &textureShader,
                        bool flipVer,
                        bool flipHor,
                        bool isGray,
                        bool binary = false,
                        float threshold = 0.035f,
                        int src = 0,
                        glm::mat4 model = glm::mat4(1.0f),
                        glm::mat4 projection = glm::mat4(1.0f),
                        glm::mat4 view = glm::mat4(1.0f));
/* engine state */
// window controls
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_width = 720;
const unsigned int cam_height = 540;
// global controls
bool debug_mode = false;
bool cam_space = false;
bool close_signal = false;
bool cmd_line_stats = false;
bool use_cuda = false;
bool simulated_camera = false;
bool freecam_mode = false;
bool use_pbo = true;
bool use_projector = false;
bool use_screen = true;
bool cam_color_mode = false;
bool icp_apply_transform = true;
bool showCamera = true;
bool showProjector = true;
bool undistortCamera = false;
int postprocess_mode = static_cast<int>(PostProcessMode::OVERLAY);
int texture_mode = static_cast<int>(TextureMode::ORIGINAL);
int material_mode = static_cast<int>(MaterialMode::DIFFUSE);
float deltaTime = 0.0f;
float masking_threshold = 0.035f;
bool threshold_flag = false;
glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 triangulated = glm::vec3(0.0f, 0.0f, 0.0f);
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
int64_t totalFrameCount = 0;
int64_t videoFrameCount = 0;
int64_t lastFrameID = 0;
bool space_modifier = false;
bool shift_modifier = false;
bool ctrl_modifier = false;
bool activateGUI = false;
bool tab_pressed = false;
bool rmb_pressed = false;
unsigned int n_bones = 0;
const unsigned int num_texels = proj_width * proj_height;
const unsigned int projected_image_size = num_texels * 3 * sizeof(uint8_t);
cv::Mat white_image(cam_height, cam_width, CV_8UC1, cv::Scalar(255));
Timer t_camera, t_leap, t_skin, t_swap, t_download, t_warp, t_app, t_misc, t_debug, t_pp, t_mls, t_mls_thread, t_bake;
std::thread producer, consumer, run_sd, run_mls;
// bake/sd controls
bool bakeRequest = false;
bool sd_succeed = false;
bool sd_running = false;
int sd_mode = static_cast<int>(SDMode::PROMPT);
int sd_mask_mode = 2;
bool saveIntermed = false;
std::vector<uint8_t> img2img_data;
int sd_outwidth, sd_outheight;
int diffuse_seed = -1;
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
// leap controls
bool leap_poll_mode = false;
bool use_pre_recorded_session = false;
float required_latency_ms = 1.0f;
std::string session_name = "s42k.0";
std::string loaded_session_name = "";
bool pre_recorded_session_loaded = false;
bool playback_video = false;
std::string playback_video_name = "s42k.0";
std::string loaded_playback_video_name = "";
bool playback_video_loaded = false;
int total_session_time_stamps = 0;
bool leap_record_session = false;
std::vector<glm::mat4> session_bones_left;
std::vector<glm::mat4> session_bones_right;
std::vector<glm::vec3> session_joints;
// int session_counter = 0;
std::string tmp_name = std::tmpnam(nullptr);
fs::path tmp_filename(tmp_name);
int leap_tracking_mode = eLeapTrackingMode_HMD;
uint64_t leap_cur_frame_id = 0;
uint32_t leap_width = 640;
uint32_t leap_height = 240;
bool useFingerWidth = false;
int magic_leap_time_delay = 10000; // us
float leap_global_scaler = 1.0f;
float magic_leap_scale_factor = 10.0f;
float leap_arm_local_scaler = 0.019f;
float leap_palm_local_scaler = 0.011f;
float leap_bone_local_scaler = 0.05f;
float magic_wrist_offset = -65.0f;
float magic_arm_forward_offset = -170.0f;
float leap_binary_threshold = 0.3f;
bool leap_threshold_flag = false;
int64_t targetFrameTime = 0;
double whole = 0.0;
LEAP_CLOCK_REBASER clockSynchronizer;
std::vector<glm::vec3> joints_left;
std::vector<glm::vec3> joints_right;
std::vector<glm::mat4> bones_to_world_left;
std::vector<glm::mat4> bones_to_world_right;
std::vector<glm::mat4> bones_to_world_right_bake;
std::vector<glm::mat4> bones_to_world_left_bake;
std::vector<glm::mat4> bones_to_world_left_interp;
glm::mat4 global_scale_right = glm::mat4(1.0f);
glm::mat4 global_scale_left = glm::mat4(1.0f);
LeapCPP leap(leap_poll_mode, false, static_cast<_eLeapTrackingMode>(leap_tracking_mode));
// calibration controls
bool ready_to_collect = false;
bool use_coaxial_calib = false;
int calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
int checkerboard_width = 10;
int checkerboard_height = 7;
int leap_collection_setting = static_cast<int>(LeapCollectionSettings::AUTO_RAW);
int leap_mark_setting = static_cast<int>(LeapMarkSettings::STREAM);
bool leap_calib_use_ransac = false;
int mark_bone_index = 17;
int leap_calibration_mark_state = 0;
int use_leap_calib_results = static_cast<int>(LeapCalibrationSettings::MANUAL);
int operation_mode = static_cast<int>(OperationMode::NORMAL);
int n_points_leap_calib = 2000;
int n_points_cam_calib = 30;
std::vector<double> calib_cam_matrix;
std::vector<double> calib_cam_distortion;
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
bool dragging = false;
int dragging_vert = 0;
int closest_vert = 0;
float min_dist = 100000.0f;
std::vector<glm::vec3> points_3d;
std::vector<glm::vec2> points_2d, points_2d_inliners;
std::vector<glm::vec2> points_2d_reprojected, points_2d_inliers_reprojected;
std::vector<std::vector<cv::Point2f>> imgpoints;
int pnp_iters = 500;
float pnp_rep_error = 2.0f;
float pnp_confidence = 0.95f;
bool showInliersOnly = true;
bool showReprojections = false;
bool showTestPoints = false;
bool calibrationSuccess = false;
cv::Mat camera_intrinsics_cv, camera_distortion_cv;
std::vector<glm::vec3> screen_verts_color_red = {{1.0f, 0.0f, 0.0f}};
std::vector<glm::vec3> screen_verts_color_green = {{0.0f, 1.0f, 0.0f}};
std::vector<glm::vec3> screen_verts_color_blue = {{0.0f, 0.0f, 1.0f}};
std::vector<glm::vec2> screen_verts = {{-1.0f, 1.0f},
                                       {-1.0f, -1.0f},
                                       {1.0f, -1.0f},
                                       {1.0f, 1.0f}};
std::vector<glm::vec2> cur_screen_verts = {{-1.0f, 1.0f},
                                           {-1.0f, -1.0f},
                                           {1.0f, -1.0f},
                                           {1.0f, 1.0f}};
glm::vec2 cur_screen_vert = {0.0f, 0.0f};
glm::vec2 marked_2d_pos1, marked_2d_pos2;
std::vector<glm::vec3> triangulated_marked;
std::vector<glm::vec2> marked_reprojected;
cv::Mat tvec_calib, rvec_calib;
// camera controls
unsigned int n_cam_channels = cam_color_mode ? 4 : 1;
unsigned int cam_buffer_format = cam_color_mode ? GL_RGBA : GL_RED;
float exposure = 1850.0f; // 1850.0f;
GLCamera gl_flycamera;
GLCamera gl_projector;
GLCamera gl_camera;
cv::Mat camImage, camImageOrig, undistort_map1, undistort_map2;
glm::mat4 w2c_auto, w2c_user;
glm::mat4 proj_project;
glm::mat4 cam_project;
std::vector<double> camera_distortion;
glm::mat4 c2p_homography;
int dst_width = cam_space ? cam_width : proj_width;
int dst_height = cam_space ? cam_height : proj_height;
DynaFlashProjector projector(true, false);
BaslerCamera camera;
moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> camera_queue(20);
moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> projector_queue(20);
// GL controls
std::string testFile("../../resource/uv.png");
std::string bakeFileLeft("../../resource/baked_left.png");
std::string bakeFileRight("../../resource/baked_right.png");
std::string diffuseTextureFile("../../resource/uv.png");
std::string sd_prompt("A natural skinned human hand with a colorful dragon tattoo, photorealistic skin");
Texture *dynamicTexture = nullptr;
Texture *bakedTextureLeft = nullptr;
Texture *bakedTextureRight = nullptr;
FBO icp_fbo(cam_width, cam_height, 4, false);
FBO icp2_fbo(dst_width, dst_height, 4, false);
FBO hands_fbo(dst_width, dst_height, 4, false);
FBO uv_fbo(dst_width, dst_height, 4, false);
FBO mls_fbo(dst_width, dst_height, 4, false);
FBO bake_fbo_right(1024, 1024, 4, false);
FBO bake_fbo_left(1024, 1024, 4, false);
FBO sd_fbo(1024, 1024, 4, false);
FBO postprocess_fbo(dst_width, dst_height, 4, false);
FBO c2p_fbo(dst_width, dst_height, 4, false);
Quad fullScreenQuad(0.0f, false);
PostProcess postProcess(cam_width, cam_height, dst_width, dst_height);
Texture camTexture;
glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
DirectionalLight dirLight(glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, 1.0f, glm::vec3(0.0f, -1.0f, -1.0f));
unsigned int skeletonVAO = 0;
unsigned int skeletonVBO = 0;
unsigned int gizmoVAO = 0;
unsigned int gizmoVBO = 0;
unsigned int frustrumVAO = 0;
unsigned int frustrumVBO = 0;
std::vector<glm::vec3> frustumCornerVertices;
std::vector<glm::vec3> near_frustrum;
std::vector<glm::vec3> mid_frustrum;
std::vector<glm::vec3> far_frustrum;
// python controls
PyObject *myModule;
PyObject *predict_single;
PyObject *predict_video;
PyObject *init_detector;
PyObject *single_detector;
// mls controls
const int grid_x_point_count = 41;
const int grid_y_point_count = 41;
const float grid_x_spacing = 0.05;
const float grid_y_spacing = 0.05;
float mls_alpha = 0.5f;
Grid deformationGrid(grid_x_point_count, grid_y_point_count, grid_x_spacing, grid_y_spacing);
std::vector<cv::Point2f> ControlPointsP;
std::vector<cv::Point2f> ControlPointsQ;
std::vector<glm::vec2> ControlPointsP_glm;
std::vector<glm::vec2> ControlPointsQ_glm;
std::vector<int> leap_selection_vector{1, 5, 11, 19, 27, 35, 9, 17, 25, 33, 41, 7, 15, 23, 31, 39};
std::vector<int> mp_selection_vector{0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20, 3, 7, 11, 15, 19};
std::vector<std::vector<glm::vec2>> prev_pred_glm;
bool mls_succeed = false;
bool mls_running = false;
bool use_mls = true;
bool show_mls_landmarks = false;
int mls_smooth_window = 0;
bool use_mp_kalman = false;
bool mls_forecast = true;
float mp_kalman_process_noise = 0.01f;
float mp_kalman_measurement_noise = 0.01f;
float mp_kalman_error = 1.0f;
float mls_depth_threshold = 215.0f;
bool mls_depth_test = false;
std::vector<Kalman2D> kalman_filters = std::vector<Kalman2D>(21, Kalman2D(mp_kalman_process_noise, mp_kalman_measurement_noise, mp_kalman_error));
int kalman_lookahead = 0;

/* main */
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
    t_app.start();
    /* py init */
    Py_Initialize();
    import_array();
    myModule = PyImport_ImportModule("predict");
    if (!myModule)
    {
        std::cout << "Import module failed!";
        PyErr_Print();
        // exit(1);
    }
    Py_INCREF(myModule);
    predict_single = PyObject_GetAttrString(myModule, (char *)"predict_single");
    if (!predict_single)
    {
        std::cout << "Import function failed!";
        PyErr_Print();
        // exit(1);
    }
    Py_INCREF(predict_single);
    predict_video = PyObject_GetAttrString(myModule, (char *)"predict_video");
    if (!predict_video)
    {
        std::cout << "Import function failed!";
        PyErr_Print();
        // exit(1);
    }
    Py_INCREF(predict_video);
    init_detector = PyObject_GetAttrString(myModule, (char *)"init_detector");
    if (!init_detector)
    {
        std::cout << "Import function failed!";
        PyErr_Print();
        // exit(1);
    }
    Py_INCREF(init_detector);
    single_detector = PyObject_CallFunction(init_detector, NULL);
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
    deformationGrid.initGLBuffers();
    glfwSwapInterval(0);                       // do not sync to monitor
    glViewport(0, 0, proj_width, proj_height); // set viewport
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glPointSize(10.0f);
    glLineWidth(5.0f);
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
    Helpers::setupSkeletonBuffers(skeletonVAO, skeletonVBO);
    Helpers::setupGizmoBuffers(gizmoVAO, gizmoVBO);
    unsigned int cubeVAO = 0;
    unsigned int cubeVBO = 0;
    Helpers::setupCubeBuffers(cubeVAO, cubeVBO);
    unsigned int tcubeVAO = 0;
    unsigned int tcubeVBO1 = 0;
    unsigned int tcubeVBO2 = 0;
    Helpers::setupCubeTexturedBuffers(tcubeVAO, tcubeVBO1, tcubeVBO2);
    Helpers::setupFrustrumBuffers(frustrumVAO, frustrumVBO);
    unsigned int pbo[2] = {0};
    initGLBuffers(pbo);
    hands_fbo.init();
    uv_fbo.init(GL_RGBA, GL_RGBA32F); // stores uv coordinates, so must be 32F
    bake_fbo_right.init();
    bake_fbo_left.init();
    sd_fbo.init();
    postprocess_fbo.init();
    icp_fbo.init();
    icp2_fbo.init();
    mls_fbo.init(GL_RGBA, GL_RGBA32F); // will possibly store uv_fbo, so must be 32F
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
    dynamicTexture->init_from_file();
    const fs::path bakeFileLeftPath{bakeFileLeft};
    const fs::path bakeFileRightPath{bakeFileRight};
    if (fs::exists(bakeFileLeftPath))
        bakedTextureLeft = new Texture(bakeFileLeft.c_str(), GL_TEXTURE_2D);
    else
        bakedTextureLeft = new Texture(testFile.c_str(), GL_TEXTURE_2D);
    if (fs::exists(bakeFileRightPath))
        bakedTextureRight = new Texture(bakeFileRight.c_str(), GL_TEXTURE_2D);
    else
        bakedTextureRight = new Texture(testFile.c_str(), GL_TEXTURE_2D);
    bakedTextureLeft->init_from_file();
    bakedTextureRight->init_from_file();
    // SkinnedModel dinosaur("../../resource/reconst.ply", "", proj_width, proj_height, cam_width, cam_height);
    n_bones = leftHandModel.NumBones();
    postProcess.initGLBuffers();
    fullScreenQuad.init();
    Quad topHalfQuad("top_half", 0.0f);
    Quad bottomLeftQuad("bottom_left", 0.0f);
    Quad bottomRightQuad("bottom_right", 0.0f);
    glm::vec3 coa = leftHandModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    Text text("../../resource/arial.ttf");
    near_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, -1.0f},
                                           {-1.0f, -1.0f, -1.0f},
                                           {1.0f, -1.0f, -1.0f},
                                           {1.0f, 1.0f, -1.0f}};
    mid_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, 0.7f},
                                          {-1.0f, -1.0f, 0.7f},
                                          {1.0f, -1.0f, 0.7f},
                                          {1.0f, 1.0f, 0.7f}};
    far_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, 1.0f},
                                          {-1.0f, -1.0f, 1.0f},
                                          {1.0f, -1.0f, 1.0f},
                                          {1.0f, 1.0f, 1.0f}};
    frustumCornerVertices = std::vector<glm::vec3>{
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
    Shader uv_NNShader("../../src/shaders/uv_NN_Shader.vs", "../../src/shaders/uv_NN_Shader.fs");
    Shader maskShader("../../src/shaders/mask.vs", "../../src/shaders/mask.fs");
    Shader jfaInitShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa_init.fs");
    Shader jfaShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa.fs");
    Shader fastTrackerShader("../../src/shaders/fast_tracker.vs", "../../src/shaders/fast_tracker.fs");
    Shader debugShader("../../src/shaders/debug.vs", "../../src/shaders/debug.fs");
    Shader projectorShader("../../src/shaders/projector_shader.vs", "../../src/shaders/projector_shader.fs");
    Shader projectorOnlyShader("../../src/shaders/projector_only.vs", "../../src/shaders/projector_only.fs");
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader gridShader("../../src/shaders/grid_texture.vs", "../../src/shaders/grid_texture.fs");
    Shader gridColorShader("../../src/shaders/grid_color.vs", "../../src/shaders/grid_color.fs");
    Shader lineShader("../../src/shaders/line_shader.vs", "../../src/shaders/line_shader.fs");
    Shader coordShader("../../src/shaders/coords.vs", "../../src/shaders/coords.fs");
    Shader vcolorShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    SkinningShader skinnedShader("../../src/shaders/skin_hand_simple.vs", "../../src/shaders/skin_hand_simple.fs");
    Shader bakeSimple("../../src/shaders/bake_proj_simple.vs", "../../src/shaders/bake_proj_simple.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    // render the baked texture into fbo
    bake_fbo_right.bind(true);
    bakedTextureRight->bind();
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    set_texture_shader(textureShader, false, false, false);
    fullScreenQuad.render();
    bake_fbo_right.unbind();
    bake_fbo_left.bind(true);
    bakedTextureLeft->bind();
    set_texture_shader(textureShader, false, false, false);
    fullScreenQuad.render();
    bake_fbo_left.unbind();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    // settings for text shader
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    /* more inits */
    double previousAppTime = t_app.getElapsedTimeInSec();
    double previousSecondAppTime = t_app.getElapsedTimeInSec();
    double currentAppTime = t_app.getElapsedTimeInSec();
    long frameCount = 0;
    uint64_t targetFrameSize = 0;
    size_t n_skeleton_primitives = 0;
    uint8_t *colorBuffer = new uint8_t[projected_image_size];
    CGrabResultPtr ptrGrabResult;
    camTexture = Texture();
    // Texture flowTexture = Texture();
    Texture displayTexture = Texture();
    displayTexture.init(cam_width, cam_height, n_cam_channels);
    camTexture.init(cam_width, cam_height, n_cam_channels);
    // flowTexture.init(cam_width, cam_height, 2);
    if (use_projector)
    {
        if (!projector.init())
        {
            std::cerr << "Failed to initialize projector\n";
            use_projector = false;
        }
    }
    LeapCreateClockRebaser(&clockSynchronizer);
    // load calibration results if they exist
    Camera_Mode camera_mode = freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    if (loadLeapCalibrationResults(proj_project, cam_project,
                                   w2c_auto, w2c_user,
                                   points_2d, points_3d,
                                   undistort_map1, undistort_map2,
                                   camera_intrinsics_cv, camera_distortion_cv))
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
    /* thread loops */
    // camera.init(camera_queue, close_signal, cam_height, cam_width, exposure);
    if (!simulated_camera)
    {
        /* real producer */
        camera.init_poll(cam_height, cam_width, exposure);
        std::cout << "Using real camera to produce images" << std::endl;
        projector.show();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        camera.balance_white();
        camera.acquire();
    }
    else
    {
        /* fake producer */
        std::cout << "Using fake camera to produce images" << std::endl;
        // simulated_camera = true;
        // std::string path = "../../resource/hand_capture";
        // std::vector<cv::Mat> fake_cam_images;
        // white image
        // fake_cam_images.push_back(white_image);
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
        // producer = std::thread([fake_cam_images]() { //, &projector
        //     CPylonImage image = CPylonImage::Create(PixelType_BGRA8packed, cam_width, cam_height);
        //     Timer t_block;
        //     int counter = 0;
        //     t_block.start();
        //     while (!close_signal)
        //     {
        //         camera_queue.wait_enqueue(image);
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
    // image consumer
    consumer = std::thread([]() { //, &projector
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
            std::modf(t_app.getElapsedTimeInMicroSec(), &whole);
            LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
        }
        frameCount++;
        /* display stats */
        if (currentAppTime - previousSecondAppTime >= 1.0)
        {
            fps = frameCount;
            ms_per_frame = 1000.0f / frameCount;
            if (cmd_line_stats)
            {
                std::cout << "avg ms: " << 1000.0f / frameCount << " FPS: " << frameCount << std::endl;
                std::cout << "total app: " << t_app.getElapsedTimeInSec() << "s" << std::endl;
                std::cout << "misc: " << t_misc.averageLapInMilliSec() << std::endl;
                std::cout << "cam_enqueue: " << camera.getAvgEnqueueTimeAndReset() << std::endl;
                std::cout << "cam_dequeue: " << t_camera.averageLapInMilliSec() << std::endl;
                std::cout << "leap: " << t_leap.averageLapInMilliSec() << std::endl;
                std::cout << "skinning: " << t_skin.averageLapInMilliSec() << std::endl;
                std::cout << "debug: " << t_debug.averageLapInMilliSec() << std::endl;
                std::cout << "post process: " << t_pp.averageLapInMilliSec() << std::endl;
                std::cout << "mls: " << t_mls.averageLapInMilliSec() << std::endl;
                std::cout << "mls thread: " << t_mls_thread.averageLapInMilliSec() << std::endl;
                std::cout << "warp: " << t_warp.averageLapInMilliSec() << std::endl;
                std::cout << "swap buffers: " << t_swap.averageLapInMilliSec() << std::endl;
                std::cout << "GPU->CPU: " << t_download.averageLapInMilliSec() << std::endl;
                // std::cout << "project time: " << t4.averageLap() << std::endl;
                std::cout << "cam q1 size: " << camera_queue.size_approx() << std::endl;
                // std::cout << "cam q2 size: " << camera_queue_cv.size() << std::endl;
                std::cout << "proj q size: " << projector_queue.size_approx() << std::endl;
            }
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
            t_mls.reset();
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
            switch (operation_mode)
            {
            case static_cast<int>(OperationMode::NORMAL):
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                break;
            }
            default:
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                break;
            }
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t_misc.stop();

        /* camera transforms */
        glm::mat4 cam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 cam_projection_transform = gl_camera.getProjectionMatrix();

        /* main switch on operation mode of engine */
        switch (operation_mode)
        {
        case static_cast<int>(OperationMode::NORMAL): // fast path
        {
            /* deal with camera input */
            t_camera.start();
            handleCameraInput(ptrGrabResult);
            t_camera.stop();

            /* deal with leap input */
            t_leap.start();
            LEAP_STATUS leap_status = handleLeapInput(0);
            t_leap.stop();

            /* skin hand meshes */
            t_skin.start();
            handleSkinning(skinnedShader, vcolorShader, rightHandModel, cam_view_transform, cam_projection_transform, true);
            handleSkinning(skinnedShader, vcolorShader, leftHandModel, cam_view_transform, cam_projection_transform, false);
            t_skin.stop();

            /* deal with bake request */
            t_bake.start();
            handleBaking(skinnedShader, textureShader, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, true);
            t_bake.stop();

            /* run MLS on MP prediction to reduce bias */
            t_mls.start();
            handleMLS(gridShader);
            t_mls.stop();

            /* post process fbo using camera input */
            t_pp.start();
            handlePostProcess(rightHandModel.GetMaterial().pDiffuse, textureShader, maskShader, jfaInitShader, jfaShader, NNShader, uv_NNShader, vcolorShader);
            /* render final output to screen */
            if (!debug_mode)
            {
                glViewport(0, 0, proj_width, proj_height); // set viewport
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                set_texture_shader(textureShader, false, false, false);
                c2p_fbo.getTexture()->bind();
                fullScreenQuad.render();
            }
            t_pp.stop();

            /* debug mode */
            t_debug.start();
            handleDebugMode(skinnedShader, textureShader, vcolorShader, textShader, lineShader, rightHandModel, leftHandModel, text);
            t_debug.stop();
            break;
        }
        case static_cast<int>(OperationMode::VIDEO): // record and playback session
        {
            if (pre_recorded_session_loaded && use_pre_recorded_session)
            {
                float diff_between_frames_ms = 1000.0f / 120.0f;             // ~8.33f
                float factor = required_latency_ms / diff_between_frames_ms; // ~0.6f
                float interp_factor, whole;
                interp_factor = std::modf(factor, &whole);
                int n_frames_ahead = static_cast<int>(whole);
                // get pre-recorded leap info i
                LEAP_STATUS leap_status = handleLeapInput(videoFrameCount);
                std::vector<glm::mat4> bones_to_world_cur, bones_to_world_left_interp1, bones_to_world_left_interp2;
                if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
                {
                    break;
                }
                // get pre-recorded leap info i
                bones_to_world_cur = bones_to_world_left;
                if (n_frames_ahead == 0)
                {
                    bones_to_world_left_interp1 = bones_to_world_cur;
                }
                else
                {
                    leap_status = handleLeapInput(videoFrameCount + n_frames_ahead);
                    if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
                    {
                        break;
                    }
                    bones_to_world_left_interp1 = bones_to_world_left;
                }
                leap_status = handleLeapInput(videoFrameCount + n_frames_ahead + 1);
                if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
                {
                    break;
                }
                bones_to_world_left_interp2 = bones_to_world_left;
                // produce fake cam image, using interpolated leap info between i and i + 1, to achieve required simulated latency

                bones_to_world_left_interp.clear();
                for (int i = 0; i < bones_to_world_left.size(); i++)
                {
                    glm::mat4 interpolated = Helpers::interpolate(bones_to_world_left_interp1[i], bones_to_world_left_interp2[i], interp_factor, true);
                    bones_to_world_left_interp.push_back(interpolated);
                }
                // render scene
                // render left hand
                bones_to_world_left = bones_to_world_left_interp;
                handleSkinning(skinnedShader, vcolorShader, leftHandModel, cam_view_transform, cam_projection_transform, false);
                // convert to gray scale single channel image
                FBO fake_cam_fbo(cam_width, cam_height, 1);
                fake_cam_fbo.bind();
                set_texture_shader(textureShader, true, true, false, true, masking_threshold);
                hands_fbo.getTexture()->bind();
                fullScreenQuad.render();
                fake_cam_fbo.unbind();
                // fake_cam_fbo.saveColorToFile(std::format("../../debug/video/cam_{:05d}.png", totalFrameCount), false);
                std::vector<uchar> fake_cam_image = fake_cam_fbo.getBuffer(1);
                // load as a regular camera image
                camTexture.load(fake_cam_image.data(), true, cam_buffer_format);
                camImageOrig = cv::Mat(cam_height, cam_width, CV_8UC1, fake_cam_image.data()).clone();
                // cv::imwrite("test2.png", camImageOrig);
                // hands_fbo.bind(false, true, false);
                // glBlitFramebuffer(0, 0, dst_width, dst_height, 0, 0, dst_width, dst_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                // render the scene as normal
                bones_to_world_left = bones_to_world_cur;
                handleSkinning(skinnedShader, vcolorShader, leftHandModel, cam_view_transform, cam_projection_transform, false);
                // hands_fbo.saveColorToFile(std::format("../../debug/video/render_{:05d}.png", totalFrameCount), false);
                // post process them with any required effect
                handlePostProcess(leftHandModel.GetMaterial().pDiffuse, textureShader, maskShader, jfaInitShader, jfaShader, NNShader, uv_NNShader, vcolorShader);
                if (!debug_mode)
                {
                    glViewport(0, 0, proj_width, proj_height); // set viewport
                    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                    set_texture_shader(textureShader, false, false, false);
                    c2p_fbo.getTexture()->bind();
                    fullScreenQuad.render();
                }
                // save render to disk as png
                // c2p_fbo.saveColorToFile(std::format("../../debug/video/final_{:05d}.png", totalFrameCount), false);
                // offline: produce a compressed video from the pngs
                // online: load video into memory and stream to projector in full fps
                videoFrameCount += 1;
            }
            else
            {
                if (playback_video)
                {
                    std::cout << "not implemented" << std::endl;
                }
            }
            break;
        }
        case static_cast<int>(OperationMode::CAMERA): // calibrate camera
        {
            switch (calibration_state)
            {
            case static_cast<int>(CalibrationStateMachine::COLLECT):
            {
                std::vector<cv::Point2f> corner_pts;
                camera.capture_single_image(ptrGrabResult);
                camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                cv::flip(camImageOrig, camImage, 1);
                bool success = cv::findChessboardCorners(camImage, cv::Size(checkerboard_width, checkerboard_height), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
                if (success)
                {
                    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
                    cv::cornerSubPix(camImage, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
                    cv::drawChessboardCorners(camImage, cv::Size(checkerboard_width, checkerboard_height), corner_pts, success);
                    if (ready_to_collect)
                        imgpoints.push_back(corner_pts);
                }
                displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                set_texture_shader(textureShader, true, false, true);
                displayTexture.bind();
                fullScreenQuad.render();
                if (imgpoints.size() >= n_points_cam_calib)
                    ready_to_collect = false;
                break;
            }
            case static_cast<int>(CalibrationStateMachine::CALIBRATE):
            {
                std::vector<std::vector<cv::Point3f>> objpoints;
                std::vector<cv::Point3f> objp;
                for (int i = 0; i < checkerboard_height; i++)
                {
                    for (int j{0}; j < checkerboard_width; j++)
                        objp.push_back(cv::Point3f(j, i, 0));
                }
                for (int i = 0; i < imgpoints.size(); i++)
                    objpoints.push_back(objp);
                // cv::calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
                // objectPoints.resize(imagePoints.size(), objectPoints[0]);
                cv::Mat cameraMatrix, distCoeffs, R, T;
                cv::calibrateCamera(objpoints, imgpoints, cv::Size(cam_height, cam_width), cameraMatrix, distCoeffs, R, T);
                std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
                std::cout << "distCoeffs : " << distCoeffs << std::endl;
                std::vector<double> camMat(cameraMatrix.begin<double>(), cameraMatrix.end<double>());
                std::vector<double> camDist(distCoeffs.begin<double>(), distCoeffs.end<double>());
                calib_cam_matrix = camMat;
                calib_cam_distortion = camDist;
                // std::cout << "Rotation vector : " << R << std::endl;
                // std::cout << "Translation vector : " << T << std::endl;
                calibration_state = static_cast<int>(CalibrationStateMachine::SHOW);
                break;
            }
            case static_cast<int>(CalibrationStateMachine::SHOW):
            {
                break;
            }
            default:
                break;
            }

            break;
        }
        case static_cast<int>(OperationMode::COAXIAL): // calibrate projector <-> camera
        {
            t_camera.start();
            handleCameraInput(ptrGrabResult);
            t_camera.stop();
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
            set_texture_shader(textureShader, true, true, true, false, masking_threshold, 0, glm::mat4(1.0f), glm::mat4(1.0f), viewMatrix);
            camTexture.bind();
            fullScreenQuad.render();
            PointCloud cloud(cur_screen_verts, screen_verts_color_red);
            vcolorShader.use();
            vcolorShader.setMat4("MVP", glm::mat4(1.0f));
            cloud.render();
            break;
        }
        case static_cast<int>(OperationMode::LEAP): // calibrate camera <-> leap
        {
            /* deal with leap input */
            t_leap.start();
            LEAP_STATUS leap_status = handleLeapInput(0);
            t_leap.stop();
            switch (calibration_state)
            {
            case static_cast<int>(CalibrationStateMachine::COLLECT):
            {
                switch (leap_collection_setting)
                {
                case static_cast<int>(LeapCollectionSettings::AUTO_RAW):
                {
                    std::vector<uint8_t> buffer1, buffer2;
                    uint32_t ignore1, ignore2;
                    if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                    {
                        uint64_t new_frame_id = leap.getImageFrameID();
                        if (leap_cur_frame_id != new_frame_id)
                        {
                            // capture cam image asap
                            camera.capture_single_image(ptrGrabResult);
                            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                            cv::flip(camImageOrig, camImage, 1);
                            cv::Mat thr;
                            cv::threshold(camImage, thr, static_cast<int>(masking_threshold * 255), 255, cv::THRESH_BINARY);
                            glm::vec2 center, center_leap1, center_leap2;
                            // render binary leap texture to bottom half of screen
                            Texture leapTexture1 = Texture();
                            Texture leapTexture2 = Texture();
                            leapTexture1.init(leap_width, leap_height, 1);
                            leapTexture2.init(leap_width, leap_height, 1);
                            leapTexture1.load(buffer1, true, cam_buffer_format);
                            leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture1.bind();
                            set_texture_shader(textureShader, true, false, true, leap_threshold_flag, leap_binary_threshold);
                            bottomLeftQuad.render();
                            leapTexture2.bind();
                            bottomRightQuad.render();
                            displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                            set_texture_shader(textureShader, true, false, true, threshold_flag, masking_threshold);
                            displayTexture.bind();
                            topHalfQuad.render();
                            glm::vec2 center_NDC;
                            bool found_centroid = false;
                            if (extract_centroid(thr, center))
                            {
                                found_centroid = true;
                                // render point on centroid in camera image
                                vcolorShader.use();
                                vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                                center_NDC = Helpers::ScreenToNDC(center, cam_width, cam_height, true);
                                glm::vec2 vert = center_NDC;
                                vert.y = (vert.y + 1.0f) / 2.0f; // for display, use top of screen
                                std::vector<glm::vec2> pc1 = {vert};
                                PointCloud pointCloud(pc1, screen_verts_color_red);
                                pointCloud.render(5.0f);
                            }
                            cv::Mat leap1_thr, leap2_thr;
                            cv::Mat leap1(leap_height, leap_width, CV_8UC1, buffer1.data());
                            cv::Mat leap2(leap_height, leap_width, CV_8UC1, buffer2.data());
                            cv::threshold(leap1, leap1_thr, static_cast<int>(leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            cv::threshold(leap2, leap2_thr, static_cast<int>(leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            if (extract_centroid(leap1_thr, center_leap1) && extract_centroid(leap2_thr, center_leap2))
                            {
                                // save the 2d and 3d points
                                glm::vec2 center_NDC_leap1 = Helpers::ScreenToNDC(center_leap1, leap_width, leap_height, true);
                                glm::vec2 center_NDC_leap2 = Helpers::ScreenToNDC(center_leap2, leap_width, leap_height, true);
                                glm::vec3 cur_3d_point = triangulate(leap, center_NDC_leap1, center_NDC_leap2);
                                triangulated = cur_3d_point;
                                if (found_centroid)
                                {
                                    glm::vec2 cur_2d_point = Helpers::NDCtoScreen(center_NDC, cam_width, cam_height, true);
                                    if (ready_to_collect)
                                    {
                                        points_3d.push_back(cur_3d_point);
                                        points_2d.push_back(cur_2d_point);
                                        if (points_2d.size() >= n_points_leap_calib)
                                        {
                                            ready_to_collect = false;
                                        }
                                    }
                                }
                                // render point on centroid in left leap image
                                vcolorShader.use();
                                vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                                glm::vec2 vert1 = center_NDC_leap1;
                                glm::vec2 vert2 = center_NDC_leap2;
                                vert1.y = (vert1.y - 1.0f) / 2.0f; // use bottom left of screen
                                vert1.x = (vert1.x - 1.0f) / 2.0f; //
                                vert2.y = (vert2.y - 1.0f) / 2.0f; // use bottom right of screen
                                vert2.x = (vert2.x + 1.0f) / 2.0f; //
                                std::vector<glm::vec2> pc2 = {vert1, vert2};
                                PointCloud pointCloud2(pc2, screen_verts_color_red);
                                pointCloud2.render(5.0f);
                            }
                            leap_cur_frame_id = new_frame_id;
                        }
                    }
                    break;
                }
                case static_cast<int>(LeapCollectionSettings::AUTO_FINGER):
                {
                    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME)
                    {
                        // capture cam image asap
                        camera.capture_single_image(ptrGrabResult);
                        camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                        cv::flip(camImageOrig, camImage, 1);
                        cv::Mat thr;
                        cv::threshold(camImage, thr, static_cast<int>(masking_threshold * 255), 255, cv::THRESH_BINARY);
                        // render binary leap texture to bottom half of screen
                        displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                        set_texture_shader(textureShader, true, false, true, threshold_flag, masking_threshold);
                        displayTexture.bind();
                        fullScreenQuad.render();
                        glm::vec2 center_NDC;
                        glm::vec2 center;
                        bool found_centroid = false;
                        if (extract_centroid(thr, center))
                        {
                            found_centroid = true;
                            // render point on centroid in camera image
                            vcolorShader.use();
                            vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                            center_NDC = Helpers::ScreenToNDC(center, cam_width, cam_height, true);
                            glm::vec2 vert = center_NDC;
                            // vert.y = (vert.y + 1.0f) / 2.0f; // for display, use top of screen
                            std::vector<glm::vec2> pc1 = {vert};
                            PointCloud pointCloud(pc1, screen_verts_color_red);
                            pointCloud.render(5.0f);
                        }
                        if (joints_right.size() > 0)
                        {
                            glm::vec3 cur_3d_point = joints_right[17]; // index tip
                            triangulated = cur_3d_point;
                            if (found_centroid)
                            {
                                glm::vec2 cur_2d_point = Helpers::NDCtoScreen(center_NDC, cam_width, cam_height, true);
                                if (ready_to_collect)
                                {
                                    points_3d.push_back(cur_3d_point);
                                    points_2d.push_back(cur_2d_point);
                                    if (points_2d.size() >= n_points_leap_calib)
                                    {
                                        ready_to_collect = false;
                                    }
                                }
                            }
                        }
                        // render 3d point on leap image ... todo
                        // vcolorShader.use();
                        // vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                        // glm::vec2 vert1 = ?;
                        // glm::vec2 vert2 = ?;
                        // vert1.y = (vert1.y - 1.0f) / 2.0f; // use bottom left of screen
                        // vert1.x = (vert1.x - 1.0f) / 2.0f; //
                        // vert2.y = (vert2.y - 1.0f) / 2.0f; // use bottom right of screen
                        // vert2.x = (vert2.x + 1.0f) / 2.0f; //
                        // std::vector<glm::vec2> pc2 = {vert1, vert2};
                        // PointCloud pointCloud2(pc2, screen_verts_color_red);
                        // pointCloud2.render(5.0f);
                    }
                    break;
                }
                case static_cast<int>(LeapCollectionSettings::MANUAL_FINGER):
                {
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                    set_texture_shader(textureShader, true, false, true);
                    displayTexture.bind();
                    fullScreenQuad.render();
                    vcolorShader.use();
                    vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                    std::vector<glm::vec2> test = {cur_screen_vert};
                    PointCloud pointCloud(test, screen_verts_color_red);
                    pointCloud.render(5.0f);
                    break;
                }

                default:
                    break;
                }
                break;
            }
            case static_cast<int>(CalibrationStateMachine::CALIBRATE):
            {
                std::vector<cv::Point2f> points_2d_cv;
                std::vector<cv::Point3f> points_3d_cv;
                for (int i = 0; i < points_2d.size(); i++)
                {
                    points_2d_cv.push_back(cv::Point2f(points_2d[i].x, points_2d[i].y));
                    points_3d_cv.push_back(cv::Point3f(points_3d[i].x, points_3d[i].y, points_3d[i].z));
                }
                // initial guess
                cv::Mat transform = cv::Mat::zeros(4, 4, CV_64FC1);
                transform.at<double>(0, 0) = -1.0f;
                transform.at<double>(1, 2) = 1.0f;
                transform.at<double>(2, 1) = 1.0f;
                transform.at<double>(0, 3) = -50.0f;
                transform.at<double>(1, 3) = -200.0f;
                transform.at<double>(2, 3) = 550.0f;
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
                cv::Rodrigues(rotmat_inverse, rvec_calib);
                // std::cout << "rvec_inverse: " << rvec << std::endl;
                tvec_calib = transform(cv::Range(0, 3), cv::Range(3, 4)).clone();
                std::vector<cv::Point2f> points_2d_inliers_cv;
                std::vector<cv::Point3f> points_3d_inliers_cv;
                if (leap_calib_use_ransac)
                {
                    cv::Mat inliers;
                    cv::solvePnPRansac(points_3d_cv, points_2d_cv, camera_intrinsics_cv, camera_distortion_cv, rvec_calib, tvec_calib, true,
                                       pnp_iters, pnp_rep_error, pnp_confidence, inliers, cv::SOLVEPNP_ITERATIVE);
                    for (int inliers_index = 0; inliers_index < inliers.rows; ++inliers_index)
                    {
                        int n = inliers.at<int>(inliers_index);          // i-inlier
                        points_2d_inliers_cv.push_back(points_2d_cv[n]); // add i-inlier to list
                        points_3d_inliers_cv.push_back(points_3d_cv[n]); // add i-inlier to list
                    }
                }
                else
                {
                    cv::solvePnP(points_3d_cv, points_2d_cv, camera_intrinsics_cv, camera_distortion_cv, rvec_calib, tvec_calib, true, cv::SOLVEPNP_ITERATIVE);
                    points_2d_inliers_cv = points_2d_cv;
                    points_3d_inliers_cv = points_3d_cv;
                }
                std::vector<cv::Point2f> reprojected_cv, reprojected_inliers_cv;
                cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                cv::projectPoints(points_3d_inliers_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_inliers_cv);
                points_2d_reprojected = Helpers::opencv2glm(reprojected_cv);
                points_2d_inliers_reprojected = Helpers::opencv2glm(reprojected_inliers_cv);
                points_2d_inliners = Helpers::opencv2glm(points_2d_inliers_cv);
                std::vector<glm::vec3> points_3d_inliers = Helpers::opencv2glm(points_3d_inliers_cv);
                float mse = Helpers::MSE(points_2d, points_2d_reprojected);
                float mse_inliers = Helpers::MSE(points_2d_inliners, points_2d_inliers_reprojected);
                std::cout << "avg reprojection error: " << mse << std::endl;
                std::cout << "avg reprojection error (inliers): " << mse_inliers << std::endl;
                cv::Mat rot_mat(3, 3, CV_64FC1);
                cv::Rodrigues(rvec_calib, rot_mat);
                // std::cout << "rotmat: " << rot_mat << std::endl;
                cv::Mat w2c(4, 4, CV_64FC1);
                w2c.at<double>(0, 0) = rot_mat.at<double>(0, 0);
                w2c.at<double>(0, 1) = rot_mat.at<double>(0, 1);
                w2c.at<double>(0, 2) = rot_mat.at<double>(0, 2);
                w2c.at<double>(0, 3) = tvec_calib.at<double>(0, 0);
                w2c.at<double>(1, 0) = rot_mat.at<double>(1, 0);
                w2c.at<double>(1, 1) = rot_mat.at<double>(1, 1);
                w2c.at<double>(1, 2) = rot_mat.at<double>(1, 2);
                w2c.at<double>(1, 3) = tvec_calib.at<double>(1, 0);
                w2c.at<double>(2, 0) = rot_mat.at<double>(2, 0);
                w2c.at<double>(2, 1) = rot_mat.at<double>(2, 1);
                w2c.at<double>(2, 2) = rot_mat.at<double>(2, 2);
                w2c.at<double>(2, 3) = tvec_calib.at<double>(2, 0);
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
                calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                calibrationSuccess = true;
                break;
            }
            case static_cast<int>(CalibrationStateMachine::SHOW):
            {
                vcolorShader.use();
                vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                // todo: move this logic to CalibrationStateMachine::CALIBRATE
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
                break;
            }
            case static_cast<int>(CalibrationStateMachine::MARK):
            {
                switch (leap_mark_setting)
                {
                case static_cast<int>(LeapMarkSettings::STREAM):
                {
                    std::vector<uint8_t> buffer1, buffer2;
                    uint32_t ignore1, ignore2;
                    if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                    {
                        uint64_t new_frame_id = leap.getImageFrameID();
                        if (leap_cur_frame_id != new_frame_id)
                        {
                            // capture cam image asap
                            camera.capture_single_image(ptrGrabResult);
                            displayTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
                            glm::vec2 center, center_leap1, center_leap2;
                            // render binary leap texture to bottom half of screen
                            Texture leapTexture1 = Texture();
                            Texture leapTexture2 = Texture();
                            leapTexture1.init(leap_width, leap_height, 1);
                            leapTexture2.init(leap_width, leap_height, 1);
                            leapTexture1.load(buffer1, true, cam_buffer_format);
                            leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture1.bind();
                            set_texture_shader(textureShader, true, false, true, leap_threshold_flag, leap_binary_threshold);
                            bottomLeftQuad.render();
                            leapTexture2.bind();
                            bottomRightQuad.render();
                            set_texture_shader(textureShader, true, true, true);
                            displayTexture.bind();
                            topHalfQuad.render();
                            cv::Mat leap1_thr, leap2_thr;
                            cv::Mat leap1(leap_height, leap_width, CV_8UC1, buffer1.data());
                            cv::Mat leap2(leap_height, leap_width, CV_8UC1, buffer2.data());
                            cv::threshold(leap1, leap1_thr, static_cast<int>(leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            cv::threshold(leap2, leap2_thr, static_cast<int>(leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            if (extract_centroid(leap1_thr, center_leap1) && extract_centroid(leap2_thr, center_leap2))
                            {
                                // save the 2d and 3d points
                                glm::vec2 center_NDC_leap1 = Helpers::ScreenToNDC(center_leap1, leap_width, leap_height, true);
                                glm::vec2 center_NDC_leap2 = Helpers::ScreenToNDC(center_leap2, leap_width, leap_height, true);
                                glm::vec3 cur_3d_point = triangulate(leap, center_NDC_leap1, center_NDC_leap2);
                                triangulated = cur_3d_point;
                                /////
                                std::vector<cv::Point3f> points_3d_cv{cv::Point3f(cur_3d_point.x, cur_3d_point.y, cur_3d_point.z)};
                                std::vector<cv::Point2f> reprojected_cv;
                                cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                                std::vector<glm::vec2> reprojected = Helpers::opencv2glm(reprojected_cv);
                                reprojected = Helpers::ScreenToNDC(reprojected, cam_width, cam_height, true);
                                ////
                                vcolorShader.use();
                                vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                                glm::vec2 vert1 = center_NDC_leap1;
                                glm::vec2 vert2 = center_NDC_leap2;
                                glm::vec2 vert3 = reprojected[0];
                                vert1.y = (vert1.y - 1.0f) / 2.0f; // use bottom left of screen
                                vert1.x = (vert1.x - 1.0f) / 2.0f; //
                                vert2.y = (vert2.y - 1.0f) / 2.0f; // use bottom right of screen
                                vert2.x = (vert2.x + 1.0f) / 2.0f; //
                                vert3.y = (vert3.y + 1.0f) / 2.0f; // for display, use top of screen
                                std::vector<glm::vec2> pc2 = {vert1, vert2, vert3};
                                PointCloud pointCloud2(pc2, screen_verts_color_red);
                                pointCloud2.render(5.0f);
                                // std::cout << "leap1 2d:" << center_NDC_leap1.x << " " << center_NDC_leap1.y << std::endl;
                                // std::cout << "leap2 2d:" << center_NDC_leap2.x << " " << center_NDC_leap2.y << std::endl;
                                // std::cout << point_3d.x << " " << point_3d.y << " " << point_3d.z << std::endl;
                            }
                            leap_cur_frame_id = new_frame_id;
                        }
                    }
                    break;
                }
                case static_cast<int>(LeapMarkSettings::POINT_BY_POINT):
                {
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    switch (leap_calibration_mark_state)
                    {
                    case 0:
                    {
                        displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                        set_texture_shader(textureShader, true, false, true);
                        displayTexture.bind();
                        fullScreenQuad.render();
                        vcolorShader.use();
                        vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                        PointCloud pointCloud(marked_reprojected, screen_verts_color_red);
                        pointCloud.render();
                        break;
                    }
                    case 1:
                    {
                        std::vector<uint8_t> buffer1, buffer2;
                        uint32_t ignore1, ignore2;
                        if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                        {
                            Texture leapTexture = Texture();
                            leapTexture.init(leap_width, leap_height, 1);
                            leapTexture.load(buffer1, true, cam_buffer_format);
                            // Texture leapTexture2 = Texture();
                            // leapTexture2.init(leap_width, leap_height, 1);
                            // leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture.bind();
                            set_texture_shader(textureShader, true, false, true);
                            fullScreenQuad.render();
                            vcolorShader.use();
                            vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                            std::vector<glm::vec2> test = {cur_screen_vert};
                            PointCloud pointCloud(test, screen_verts_color_red);
                            pointCloud.render(5.0f);
                        }
                        break;
                    }
                    case 2:
                    {
                        std::vector<uint8_t> buffer1, buffer2;
                        uint32_t ignore1, ignore2;
                        if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                        {
                            Texture leapTexture = Texture();
                            leapTexture.init(leap_width, leap_height, 1);
                            leapTexture.load(buffer2, true, cam_buffer_format);
                            // Texture leapTexture2 = Texture();
                            // leapTexture2.init(leap_width, leap_height, 1);
                            // leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture.bind();
                            set_texture_shader(textureShader, true, false, true);
                            fullScreenQuad.render();
                            vcolorShader.use();
                            vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                            std::vector<glm::vec2> test = {cur_screen_vert};
                            PointCloud pointCloud(test, screen_verts_color_red);
                            pointCloud.render(5.0f);
                        }
                        break;
                    }
                    default:
                        break;
                    } // switch(leap_calibration_mark_state)
                    break;
                } // LeapMarkSettings::POINT_BY_POINT
                case static_cast<int>(LeapMarkSettings::WHOLE_HAND):
                {
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                    set_texture_shader(textureShader, true, false, true);
                    displayTexture.bind();
                    fullScreenQuad.render();
                    if (joints_right.size() > 0)
                    {
                        vcolorShader.use();
                        vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                        std::vector<cv::Point3f> points_3d_cv;
                        for (int i = 0; i < joints_right.size(); i++)
                        {
                            points_3d_cv.push_back(cv::Point3f(joints_right[i].x, joints_right[i].y, joints_right[i].z));
                        }
                        // points_3d_cv.push_back(cv::Point3f(joints_right[mark_bone_index].x, joints_right[mark_bone_index].y, joints_right[mark_bone_index].z));
                        std::vector<cv::Point2f> reprojected_cv;
                        cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                        std::vector<glm::vec2> reprojected = Helpers::opencv2glm(reprojected_cv);
                        // glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                        reprojected = Helpers::ScreenToNDC(reprojected, cam_width, cam_height, true);
                        std::vector<glm::vec2> pc = {reprojected};
                        PointCloud pointCloud(pc, screen_verts_color_red);
                        pointCloud.render();
                    }
                    break;
                } // LeapMarkSettings::WHOLE_HAND
                case static_cast<int>(LeapMarkSettings::ONE_BONE):
                {
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    displayTexture.load((uint8_t *)camImage.data, true, cam_buffer_format);
                    set_texture_shader(textureShader, true, false, true);
                    displayTexture.bind();
                    fullScreenQuad.render();
                    if (joints_right.size() > 0)
                    {
                        vcolorShader.use();
                        vcolorShader.setMat4("MVP", glm::mat4(1.0f));
                        std::vector<cv::Point3f> points_3d_cv;
                        points_3d_cv.push_back(cv::Point3f(joints_right[mark_bone_index].x, joints_right[mark_bone_index].y, joints_right[mark_bone_index].z));
                        std::vector<cv::Point2f> reprojected_cv;
                        cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                        std::vector<glm::vec2> reprojected = Helpers::opencv2glm(reprojected_cv);
                        // glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                        reprojected = Helpers::ScreenToNDC(reprojected, cam_width, cam_height, true);
                        std::vector<glm::vec2> pc = {reprojected};
                        PointCloud pointCloud(pc, screen_verts_color_red);
                        pointCloud.render();
                    }
                    break;
                } // LeapMarkSettings::ONE_BONE
                default:
                    break;
                } // switch(leap_mark_setting)
                break;
            } // CalibrationStateMachine::MARK
            default:
                break;
            } // switch (calibration_state)
            break;
        } // OperationMode::LEAP
        default:
            break;
        } // switch(calib_mode)

        if (use_projector)
        {
            // send result to projector queue
            glReadBuffer(GL_FRONT);
            if (use_pbo) // todo: investigate better way to download asynchornously
            {
                t_download.start();
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[totalFrameCount % 2]); // todo: replace with totalFrameCount
                glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, 0);
                t_download.stop();

                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[(totalFrameCount + 1) % 2]);
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
        totalFrameCount++;
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
    // if (simulated_camera)
    // {
    //     producer.join();
    // }
    if (run_mls.joinable())
        run_mls.join();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    Py_Finalize();
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
            if (operation_mode == static_cast<int>(OperationMode::LEAP))
            {
                switch (calibration_state)
                {
                case static_cast<int>(CalibrationStateMachine::COLLECT):
                {
                    if (ready_to_collect && joints_right.size() > 0)
                    {
                        points_3d.push_back(joints_right[17]);
                        glm::vec2 cur_2d_point = Helpers::NDCtoScreen(cur_screen_vert, cam_width, cam_height, false);
                        points_2d.push_back(cur_2d_point);
                    }
                    break;
                }
                case static_cast<int>(CalibrationStateMachine::MARK):
                {
                    if (leap_mark_setting == static_cast<int>(LeapMarkSettings::POINT_BY_POINT))
                    {
                        if (leap_calibration_mark_state == 1)
                        {
                            marked_2d_pos1 = cur_screen_vert;
                        }
                        if (leap_calibration_mark_state == 2)
                        {
                            marked_2d_pos2 = cur_screen_vert;
                            glm::vec3 pos_3d = triangulate(leap, marked_2d_pos1, marked_2d_pos2);
                            triangulated_marked.push_back(pos_3d);
                            std::vector<cv::Point3f> points_3d_cv{cv::Point3f(pos_3d.x, pos_3d.y, pos_3d.z)};
                            std::vector<cv::Point2f> reprojected_cv;
                            cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                            glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                            reprojected = Helpers::ScreenToNDC(reprojected, cam_width, cam_height, true);
                            marked_reprojected.push_back(reprojected);
                        }
                        leap_calibration_mark_state = (leap_calibration_mark_state + 1) % 3;
                    }
                    break;
                }
                default:
                    break;
                }
            }
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

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        rmb_pressed = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
    {
        if (rmb_pressed)
            activateGUI = !activateGUI;
        rmb_pressed = false;
    }
    if (activateGUI) // dont allow moving cameras when GUI active
        return;
    switch (operation_mode)
    {
    case static_cast<int>(OperationMode::NORMAL):
    {
        break;
    }
    case static_cast<int>(OperationMode::COAXIAL):
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
        break;
    }
    case static_cast<int>(OperationMode::LEAP):
    {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            dragging = true;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        {
            dragging = false;
        }
        break;
    }
    default:
        break;
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
    switch (operation_mode)
    {
    case static_cast<int>(OperationMode::NORMAL):
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
    case static_cast<int>(OperationMode::COAXIAL):
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
    case static_cast<int>(OperationMode::LEAP):
    {
        glm::vec2 mouse_pos_NDC = Helpers::ScreenToNDC(glm::vec2(xpos, ypos), proj_width, proj_height, true);
        if (dragging)
        {
            cur_screen_vert = mouse_pos_NDC;
        }
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
                                glm::mat4 &w2c_auto,
                                glm::mat4 &w2c_user,
                                std::vector<glm::vec2> &points_2d,
                                std::vector<glm::vec3> &points_3d,
                                cv::Mat &undistort_map1,
                                cv::Mat &undistort_map2,
                                cv::Mat &camera_intrinsics_cv,
                                cv::Mat &camera_distortion_cv)
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
    std::vector<double> camera_intrinsics_raw = cam_npz["cam_intrinsics"].as_vec<double>();
    std::vector<double> camera_distortion_raw = cam_npz["cam_distortion"].as_vec<double>();
    calib_cam_matrix = camera_intrinsics_raw;
    calib_cam_distortion = camera_distortion_raw;
    camera_intrinsics_cv = cv::Mat(3, 3, CV_64F, cam_npz["cam_intrinsics"].data<double>()).clone();
    camera_distortion_cv = cv::Mat(5, 1, CV_64F, cam_npz["cam_distortion"].data<double>()).clone();
    glm::mat3 camera_intrinsics = glm::make_mat3(camera_intrinsics_raw.data());
    // cv::Mat camera_intrinsics_cv = cv::Mat(3, 3, CV_64F, camera_intrinsics_raw.data());
    // cv::Mat camera_distortion_cv = cv::Mat(1, camera_distortion_raw.size(), CV_64F, camera_distortion_raw.data());
    std::cout << "camera_distortion: " << camera_distortion_cv << std::endl;
    std::cout << "camera_intrinsics: " << camera_intrinsics_cv << std::endl;
    cv::initUndistortRectifyMap(camera_intrinsics_cv, camera_distortion_cv, cv::Mat(), camera_intrinsics_cv, cv::Size(cam_width, cam_height), CV_32FC1, undistort_map1, undistort_map2);
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

LEAP_STATUS getLeapFramePreRecorded(std::vector<glm::mat4> &bones_to_world_left,
                                    std::vector<glm::mat4> &bones_to_world_right,
                                    std::vector<glm::vec3> &joints_left,
                                    std::vector<glm::vec3> &joints_right,
                                    uint64_t frameCounter)
{
    if (pre_recorded_session_loaded)
    {
        if (frameCounter < total_session_time_stamps)
        {
            bones_to_world_left.clear();
            bones_to_world_right.clear();
            joints_left.clear();
            joints_right.clear();
            for (int i = 0; i < 22; i++)
            {
                bones_to_world_left.push_back(session_bones_left[frameCounter * 22 + i]);
            }
            for (int i = 0; i < 42; i++)
            {
                joints_left.push_back(session_joints[frameCounter * 42 + i]);
            }
            return LEAP_STATUS::LEAP_NEWFRAME;
        }
        else
        {
            return LEAP_STATUS::LEAP_NONEWFRAME;
        }
    }
    else
    {
        return LEAP_STATUS::LEAP_FAILED;
    }
}

bool loadVideo()
{
    return true;
}

bool loadPrerecordedSession()
{
    std::string recordings("../../debug/recordings/");
    fs::path bones_left_path(recordings + session_name + std::string("_bones_left.npy"));
    // fs::path bones_right_path(recordings + tmp_filename + std::string("_bones_right.npy"));
    fs::path joints_path(recordings + session_name + std::string("_joints.npy"));
    cnpy::NpyArray bones_left_npy, joints_npy;
    if (fs::exists(bones_left_path))
    {
        bones_left_npy = cnpy::npy_load(bones_left_path.string());
        std::vector<float> raw_data = bones_left_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 16)
        {
            session_bones_left.push_back(glm::make_mat4(raw_data.data() + i));
        }
    }
    else
    {
        return false;
    }
    if (fs::exists(joints_path))
    {
        joints_npy = cnpy::npy_load(joints_path.string());
        std::vector<float> raw_data = joints_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 3)
        {
            session_joints.push_back(glm::make_vec3(raw_data.data() + i));
        }
    }
    else
    {
        return false;
    }
    total_session_time_stamps = session_joints.size() / 42;
    return true;
}

LEAP_STATUS getLeapFrame(LeapCPP &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &joints_left, // todo: remove joints, they are redundant
                         std::vector<glm::vec3> &joints_right,
                         bool leap_poll_mode,
                         int64_t &lastFrameID)
{
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
            joints_left.clear();
            joints_right.clear();
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
        joints_left.clear();
        joints_right.clear();
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
    for (uint32_t h = 0; h < frame->nHands; h++)
    {
        LEAP_HAND *hand = &frame->pHands[h];
        // if (debug_vec.x > 0)
        if (hand->type == eLeapHandType_Right)
            chirality = flip_x;
        float grab_angle = hand->grab_angle;
        std::vector<glm::mat4> bones_to_world;
        std::vector<glm::vec3> joints;
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
        if (useFingerWidth)
        {
            glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), hand->palm.width * glm::vec3(leap_palm_local_scaler, leap_palm_local_scaler, leap_palm_local_scaler));
            palm_orientation = palm_orientation * chirality * magic_leap_basis_fix * scalar * local_scaler;
        }
        else
        {
            palm_orientation = palm_orientation * chirality * magic_leap_basis_fix * scalar;
        }
        bones_to_world.push_back(glm::translate(glm::mat4(1.0f), palm_pos) * palm_orientation);
        // arm
        LEAP_VECTOR arm_j1 = hand->arm.prev_joint;
        LEAP_VECTOR arm_j2 = hand->arm.next_joint;
        joints.push_back(glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
        joints.push_back(glm::vec3(arm_j2.x, arm_j2.y, arm_j2.z));
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
        if (useFingerWidth)
        {
            glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), hand->arm.width * glm::vec3(leap_arm_local_scaler, leap_arm_local_scaler, leap_arm_local_scaler));
            bones_to_world.push_back(arm_translate * arm_rot * chirality * magic_leap_basis_fix * scalar * local_scaler);
        }
        else
        {
            bones_to_world.push_back(arm_translate * arm_rot * chirality * magic_leap_basis_fix * scalar);
        }
        // fingers
        for (uint32_t f = 0; f < 5; f++)
        {
            LEAP_DIGIT finger = hand->digits[f];
            for (uint32_t b = 0; b < 4; b++)
            {
                LEAP_VECTOR joint1 = finger.bones[b].prev_joint;
                LEAP_VECTOR joint2 = finger.bones[b].next_joint;
                joints.push_back(glm::vec3(joint1.x, joint1.y, joint1.z));
                joints.push_back(glm::vec3(joint2.x, joint2.y, joint2.z));
                glm::mat4 rot = glm::toMat4(glm::quat(finger.bones[b].rotation.w,
                                                      finger.bones[b].rotation.x,
                                                      finger.bones[b].rotation.y,
                                                      finger.bones[b].rotation.z));
                glm::vec3 translate = glm::vec3(joint1.x, joint1.y, joint1.z);
                glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
                if (useFingerWidth)
                {
                    glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), finger.bones[b].width * glm::vec3(leap_bone_local_scaler, leap_bone_local_scaler, leap_bone_local_scaler));
                    bones_to_world.push_back(trans * rot * chirality * magic_leap_basis_fix * scalar * local_scaler);
                }
                else
                {
                    bones_to_world.push_back(trans * rot * chirality * magic_leap_basis_fix * scalar);
                }
            }
        }
        if (hand->type == eLeapHandType_Right)
        {
            bones_to_world_right = std::move(bones_to_world);
            joints_right = std::move(joints);
            // grab_angle_right = grab_angle;
        }
        else
        {
            bones_to_world_left = std::move(bones_to_world);
            joints_left = std::move(joints);
            // grab_angle_left = grab_angle;
        }
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

void set_texture_shader(Shader &textureShader, bool flipVer, bool flipHor, bool isGray, bool binary, float threshold,
                        int src, glm::mat4 model, glm::mat4 projection, glm::mat4 view)
{
    textureShader.use();
    textureShader.setMat4("view", view);
    textureShader.setMat4("projection", projection);
    textureShader.setMat4("model", model);
    textureShader.setFloat("threshold", threshold);
    textureShader.setBool("flipHor", flipHor);
    textureShader.setBool("flipVer", flipVer);
    textureShader.setBool("binary", binary);
    textureShader.setBool("isGray", isGray);
    textureShader.setInt("src", src);
}

glm::vec3 triangulate(LeapCPP &leap, const glm::vec2 &leap1, const glm::vec2 &leap2)
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

std::vector<glm::vec2> mp_predict(cv::Mat origImage, int timestamp)
{
    cv::Mat image;
    cv::flip(origImage, image, 1);
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    // cv::Mat image = cv::imread("../../resource/hand.png", cv::IMREAD_GRAYSCALE);
    // std::cout << "mp received timestamp: " << timestamp << std::endl;
    // cv::Mat image;
    // cv::Mat image = cv::imread("C:/src/augmented_hands/debug/ss/sg0o.0_raw_cam.png");
    // cv::resize(image1, image, cv::Size(512, 512));
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
    // std::cout << "mp imported function" << std::endl;
    // warmup
    // PyObject* warmobj = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);
    // PyObject* res = PyObject_CallFunction(predict_single, "(O, i)", warmobj, 0);
    PyObject *image_object = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp *)&dimensions, NPY_UINT8, image.data);
    // std::cout << "mp converted data" << std::endl;
    // PyObject_CallFunction(myprint, "O", image_object);
    // PyObject* myResult = PyObject_CallFunction(iden, "O", image_object);
    PyObject *myResult = PyObject_CallFunction(predict_video, "(O,i)", image_object, timestamp);
    // std::cout << "mp called function" << std::endl;
    if (!myResult)
    {
        std::cout << "Call failed!" << std::endl;
        PyErr_Print();
        return std::vector<glm::vec2>();
        // exit(1);
    }
    // PyObject* myResult = PyObject_CallFunction(myprofile, "O", image_object);
    PyArrayObject *myNumpyArray = reinterpret_cast<PyArrayObject *>(myResult);
    glm::vec2 *data = (glm::vec2 *)PyArray_DATA(myNumpyArray);
    std::vector<glm::vec2> data_vec(data, data + PyArray_SIZE(myNumpyArray) / 2);
    // for (int j = 0; j < data_vec.size(); j+=2)
    // {
    //   std::cout << data_vec[j] << data_vec[j+1] << std::endl;
    // }
    // npy_intp* arrDims = PyArray_SHAPE( myNumpyArray );
    // int nDims = PyArray_NDIM( myNumpyArray ); // number of dimensions
    // std:: cout << nDims << std::endl;
    // for (int i = 0; i < nDims; i++)
    // {
    //   std::cout << arrDims[i] << std::endl;
    // }

    // cv::Mat python_result = cv::Mat(image.rows, image.cols, CV_8UC3, PyArray_DATA(myNumpyArray));
    // cv::imshow("result", python_result);
    // cv::waitKey(0);
    // double* array_pointer = reinterpret_cast<double*>( PyArray_DATA( your_numpy_array ) );
    // Py_XDECREF(myModule);
    // Py_XDECREF(myObject);
    // Py_XDECREF(myFunction);
    // Py_XDECREF(myResult);
    return data_vec;
}

void handleCameraInput(CGrabResultPtr ptrGrabResult)
{
    // prevFrame = curFrame.clone();
    // if (simulated_camera)
    // {
    //     cv::Mat tmp = camera_queue_cv.pop();
    //     camTexture.load((uint8_t *)tmp.data, true, cam_buffer_format);
    // }
    // else
    // {
    // std::cout << "before: " << camera_queue.size_approx() << std::endl;
    if (!simulated_camera)
    {
        bool sucess = camera.capture_single_image(ptrGrabResult);
        if (!sucess)
        {
            std::cout << "Failed to capture image" << std::endl;
            exit(1);
        }
        // camera_queue.wait_dequeue(ptrGrabResult);
        if (undistortCamera)
        {
            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
            cv::remap(camImageOrig, camImage, undistort_map1, undistort_map2, cv::INTER_LINEAR);
            camTexture.load(camImage.data, true, cam_buffer_format);
        }
        else
        {
            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
            camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, cam_buffer_format);
        }
    }
    else
    {
        camImageOrig = cv::imread("../../resource/hand.png", cv::IMREAD_GRAYSCALE);
        camTexture.load((uint8_t *)camImageOrig.data, true, cam_buffer_format);
    }
    // camera_queue.wait_dequeue(ptrGrabResult);
    // std::cout << "after: " << camera_queue.size_approx() << std::endl;
    // curCamImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
    // curCamBuf = std::vector<uint8_t>((uint8_t *)ptrGrabResult->GetBuffer(), (uint8_t *)ptrGrabResult->GetBuffer() + ptrGrabResult->GetImageSize());
}

LEAP_STATUS handleLeapInput(uint64_t frameCounter)
{
    LEAP_STATUS leap_status;
    if (use_pre_recorded_session)
    {
        leap_status = getLeapFramePreRecorded(bones_to_world_left, bones_to_world_right, joints_left, joints_right, frameCounter);
    }
    else
    {
        if (!leap_poll_mode)
        {
            // sync leap clock
            std::modf(t_app.getElapsedTimeInMicroSec(), &whole);
            LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
        }
        leap_status = getLeapFrame(leap, targetFrameTime, bones_to_world_left, bones_to_world_right, joints_left, joints_right, leap_poll_mode, lastFrameID);
        if (leap_record_session)
        {
            std::string savepath(std::string("../../debug/recordings/"));
            fs::path bones_left(savepath + tmp_filename.filename().string() + std::string("_bones_left.npy"));
            // fs::path bones_right(savepath + tmp_filename.filename().string() + std::string("_bones_right.npy"));
            fs::path joints(savepath + tmp_filename.filename().string() + std::string("_joints.npy"));
            // fs::path session(savepath + tmp_filename.filename().string() + std::string("_session.npz"));
            if ((joints_left.size() > 0) && (bones_to_world_left.size() > 0))
            {
                // cnpy::npy_save(session.string().c_str(), "joints", &skeleton_vertices[0].x, {skeleton_vertices.size(), 3}, "a");
                // cnpy::npy_save(session.string().c_str(), "bones_left", &bones_to_world_left[0][0].x, {bones_to_world_left.size(), 4, 4}, "a");
                // cnpy::npz_save(session.string().c_str(), "bones_right", &bones_to_world_right[0][0].x, {bones_to_world_right.size(), 4, 4}, "a");
                cnpy::npy_save(bones_left.string().c_str(), &bones_to_world_left[0][0].x, {1, bones_to_world_left.size(), 4, 4}, "a");
                // cnpy::npy_save(bones_right.string().c_str(), &bones_to_world_left[0][0], {bones_to_world_right.size(), 4, 4}, "a");
                cnpy::npy_save(joints.string().c_str(), &joints_left[0].x, {1, joints_left.size(), 3}, "a");
            }
        }
    }
    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME) // deal with user setting a global scale transform
    {
        glm::mat4 global_scale_transform = glm::scale(glm::mat4(1.0f), glm::vec3(leap_global_scaler));
        if (bones_to_world_right.size() > 0)
        {
            glm::mat4 right_translate = glm::translate(glm::mat4(1.0f), glm::vec3(bones_to_world_right[0][3][0], bones_to_world_right[0][3][1], bones_to_world_right[0][3][2]));
            global_scale_right = right_translate * global_scale_transform * glm::inverse(right_translate);
        }
        if (bones_to_world_left.size() > 0)
        {
            glm::mat4 left_translate = glm::translate(glm::mat4(1.0f), glm::vec3(bones_to_world_left[0][3][0], bones_to_world_left[0][3][1], bones_to_world_left[0][3][2]));
            global_scale_left = left_translate * global_scale_transform * glm::inverse(left_translate);
        }
    }
    return leap_status;
}

void handlePostProcess(Texture *handTexture,
                       Shader &textureShader,
                       Shader &maskShader,
                       Shader &jfaInitShader,
                       Shader &jfaShader,
                       Shader &NNShader,
                       Shader &uv_NNShader,
                       Shader &vcolorShader)
{
    switch (postprocess_mode)
    {
    case static_cast<int>(PostProcessMode::NONE):
    {
        // bind fbo
        postprocess_fbo.bind();
        // bind texture
        if (use_mls)
            mls_fbo.getTexture()->bind();
        else
            hands_fbo.getTexture()->bind();
        // render
        set_texture_shader(textureShader, false, false, false);
        fullScreenQuad.render(false, false, true);
        // if (skeleton_vertices.size() > 0)
        // {
        //     vcolorShader.use();
        //     vcolorShader.setMat4("MVP", glm::mat4(1.0f));
        //     std::vector<glm::vec2> skele_verts = Helpers::project_points(skeleton_vertices, glm::mat4(1.0f), cam_view_transform, cam_projection_transform);
        //     PointCloud cloud3(skele_verts, screen_verts_color_green);
        //     cloud3.render();
        // }
        // unbind fbo
        postprocess_fbo.unbind();
        break;
    }
    case static_cast<int>(PostProcessMode::CAM_FEED):
    {
        set_texture_shader(textureShader, true, true, true, threshold_flag, masking_threshold);
        camTexture.bind();
        postprocess_fbo.bind();
        fullScreenQuad.render();
        postprocess_fbo.unbind();
        break;
    }
    case static_cast<int>(PostProcessMode::OVERLAY):
    {
        if (use_mls)
            postProcess.mask(maskShader, mls_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
        else
            postProcess.mask(maskShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
        break;
    }
    case static_cast<int>(PostProcessMode::JUMP_FLOOD):
    {
        if (use_mls)
            postProcess.jump_flood(jfaInitShader, jfaShader, NNShader, mls_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
        else
            postProcess.jump_flood(jfaInitShader, jfaShader, NNShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
        break;
    }
    case static_cast<int>(PostProcessMode::JUMP_FLOOD_UV):
    {
        if (use_mls)
            postProcess.jump_flood_uv(jfaInitShader, jfaShader, uv_NNShader, mls_fbo.getTexture()->getTexture(),
                                      handTexture->getTexture(),
                                      camTexture.getTexture(),
                                      &postprocess_fbo, masking_threshold);
        else
            postProcess.jump_flood_uv(jfaInitShader, jfaShader, uv_NNShader, uv_fbo.getTexture()->getTexture(),
                                      handTexture->getTexture(),
                                      camTexture.getTexture(),
                                      &postprocess_fbo, masking_threshold);
        break;
    }
    case static_cast<int>(PostProcessMode::ICP):
    {
        // get render in camera space
        icp_fbo.bind();
        hands_fbo.getTexture()->bind();
        set_texture_shader(textureShader, true, false, false, true, masking_threshold);
        fullScreenQuad.render();
        icp_fbo.unbind();
        std::vector<uchar> rendered_data = icp_fbo.getBuffer(1);
        cv::Mat render_image = cv::Mat(cam_height, cam_width, CV_8UC1, rendered_data.data());
        // std::vector<uchar> rendered_data = hands_fbo.getBuffer(4);
        // cv::Mat render_image = cv::Mat(dst_height, dst_width, CV_8UC4, rendered_data.data());
        // cv::Mat bgra[4];               // destination array
        // cv::split(render_image, bgra); // split source
        glm::mat4 transform = glm::mat4(1.0f);
        cv::Mat debug = postProcess.icp(render_image, camImageOrig, masking_threshold, transform);
        // postprocess_fbo.bind();
        icp2_fbo.bind();
        hands_fbo.getTexture()->bind();
        if (icp_apply_transform)
        {
            // displayTexture.load((uint8_t *)debug.data, true, cam_buffer_format);
            // displayTexture.bind();
            set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, transform);
        }
        else
        {
            // icp_fbo.getTexture()->bind();
            set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, glm::mat4(1.0f));
        }
        fullScreenQuad.render(false, false, true);
        icp2_fbo.unbind();
        postProcess.mask(maskShader, icp2_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, masking_threshold);
        // fullScreenQuad.render();
        // postprocess_fbo.unbind();
        break;
    }
    default:
        break;
    }
    c2p_fbo.bind();
    if (use_coaxial_calib)
        set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, glm::mat4(1.0f), c2p_homography);
    else
        set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, glm::mat4(1.0f), glm::mat4(1.0f));
    postprocess_fbo.getTexture()->bind();
    fullScreenQuad.render();
    if (show_mls_landmarks && use_mls)
    {
        vcolorShader.use();
        if (size(ControlPointsP_glm) > 0)
        {
            PointCloud cloud_src(ControlPointsP_glm, screen_verts_color_red);
            PointCloud cloud_dst(ControlPointsQ_glm, screen_verts_color_green);
            if (use_coaxial_calib)
                vcolorShader.setMat4("MVP", c2p_homography);
            else
                vcolorShader.setMat4("MVP", glm::mat4(1.0f));
            cloud_src.render(5.0f);
            cloud_dst.render(5.0f);
        }
    }
    c2p_fbo.unbind();
}

void handleSkinning(SkinningShader &skinnedShader,
                    Shader &vcolorShader,
                    SkinnedModel &handModel,
                    glm::mat4 cam_view_transform,
                    glm::mat4 cam_projection_transform,
                    bool isRightHand)
{
    std::vector<glm::mat4> bones_to_world = isRightHand ? bones_to_world_right : bones_to_world_left;
    if (bones_to_world.size() > 0)
    {
        glm::mat4 global_scale = isRightHand ? global_scale_right : global_scale_left;
        if (isRightHand || bones_to_world_right.size() == 0)
            hands_fbo.bind(true);
        else
            hands_fbo.bind(false);
        glEnable(GL_DEPTH_TEST); // depth test on (todo: why is it off to begin with?)
        switch (material_mode)
        {
        case static_cast<int>(MaterialMode::DIFFUSE):
        {
            /* render skinned mesh to fbo, in camera space*/
            skinnedShader.use();
            skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform * global_scale);
            skinnedShader.setBool("bake", false);
            skinnedShader.setBool("flipTexVertically", false);
            skinnedShader.setInt("src", 0);
            skinnedShader.setBool("useGGX", false);
            skinnedShader.setBool("renderUV", false);
            switch (texture_mode)
            {
            case static_cast<int>(TextureMode::ORIGINAL): // original texture loaded with mesh
                skinnedShader.setBool("useProjector", false);
                handModel.Render(skinnedShader, bones_to_world, rotx, false, nullptr);
                break;
            case static_cast<int>(TextureMode::BAKED): // a baked texture from a bake operation
                skinnedShader.setBool("useProjector", false);
                if (isRightHand)
                    handModel.Render(skinnedShader, bones_to_world, rotx, false, bake_fbo_right.getTexture());
                else
                    handModel.Render(skinnedShader, bones_to_world, rotx, false, bake_fbo_left.getTexture());
                break;
            case static_cast<int>(TextureMode::PROJECTIVE): // a projective texture from the virtual cameras viewpoint
                skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform);
                skinnedShader.setBool("useProjector", true);
                skinnedShader.setBool("flipTexVertically", true);
                handModel.Render(skinnedShader, bones_to_world, rotx, false, dynamicTexture);
                break;
            default:
                skinnedShader.setBool("useProjector", false); // original texture loaded with mesh
                handModel.Render(skinnedShader, bones_to_world, rotx, false, nullptr);
                break;
            }
            break;
        }
        case static_cast<int>(MaterialMode::GGX): // uses GGX material with the original diffuse texture loaded with mesh
        {
            skinnedShader.use();
            skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform * global_scale);
            skinnedShader.setBool("bake", false);
            skinnedShader.setBool("flipTexVertically", false);
            skinnedShader.setInt("src", 0);
            skinnedShader.setBool("useGGX", true);
            skinnedShader.setBool("renderUV", false);
            dirLight.calcLocalDirection(bones_to_world[0]);
            skinnedShader.SetDirectionalLight(dirLight);
            glm::vec3 camWorldPos = glm::vec3(cam_view_transform[3][0], cam_view_transform[3][1], cam_view_transform[3][2]);
            skinnedShader.SetCameraLocalPos(camWorldPos);
            skinnedShader.setBool("useProjector", false);
            handModel.Render(skinnedShader, bones_to_world, rotx, false, nullptr);
            break;
        }
        case static_cast<int>(MaterialMode::SKELETON): // mesh is rendered as a skeleton connecting the joints
        {
            glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO);
            std::vector<glm::vec3> skele_for_upload;
            glm::vec3 green = glm::vec3(0.0f, 1.0f, 0.0f);
            std::vector<glm::vec3> my_joints = isRightHand ? joints_right : joints_left;
            for (int i = 0; i < my_joints.size(); i++)
            {
                skele_for_upload.push_back(my_joints[i]);
                skele_for_upload.push_back(green);
            }
            glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * skele_for_upload.size(), skele_for_upload.data(), GL_STATIC_DRAW);
            vcolorShader.use();
            vcolorShader.setMat4("MVP", cam_projection_transform * cam_view_transform * global_scale);
            glBindVertexArray(skeletonVAO);
            glDrawArrays(GL_LINES, 0, static_cast<int>(skele_for_upload.size() / 2));
            break;
        }
        default:
        {
            break;
        }
        }
        hands_fbo.unbind();
        /* render the uvs into a seperate texture for JUMP_FLOOD_UV post process */
        // todo: if this pp is enabled, hands_fbo above performs redundant work: refactor.
        if (postprocess_mode == static_cast<int>(PostProcessMode::JUMP_FLOOD_UV))
        {
            uv_fbo.bind();
            skinnedShader.use();
            skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform * global_scale);
            skinnedShader.setBool("bake", false);
            skinnedShader.setBool("flipTexVertically", false);
            skinnedShader.setInt("src", 0);
            skinnedShader.setBool("useGGX", false);
            skinnedShader.setBool("renderUV", true);
            skinnedShader.setBool("useProjector", false);
            handModel.Render(skinnedShader, bones_to_world, rotx, false, nullptr);
            uv_fbo.unbind();
        }
        glDisable(GL_DEPTH_TEST); // todo: why not keep it on ?
    }
}

void handleBaking(SkinningShader &skinnedShader,
                  Shader &textureShader,
                  SkinnedModel &leftHandModel,
                  SkinnedModel &rightHandModel,
                  glm::mat4 cam_view_transform,
                  glm::mat4 cam_projection_transform,
                  bool isRightHand)
{
    if ((bones_to_world_right.size() > 0) || (bones_to_world_left.size() > 0)) // refactor into function
    {
        if (bakeRequest)
        {
            if (!sd_running)
            {
                sd_running = true;
                bones_to_world_right_bake = bones_to_world_right;
                bones_to_world_left_bake = bones_to_world_left;
                // launch thread etc.
                // download camera image to cpu (resizing to 1024x1024)
                sd_fbo.bind(true);
                set_texture_shader(textureShader, false, true, true, false);
                camTexture.bind();
                fullScreenQuad.render();
                sd_fbo.unbind();
                if (saveIntermed)
                    sd_fbo.saveColorToFile("../../resource/camera_image.png", false);
                std::vector<uint8_t> buf = sd_fbo.getBuffer(1);
                // download camera image thresholded to cpu (todo: consider doing all this on CPU)
                sd_fbo.bind(true);
                set_texture_shader(textureShader, false, true, true, true, masking_threshold);
                camTexture.bind();
                fullScreenQuad.render();
                sd_fbo.unbind();
                if (saveIntermed)
                    sd_fbo.saveColorToFile("../../resource/camera_mask.png", false);
                std::vector<uint8_t> buf_mask = sd_fbo.getBuffer(1);
                run_sd = std::thread([buf, buf_mask]() { // send camera image to stable diffusion
                    std::string myprompt;
                    std::vector<std::string> random_animal;
                    switch (sd_mode)
                    {
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
                        img2img_data = Diffuse::img2img(myprompt.c_str(),
                                                        sd_outwidth, sd_outheight,
                                                        buf, buf_mask, diffuse_seed,
                                                        1024, 1024, 1,
                                                        512, 512, false, false, sd_mask_mode);
                        if (saveIntermed)
                        {
                            cv::Mat img2img_result = cv::Mat(sd_outheight, sd_outwidth, CV_8UC3, img2img_data.data()).clone();
                            cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
                            cv::imwrite("../../resource/sd_result.png", img2img_result);
                        }
                        sd_succeed = true;
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    sd_running = false;
                });
            }
            bakeRequest = false;
        }
        if (sd_succeed)
        {
            if (dynamicTexture != nullptr)
                delete dynamicTexture;
            dynamicTexture = new Texture(GL_TEXTURE_2D);
            dynamicTexture->init(sd_outwidth, sd_outheight, 3);
            dynamicTexture->load(img2img_data.data(), true, GL_RGB);
            // bake dynamic texture
            bake_fbo_right.bind(true);
            /* hand */
            glDisable(GL_CULL_FACE); // todo: why disbale backface ? doesn't this mean bake will reach the back of the hand ?
            glEnable(GL_DEPTH_TEST);
            skinnedShader.use();
            skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform * global_scale_right);
            skinnedShader.setBool("useProjector", true);
            skinnedShader.setBool("bake", true);
            skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform * global_scale_right);
            skinnedShader.setBool("flipTexVertically", true);
            skinnedShader.setBool("renderUV", false);
            skinnedShader.setBool("useGGX", false);
            skinnedShader.setInt("src", 0);
            rightHandModel.Render(skinnedShader, bones_to_world_right_bake, rotx, false, dynamicTexture);
            bake_fbo_right.unbind();
            bake_fbo_left.bind();
            /* hand */
            skinnedShader.use();
            skinnedShader.SetWorldTransform(cam_projection_transform * cam_view_transform * global_scale_left);
            skinnedShader.setBool("useProjector", true);
            skinnedShader.setBool("bake", true);
            skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform * global_scale_left);
            skinnedShader.setBool("flipTexVertically", true);
            skinnedShader.setBool("renderUV", false);
            skinnedShader.setBool("useGGX", false);
            skinnedShader.setInt("src", 0);
            leftHandModel.Render(skinnedShader, bones_to_world_left_bake, rotx, false, dynamicTexture);
            /* debug points */
            // vcolorShader.use();
            // vcolorShader.setMat4("MVP", glm::mat4(1.0f));
            // std::vector<glm::vec2> points;
            // rightHandModel.getUnrolledTexCoords(points);
            // Helpers::UV2NDC(points);
            // std::vector<glm::vec3> screen_vert_color = {{1.0f, 0.0f, 0.0f}};
            // PointCloud cloud(points, screen_vert_color);
            // cloud.render();
            bake_fbo_left.unbind();
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
            // glDisable(GL_DEPTH_TEST);
            if (saveIntermed)
            {
                bake_fbo_right.saveColorToFile(bakeFileRight);
                bake_fbo_left.saveColorToFile(bakeFileLeft);
            }
            // if (bakedTexture != nullptr)
            // {
            //     delete bakedTexture;
            //     bakedTexture = nullptr;
            // }
            // bakedTexture = new Texture(bakeFile.c_str(), GL_TEXTURE_2D);
            // bakedTexture->init();
            sd_succeed = false;
            run_sd.join();
        }
    }
}

void handleMLS(Shader &gridShader)
{
    if (use_mls) // todo: support two hands
    {
        if (!mls_running && !mls_succeed)
        {
            if (joints_left.size() > 0)
            {
                t_mls_thread.start();
                std::vector<glm::vec3> to_project = joints_left;
                bool any_visible = false;
                std::vector<float> rendered_depths;
                std::vector<glm::vec3> projected_with_depth;
                if (mls_depth_test)
                {
                    projected_with_depth = Helpers::project_points_w_depth(to_project,
                                                                           glm::mat4(1.0f),
                                                                           gl_camera.getViewMatrix(),
                                                                           gl_camera.getProjectionMatrix());
                    std::vector<glm::vec2> screen_space = Helpers::NDCtoScreen(Helpers::vec3to2(projected_with_depth), dst_width, dst_height, false);
                    rendered_depths = hands_fbo.sampleDepthBuffer(screen_space); // todo: make async
                }
                if (run_mls.joinable())
                    run_mls.join();
                mls_running = true;
                camImage = camImageOrig.clone();
                run_mls = std::thread([to_project, rendered_depths, projected_with_depth]() { // send (forecasted) leap joints to MP
                    try
                    {
                        // Timer mls_profile;
                        // mls_profile.start();
                        std::vector<glm::vec2> projected = Helpers::project_points(to_project, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                        // mls_profile.stop();
                        // std::cout << "MLS projection time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                        // mls_profile.start();
                        std::vector<glm::vec2> cur_pred_glm = mp_predict(camImage, totalFrameCount);
                        std::vector<glm::vec2> pred_glm, kalman_forecast; // todo preallocate
                        std::vector<glm::vec2> projected_diff(projected.size(), glm::vec2(0.0f, 0.0f));
                        // mls_profile.stop();
                        // std::cout << "MLS prediction time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                        if (cur_pred_glm.size() == 21)
                        {
                            prev_pred_glm.push_back(cur_pred_glm);
                            pred_glm = Helpers::accumulate(prev_pred_glm);
                            int diff = prev_pred_glm.size() - mls_smooth_window;
                            if (diff > 0)
                            {
                                for (int i = 0; i < diff; i++)
                                    prev_pred_glm.erase(prev_pred_glm.begin());
                            }
                            if (use_mp_kalman)
                            {
                                for (int i = 0; i < pred_glm.size(); i++)
                                {
                                    cv::Mat pred = kalman_filters[i].predict();
                                    cv::Mat measurement(2, 1, CV_32F);
                                    measurement.at<float>(0) = pred_glm[i].x;
                                    measurement.at<float>(1) = pred_glm[i].y;
                                    cv::Mat corr = kalman_filters[i].correct(measurement);
                                    cv::Mat forecast = kalman_filters[i].forecast(kalman_lookahead);
                                    if (i == 0)
                                    {
                                        std::cout << "pred: " << pred << std::endl;
                                        std::cout << "meas: " << measurement << std::endl;
                                        std::cout << "corr: " << corr << std::endl;
                                        std::cout << "forecast: " << forecast << std::endl;
                                    }
                                    kalman_forecast.push_back(glm::vec2(forecast.at<float>(0), forecast.at<float>(1)));
                                    pred_glm = kalman_forecast;
                                }
                                // pred_glm = kalman_forecast;
                            }
                            std::vector<cv::Point2f> leap_keypoints, diff_keypoints, mp_keypoints;
                            if (mls_forecast)
                            {
                                // fix prediction with current leap info
                                // sync leap clock
                                std::modf(t_app.getElapsedTimeInMicroSec(), &whole);
                                LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
                                std::modf(t_app.getElapsedTimeInMicroSec(), &whole);
                                LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
                                std::vector<glm::mat4> cur_left_bones, cur_right_bones;
                                std::vector<glm::vec3> cur_vertices_left, cur_vertices_right;
                                LEAP_STATUS status = getLeapFrame(leap, targetFrameTime, cur_left_bones, cur_right_bones, cur_vertices_left, cur_vertices_right, leap_poll_mode, lastFrameID);
                                if (status == LEAP_STATUS::LEAP_NEWFRAME && cur_vertices_left.size() > 0)
                                {
                                    std::vector<glm::vec2> projected_new = Helpers::project_points(cur_vertices_left, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                                    for (int i = 0; i < projected_new.size(); i++)
                                    {
                                        projected_diff[i] += projected_new[i] - projected[i];
                                    }
                                    leap_keypoints = Helpers::glm2opencv(projected_new);
                                }
                                else
                                {
                                    leap_keypoints = Helpers::glm2opencv(projected);
                                }
                            }
                            else
                            {
                                leap_keypoints = Helpers::glm2opencv(projected);
                            }
                            diff_keypoints = Helpers::glm2opencv(projected_diff);
                            mp_keypoints = Helpers::glm2opencv(pred_glm);
                            // mls_profile.start();
                            ControlPointsP.clear();
                            ControlPointsQ.clear();
                            std::vector<bool> visible_landmarks(leap_keypoints.size(), true);
                            for (int i = 0; i < projected_with_depth.size(); i++)
                            {
                                float rendered_depth = rendered_depths[i];
                                float projected_depth = projected_with_depth[i].z;
                                float cam_near = 1.0f;
                                float cam_far = 1500.0f;
                                rendered_depth = (2.0 * rendered_depth) - 1.0; // logarithmic NDC
                                rendered_depth = (2.0 * cam_near * cam_far) / (cam_far + cam_near - (rendered_depth * (cam_far - cam_near)));
                                projected_depth = (2.0 * projected_depth) - 1.0; // logarithmic NDC
                                projected_depth = (2.0 * cam_near * cam_far) / (cam_far + cam_near - (projected_depth * (cam_far - cam_near)));
                                if ((std::abs(rendered_depth - projected_depth) <= mls_depth_threshold) && (i != 1)) // always include wrist
                                {
                                    visible_landmarks[i] = false;
                                }
                            }
                            for (int i = 0; i < leap_selection_vector.size(); i++)
                            {
                                if (visible_landmarks[leap_selection_vector[i]])
                                {
                                    ControlPointsP.push_back(leap_keypoints[leap_selection_vector[i]]);
                                    ControlPointsQ.push_back(mp_keypoints[mp_selection_vector[i]] + diff_keypoints[mp_selection_vector[i]]);
                                }
                            }
                            // deform grid using prediction
                            if (ControlPointsP.size() > 0)
                            {
                                // deform grid using prediction
                                // todo: refactor control points to avoid this part
                                cv::Mat p = cv::Mat::zeros(2, ControlPointsP.size(), CV_32F);
                                cv::Mat q = cv::Mat::zeros(2, ControlPointsQ.size(), CV_32F);
                                for (int i = 0; i < ControlPointsP.size(); i++)
                                {
                                    p.at<float>(0, i) = (ControlPointsP.at(i)).x;
                                    p.at<float>(1, i) = (ControlPointsP.at(i)).y;
                                }
                                for (int i = 0; i < ControlPointsQ.size(); i++)
                                {
                                    q.at<float>(0, i) = (ControlPointsQ.at(i)).x;
                                    q.at<float>(1, i) = (ControlPointsQ.at(i)).y;
                                }
                                // mls_profile.stop();
                                // std::cout << "MLS precomputation time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                                // mls_profile.start();
                                // weights
                                cv::Mat w = MLSprecomputeWeights(p, deformationGrid.getM(), mls_alpha);
                                // mls_profile.stop();
                                // std::cout << "MLS weight time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                                // affine
                                // mls_profile.start();
                                cv::Mat A = MLSprecomputeAffine(p, deformationGrid.getM(), w);
                                // mls_profile.stop();
                                // std::cout << "MLS affine time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                                // mls_profile.start();
                                cv::Mat fv = MLSPointsTransformAffine(w, A, q);
                                deformationGrid.constructDeformedGrid(fv);
                                // mls_profile.stop();
                                // std::cout << "MLS transform time: " << mls_profile.getElapsedTimeInMilliSec() << std::endl;
                            }
                            else
                            {
                                deformationGrid.constructGrid();
                            }
                            mls_succeed = true;
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    // std::cout << "mls thread dying !" << std::endl;
                    mls_running = false;
                });
            } // if (skeleton_vertices.size() > 0)
        }     // if (!mls_running && !mls_succeed)
        if (mls_succeed)
        {
            deformationGrid.updateGLBuffers();
            mls_succeed = false;
            ControlPointsP_glm = Helpers::opencv2glm(ControlPointsP);
            ControlPointsQ_glm = Helpers::opencv2glm(ControlPointsQ);
            if (run_mls.joinable())
                run_mls.join();
            t_mls_thread.stop();
        }
        // render as post process
        mls_fbo.bind();
        // PointCloud cloud_src(ControlPointsP_glm, screen_verts_color_red);
        // PointCloud cloud_dst(ControlPointsQ_glm, screen_verts_color_green);
        // vcolorShader.use();
        // vcolorShader.setMat4("MVP", glm::mat4(1.0f));
        // cloud_src.render();
        // cloud_dst.render();
        glDisable(GL_CULL_FACE); // todo: why is this necessary? flip grid triangles...
        if (postprocess_mode == static_cast<int>(PostProcessMode::JUMP_FLOOD_UV))
            uv_fbo.getTexture()->bind();
        else
            hands_fbo.getTexture()->bind();
        gridShader.use();
        gridShader.setBool("flipVer", false);
        deformationGrid.render();
        // gridColorShader.use();
        // deformationGrid.renderGridLines();
        mls_fbo.unbind(); // mls_fbo
        glEnable(GL_CULL_FACE);
    }
}

void handleDebugMode(SkinningShader &skinnedShader,
                     Shader &textureShader,
                     Shader &vcolorShader,
                     Shader &textShader,
                     Shader &lineShader,
                     SkinnedModel &rightHandModel,
                     SkinnedModel &leftHandModel,
                     Text &text)
{
    if (debug_mode && operation_mode == static_cast<int>(OperationMode::NORMAL))
    {
        glm::mat4 proj_view_transform = gl_projector.getViewMatrix();
        glm::mat4 proj_projection_transform = gl_projector.getProjectionMatrix();
        glm::mat4 flycam_view_transform = gl_flycamera.getViewMatrix();
        glm::mat4 flycam_projection_transform = gl_flycamera.getProjectionMatrix();
        glm::mat4 cam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 cam_projection_transform = gl_camera.getProjectionMatrix();
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
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_right);
                    skinnedShader.setBool("useProjector", false);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
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
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_right);
                    skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform * global_scale_right);
                    skinnedShader.setBool("useProjector", true);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
                    skinnedShader.setBool("flipTexVertically", false);
                    skinnedShader.setInt("src", 0);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, dynamicTexture);
                    break;
                }
                case static_cast<int>(TextureMode::BAKED):
                {
                    skinnedShader.use();
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_right);
                    skinnedShader.setBool("useProjector", false);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
                    skinnedShader.setBool("flipTexVertically", false);
                    skinnedShader.setInt("src", 0);
                    rightHandModel.Render(skinnedShader, bones_to_world_right, rotx, false, bake_fbo_right.getTexture());
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
                std::vector<glm::mat4> BoneToLocalTransforms;
                leftHandModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                glBindVertexArray(gizmoVAO);
                // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                {
                    // in bind pose
                    vcolorShader.setMat4("MVP", flycam_projection_transform * flycam_view_transform * rotx * BoneToLocalTransforms[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                for (unsigned int i = 0; i < bones_to_world_left.size(); i++)
                {
                    // in leap motion pose
                    vcolorShader.setMat4("MVP", flycam_projection_transform * flycam_view_transform * bones_to_world_left[i]);
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
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_left);
                    skinnedShader.setBool("useProjector", false);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
                    skinnedShader.setBool("flipTexVertically", false);
                    skinnedShader.setInt("src", 0);
                    leftHandModel.Render(skinnedShader, bones_to_world_left, rotx);
                    break;
                }
                case static_cast<int>(TextureMode::FROM_FILE):
                {
                    break;
                }
                case static_cast<int>(TextureMode::PROJECTIVE):
                {
                    skinnedShader.use();
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_left);
                    skinnedShader.setMat4("projTransform", cam_projection_transform * cam_view_transform * global_scale_left);
                    skinnedShader.setBool("useProjector", true);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
                    skinnedShader.setBool("flipTexVertically", false);
                    skinnedShader.setInt("src", 0);
                    leftHandModel.Render(skinnedShader, bones_to_world_left, rotx, false, dynamicTexture);
                    break;
                }
                case static_cast<int>(TextureMode::BAKED):
                {
                    skinnedShader.use();
                    skinnedShader.SetWorldTransform(flycam_projection_transform * flycam_view_transform * global_scale_left);
                    skinnedShader.setBool("useProjector", false);
                    skinnedShader.setBool("bake", false);
                    skinnedShader.setBool("useGGX", false);
                    skinnedShader.setBool("renderUV", false);
                    skinnedShader.setBool("flipTexVertically", false);
                    skinnedShader.setInt("src", 0);
                    leftHandModel.Render(skinnedShader, bones_to_world_left, rotx, false, bake_fbo_left.getTexture());
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
                set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, glm::mat4(1.0f), flycam_projection_transform, flycam_view_transform);
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
                set_texture_shader(textureShader, false, false, false, false, masking_threshold, 0, glm::mat4(1.0f), flycam_projection_transform, flycam_view_transform);
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
    }
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
            ImGui::Checkbox("Command Line Stats", &cmd_line_stats);
            ImGui::SameLine();
            if (ImGui::Checkbox("Freecam Mode", &freecam_mode))
            {
                create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            ImGui::Checkbox("PBO", &use_pbo);
            ImGui::Text("Operation Mode");
            if (ImGui::RadioButton("Normal", &operation_mode, 0))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Video", &operation_mode, 1))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                use_mls = false;
                exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Camera", &operation_mode, 2))
            {
                leap.setImageMode(false);
                exposure = 10000.0f;
                camera.set_exposure_time(exposure);
            }
            if (ImGui::RadioButton("Coaxial Calibration", &operation_mode, 3))
            {
                debug_mode = false;
                leap.setImageMode(false);
                exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(exposure);
            }
            if (ImGui::RadioButton("Leap Calibration", &operation_mode, 4))
            {
                projector.kill();
                use_projector = false;
                leap.setImageMode(true);
                leap.setPollMode(true);
                std::vector<uint8_t> buffer1, buffer2;
                while (!leap.getImage(buffer1, buffer2, leap_width, leap_height))
                    continue;
                // throttle down producer speed to allow smooth display
                // see https://docs.baslerweb.com/pylonapi/cpp/pylon_advanced_topics#grab-strategies
                exposure = 10000.0f;
                camera.set_exposure_time(exposure);
                debug_mode = false;
            }
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Calibration"))
        {
            switch (operation_mode)
            {
            case static_cast<int>(OperationMode::NORMAL):
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
            case static_cast<int>(OperationMode::CAMERA):
            {
                if (ImGui::Checkbox("Ready To Collect", &ready_to_collect))
                {
                    if (ready_to_collect)
                    {
                        imgpoints.clear();
                        calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                    }
                }
                float calib_progress = imgpoints.size() / (float)n_points_cam_calib;
                char buf[32];
                sprintf(buf, "%d/%d points", (int)(calib_progress * n_points_cam_calib), n_points_cam_calib);
                ImGui::ProgressBar(calib_progress, ImVec2(-1.0f, 0.0f), buf);
                if (ImGui::Button("Calibrate Camera"))
                {
                    if (imgpoints.size() >= n_points_cam_calib)
                    {
                        calibration_state = static_cast<int>(CalibrationStateMachine::CALIBRATE);
                    }
                }
                ImGui::Text("Camera Intrinsics");
                if (ImGui::BeginTable("Camera Matrix", 3))
                {
                    for (int row = 0; row < 3; row++)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%f", calib_cam_matrix[row * 3]);
                        ImGui::TableNextColumn();
                        ImGui::Text("%f", calib_cam_matrix[row * 3 + 1]);
                        ImGui::TableNextColumn();
                        ImGui::Text("%f", calib_cam_matrix[row * 3 + 2]);
                    }
                    ImGui::EndTable();
                }
                ImGui::Text("Camera Distortion");
                ImGui::Text("%f, %f, %f, %f, %f", calib_cam_distortion[0], calib_cam_distortion[1], calib_cam_distortion[2], calib_cam_distortion[3], calib_cam_distortion[4]);
                if (ImGui::Button("Save Camera Params"))
                {
                    cnpy::npz_save("../../resource/calibrations/cam_calibration/cam_calibration.npz", "cam_intrinsics", calib_cam_matrix, "a");
                    cnpy::npz_save("../../resource/calibrations/cam_calibration/cam_calibration.npz", "cam_distortion", calib_cam_distortion, "a");
                }
                break;
            }
            case static_cast<int>(OperationMode::COAXIAL):
            {
                if (ImGui::Button("Save Coaxial Calibration"))
                {
                    cnpy::npy_save("../../resource/calibrations/coaxial_calibration/coax_user.npy", cur_screen_verts.data(), {4, 2}, "w");
                }
                if (ImGui::BeginTable("Cam2Proj Vertices", 2))
                {
                    std::vector<glm::vec2> tmpVerts;
                    if (use_coaxial_calib)
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
            case static_cast<int>(OperationMode::LEAP):
            {
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
                ImGui::SliderInt("# Points to Collect", &n_points_leap_calib, 1000, 100000);
                ImGui::Checkbox("Use RANSAC", &leap_calib_use_ransac);
                ImGui::Text("Collection Procedure");
                ImGui::RadioButton("Manual Raw", &leap_collection_setting, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Manual Finger", &leap_collection_setting, 1);
                ImGui::SameLine();
                ImGui::RadioButton("Auto Raw", &leap_collection_setting, 2);
                ImGui::SameLine();
                ImGui::RadioButton("Auto Finger", &leap_collection_setting, 3);
                ImGui::SliderFloat("Binary Threshold", &leap_binary_threshold, 0.0f, 1.0f);
                if (ImGui::IsItemActive())
                {
                    leap_threshold_flag = true;
                }
                else
                {
                    leap_threshold_flag = false;
                }
                if (ImGui::Checkbox("Ready To Collect", &ready_to_collect))
                {
                    if (ready_to_collect)
                    {
                        points_2d.clear();
                        points_3d.clear();
                        points_2d_inliners.clear();
                        points_2d_reprojected.clear();
                        points_2d_inliers_reprojected.clear();
                        calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                    }
                }
                float calib_progress = points_2d.size() / (float)n_points_leap_calib;
                char buf[32];
                sprintf(buf, "%d/%d points", (int)(calib_progress * n_points_leap_calib), n_points_leap_calib);
                ImGui::ProgressBar(calib_progress, ImVec2(-1.0f, 0.0f), buf);
                ImGui::Text("cur. triangulated: %05.1f, %05.1f, %05.1f", triangulated.x, triangulated.y, triangulated.z);
                if (joints_right.size() > 0)
                {
                    ImGui::Text("cur. skeleton: %05.1f, %05.1f, %05.1f", joints_right[mark_bone_index].x, joints_right[mark_bone_index].y, joints_right[mark_bone_index].z);
                    float distance = glm::l2Norm(joints_right[mark_bone_index] - triangulated);
                    ImGui::Text("diff: %05.2f", distance);
                }
                ImGui::SliderInt("Selected Bone Index", &mark_bone_index, 0, 30);
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
                    if (points_2d.size() >= n_points_leap_calib)
                    {
                        calibration_state = static_cast<int>(CalibrationStateMachine::CALIBRATE);
                    }
                }
                if (calibrationSuccess)
                {
                    if (ImGui::Checkbox("Show Calib Reprojections", &showReprojections))
                    {
                        calibration_state = static_cast<int>(CalibrationStateMachine::SHOW);
                        showTestPoints = false;
                    }
                    if (ImGui::Checkbox("Show Test Points", &showTestPoints))
                    {
                        calibration_state = static_cast<int>(CalibrationStateMachine::MARK);
                        showReprojections = false;
                    }
                    if (showReprojections)
                    {
                        ImGui::Checkbox("Show only inliers", &showInliersOnly);
                    }
                    if (showTestPoints)
                    {
                        ImGui::RadioButton("Stream", &leap_mark_setting, 0);
                        ImGui::SameLine();
                        ImGui::RadioButton("Point by Point", &leap_mark_setting, 1);
                        ImGui::SameLine();
                        ImGui::RadioButton("Whole Hand", &leap_mark_setting, 2);
                        ImGui::SameLine();
                        ImGui::RadioButton("Single Bone", &leap_mark_setting, 3);
                        // ImGui::ListBox("listbox", &item_current, items, IM_ARRAYSIZE(items), 4);
                        if (leap_mark_setting == static_cast<int>(LeapMarkSettings::POINT_BY_POINT))
                        {
                            if (ImGui::BeginTable("Triangulated", 2))
                            {
                                for (int row = 0; row < triangulated_marked.size(); row++)
                                {
                                    ImGui::TableNextRow();
                                    ImGui::TableNextColumn();
                                    ImGui::Text("Vert %d", row);
                                    ImGui::TableNextColumn();
                                    ImGui::Text("%f, %f, %f", triangulated_marked[row][0], triangulated_marked[row][1], triangulated_marked[row][2]);
                                }
                                ImGui::EndTable();
                            }
                        }
                    }
                    if (!showReprojections && !showTestPoints)
                        calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
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
            if (ImGui::Checkbox("Use Coaxial Calib", &use_coaxial_calib))
            {
                if (use_coaxial_calib)
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
            if (ImGui::SliderFloat("Camera Exposure [us]", &exposure, 300.0f, 10000.0f))
            {
                // std::cout << "current exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
                camera.set_exposure_time(exposure);
                // std::cout << "new exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
            }
            ImGui::SliderFloat("Masking Threshold", &masking_threshold, 0.0f, 1.0f);
            if (ImGui::IsItemActive())
            {
                threshold_flag = true;
            }
            else
            {
                threshold_flag = false;
            }
            if (ImGui::Button("Screen Shot"))
            {
                std::string name = std::tmpnam(nullptr);
                fs::path filename(name);
                std::string savepath(std::string("../../debug/"));
                // std::cout << "unique file name: " << filename.filename().string() << std::endl;
                fs::path raw_image(savepath + filename.filename().string() + std::string("_raw_cam.png"));
                fs::path mask_path(savepath + filename.filename().string() + std::string("_mask.png"));
                fs::path render(savepath + filename.filename().string() + std::string("_render.png"));
                fs::path uv(savepath + filename.filename().string() + std::string("_uv.png"));
                fs::path mls(savepath + filename.filename().string() + std::string("_mls_fbo.png"));
                fs::path pp(savepath + filename.filename().string() + std::string("_pp.png"));
                fs::path coax(savepath + filename.filename().string() + std::string("_coax.png"));
                fs::path keypoints(savepath + filename.filename().string() + std::string("_keypoints.npy"));
                cv::Mat tmp, mask;
                cv::flip(camImageOrig, tmp, 1); // flip horizontally
                cv::resize(tmp, tmp, cv::Size(proj_width, proj_height));
                cv::threshold(tmp, mask, static_cast<int>(masking_threshold * 255), 255, cv::THRESH_BINARY);
                cv::imwrite(raw_image.string(), tmp);
                cv::imwrite(mask_path.string(), mask);
                hands_fbo.saveColorToFile(render.string());
                uv_fbo.saveColorToFile(uv.string());
                mls_fbo.saveColorToFile(mls.string());
                postprocess_fbo.saveColorToFile(pp.string());
                c2p_fbo.saveColorToFile(coax.string());
                if (joints_right.size() > 0)
                {
                    std::vector<glm::vec2> projected = Helpers::project_points(joints_right, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                    cnpy::npy_save(keypoints.string().c_str(), &projected[0].x, {projected.size(), 2}, "w");
                }
                if (joints_left.size() > 0)
                {
                    std::vector<glm::vec2> projected = Helpers::project_points(joints_left, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                    cnpy::npy_save(keypoints.string().c_str(), &projected[0].x, {projected.size(), 2}, "w");
                }
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
            ImGui::Checkbox("Undistort Camera Input", &undistortCamera);
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
            ImGui::Text("Camera Projection Matrix (column major, OpenGL convention)");
            if (ImGui::BeginTable("CamProj", 4))
            {
                glm::mat4 proj = gl_camera.getProjectionMatrix();
                for (int row = 0; row < 4; row++)
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", proj[row][0]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", proj[row][1]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", proj[row][2]);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", proj[row][3]);
                }
                ImGui::EndTable();
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
            ImGui::RadioButton("Wireframe", &material_mode, 2);

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
            ImGui::InputText("Baked Texture File (right)", &bakeFileRight);
            ImGui::InputText("Baked Texture File (left)", &bakeFileLeft);
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
            ImGui::Checkbox("MLS", &use_mls);
            ImGui::SameLine();
            ImGui::Checkbox("Forecast", &mls_forecast);
            ImGui::Checkbox("Kalman Filter", &use_mp_kalman);
            ImGui::SliderInt("Kalman lookahead", &kalman_lookahead, 0, 100);
            ImGui::SliderFloat("Kalman Pnoise", &mp_kalman_process_noise, 0.00001f, 1.0f, "%.6f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                kalman_filters.clear();
                kalman_filters = std::vector<Kalman2D>(21, Kalman2D(mp_kalman_process_noise, mp_kalman_measurement_noise, mp_kalman_error));
            }
            ImGui::SliderFloat("Kalman Mnoise", &mp_kalman_measurement_noise, 0.00001f, 1.0f, "%.6f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                kalman_filters.clear();
                kalman_filters = std::vector<Kalman2D>(21, Kalman2D(mp_kalman_process_noise, mp_kalman_measurement_noise, mp_kalman_error));
            }
            ImGui::SliderFloat("Kalman err", &mp_kalman_error, 0.01f, 10.0f);
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                kalman_filters.clear();
                kalman_filters = std::vector<Kalman2D>(21, Kalman2D(mp_kalman_process_noise, mp_kalman_measurement_noise, mp_kalman_error));
            }
            ImGui::SliderInt("MLS smooth window", &mls_smooth_window, 0, 10);
            ImGui::Checkbox("Show MLS landmarks", &show_mls_landmarks);
            ImGui::SliderFloat("MLS alpha", &mls_alpha, 0.01f, 5.0f);
            // ImGui::SliderFloat("MLS grab threshold", &mls_grab_threshold, -1.0f, 5.0f);
            ImGui::Checkbox("MLS depth test?", &mls_depth_test);
            ImGui::SameLine();
            ImGui::SliderFloat("d. thre", &mls_depth_threshold, 1.0f, 1500.0f, "%.6f");
            ImGui::SliderFloat("Masking threshold", &masking_threshold, 0.0f, 1.0f);
            if (ImGui::IsItemActive())
            {
                threshold_flag = true;
            }
            else
            {
                threshold_flag = false;
            }
            ImGui::Text("Post Processing Mode");
            ImGui::RadioButton("Render Only", &postprocess_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Camera Feed", &postprocess_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Overlay", &postprocess_mode, 2);
            ImGui::RadioButton("Jump Flood", &postprocess_mode, 3);
            ImGui::SameLine();
            ImGui::RadioButton("Jump Flood UV", &postprocess_mode, 4);
            ImGui::SameLine();
            ImGui::RadioButton("ICP", &postprocess_mode, 5);
            if (postprocess_mode == static_cast<int>(PostProcessMode::ICP))
            {
                ImGui::Checkbox("ICP on?", &icp_apply_transform);
            }
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Leap Controls"))
        {
            if (ImGui::Checkbox("Leap Polling Mode", &leap_poll_mode))
            {
                leap.setPollMode(leap_poll_mode);
            }
            ImGui::Checkbox("Record Session", &leap_record_session);
            if (ImGui::Checkbox("Use Recorded Session", &use_pre_recorded_session))
            {
                if (session_name != loaded_session_name)
                {
                    if (loadPrerecordedSession())
                        pre_recorded_session_loaded = true;
                    else
                        std::cout << "Failed to load prerecorded session: " << session_name << std::endl;
                }
            }
            ImGui::SameLine();
            ImGui::InputText("Session Name", &session_name);
            ImGui::SliderFloat("Desired Latency [ms]", &required_latency_ms, 1.0f, 50.0f);
            if (ImGui::Checkbox("Playback Video", &playback_video))
            {
                if (playback_video_name != loaded_playback_video_name)
                {
                    if (loadVideo())
                        playback_video_loaded = true;
                    else
                        std::cout << "Failed to load video: " << session_name << std::endl;
                }
            }
            ImGui::SameLine();
            ImGui::InputText("Video Name", &playback_video_name);
            ImGui::SameLine();
            if (ImGui::RadioButton("Desktop", &leap_tracking_mode, 0))
            {
                leap.setTrackingMode(eLeapTrackingMode_Desktop);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("HMD", &leap_tracking_mode, 1))
            {
                leap.setTrackingMode(eLeapTrackingMode_ScreenTop);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Screentop", &leap_tracking_mode, 2))
            {
                leap.setTrackingMode(eLeapTrackingMode_HMD);
            }
            ImGui::Checkbox("Use Finger Width", &useFingerWidth);
            ImGui::SliderInt("Leap Prediction [us]", &magic_leap_time_delay, -50000, 50000);
            ImGui::SliderFloat("Leap Global Scale", &leap_global_scaler, 0.1f, 10.0f);
            ImGui::SliderFloat("Leap Bone Scale", &magic_leap_scale_factor, 1.0f, 20.0f);
            ImGui::SliderFloat("Leap Wrist Offset", &magic_wrist_offset, -100.0f, 100.0f);
            ImGui::SliderFloat("Leap Arm Offset", &magic_arm_forward_offset, -300.0f, 200.0f);
            ImGui::SliderFloat("Leap Local Bone Scale", &leap_bone_local_scaler, 0.001f, 0.1f);
            ImGui::SliderFloat("Leap Palm Scale", &leap_palm_local_scaler, 0.001f, 0.1f);
            ImGui::SliderFloat("Leap Arm Scale", &leap_arm_local_scaler, 0.001f, 0.1f);
            ImGui::TreePop();
        }
    }
    ImGui::End();
}