#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>
#include <filesystem>
#include "texture.h"
#include "display.h"
#include "readerwritercircularbuffer.h"
#include "cxxopts.h"
#include "camera.h"
#include "gl_camera.h"
#include "shader.h"
#include "engine_state.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "point_cloud.h"
#include "leapCPP.h"
#include "stubs.h"
#ifdef OPENCV_WITH_CUDA
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#endif
#include "text.h"
#include "post_process.h"
#include "utils.h"
#include "cnpy.h"
#include "helpers.h"
#include "dear_widgets.h"
#include "imgui.h"
#include "kalman.h"
#include "imgui_stdlib.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "diffuse.h"
#include "user_study.h"
#include "guess_pose_game.h"
#include "guess_char_game.h"
#include "guess_animal_game.h"
#include "grid.h"
#include "moving_least_squares.h"
#include "MidiControllerAPI.h"
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
namespace fs = std::filesystem;

/* forward declarations */
void openIMGUIFrame();
void handleCameraInput(CGrabResultPtr ptrGrabResult, bool simulatedCam, cv::Mat simulatedImage);
LEAP_STATUS handleLeapInput();
LEAP_STATUS handleLeapInput(int num_frames);
void saveLeapData(LEAP_STATUS leap_status, uint64_t image_timestamp, bool record_images);
void saveSession(std::string savepath, bool record_images);
void saveSession(std::string savepath, LEAP_STATUS leap_status, uint64_t image_timestamp, bool record_images);
bool loadGamePoses(std::string loadPath, std::vector<std::vector<glm::mat4>> &poses);
void handlePostProcess(SkinnedModel &leftHandModel,
                       SkinnedModel &rightHandModel,
                       Texture &camTexture,
                       std::unordered_map<std::string, Shader *> &shader_map);
void handleSkinning(const std::vector<glm::mat4> &bones2world,
                    bool isRightHand,
                    bool isFirstHand,
                    std::unordered_map<std::string, Shader *> &shader_map,
                    SkinnedModel &handModel,
                    glm::mat4 cam_view_transform,
                    glm::mat4 cam_projection_transform);
void handleBaking(std::unordered_map<std::string, Shader *> &shader_map,
                  SkinnedModel &leftHandModel,
                  SkinnedModel &rightHandModel,
                  glm::mat4 cam_view_transform,
                  glm::mat4 cam_projection_transform);
void handleBakeConfig();
void handleBakingInternal(std::unordered_map<std::string, Shader *> &shader_map,
                          Texture &bakeTexture,
                          SkinnedModel &leftHandModel,
                          SkinnedModel &rightHandModel,
                          glm::mat4 cam_view_transform,
                          glm::mat4 cam_projection_transform,
                          bool flipVertical,
                          bool flipHorizontal,
                          bool projSingleChannel,
                          bool ignoreGlobalScale);
void handleMLS(Shader &gridShader, bool blocking = false, bool detect_landmarks = true, bool new_frame = true, bool simulation = false);
void handleOF(Shader *gridShader);
void landmarkDetection(bool blocking = false);
void landmarkDetectionThread(std::vector<glm::vec3> projected_filtered_left,
                             std::vector<float> rendered_depths_left,
                             bool isLeftHandVis,
                             std::vector<glm::vec3> projected_filtered_right,
                             std::vector<float> rendered_depths_right,
                             bool isRightHandVis);
void projectAndFilterJoints(const std::vector<glm::vec3> &raw_joints_left,
                            const std::vector<glm::vec3> &raw_joints_right,
                            std::vector<glm::vec3> &out_joints_left,
                            std::vector<glm::vec3> &out_joints_right);
cv::Mat computeGridDeformation(const std::vector<cv::Point2f> &P,
                               const std::vector<cv::Point2f> &Q,
                               int deformation_mode, float alpha,
                               Grid &grid);
void handleDebugMode(std::unordered_map<std::string, Shader *> &shader_map,
                     SkinnedModel &rightHandModel,
                     SkinnedModel &leftHandModel,
                     SkinnedModel &otherObject,
                     TextModel &text);
void handleGuessPoseGame(std::unordered_map<std::string, Shader *> &shaderMap,
                         SkinnedModel &leftHandModel,
                         SkinnedModel &rightHandModel,
                         Quad &topRightQuad,
                         glm::mat4 &cam_view_transform,
                         glm::mat4 &cam_projection_transform);
void handleGuessCharGame(std::unordered_map<std::string, Shader *> &shaderMap,
                         SkinnedModel &leftHandModel,
                         SkinnedModel &rightHandModel,
                         TextModel &textModel,
                         glm::mat4 &cam_view_transform,
                         glm::mat4 &cam_projection_transform);
void handleGuessAnimalGame(std::unordered_map<std::string, Shader *> &shaderMap,
                           SkinnedModel &leftHandModel,
                           SkinnedModel &rightHandModel,
                           TextModel &textModel,
                           glm::mat4 &cam_view_transform,
                           glm::mat4 &cam_projection_transform);
void handleUserStudy(std::unordered_map<std::string, Shader *> &shader_map,
                     GLFWwindow *window,
                     SkinnedModel &leftHandModel,
                     SkinnedModel &rightHandModel,
                     TextModel &textModel,
                     glm::mat4 &cam_view_transform,
                     glm::mat4 &cam_projection_transform);
void handleSimulation(std::unordered_map<std::string, Shader *> &shaderMap,
                      SkinnedModel &leftHandModel,
                      SkinnedModel &rightHandModel,
                      TextModel &textModel,
                      glm::mat4 &cam_view_transform,
                      glm::mat4 &cam_projection_transform);
bool interpolateBones(float time, std::vector<glm::mat4> &bones_out, const std::vector<glm::mat4> &session, bool isRightHand);
bool getSkeletonByTimestamp(uint32_t timestamp,
                            std::vector<glm::mat4> &bones_out,
                            std::vector<glm::vec3> &joints_out,
                            const std::vector<glm::mat4> &bones_session,
                            const std::vector<glm::vec3> &joints_session,
                            bool isRightHand);
bool handleGetSkeletonByTimestamp(uint32_t timestamp,
                                  std::vector<glm::mat4> &bones2world_left,
                                  std::vector<glm::vec3> &joints_left,
                                  std::vector<glm::mat4> &bones2world_right,
                                  std::vector<glm::vec3> &joints_right);
bool handleInterpolateFrames(std::vector<glm::mat4> &bones2world_left_cur,
                             std::vector<glm::mat4> &bones2world_right_cur,
                             std::vector<glm::mat4> &bones2world_left_lag,
                             std::vector<glm::mat4> &bones2world_right_lag);
bool mp_predict(cv::Mat origImage, int timestamp, std::vector<glm::vec2> &left, std::vector<glm::vec2> &right, bool &detected_left, bool &detected_right);
bool mp_predict_single(cv::Mat origImage, std::vector<glm::vec2> &left, std::vector<glm::vec2> &right, bool &left_detected, bool &right_detected);
void create_virtual_cameras(GLCamera &gl_flycamera, GLCamera &gl_projector, GLCamera &gl_camera);
void getLightTransform(glm::mat4 &lightProjection,
                       glm::mat4 &lightView,
                       const std::vector<glm::mat4> &bones2world);
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
                         std::vector<uint32_t> &leftFingersExtended,
                         std::vector<uint32_t> &rightFingersExtended,
                         bool leap_poll_mode,
                         int64_t &curFrameID,
                         int64_t &curFrameTimeStamp);
LEAP_STATUS getLeapFramePreRecorded(std::vector<glm::mat4> &bones,
                                    std::vector<glm::vec3> &joints,
                                    uint64_t frameCounter,
                                    uint64_t totalFrameCount,
                                    const std::vector<glm::mat4> &bones_session,
                                    const std::vector<glm::vec3> &joints_session);
bool loadSession();
void loadImagesFromFolder(std::string loadpath);
bool playVideo(std::unordered_map<std::string, Shader *> &shader_map,
               SkinnedModel &leftHandModel,
               SkinnedModel &rightHandModel,
               TextModel &textModel,
               glm::mat4 &cam_view_transform,
               glm::mat4 &cam_projection_transform);
uint32_t getCurrentSimulationIndex();
bool playVideoReal(std::unordered_map<std::string, Shader *> &shader_map,
                   SkinnedModel &leftHandModel,
                   SkinnedModel &rightHandModel,
                   TextModel &textModel,
                   glm::mat4 &cam_view_transform,
                   glm::mat4 &cam_projection_transform);
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
void set_texture_shader(Shader *textureShader,
                        bool flipVer,
                        bool flipHor,
                        bool isGray,
                        bool binary = false,
                        float threshold = 0.035f,
                        int src = 0,
                        glm::mat4 model = glm::mat4(1.0f),
                        glm::mat4 projection = glm::mat4(1.0f),
                        glm::mat4 view = glm::mat4(1.0f),
                        bool gammaCorrection = false,
                        glm::vec3 bgColor = glm::vec3(0.0f));
void set_skinned_shader(SkinningShader *skinnedShader,
                        glm::mat4 transform,
                        bool flipVer = false, bool flipHor = false,
                        bool useGGX = false, bool renderUV = false,
                        bool bake = false, bool useProjector = false, bool projOnly = true, bool projIsSingleC = false,
                        glm::mat4 projTransform = glm::mat4(1.0f),
                        bool useMetric = false, std::vector<float> scalarPerBone = std::vector<float>(), int src = 0);
void initKalmanFilters();
std::vector<float> computeDistanceFromPose(const std::vector<glm::mat4> &bones_to_world, const std::vector<glm::mat4> &required_pose_bones_to_world);
/* global engine state */
EngineState es;
Timer t_camera, t_leap, t_skin, t_swap, t_download, t_app, t_misc, t_debug, t_pp, t_mls, t_of, t_mls_thread, t_bake, t_profile0, t_profile1;
std::thread sd_thread, mls_thread;
GuessPoseGame guessPoseGame = GuessPoseGame();
GuessCharGame guessCharGame = GuessCharGame();
GuessAnimalGame guessAnimalGame = GuessAnimalGame();
UserStudy user_study = UserStudy();
ControlNetClient controlNetClient = ControlNetClient();
// record & playback controls
std::vector<glm::vec2> filtered_cur, filtered_next, kalman_pred, kalman_corrected, kalman_forecast;
// const int max_supported_frames = 5000;
// static uint8_t *pFrameData[max_supported_frames] = {NULL};
std::vector<glm::mat4> session_bones_left;
std::vector<glm::mat4> session_bones_right;
std::vector<glm::vec3> session_joints_left;
std::vector<glm::vec3> session_joints_right;
std::vector<float> session_timestamps;
std::unordered_map<std::string, std::vector<glm::mat4>> sessions_bones_left;
std::unordered_map<std::string, std::vector<glm::vec3>> sessions_joints_left;
std::unordered_map<std::string, std::vector<float>> sessions_timestamps;
std::vector<float> raw_session_timestamps;
std::vector<std::vector<glm::mat4>> savedLeapBonesLeft;
std::vector<std::vector<glm::vec3>> savedLeapJointsLeft;
std::vector<std::vector<glm::mat4>> savedLeapBonesRight;
std::vector<std::vector<glm::vec3>> savedLeapJointsRight;
std::vector<int64_t> savedLeapTimestamps;
std::vector<cv::Mat> recordedImages;
std::vector<uint64_t> savedCameraTimestamps;
// leap controls
LeapCPP leap(es.leap_poll_mode, false, static_cast<_eLeapTrackingMode>(es.leap_tracking_mode), false);
// LEAP_CLOCK_REBASER clockSynchronizer;
std::vector<glm::vec3> joints_left, joints_right;
std::vector<glm::vec3> projected_filtered_left, projected_filtered_right;
std::vector<glm::mat4> bones_to_world_left, bones_to_world_right;
std::vector<uint32_t> left_fingers_extended, right_fingers_extended;
std::vector<glm::mat4> required_pose_bones_to_world_left;
std::vector<glm::mat4> bones_to_world_left_bake, bones_to_world_right_bake;
// calibration controls
std::vector<double> calib_cam_matrix;
std::vector<double> calib_cam_distortion;
std::vector<glm::vec3> points_3d;
std::vector<glm::vec2> points_2d, points_2d_inliners;
std::vector<glm::vec2> points_2d_reprojected, points_2d_inliers_reprojected;
std::vector<std::vector<cv::Point2f>> imgpoints;
cv::Mat camera_intrinsics_cv, camera_distortion_cv;
glm::vec2 marked_2d_pos1, marked_2d_pos2;
std::vector<glm::vec3> triangulated_marked;
std::vector<glm::vec2> marked_reprojected;
cv::Mat tvec_calib, rvec_calib;
// camera/projector controls
Display *projector = nullptr;
BaslerCamera camera;
moodycamel::BlockingReaderWriterCircularBuffer<CGrabResultPtr> camera_queue(20);
// moodycamel::BlockingReaderWriterCircularBuffer<uint8_t *> projector_queue(20);
GLCamera gl_flycamera;
GLCamera gl_projector;
GLCamera gl_camera;
cv::Mat camImagePrev;
cv::Mat flow, rawFlow;
cv::Mat camImage, camImageOrig, undistort_map1, undistort_map2;
#ifdef OPENCV_WITH_CUDA
cv::cuda::GpuMat gcur, gprev;
cv::cuda::GpuMat gflow;
cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbof;
cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> nvof;
#else
GPUMATStub gcur, gprev, gflow;
OFStub *fbof = nullptr;
OFStub *nvof = nullptr;
#endif
glm::mat4 w2c_auto, w2c_user;
glm::mat4 proj_project;
glm::mat4 cam_project;
std::vector<double> camera_distortion;
glm::mat4 c2p_homography;
// GL controls
std::unordered_map<std::string, Texture *> texturePack;
SkinnedModel *extraLeftHandModel = nullptr;
Texture *dynamicTexture = nullptr;
Texture *projectiveTexture = nullptr;
Texture *normalMap = nullptr;
Texture *armMap = nullptr;
Texture *dispMap = nullptr;
Texture *bakedTextureLeft = nullptr;
Texture *bakedTextureRight = nullptr;
Texture *OFTexture = nullptr;
Texture toBakeTexture;
Texture camTexture;
FBO hands_fbo(es.dst_width, es.dst_height, 4, false);
FBO game_fbo(es.dst_width, es.dst_height, 4, false);
FBO game_fbo_aux1(es.dst_width, es.dst_height, 4, false);
FBO game_fbo_aux2(es.dst_width, es.dst_height, 4, false);
FBO game_fbo_aux3(es.dst_width, es.dst_height, 4, false);
FBO fake_cam_fbo(es.dst_width, es.dst_height, 4, false);
FBO fake_cam_binary_fbo(es.dst_width, es.dst_height, 1, false);
FBO uv_fbo(es.dst_width, es.dst_height, 4, false);
FBO mls_fbo(es.dst_width, es.dst_height, 4, false);
FBO bake_fbo_right(1024, 1024, 4, false);
FBO bake_fbo_left(1024, 1024, 4, false);
FBO pre_bake_fbo(1024, 1024, 4, false);
FBO sd_fbo(512, 512, 4, false);
FBO shadowmap_fbo(1024, 1024, 1, false);
FBO postprocess_fbo(es.dst_width, es.dst_height, 4, false);
FBO postprocess_fbo2(es.dst_width, es.dst_height, 4, false);
FBO dynamic_fbo(es.dst_width, es.dst_height, 4, false);
FBO c2p_fbo(es.dst_width, es.dst_height, 4, false);
Quad fullScreenQuad(0.0f, false);
PostProcess postProcess(es.cam_width, es.cam_height, es.dst_width, es.dst_height);
DirectionalLight dirLight(glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, 1.0f, glm::vec3(0.0f, -1.0f, -1.0f));
unsigned int skeletonVAO = 0;
unsigned int skeletonVBO = 0;
unsigned int gizmoVAO = 0;
unsigned int gizmoVBO = 0;
unsigned int frustrumVAO = 0;
unsigned int frustrumVBO = 0;
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
unsigned int cubeEBO = 0;
// python controls
PyObject *myModule;
PyObject *predict_single;
PyObject *predict_video;
PyObject *init_detector;
PyObject *single_detector;
// mls controls
Grid deformationGrid(es.grid_x_point_count, es.grid_y_point_count, es.grid_x_spacing, es.grid_y_spacing);
std::vector<cv::Point2f> ControlPointsP;
std::vector<cv::Point2f> ControlPointsQ;
std::vector<cv::Point2f> prev_leap_keypoints;
// std::vector<glm::vec2> ControlPointsP_glm;
std::vector<glm::vec3> ControlPointsP_input_left, ControlPointsP_input_right;
// std::vector<glm::vec2> ControlPointsP_input_glm_left, ControlPointsP_input_glm_right;
std::vector<glm::vec2> mp_keypoints_left_result, mp_keypoints_right_result;
// std::vector<glm::vec2> ControlPointsQ_glm;
std::vector<std::vector<glm::vec2>> prev_pred_glm_left, prev_pred_glm_right;
std::vector<Kalman2D_ConstantV> kalman_filters_left = std::vector<Kalman2D_ConstantV>(16);
std::vector<Kalman2D_ConstantV> kalman_filters_right = std::vector<Kalman2D_ConstantV>(16);
std::vector<Kalman2D_ConstantV2> kalman_filters_vleft = std::vector<Kalman2D_ConstantV2>(16);
std::vector<Kalman2D_ConstantV2> kalman_filters_vright = std::vector<Kalman2D_ConstantV2>(16);
std::vector<glm::vec2> projected_left_prev, projected_right_prev;
// std::vector<Kalman2D_ConstantV> kalman_filters = std::vector<Kalman2D_ConstantV>(16);
// std::vector<Kalman2D_ConstantV2> grid_kalman = std::vector<Kalman2D_ConstantV2>(es.grid_x_point_count * es.grid_y_point_count);

/* main */
int main(int argc, char *argv[])
{
    t_app.start();
    /* parse cmd line options */
    cxxopts::Options options("casper", "casper.exe: A graphics engine for performing projection mapping onto human hands");
    options.add_options()                                                                                                        //
        ("mode", "the operation mode [normal, user_study, cam_calib, coax_calib, leap_calib, guess_char_game, guess_pose_game]", //
         cxxopts::value<std::string>()->default_value("normal"))                                                                 //
        ("mesh", "A .fbx mesh file to use for skinning",                                                                         //
         cxxopts::value<std::string>()->default_value("../../resource/Default.fbx"))                                             //
        ("simcam", "A simulated camera is used", cxxopts::value<bool>()->default_value("false"))                                 //
        ("simproj", "A simulated projector is used", cxxopts::value<bool>()->default_value("false"))                             //
        ("emesh", "A .fbx mesh file to use for hot swapping (used in some apps)",                                                //
         cxxopts::value<std::string>()->default_value("../../resource/GuessCharGame_palm.fbx"))                                  //
        ("h,help", "Prints usage")                                                                                               //
        ;
    try
    {
        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        es.meshFile = result["mesh"].as<std::string>();
        es.extraMeshFile = result["emesh"].as<std::string>();
        es.simulated_camera = result["simcam"].as<bool>();
        es.simulated_projector = result["simproj"].as<bool>();
        es.proj_channel_order = es.simulated_projector ? GL_RGB : GL_BGR;
        std::unordered_map<std::string, int> mode_map{
            {"normal", static_cast<int>(OperationMode::SANDBOX)},
            {"user_study", static_cast<int>(OperationMode::USER_STUDY)},
            {"cam_calib", static_cast<int>(OperationMode::CAMERA)},
            {"coax_calib", static_cast<int>(OperationMode::COAXIAL)},
            {"leap_calib", static_cast<int>(OperationMode::LEAP)},
            {"guess_char_game", static_cast<int>(OperationMode::GUESS_CHAR_GAME)},
            {"guess_pose_game", static_cast<int>(OperationMode::GUESS_POSE_GAME)},
            {"guess_animal_game", static_cast<int>(OperationMode::GUESS_ANIMAL_GAME)},
            {"simulation", static_cast<int>(OperationMode::SIMULATION)}};
        // check if mode is valid
        if (mode_map.find(result["mode"].as<std::string>()) == mode_map.end())
        {
            std::cout << "Invalid mode: " << result["mode"].as<std::string>() << std::endl;
            es.operation_mode = static_cast<int>(OperationMode::SANDBOX);
        }
        else
        {
            es.operation_mode = mode_map[result["mode"].as<std::string>()];
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cout << options.help() << std::endl;
        exit(0);
    }
    /* embedded python init */
    Py_Initialize();
    controlNetClient.init();
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
    GLFWwindow *window = glfwCreateWindow(es.proj_width, es.proj_height, "casper", NULL, NULL); // monitors[0], NULL for full screen
    int secondary_screen_x, secondary_screen_y;
    glfwGetMonitorPos(monitors[num_of_monitors - 1], &secondary_screen_x, &secondary_screen_y);
    glfwSetWindowPos(window, secondary_screen_x + 300, secondary_screen_y + 100);
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
    glfwSwapInterval(0);                             // do not sync to monitor
    glViewport(0, 0, es.proj_width, es.proj_height); // set viewport
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
    Helpers::setupFrustrumBuffers(frustrumVAO, frustrumVBO);
    Helpers::setupCubeBuffers(cubeVAO, cubeVBO, cubeEBO);
    unsigned int pbo[2] = {0};
    initGLBuffers(pbo);
    hands_fbo.init();
    game_fbo.init(GL_BGRA, GL_RGBA, GL_LINEAR, GL_CLAMP_TO_EDGE);
    game_fbo_aux1.init(GL_RGBA, GL_RGBA32F);
    game_fbo_aux2.init(GL_RGBA, GL_RGBA32F);
    game_fbo_aux3.init(GL_RGBA, GL_RGBA32F);
    fake_cam_fbo.init();
    fake_cam_binary_fbo.init();
    uv_fbo.init(GL_RGBA, GL_RGBA32F, GL_NEAREST); // stores uv coordinates, so must be 32F
    bake_fbo_right.init();
    bake_fbo_left.init();
    pre_bake_fbo.init();
    sd_fbo.init();
    shadowmap_fbo.initDepthOnly();
    postprocess_fbo.init();
    postprocess_fbo2.init();
    dynamic_fbo.init();
    mls_fbo.init(GL_RGBA, GL_RGBA32F); // will possibly store uv_fbo, so must be 32F
    c2p_fbo.init();
    std::cout << "Loading Meshes..." << std::endl;
    SkinnedModel leftHandModel(es.meshFile,
                               es.userTextureFile,
                               es.proj_width, es.proj_height,
                               es.cam_width, es.cam_height); // GenericHand.fbx is a left hand model
    SkinnedModel rightHandModel(es.meshFile,
                                es.userTextureFile,
                                es.proj_width, es.proj_height,
                                es.cam_width, es.cam_height,
                                false);
    SkinnedModel dinosaur("../../resource/reconst.ply",
                          "",
                          es.proj_width, es.proj_height,
                          es.cam_width, es.cam_height);
    switch (es.operation_mode) // set some default values for user study depending on cmd line
    {
    case static_cast<int>(OperationMode::SIMULATION):
    {
        es.postprocess_mode = static_cast<int>(PostProcessMode::OVERLAY);
        es.mask_missing_color_is_camera = true;
        es.mask_unused_info_color = es.mask_bg_color;
        es.mls_blocking = true;
        es.use_coaxial_calib = false;
        es.deformation_mode = static_cast<int>(DeformationMode::SIMILARITY);
        es.simulated_projector = true;
        es.simulated_camera = true;
        es.pseudo_vid_playback_speed = 0.02f;
        if (!loadSession())
        {
            std::cout << "Failed to load recording: " << es.recording_name << std::endl;
        }
        else
        {
            es.debug_playback = true;
            es.simulationTime = 0.0f;
            es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
        }
        break;
    }
    case static_cast<int>(OperationMode::USER_STUDY):
    {
        es.postprocess_mode = static_cast<int>(PostProcessMode::NONE);
        es.vid_simulated_latency_ms = 35.0f;
        es.jfa_distance_threshold = 100.0f;
        es.use_mls = false;
        break;
    }
    case static_cast<int>(OperationMode::GUESS_ANIMAL_GAME):
    {
        es.postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD_UV);
        es.texture_mode = static_cast<int>(TextureMode::BAKED);
        es.material_mode = static_cast<int>(MaterialMode::DIFFUSE);
        es.bake_mode = static_cast<int>(BakeMode::CONTROL_NET);
        es.prompt_mode = static_cast<int>(PromptMode::AUTO_PROMPT);
        es.no_preprompt = false;
        // es.cur_prompt = es.selected_listed_prompt;
        // es.use_mls = false;
        // es.use_of = false;
        break;
    }
    case static_cast<int>(OperationMode::GUESS_CHAR_GAME):
    {
        extraLeftHandModel = new SkinnedModel(es.extraMeshFile,
                                              es.userTextureFile,
                                              es.proj_width, es.proj_height,
                                              es.cam_width, es.cam_height);
        break;
    }
    default:
        break;
    }
    // load all image files from supplied texture paths to GPU
    std::cout << "Loading Textures..." << std::endl;
    for (auto &it : es.texturePaths)
    {
        if (!fs::is_directory(it))
        {
            std::cout << it + " is not a folder" << std::endl;
            exit(1);
        }
        for (const auto &entry : fs::directory_iterator(it))
        {
            const auto full_name = entry.path().string();
            const auto stem = entry.path().stem();
            const auto ext = entry.path().extension();
            if (texturePack.find(stem.string()) == texturePack.end())
            {
                if (entry.is_regular_file())
                {
                    if ((ext != ".png") && (ext != ".jpg"))
                        continue;
                    Texture *t = new Texture(full_name.c_str(), GL_TEXTURE_2D);
                    t->init_from_file();
                    texturePack.insert({stem.string(), t});
                }
            }
        }
    }
    projectiveTexture = texturePack["uv"];
    dynamicTexture = texturePack["uv"];
    normalMap = texturePack["wood_floor_deck_nor_gl_1k"];
    armMap = texturePack["wood_floor_deck_arm_1k"];
    // dispMap = texturePack["wood_floor_deck_disp_1k"];
    const fs::path bakeFileLeftPath{es.bakeFileLeft};
    const fs::path bakeFileRightPath{es.bakeFileRight};
    if (fs::exists(bakeFileLeftPath))
    {
        bakedTextureLeft = new Texture(es.bakeFileLeft.c_str(), GL_TEXTURE_2D);
        bakedTextureLeft->init_from_file();
    }
    else
        bakedTextureLeft = texturePack["uv"];
    if (fs::exists(bakeFileRightPath))
    {
        bakedTextureRight = new Texture(es.bakeFileRight.c_str(), GL_TEXTURE_2D);
        bakedTextureRight->init_from_file();
    }
    else
        bakedTextureRight = texturePack["uv"];
    postProcess.initGLBuffers();
    fullScreenQuad.init();
    Quad topHalfQuad("top_half", 0.0f);
    Quad bottomLeftQuad("bottom_left", 0.0f);
    Quad bottomRightQuad("bottom_right", 0.0f);
    Quad topRightQuad("tiny_top_right", 0.0f);
    glm::vec3 coa = leftHandModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    TextModel textModel("../../resource/arial.ttf");
    /* setup shaders*/
    Shader NNShader("../../src/shaders/NN_shader.vs", "../../src/shaders/NN_shader.fs");
    Shader uv_NNShader("../../src/shaders/uv_NN_Shader.vs", "../../src/shaders/uv_NN_Shader.fs");
    Shader maskShader("../../src/shaders/mask.vs", "../../src/shaders/mask.fs");
    Shader jfaInitShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa_init.fs");
    Shader jfaShader("../../src/shaders/jfa.vs", "../../src/shaders/jfa.fs");
    Shader debugShader("../../src/shaders/debug.vs", "../../src/shaders/debug.fs");
    Shader projectorShader("../../src/shaders/projector_shader.vs", "../../src/shaders/projector_shader.fs");
    Shader blurShader("../../src/shaders/blur.vs", "../../src/shaders/blur.fs");
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    Shader overlayShader("../../src/shaders/overlay.vs", "../../src/shaders/overlay.fs");
    Shader thresholdAlphaShader("../../src/shaders/threshold_alpha.vs", "../../src/shaders/threshold_alpha.fs");
    Shader gridShader("../../src/shaders/grid_texture.vs", "../../src/shaders/grid_texture.fs");
    Shader gridColorShader("../../src/shaders/grid_color.vs", "../../src/shaders/grid_color.fs");
    Shader lineShader("../../src/shaders/line_shader.vs", "../../src/shaders/line_shader.fs");
    Shader vcolorShader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    Shader uvDilateShader("../../src/shaders/uv_dilate.vs", "../../src/shaders/uv_dilate.fs");
    Shader textShader("../../src/shaders/text.vs", "../../src/shaders/text.fs");
    Shader shaderToySea("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_Ms2SD1.fs");
    // Shader shaderToyCloud("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_3l23Rh.fs");
    // Shader shaderToyGameBufferA("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_XsdGDX_BufferA.fs");
    // Shader shaderToyGameImage("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_XsdGDX_Image.fs");
    Shader shaderToyGameBufferA("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_XldGDN_BufferA.fs");
    Shader shaderToyGameBufferB("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_XldGDN_BufferB.fs");
    Shader shaderToyGameImage("../../src/shaders/shadertoy.vs", "../../src/shaders/shadertoy_XldGDN_Image.fs");
    SkinningShader skinnedShader("../../src/shaders/skin.vs", "../../src/shaders/skin.fs");
    std::unordered_map<std::string, Shader *> shaderMap = {
        {"NNShader", &NNShader},
        {"uv_NNShader", &uv_NNShader},
        {"maskShader", &maskShader},
        {"jfaInitShader", &jfaInitShader},
        {"jfaShader", &jfaShader},
        {"debugShader", &debugShader},
        {"projectorShader", &projectorShader},
        {"blurShader", &blurShader},
        {"textureShader", &textureShader},
        {"overlayShader", &overlayShader},
        {"thresholdAlphaShader", &thresholdAlphaShader},
        {"gridShader", &gridShader},
        {"gridColorShader", &gridColorShader},
        {"lineShader", &lineShader},
        {"vcolorShader", &vcolorShader},
        {"uvDilateShader", &uvDilateShader},
        {"textShader", &textShader},
        {"skinnedShader", &skinnedShader},
        {"shaderToySea", &shaderToySea},
        // {"shaderToyCloud", &shaderToyCloud},
        {"shaderToyGameBufferA", &shaderToyGameBufferA},
        {"shaderToyGameBufferB", &shaderToyGameBufferB},
        {"shaderToyGameImage", &shaderToyGameImage}};
    // settings for text shader
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(es.proj_width), 0.0f, static_cast<float>(es.proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    textShader.setFloat("threshold", 0.5f);
    // render the baked texture into fbo
    bake_fbo_right.bind(true);
    bakedTextureRight->bind();
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    set_texture_shader(&textureShader, false, false, false);
    fullScreenQuad.render();
    bake_fbo_right.unbind();
    bake_fbo_left.bind(true);
    bakedTextureLeft->bind();
    set_texture_shader(&textureShader, false, false, false);
    fullScreenQuad.render();
    bake_fbo_left.unbind();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    /* more inits */
    initKalmanFilters();
    double previousAppTime = t_app.getElapsedTimeInSec();
    double previousSecondAppTime = t_app.getElapsedTimeInSec();
    double currentAppTime = t_app.getElapsedTimeInSec();
    es.prev_vid_time = t_app.getElapsedTimeInMilliSec();
    es.cur_vid_time = t_app.getElapsedTimeInMilliSec();
    long frameCount = 0;
    uint64_t targetFrameSize = 0;
    size_t n_skeleton_primitives = 0;
    uint8_t *colorBuffer = new uint8_t[es.projected_image_size];
    camTexture = Texture();
    toBakeTexture = Texture();
    Texture displayTexture = Texture();
    OFTexture = new Texture();
    OFTexture->init(es.cam_width / es.of_resize_factor, es.cam_height / es.of_resize_factor, 3);
    displayTexture.init(es.cam_width, es.cam_height, es.n_cam_channels);
    camTexture.init(es.cam_width, es.cam_height, es.n_cam_channels);
    toBakeTexture.init(es.dst_width, es.dst_height, 4);
    if (es.simulated_projector)
    {
        projector = new SaveToDisk("../../debug/video/", es.proj_height, es.proj_width);
    }
    else
    {
        projector = new DynaFlashProjector(true, false);
    }
    // LeapCreateClockRebaser(&clockSynchronizer);
    // load calibration results if they exist
    Camera_Mode camera_mode = es.freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    std::cout << "Loading calibration results..." << std::endl;
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
        std::cout << "Failed. Using hard-coded values for camera and projector settings" << std::endl;
        gl_projector = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                camera_mode, es.proj_width, es.proj_height, 500.0f, 2.0f);
        gl_camera = GLCamera(glm::vec3(-4.76f, 18.2f, 38.6f),
                             glm::vec3(0.0f, 0.0f, 0.0f),
                             glm::vec3(0.0f, -1.0f, 0.0f),
                             camera_mode, es.proj_width, es.proj_height, 500.0f, 2.0f);
        gl_flycamera = GLCamera(glm::vec3(-4.72f, 16.8f, 38.9f),
                                glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                camera_mode, es.proj_width, es.proj_height, 1500.0f, 50.0f);
    }
    loadCoaxialCalibrationResults(es.cur_screen_verts);
    c2p_homography = PostProcess::findHomography(es.cur_screen_verts);
    /* thread loops */
    // camera.init(camera_queue, close_signal, cam_height, cam_width, exposure);
    /* real producer */
    camera.init_poll(es.cam_height, es.cam_width, es.exposure);
    projector->show();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    // camera.balance_white();
    camera.acquire();
    /* main loop */
    while (!glfwWindowShouldClose(window))
    {
        /* update / sync clocks */
        t_misc.start();
        currentAppTime = t_app.getElapsedTimeInSec(); // glfwGetTime();
        es.deltaTime = static_cast<float>(currentAppTime - previousAppTime);
        previousAppTime = currentAppTime;
        // if (!leap_poll_mode)
        // {
        //     std::modf(t_app.getElapsedTimeInMicroSec(), &whole);
        //     LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
        // }
        frameCount++;
        /* display stats */
        if (currentAppTime - previousSecondAppTime >= 1.0)
        {
            es.fps = frameCount;
            es.ms_per_frame = 1000.0f / frameCount;
            if (es.cmd_line_stats)
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
                std::cout << "of: " << t_of.averageLapInMilliSec() << std::endl;
                // std::cout << "profile0: " << t_profile0.averageLapInMilliSec() << std::endl;
                // std::cout << "profile1: " << t_profile1.averageLapInMilliSec() << std::endl;
                // std::cout << "warp: " << t_warp.averageLapInMilliSec() << std::endl;
                std::cout << "swap buffers: " << t_swap.averageLapInMilliSec() << std::endl;
                std::cout << "GPU->CPU: " << t_download.averageLapInMilliSec() << std::endl;
                // std::cout << "project time: " << t4.averageLap() << std::endl;
                std::cout << "cam q1 size: " << camera_queue.size_approx() << std::endl;
                // std::cout << "cam q2 size: " << camera_queue_cv.size() << std::endl;
                std::cout << "proj q size: " << projector->get_queue_size() << std::endl;
                std::cout << "mls ops: " << Helpers::average(es.mls_succeed_counters) << std::endl;
            }
            es.mls_succeed_counters.clear();
            frameCount = 0;
            previousSecondAppTime = currentAppTime;
            t_camera.reset();
            t_leap.reset();
            t_skin.reset();
            t_swap.reset();
            t_download.reset();
            // t_warp.reset();
            t_misc.reset();
            t_pp.reset();
            t_mls.reset();
            t_mls_thread.reset();
            t_of.reset();
            // t_profile0.reset();
            // t_profile1.reset();
            t_debug.reset();
        }
        /* deal with user input */
        glfwPollEvents();
        process_input(window);
        if (es.activateGUI)
        {
            openIMGUIFrame(); // create imgui frame
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else
        {
            switch (es.operation_mode)
            {
            case static_cast<int>(OperationMode::SANDBOX):
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
        switch (es.operation_mode)
        {
        case static_cast<int>(OperationMode::SANDBOX):
        {
            /* deal with camera input */
            t_camera.start();
            CGrabResultPtr ptrGrabResult;
            if (es.simulated_camera)
            {
                cv::Mat sim = cv::Mat(es.cam_height, es.cam_width, CV_8UC1, 255);
                handleCameraInput(ptrGrabResult, true, sim);
            }
            else
            {
                handleCameraInput(ptrGrabResult, false, cv::Mat());
            }
            t_camera.stop();

            /* deal with leap input */
            t_leap.start();
            LEAP_STATUS leap_status = handleLeapInput();
            if (es.record_session)
            {
                if ((t_app.getElapsedTimeInSec() - es.recordStartTime) > es.recordDuration)
                {
                    es.record_session = false;
                    saveSession(std::format("../../resource/recordings/{}", es.recording_name), es.recordImages);
                    std::cout << "Recording stopped" << std::endl;
                }
                else
                {
                    saveLeapData(leap_status, es.totalFrameCount, es.recordImages);
                }
            }
            if (es.record_single_pose)
            {
                saveSession(std::format("../../resource/recordings/{}", es.recording_name), leap_status, es.totalFrameCount, es.recordImages);
                es.record_single_pose = false;
            }
            projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
            t_leap.stop();
            /* skin hand meshes */
            t_skin.start();
            handleSkinning(bones_to_world_right, true, true, shaderMap, rightHandModel, cam_view_transform, cam_projection_transform);
            handleSkinning(bones_to_world_left, false, bones_to_world_right.size() == 0, shaderMap, leftHandModel, cam_view_transform, cam_projection_transform);
            t_skin.stop();

            /* deal with bake request */
            t_bake.start();
            handleBaking(shaderMap, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform);
            t_bake.stop();

            /* run MLS on MP prediction to reduce bias */
            t_mls.start();
            handleMLS(gridShader, es.mls_blocking);
            t_mls.stop();

            /* post process fbo using camera input */
            t_pp.start();
            handlePostProcess(leftHandModel, rightHandModel, camTexture, shaderMap);
            /* render final output to screen */
            if (!es.debug_mode)
            {
                glViewport(0, 0, es.proj_width, es.proj_height); // set viewport
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                set_texture_shader(&textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
                c2p_fbo.getTexture()->bind();
                fullScreenQuad.render();
            }
            t_pp.stop();

            /* debug mode */
            t_debug.start();
            handleDebugMode(shaderMap, rightHandModel, leftHandModel, dinosaur, textModel);
            t_debug.stop();
            break;
        }
        case static_cast<int>(OperationMode::USER_STUDY): // in this mode we can also playback recorded sessions
        {
            handleUserStudy(shaderMap,
                            window,
                            leftHandModel,
                            rightHandModel,
                            textModel,
                            cam_view_transform,
                            cam_projection_transform);
            break;
        }
        case static_cast<int>(OperationMode::SIMULATION): // discrete simulation on prerecorded data
        {
            handleSimulation(shaderMap,
                             leftHandModel,
                             rightHandModel,
                             textModel,
                             cam_view_transform,
                             cam_projection_transform);
            break;
        }
        case static_cast<int>(OperationMode::GUESS_POSE_GAME): // a game to guess the hand pose using visual color cues
        {
            handleGuessPoseGame(shaderMap, leftHandModel, rightHandModel, topRightQuad, cam_view_transform, cam_projection_transform);
            break;
        }
        case static_cast<int>(OperationMode::GUESS_CHAR_GAME): // a game to guess which character is shown on backhand as quickly as possible
        {
            handleGuessCharGame(shaderMap, leftHandModel, rightHandModel, textModel, cam_view_transform, cam_projection_transform);
            break;
        }
        case static_cast<int>(OperationMode::GUESS_ANIMAL_GAME): // a game to guess which character is shown on backhand as quickly as possible
        {
            handleGuessAnimalGame(shaderMap, leftHandModel, rightHandModel, textModel, cam_view_transform, cam_projection_transform);
            break;
        }
        case static_cast<int>(OperationMode::CAMERA): // calibrate camera todo: move to seperate handler
        {
            switch (es.calibration_state)
            {
            case static_cast<int>(CalibrationStateMachine::COLLECT):
            {
                CGrabResultPtr ptrGrabResult;
                camera.capture_single_image(ptrGrabResult);
                camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                cv::flip(camImageOrig, camImage, 1);
                std::vector<cv::Point2f> corner_pts;
                bool success = cv::findChessboardCorners(camImage, cv::Size(es.checkerboard_width, es.checkerboard_height), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
                if (success)
                {
                    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
                    cv::cornerSubPix(camImage, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);
                    cv::drawChessboardCorners(camImage, cv::Size(es.checkerboard_width, es.checkerboard_height), corner_pts, success);
                    if (es.ready_to_collect)
                        imgpoints.push_back(corner_pts);
                }
                displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                set_texture_shader(&textureShader, true, false, true);
                displayTexture.bind();
                fullScreenQuad.render();
                if (imgpoints.size() >= es.n_points_cam_calib)
                    es.ready_to_collect = false;
                break;
            }
            case static_cast<int>(CalibrationStateMachine::SOLVE):
            {
                std::vector<std::vector<cv::Point3f>> objpoints;
                std::vector<cv::Point3f> objp;
                for (int i = 0; i < es.checkerboard_height; i++)
                {
                    for (int j{0}; j < es.checkerboard_width; j++)
                        objp.push_back(cv::Point3f(j, i, 0));
                }
                for (int i = 0; i < imgpoints.size(); i++)
                    objpoints.push_back(objp);
                // cv::calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
                // objectPoints.resize(imagePoints.size(), objectPoints[0]);
                cv::Mat cameraMatrix, distCoeffs, R, T;
                cv::calibrateCamera(objpoints, imgpoints, cv::Size(es.cam_height, es.cam_width), cameraMatrix, distCoeffs, R, T);
                std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
                std::cout << "distCoeffs : " << distCoeffs << std::endl;
                std::vector<double> camMat(cameraMatrix.begin<double>(), cameraMatrix.end<double>());
                std::vector<double> camDist(distCoeffs.begin<double>(), distCoeffs.end<double>());
                calib_cam_matrix = camMat;
                calib_cam_distortion = camDist;
                // std::cout << "Rotation vector : " << R << std::endl;
                // std::cout << "Translation vector : " << T << std::endl;
                es.calibration_state = static_cast<int>(CalibrationStateMachine::SHOW);
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
            CGrabResultPtr ptrGrabResult;
            handleCameraInput(ptrGrabResult, false, cv::Mat());
            t_camera.stop();
            std::vector<cv::Point2f> origpts, newpts;
            for (int i = 0; i < 4; ++i)
            {
                origpts.push_back(cv::Point2f(es.screen_verts[i].x, es.screen_verts[i].y));
                newpts.push_back(cv::Point2f(es.cur_screen_verts[i].x, es.cur_screen_verts[i].y));
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
                cv::Vec4f cord = cv::Vec4f(es.screen_verts[i].x, es.screen_verts[i].y, 0.0f, 1.0f);
                cv::Mat tmp = perspective * cv::Mat(cord);
                es.cur_screen_verts[i].x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);
                es.cur_screen_verts[i].y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);
            }
            glm::mat4 viewMatrix;
            GLMHelpers::CV2GLM(perspective, &viewMatrix);
            set_texture_shader(&textureShader, true, true, true, false, es.masking_threshold, 0, glm::mat4(1.0f), glm::mat4(1.0f), viewMatrix);
            camTexture.bind();
            fullScreenQuad.render();
            PointCloud cloud(es.cur_screen_verts, es.screen_verts_color_red);
            vcolorShader.use();
            vcolorShader.setMat4("mvp", glm::mat4(1.0f));
            cloud.render();
            break;
        }
        case static_cast<int>(OperationMode::LEAP): // calibrate camera <-> leap
        {
            /* deal with leap input */
            t_leap.start();
            LEAP_STATUS leap_status = handleLeapInput();
            t_leap.stop();
            switch (es.calibration_state)
            {
            case static_cast<int>(CalibrationStateMachine::COLLECT):
            {
                switch (es.leap_collection_setting)
                {
                case static_cast<int>(LeapCollectionSettings::AUTO_RAW): // triangulates brightest point in leap images to extract a 3D point
                {
                    std::vector<uint8_t> buffer1, buffer2;
                    uint32_t ignore1, ignore2;
                    if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                    {
                        uint64_t new_frame_id = leap.getImageFrameID();
                        if (es.leap_cur_frame_id != new_frame_id)
                        {
                            // capture cam image asap
                            CGrabResultPtr ptrGrabResult;
                            camera.capture_single_image(ptrGrabResult);
                            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                            cv::flip(camImageOrig, camImage, 1);
                            cv::Mat thr;
                            cv::threshold(camImage, thr, static_cast<int>(es.masking_threshold * 255), 255, cv::THRESH_BINARY);
                            glm::vec2 center, center_leap1, center_leap2;
                            // render binary leap texture to bottom half of screen
                            Texture leapTexture1 = Texture();
                            Texture leapTexture2 = Texture();
                            leapTexture1.init(es.leap_width, es.leap_height, 1);
                            leapTexture2.init(es.leap_width, es.leap_height, 1);
                            leapTexture1.load(buffer1, true, es.cam_buffer_format);
                            leapTexture2.load(buffer2, true, es.cam_buffer_format);
                            leapTexture1.bind();
                            set_texture_shader(&textureShader, true, false, true, es.leap_threshold_flag, es.leap_binary_threshold);
                            bottomLeftQuad.render();
                            leapTexture2.bind();
                            bottomRightQuad.render();
                            displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                            set_texture_shader(&textureShader, true, false, true, es.threshold_flag, es.masking_threshold);
                            displayTexture.bind();
                            topHalfQuad.render();
                            glm::vec2 center_NDC;
                            bool found_centroid = false;
                            if (extract_centroid(thr, center))
                            {
                                found_centroid = true;
                                // render point on centroid in camera image
                                vcolorShader.use();
                                vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                                center_NDC = Helpers::ScreenToNDC(center, es.cam_width, es.cam_height, true);
                                glm::vec2 vert = center_NDC;
                                vert.y = (vert.y + 1.0f) / 2.0f; // for display, use top of screen
                                std::vector<glm::vec2> pc1 = {vert};
                                PointCloud pointCloud(pc1, es.screen_verts_color_red);
                                pointCloud.render(5.0f);
                            }
                            cv::Mat leap1_thr, leap2_thr;
                            cv::Mat leap1(es.leap_height, es.leap_width, CV_8UC1, buffer1.data());
                            cv::Mat leap2(es.leap_height, es.leap_width, CV_8UC1, buffer2.data());
                            cv::threshold(leap1, leap1_thr, static_cast<int>(es.leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            cv::threshold(leap2, leap2_thr, static_cast<int>(es.leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            if (extract_centroid(leap1_thr, center_leap1) && extract_centroid(leap2_thr, center_leap2))
                            {
                                // save the 2d and 3d points
                                glm::vec2 center_NDC_leap1 = Helpers::ScreenToNDC(center_leap1, es.leap_width, es.leap_height, true);
                                glm::vec2 center_NDC_leap2 = Helpers::ScreenToNDC(center_leap2, es.leap_width, es.leap_height, true);
                                glm::vec3 cur_3d_point = triangulate(leap, center_NDC_leap1, center_NDC_leap2);
                                es.triangulated = cur_3d_point;
                                if (found_centroid)
                                {
                                    glm::vec2 cur_2d_point = Helpers::NDCtoScreen(center_NDC, es.cam_width, es.cam_height, true);
                                    if (es.ready_to_collect)
                                    {
                                        points_3d.push_back(cur_3d_point);
                                        points_2d.push_back(cur_2d_point);
                                        if (points_2d.size() >= es.n_points_leap_calib)
                                        {
                                            es.ready_to_collect = false;
                                        }
                                    }
                                }
                                // render point on centroid in left leap image
                                vcolorShader.use();
                                vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                                glm::vec2 vert1 = center_NDC_leap1;
                                glm::vec2 vert2 = center_NDC_leap2;
                                vert1.y = (vert1.y - 1.0f) / 2.0f; // use bottom left of screen
                                vert1.x = (vert1.x - 1.0f) / 2.0f; //
                                vert2.y = (vert2.y - 1.0f) / 2.0f; // use bottom right of screen
                                vert2.x = (vert2.x + 1.0f) / 2.0f; //
                                std::vector<glm::vec2> pc2 = {vert1, vert2};
                                PointCloud pointCloud2(pc2, es.screen_verts_color_red);
                                pointCloud2.render(5.0f);
                            }
                            es.leap_cur_frame_id = new_frame_id;
                        }
                    }
                    break;
                }
                case static_cast<int>(LeapCollectionSettings::AUTO_FINGER): // uses the high level API to get index tip as the extracted 3D point
                {
                    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME)
                    {
                        // capture cam image asap
                        CGrabResultPtr ptrGrabResult;
                        camera.capture_single_image(ptrGrabResult);
                        camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                        cv::flip(camImageOrig, camImage, 1);
                        cv::Mat thr;
                        cv::threshold(camImage, thr, static_cast<int>(es.masking_threshold * 255), 255, cv::THRESH_BINARY);
                        // render binary leap texture to bottom half of screen
                        displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                        set_texture_shader(&textureShader, true, false, true, es.threshold_flag, es.masking_threshold);
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
                            vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                            center_NDC = Helpers::ScreenToNDC(center, es.cam_width, es.cam_height, true);
                        }
                        if (found_centroid)
                        {
                            if (joints_left.size() > 0)
                            {
                                glm::vec3 cur_3d_point = joints_left[17]; // index tip
                                es.triangulated = cur_3d_point;
                                glm::vec2 cur_2d_point = Helpers::NDCtoScreen(center_NDC, es.cam_width, es.cam_height, true);
                                if (es.ready_to_collect)
                                {
                                    points_3d.push_back(cur_3d_point);
                                    points_2d.push_back(cur_2d_point);
                                    if (points_2d.size() >= es.n_points_leap_calib)
                                    {
                                        es.ready_to_collect = false;
                                    }
                                }
                                glm::vec2 vert = center_NDC;
                                std::vector<glm::vec2> pc1 = {vert};
                                PointCloud pointCloud(pc1, es.screen_verts_color_green);
                                pointCloud.render(5.0f);
                            }
                            else
                            {
                                glm::vec2 vert = center_NDC;
                                // vert.y = (vert.y + 1.0f) / 2.0f; // for display, use top of screen
                                std::vector<glm::vec2> pc1 = {vert};
                                PointCloud pointCloud(pc1, es.screen_verts_color_red);
                                pointCloud.render(5.0f);
                            }
                        }
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case static_cast<int>(CalibrationStateMachine::SOLVE):
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
                if (es.leap_calib_use_ransac)
                {
                    cv::Mat inliers;
                    cv::solvePnPRansac(points_3d_cv, points_2d_cv, camera_intrinsics_cv, camera_distortion_cv, rvec_calib, tvec_calib, true,
                                       es.pnp_iters, es.pnp_rep_error, es.pnp_confidence, inliers, cv::SOLVEPNP_ITERATIVE);
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
                points_2d_reprojected = Helpers::cv2glm(reprojected_cv);
                points_2d_inliers_reprojected = Helpers::cv2glm(reprojected_inliers_cv);
                points_2d_inliners = Helpers::cv2glm(points_2d_inliers_cv);
                std::vector<glm::vec3> points_3d_inliers = Helpers::cv2glm(points_3d_inliers_cv);
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
                es.use_leap_calib_results = static_cast<int>(LeapCalibrationSettings::AUTO);
                create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
                es.calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                es.calibrationSuccess = true;
                break;
            }
            case static_cast<int>(CalibrationStateMachine::SHOW):
            {
                vcolorShader.use();
                vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                // todo: move this logic to CalibrationStateMachine::SOLVE
                std::vector<glm::vec2> NDCs;
                std::vector<glm::vec2> NDCs_reprojected;
                if (es.showInliersOnly)
                {
                    NDCs = Helpers::ScreenToNDC(points_2d_inliners, es.cam_width, es.cam_height, true);
                    NDCs_reprojected = Helpers::ScreenToNDC(points_2d_inliers_reprojected, es.cam_width, es.cam_height, true);
                }
                else
                {
                    NDCs = Helpers::ScreenToNDC(points_2d, es.cam_width, es.cam_height, true);
                    NDCs_reprojected = Helpers::ScreenToNDC(points_2d_reprojected, es.cam_width, es.cam_height, true);
                }
                PointCloud pointCloud1(NDCs, es.screen_verts_color_red);
                pointCloud1.render();
                PointCloud pointCloud2(NDCs_reprojected, es.screen_verts_color_blue);
                pointCloud2.render();
                break;
            }
            case static_cast<int>(CalibrationStateMachine::MARK):
            {
                switch (es.leap_mark_setting)
                {
                case static_cast<int>(LeapMarkSettings::STREAM):
                {
                    std::vector<uint8_t> buffer1, buffer2;
                    uint32_t ignore1, ignore2;
                    if (leap.getImage(buffer1, buffer2, ignore1, ignore2))
                    {
                        uint64_t new_frame_id = leap.getImageFrameID();
                        if (es.leap_cur_frame_id != new_frame_id)
                        {
                            // capture cam image asap
                            CGrabResultPtr ptrGrabResult;
                            camera.capture_single_image(ptrGrabResult);
                            displayTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, es.cam_buffer_format);
                            glm::vec2 center, center_leap1, center_leap2;
                            // render binary leap texture to bottom half of screen
                            Texture leapTexture1 = Texture();
                            Texture leapTexture2 = Texture();
                            leapTexture1.init(es.leap_width, es.leap_height, 1);
                            leapTexture2.init(es.leap_width, es.leap_height, 1);
                            leapTexture1.load(buffer1, true, es.cam_buffer_format);
                            leapTexture2.load(buffer2, true, es.cam_buffer_format);
                            leapTexture1.bind();
                            set_texture_shader(&textureShader, true, false, true, es.leap_threshold_flag, es.leap_binary_threshold);
                            bottomLeftQuad.render();
                            leapTexture2.bind();
                            bottomRightQuad.render();
                            set_texture_shader(&textureShader, true, true, true);
                            displayTexture.bind();
                            topHalfQuad.render();
                            cv::Mat leap1_thr, leap2_thr;
                            cv::Mat leap1(es.leap_height, es.leap_width, CV_8UC1, buffer1.data());
                            cv::Mat leap2(es.leap_height, es.leap_width, CV_8UC1, buffer2.data());
                            cv::threshold(leap1, leap1_thr, static_cast<int>(es.leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            cv::threshold(leap2, leap2_thr, static_cast<int>(es.leap_binary_threshold * 255), 255, cv::THRESH_BINARY);
                            if (extract_centroid(leap1_thr, center_leap1) && extract_centroid(leap2_thr, center_leap2))
                            {
                                // save the 2d and 3d points
                                glm::vec2 center_NDC_leap1 = Helpers::ScreenToNDC(center_leap1, es.leap_width, es.leap_height, true);
                                glm::vec2 center_NDC_leap2 = Helpers::ScreenToNDC(center_leap2, es.leap_width, es.leap_height, true);
                                glm::vec3 cur_3d_point = triangulate(leap, center_NDC_leap1, center_NDC_leap2);
                                es.triangulated = cur_3d_point;
                                /////
                                std::vector<cv::Point3f> points_3d_cv{cv::Point3f(cur_3d_point.x, cur_3d_point.y, cur_3d_point.z)};
                                std::vector<cv::Point2f> reprojected_cv;
                                cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                                std::vector<glm::vec2> reprojected = Helpers::cv2glm(reprojected_cv);
                                reprojected = Helpers::ScreenToNDC(reprojected, es.cam_width, es.cam_height, true);
                                ////
                                vcolorShader.use();
                                vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                                glm::vec2 vert1 = center_NDC_leap1;
                                glm::vec2 vert2 = center_NDC_leap2;
                                glm::vec2 vert3 = reprojected[0];
                                vert1.y = (vert1.y - 1.0f) / 2.0f; // use bottom left of screen
                                vert1.x = (vert1.x - 1.0f) / 2.0f; //
                                vert2.y = (vert2.y - 1.0f) / 2.0f; // use bottom right of screen
                                vert2.x = (vert2.x + 1.0f) / 2.0f; //
                                vert3.y = (vert3.y + 1.0f) / 2.0f; // for display, use top of screen
                                std::vector<glm::vec2> pc2 = {vert1, vert2, vert3};
                                PointCloud pointCloud2(pc2, es.screen_verts_color_red);
                                pointCloud2.render(5.0f);
                                // std::cout << "leap1 2d:" << center_NDC_leap1.x << " " << center_NDC_leap1.y << std::endl;
                                // std::cout << "leap2 2d:" << center_NDC_leap2.x << " " << center_NDC_leap2.y << std::endl;
                                // std::cout << point_3d.x << " " << point_3d.y << " " << point_3d.z << std::endl;
                            }
                            es.leap_cur_frame_id = new_frame_id;
                        }
                    }
                    break;
                }
                case static_cast<int>(LeapMarkSettings::POINT_BY_POINT):
                {
                    CGrabResultPtr ptrGrabResult;
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    switch (es.leap_calibration_mark_state)
                    {
                    case 0:
                    {
                        displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                        set_texture_shader(&textureShader, true, false, true);
                        displayTexture.bind();
                        fullScreenQuad.render();
                        vcolorShader.use();
                        vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                        PointCloud pointCloud(marked_reprojected, es.screen_verts_color_red);
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
                            leapTexture.init(es.leap_width, es.leap_height, 1);
                            leapTexture.load(buffer1, true, es.cam_buffer_format);
                            // Texture leapTexture2 = Texture();
                            // leapTexture2.init(leap_width, leap_height, 1);
                            // leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture.bind();
                            set_texture_shader(&textureShader, true, false, true);
                            fullScreenQuad.render();
                            vcolorShader.use();
                            vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                            std::vector<glm::vec2> test = {es.cur_screen_vert};
                            PointCloud pointCloud(test, es.screen_verts_color_red);
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
                            leapTexture.init(es.leap_width, es.leap_height, 1);
                            leapTexture.load(buffer2, true, es.cam_buffer_format);
                            // Texture leapTexture2 = Texture();
                            // leapTexture2.init(leap_width, leap_height, 1);
                            // leapTexture2.load(buffer2, true, cam_buffer_format);
                            leapTexture.bind();
                            set_texture_shader(&textureShader, true, false, true);
                            fullScreenQuad.render();
                            vcolorShader.use();
                            vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                            std::vector<glm::vec2> test = {es.cur_screen_vert};
                            PointCloud pointCloud(test, es.screen_verts_color_red);
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
                    CGrabResultPtr ptrGrabResult;
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                    set_texture_shader(&textureShader, true, false, true);
                    displayTexture.bind();
                    fullScreenQuad.render();
                    if (joints_right.size() > 0)
                    {
                        vcolorShader.use();
                        vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                        std::vector<cv::Point3f> points_3d_cv;
                        for (int i = 0; i < joints_right.size(); i++)
                        {
                            points_3d_cv.push_back(cv::Point3f(joints_right[i].x, joints_right[i].y, joints_right[i].z));
                        }
                        // points_3d_cv.push_back(cv::Point3f(joints_right[mark_bone_index].x, joints_right[mark_bone_index].y, joints_right[mark_bone_index].z));
                        std::vector<cv::Point2f> reprojected_cv;
                        cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                        std::vector<glm::vec2> reprojected = Helpers::cv2glm(reprojected_cv);
                        // glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                        reprojected = Helpers::ScreenToNDC(reprojected, es.cam_width, es.cam_height, true);
                        std::vector<glm::vec2> pc = {reprojected};
                        PointCloud pointCloud(pc, es.screen_verts_color_red);
                        pointCloud.render();
                    }
                    break;
                } // LeapMarkSettings::WHOLE_HAND
                case static_cast<int>(LeapMarkSettings::ONE_BONE):
                {
                    CGrabResultPtr ptrGrabResult;
                    camera.capture_single_image(ptrGrabResult);
                    camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
                    cv::flip(camImageOrig, camImage, 1);
                    displayTexture.load((uint8_t *)camImage.data, true, es.cam_buffer_format);
                    set_texture_shader(&textureShader, true, false, true);
                    displayTexture.bind();
                    fullScreenQuad.render();
                    if (joints_right.size() > 0)
                    {
                        vcolorShader.use();
                        vcolorShader.setMat4("mvp", glm::mat4(1.0f));
                        std::vector<cv::Point3f> points_3d_cv;
                        points_3d_cv.push_back(cv::Point3f(joints_right[es.mark_bone_index].x, joints_right[es.mark_bone_index].y, joints_right[es.mark_bone_index].z));
                        std::vector<cv::Point2f> reprojected_cv;
                        cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                        std::vector<glm::vec2> reprojected = Helpers::cv2glm(reprojected_cv);
                        // glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                        reprojected = Helpers::ScreenToNDC(reprojected, es.cam_width, es.cam_height, true);
                        std::vector<glm::vec2> pc = {reprojected};
                        PointCloud pointCloud(pc, es.screen_verts_color_red);
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
        } // switch(operation_mode)

        if (es.use_projector && es.project_this_frame)
        {
            // send result to projector queue
            glReadBuffer(GL_BACK);
            if (es.use_pbo || es.double_pbo) // todo: investigate better way to download asynchornously
            {
                if (es.double_pbo)
                {
                    t_download.start();
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[es.totalFrameCount % 2]); // todo: replace with totalFrameCount
                    glReadPixels(0, 0, es.proj_width, es.proj_height, es.proj_channel_order, GL_UNSIGNED_BYTE, 0);
                    t_download.stop();

                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[(es.totalFrameCount + 1) % 2]);
                    GLubyte *src = (GLubyte *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                    if (src)
                    {
                        // std::vector<uint8_t> data(src, src + image_size);
                        // tmpdata.assign(src, src + image_size);
                        // std::copy(src, src + tmpdata.size(), tmpdata.begin());
                        memcpy(colorBuffer, src, es.projected_image_size);
                        projector->show_buffer(colorBuffer); // this assumes show_buffer is faster than render cycle, or performs internal copy.
                        glUnmapBuffer(GL_PIXEL_PACK_BUFFER); // release pointer to the mapped buffer
                    }
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                }
                else
                {
                    t_download.start();
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]); // todo: replace with totalFrameCount
                    glReadPixels(0, 0, es.proj_width, es.proj_height, es.proj_channel_order, GL_UNSIGNED_BYTE, 0);
                    GLubyte *src = (GLubyte *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                    if (src)
                    {
                        memcpy(colorBuffer, src, es.projected_image_size);
                        projector->show_buffer(colorBuffer); // this assumes show_buffer is faster than render cycle, or performs internal copy.
                        glUnmapBuffer(GL_PIXEL_PACK_BUFFER); // release pointer to the mapped buffer
                    }
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                    t_download.stop();
                }
            }
            else
            {
                t_download.start();
                glReadPixels(0, 0, es.proj_width, es.proj_height, es.proj_channel_order, GL_UNSIGNED_BYTE, colorBuffer);
                t_download.stop();
                // std::vector<uint8_t> data(colorBuffer, colorBuffer + image_size);
                projector->show_buffer(colorBuffer); // this assumes show_buffer is faster than render cycle, or performs internal copy.
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
        if (es.activateGUI)
        {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
        glfwSwapBuffers(window);
        es.totalFrameCount++;
        t_swap.stop();
    }
    // cleanup
    if (mls_thread.joinable())
        mls_thread.join();
    if (sd_thread.joinable())
        sd_thread.join();
    projector->kill();
    camera.kill();
    glfwTerminate();
    delete[] colorBuffer;
    // if (simulated_camera)
    // {
    //     producer.join();
    // }
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

void initKalmanFilters()
{
    kalman_filters_left.clear();
    kalman_filters_vleft.clear();
    kalman_filters_right.clear();
    kalman_filters_vright.clear();
    for (int i = 0; i < 16; i++)
    {
        kalman_filters_left.push_back(Kalman2D_ConstantV(es.kalman_process_noise,
                                                         es.kalman_measurement_noise));
        kalman_filters_vleft.push_back(Kalman2D_ConstantV2(es.kalman_process_noise,
                                                           es.kalman_measurement_noise));
    }
    for (int i = 0; i < 16; i++)
    {
        kalman_filters_right.push_back(Kalman2D_ConstantV(es.kalman_process_noise,
                                                          es.kalman_measurement_noise));
        kalman_filters_vright.push_back(Kalman2D_ConstantV2(es.kalman_process_noise,
                                                            es.kalman_measurement_noise));
    }
}

std::vector<float> computeDistanceFromPose(const std::vector<glm::mat4> &bones_to_world, const std::vector<glm::mat4> &required_pose_bones_to_world)
{
    // compute the rotational difference between the current pose and the required pose per bone
    std::vector<float> distances;
    for (int i = 0; i < bones_to_world.size(); i++)
    {
        glm::mat4 bone_to_world_rot = glm::scale(bones_to_world[i], glm::vec3(1 / es.magic_leap_scale_factor));
        glm::mat4 required_bone_to_world_rot = glm::scale(required_pose_bones_to_world[i], glm::vec3(1 / es.magic_leap_scale_factor));
        glm::mat3 diff = glm::mat3(bone_to_world_rot) * glm::transpose(glm::mat3(required_bone_to_world_rot));
        float trace = diff[0][0] + diff[1][1] + diff[2][2];
        float angle = glm::acos((trace - 1.0f) / 2.0f);
        if (std::isnan(angle)) // because the rotations are not exactly orthogonal acos can return nan
            angle = 0.0f;
        // normalize to 01
        angle = angle / glm::pi<float>();
        angle = 1.0f - angle;
        distances.push_back(angle);
    }
    return distances;
}
void initGLBuffers(unsigned int *pbo)
{
    // set up vertex data parameter
    // void *data = malloc(es.projected_image_size);
    // create ping pong pbos
    glGenBuffers(2, pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, es.projected_image_size, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, es.projected_image_size, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    // free(data);
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void process_input(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
    {
        es.tab_pressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE)
    {
        if (es.tab_pressed)
            es.activateGUI = !es.activateGUI;
        es.tab_pressed = false;
    }
    if (es.activateGUI) // dont allow moving cameras when GUI active
        return;
    bool mod = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        mod = true;
        es.shift_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            gl_projector.processKeyboard(FORWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            gl_projector.processKeyboard(BACKWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            gl_projector.processKeyboard(LEFT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            gl_projector.processKeyboard(RIGHT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            gl_projector.processKeyboard(UP, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            gl_projector.processKeyboard(DOWN, es.deltaTime);
    }
    else
    {
        es.shift_modifier = false;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        mod = true;
        es.ctrl_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            gl_camera.processKeyboard(FORWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            gl_camera.processKeyboard(BACKWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            gl_camera.processKeyboard(LEFT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            gl_camera.processKeyboard(RIGHT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            gl_camera.processKeyboard(UP, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            gl_camera.processKeyboard(DOWN, es.deltaTime);
    }
    else
    {
        es.ctrl_modifier = false;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        mod = true;
        es.space_modifier = true;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            es.debug_vec.x += es.deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            es.debug_vec.x -= es.deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            es.debug_vec.y += es.deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            es.debug_vec.y -= es.deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            es.debug_vec.z += es.deltaTime * 10.0f;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            es.debug_vec.z -= es.deltaTime * 10.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE)
    {
        if (es.space_modifier)
        {
            es.space_modifier = false;
            if (es.operation_mode == static_cast<int>(OperationMode::LEAP))
            {
                switch (es.calibration_state)
                {
                case static_cast<int>(CalibrationStateMachine::COLLECT):
                {
                    if (es.ready_to_collect && joints_right.size() > 0)
                    {
                        points_3d.push_back(joints_right[17]);
                        glm::vec2 cur_2d_point = Helpers::NDCtoScreen(es.cur_screen_vert, es.cam_width, es.cam_height, false);
                        points_2d.push_back(cur_2d_point);
                    }
                    break;
                }
                case static_cast<int>(CalibrationStateMachine::MARK):
                {
                    if (es.leap_mark_setting == static_cast<int>(LeapMarkSettings::POINT_BY_POINT))
                    {
                        if (es.leap_calibration_mark_state == 1)
                        {
                            marked_2d_pos1 = es.cur_screen_vert;
                        }
                        if (es.leap_calibration_mark_state == 2)
                        {
                            marked_2d_pos2 = es.cur_screen_vert;
                            glm::vec3 pos_3d = triangulate(leap, marked_2d_pos1, marked_2d_pos2);
                            triangulated_marked.push_back(pos_3d);
                            std::vector<cv::Point3f> points_3d_cv{cv::Point3f(pos_3d.x, pos_3d.y, pos_3d.z)};
                            std::vector<cv::Point2f> reprojected_cv;
                            cv::projectPoints(points_3d_cv, rvec_calib, tvec_calib, camera_intrinsics_cv, camera_distortion_cv, reprojected_cv);
                            glm::vec2 reprojected = glm::vec2(reprojected_cv[0].x, reprojected_cv[0].y);
                            reprojected = Helpers::ScreenToNDC(reprojected, es.cam_width, es.cam_height, true);
                            marked_reprojected.push_back(reprojected);
                        }
                        es.leap_calibration_mark_state = (es.leap_calibration_mark_state + 1) % 3;
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
            gl_flycamera.processKeyboard(FORWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            gl_flycamera.processKeyboard(BACKWARD, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            gl_flycamera.processKeyboard(LEFT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            gl_flycamera.processKeyboard(RIGHT, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            gl_flycamera.processKeyboard(UP, es.deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            gl_flycamera.processKeyboard(DOWN, es.deltaTime);
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
        es.rmb_pressed = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
    {
        if (es.rmb_pressed)
            es.activateGUI = !es.activateGUI;
        es.rmb_pressed = false;
    }
    if (es.activateGUI) // dont allow moving cameras when GUI active
        return;
    switch (es.operation_mode)
    {
    case static_cast<int>(OperationMode::SANDBOX):
    {
        break;
    }
    case static_cast<int>(OperationMode::COAXIAL):
    {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            if (es.dragging == false)
            {
                if (es.min_dist < 1.0f)
                {
                    es.dragging = true;
                    es.dragging_vert = es.closest_vert;
                }
            }
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        {
            es.dragging = false;
        }
        break;
    }
    case static_cast<int>(OperationMode::LEAP):
    {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            es.dragging = true;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
        {
            es.dragging = false;
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

    if (es.firstMouse)
    {
        es.lastX = xpos;
        es.lastY = ypos;
        es.firstMouse = false;
    }

    float xoffset = xpos - es.lastX;
    float yoffset = es.lastY - ypos; // reversed since y-coordinates go from bottom to top

    es.lastX = xpos;
    es.lastY = ypos;
    if (es.activateGUI)
        return;
    switch (es.operation_mode)
    {
    case static_cast<int>(OperationMode::SANDBOX):
    {
        if (es.shift_modifier)
        {
            gl_projector.processMouseMovement(xoffset, yoffset);
        }
        else
        {
            if (es.ctrl_modifier)
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
        glm::vec2 mouse_pos = Helpers::ScreenToNDC(glm::vec2(xpos, ypos), es.proj_width, es.proj_height, true);
        float cur_min_dist = 100.0f;
        for (int i = 0; i < es.cur_screen_verts.size(); i++)
        {
            glm::vec2 v = glm::vec2(es.cur_screen_verts[i]);

            float dist = glm::distance(v, mouse_pos);
            if (dist < cur_min_dist)
            {
                cur_min_dist = dist;
                es.closest_vert = i;
            }
        }
        es.min_dist = cur_min_dist;
        if (es.dragging)
        {
            es.cur_screen_verts[es.dragging_vert].x = mouse_pos.x;
            es.cur_screen_verts[es.dragging_vert].y = mouse_pos.y;
        }
        break;
    }
    case static_cast<int>(OperationMode::LEAP):
    {
        glm::vec2 mouse_pos_NDC = Helpers::ScreenToNDC(glm::vec2(xpos, ypos), es.proj_width, es.proj_height, true);
        if (es.dragging)
        {
            es.cur_screen_vert = mouse_pos_NDC;
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
    if (es.activateGUI)
        return;
    if (es.shift_modifier)
    {
        gl_projector.processMouseScroll(static_cast<float>(yoffset));
    }
    else
    {
        if (es.ctrl_modifier)
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
    glm::mat4 flipYZ = glm::mat4(1.0f);
    flipYZ[1][1] = -1.0f;
    flipYZ[2][2] = -1.0f;
    cnpy::NpyArray points2d_npy, points3d_npy;
    cnpy::NpyArray w2c_user_npy, w2c_auto_npy;
    cnpy::npz_t cam_npz;
    try
    {
        const fs::path user_path{"../../resource/calibrations/leap_calibration/w2c_user.npy"};
        const fs::path auto_path{"../../resource/calibrations/leap_calibration/w2c.npy"};
        const fs::path points2d_path{"../../resource/calibrations/leap_calibration/2dpoints.npy"};
        const fs::path points3d_path{"../../resource/calibrations/leap_calibration/3dpoints.npy"};
        const fs::path cam_calib_path{"../../resource/calibrations/cam_calibration/cam_calibration.npz"};
        // const fs::path projcam_calib_path{"../../resource/calibrations/camproj_calibration/calibration.npz"};
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
        // if (!fs::exists(projcam_calib_path))
        //     return false;
        w2c_user_npy = cnpy::npy_load(user_path.string());
        w2c_auto_npy = cnpy::npy_load(auto_path.string());
        points2d_npy = cnpy::npy_load(points2d_path.string());
        points3d_npy = cnpy::npy_load(points3d_path.string());
        cam_npz = cnpy::npz_load(cam_calib_path.string());
        // projcam_npz = cnpy::npz_load(projcam_calib_path.string());
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
    // std::cout << "camera_distortion: " << camera_distortion_cv << std::endl;
    // std::cout << "camera_intrinsics: " << camera_intrinsics_cv << std::endl;
    cv::initUndistortRectifyMap(camera_intrinsics_cv, camera_distortion_cv, cv::Mat(), camera_intrinsics_cv, cv::Size(es.cam_width, es.cam_height), CV_32FC1, undistort_map1, undistort_map2);
    float cfx = camera_intrinsics[0][0];
    float cfy = camera_intrinsics[1][1];
    float ccx = camera_intrinsics[0][2];
    float ccy = camera_intrinsics[1][2];
    cam_project = glm::mat4(0.0);
    cam_project[0][0] = 2 * cfx / es.cam_width;
    cam_project[0][2] = (es.cam_width - 2 * ccx) / es.cam_width;
    cam_project[1][1] = 2 * cfy / es.cam_height;
    cam_project[1][2] = -(es.cam_height - 2 * ccy) / es.cam_height;
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

LEAP_STATUS getLeapFramePreRecorded(std::vector<glm::mat4> &bones,
                                    std::vector<glm::vec3> &joints,
                                    uint64_t frameCounter,
                                    uint64_t totalFrameCount,
                                    const std::vector<glm::mat4> &bones_session,
                                    const std::vector<glm::vec3> &joints_session)
{
    if (frameCounter < totalFrameCount)
    {
        bones.clear();
        joints.clear();
        if (bones_session.size() > 0)
        {
            for (int i = 0; i < 22; i++)
            {
                bones.push_back(bones_session[frameCounter * 22 + i]);
            }
            if (joints_session.size() > 0) // the session might not have joints saved
            {
                for (int i = 0; i < 42; i++)
                {
                    joints.push_back(joints_session[frameCounter * 42 + i]);
                }
            }
        }
        return LEAP_STATUS::LEAP_NEWFRAME;
    }
    else
    {
        return LEAP_STATUS::LEAP_NONEWFRAME;
    }
}

void loadImagesFromFolder(std::string loadpath)
{
    recordedImages.clear();
    fs::path video_path(loadpath);
    int frame_size = es.cam_width * es.cam_height * 1;
    int counter = 0;
    std::cout << "Loading Images From: " << video_path.string() << " ..." << std::endl;
    for (const auto &entry : fs::directory_iterator(video_path))
    {
        if (entry.path().extension() != ".png")
            continue;
        // if (counter >= max_supported_frames)
        // {
        //     std::cout << "too many images in folder" << std::endl;
        //     exit(1);
        // }
        cv::Mat origImage = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        cv::Mat image;
        cv::flip(origImage, image, 1);
        recordedImages.push_back(image);
        // pFrameData[counter] = (uint8_t *)malloc(frame_size);
        // memcpy((void *)pFrameData[counter], (void *)image.data, frame_size);
        counter += 1;
    }
    es.maxVideoFrameCount = counter;
}

bool loadSession()
{
    // load raw session data (hand poses n x 22 x 4 x 4, and timestamps n x 1)
    std::string recordings("../../resource/recordings/");
    fs::path recording_path = fs::path(recordings) / fs::path(es.recording_name);
    fs::path bones_left_path = recording_path / fs::path(std::string("bones_left.npy"));
    fs::path bones_right_path = recording_path / fs::path(std::string("bones_right.npy"));
    fs::path joints_left_path = recording_path / fs::path(std::string("joints_left.npy"));
    fs::path joints_right_path = recording_path / fs::path(std::string("joints_right.npy"));
    fs::path timestamps_path = recording_path / fs::path(std::string("timestamps.npy"));
    cnpy::NpyArray bones_left_npy, bones_right_npy, joints_left_npy, joints_right_npy, timestamps_npy;
    std::vector<glm::mat4> raw_session_bones_left, raw_session_bones_right;
    std::vector<glm::vec3> raw_session_joints_left, raw_session_joints_right;
    bool found_bones = false;
    bool found_joints = false;
    if (fs::exists(bones_left_path))
    {
        bones_left_npy = cnpy::npy_load(bones_left_path.string());
        std::vector<float> raw_data = bones_left_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 16)
        {
            raw_session_bones_left.push_back(glm::make_mat4(raw_data.data() + i));
        }
        found_bones = true;
    }
    if (fs::exists(bones_right_path))
    {
        bones_right_npy = cnpy::npy_load(bones_right_path.string());
        std::vector<float> raw_data = bones_right_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 16)
        {
            raw_session_bones_right.push_back(glm::make_mat4(raw_data.data() + i));
        }
        found_bones = true;
    }
    if (!found_bones)
    {
        return false;
    }
    if (fs::exists(joints_left_path))
    {
        joints_left_npy = cnpy::npy_load(joints_left_path.string());
        std::vector<float> raw_data = joints_left_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 3)
        {
            raw_session_joints_left.push_back(glm::make_vec3(raw_data.data() + i));
        }
        found_joints = true;
    }
    if (fs::exists(joints_right_path))
    {
        joints_right_npy = cnpy::npy_load(joints_right_path.string());
        std::vector<float> raw_data = joints_right_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 3)
        {
            raw_session_joints_right.push_back(glm::make_vec3(raw_data.data() + i));
        }
        found_joints = true;
    }
    if (!found_joints)
    {
        return false;
    }
    std::vector<float> raw_session_timestamps;
    if (fs::exists(timestamps_path))
    {
        timestamps_npy = cnpy::npy_load(timestamps_path.string());
        std::vector<uint64_t> raw_data = timestamps_npy.as_vec<uint64_t>();
        // todo: convert to std iterators
        uint64_t first_timestamp = raw_data[0];
        for (int i = 0; i < raw_data.size(); i++)
        {
            raw_session_timestamps.push_back((raw_data[i] - first_timestamp) / 1000.0f);
        }
    }
    else
    {
        return false;
    }
    es.total_session_time_stamps = raw_session_timestamps.size();
    session_bones_left = raw_session_bones_left;
    session_bones_right = raw_session_bones_right;
    session_joints_left = raw_session_joints_left;
    session_joints_right = raw_session_joints_right;
    session_timestamps = raw_session_timestamps;
    loadImagesFromFolder(recording_path.string());
    if (es.maxVideoFrameCount == raw_session_timestamps.size())
    {
        es.canUseRecordedImages = true;
    }
    else
    {
        es.canUseRecordedImages = false;
    }
    es.pre_recorded_session_loaded = true;
    es.loaded_session_name = es.recording_name;
    return true;
}

LEAP_STATUS getLeapFrame(LeapCPP &leap, const int64_t &targetFrameTime,
                         std::vector<glm::mat4> &bones_to_world_left,
                         std::vector<glm::mat4> &bones_to_world_right,
                         std::vector<glm::vec3> &joints_left,
                         std::vector<glm::vec3> &joints_right,
                         std::vector<uint32_t> &leftFingersExtended,
                         std::vector<uint32_t> &rightFingersExtended,
                         bool leap_poll_mode,
                         int64_t &curFrameID,
                         int64_t &curFrameTimeStamp)
{
    //  some defs
    // t_profile1.start();
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
    glm::mat4 scalar = glm::scale(glm::mat4(1.0f), glm::vec3(es.magic_leap_scale_factor));
    uint64_t targetFrameSize = 0;
    LEAP_TRACKING_EVENT *frame = nullptr;
    // t_profile1.stop();
    // std::cout << "init defs: " << t_profile1.averageLapInMilliSec() << std::endl;
    // t_profile1.start();
    if (leap_poll_mode)
    {
        frame = leap.getFrame();
        if (frame != NULL && (frame->tracking_frame_id > curFrameID))
        {
            curFrameID = frame->tracking_frame_id;
            joints_left.clear();
            joints_right.clear();
            bones_to_world_left.clear();
            bones_to_world_right.clear();
            leftFingersExtended.clear();
            rightFingersExtended.clear();
            curFrameTimeStamp = frame->info.timestamp;
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
        leftFingersExtended.clear();
        rightFingersExtended.clear();
        // Get the buffer size needed to hold the tracking data
        eLeapRS retVal = LeapGetFrameSize(*leap.getConnectionHandle(), LeapGetNow() + targetFrameTime, &targetFrameSize);
        if (retVal != eLeapRS_Success)
        {
            // std::cout << "ERROR: LeapGetFrameSize() returned " << retVal << std::endl;
            return LEAP_STATUS::LEAP_FAILED;
        }
        // Allocate enough memory
        frame = (LEAP_TRACKING_EVENT *)malloc((size_t)targetFrameSize);
        // Get the frame data
        retVal = LeapInterpolateFrame(*leap.getConnectionHandle(), LeapGetNow() + targetFrameTime, frame, targetFrameSize);
        if (retVal != eLeapRS_Success)
        {
            // std::cout << "ERROR: LeapInterpolateFrame() returned " << retVal << std::endl;
            return LEAP_STATUS::LEAP_FAILED;
        }
        curFrameTimeStamp = frame->info.timestamp;
    }
    // t_profile1.stop();
    // std::cout << "get_frame: " << t_profile1.averageLapInMilliSec() << std::endl;
    // Use the data...
    //  std::cout << "frame id: " << interpolatedFrame->tracking_frame_id << std::endl;
    //  std::cout << "frame delay (us): " << (long long int)LeapGetNow() - interpolatedFrame->info.timestamp << std::endl;
    //  std::cout << "frame hands: " << interpolatedFrame->nHands << std::endl;
    // t_profile1.start();
    for (uint32_t h = 0; h < frame->nHands; h++)
    {
        LEAP_HAND *hand = &frame->pHands[h];
        // if (debug_vec.x > 0)
        if (hand->type == eLeapHandType_Right)
            chirality = flip_x;
        float grab_angle = hand->grab_angle;
        std::vector<glm::mat4> bones_to_world;
        std::vector<glm::vec3> joints;
        std::vector<uint32_t> fingersExtended;
        // palm
        glm::vec3 palm_pos_raw = glm::vec3(hand->palm.position.x,
                                           hand->palm.position.y,
                                           hand->palm.position.z);
        glm::vec3 towards_hand_tips = glm::vec3(hand->palm.direction.x, hand->palm.direction.y, hand->palm.direction.z);
        towards_hand_tips = glm::normalize(towards_hand_tips);
        // we offset the palm to coincide with wrist, as a real hand has a wrist joint that needs to be controlled
        glm::vec3 palm_pos = palm_pos_raw + towards_hand_tips * es.magic_wrist_offset;
        glm::mat4 palm_orientation = glm::toMat4(glm::quat(hand->palm.orientation.w,
                                                           hand->palm.orientation.x,
                                                           hand->palm.orientation.y,
                                                           hand->palm.orientation.z));
        // for some reason using the "basis" from leap rotates and flips the coordinate system of the palm
        // also there is an arbitrary scale factor associated with the 3d mesh
        // so we need to fix those
        if (es.useFingerWidth)
        {
            glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), hand->palm.width * glm::vec3(es.leap_palm_local_scaler, es.leap_palm_local_scaler, es.leap_palm_local_scaler));
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
        if (es.leap_use_arm)
        {
            /* <using the arm bone (very inaccurate)> */
            glm::mat4 arm_rot = glm::toMat4(glm::quat(hand->arm.rotation.w,
                                                      hand->arm.rotation.x,
                                                      hand->arm.rotation.y,
                                                      hand->arm.rotation.z));
            // arm_rot = glm::rotate(arm_rot, glm::radians(debug_vec.x), glm::vec3(arm_rot[0][0], arm_rot[0][1], arm_rot[0][2]));
            glm::mat4 arm_translate = glm::translate(glm::mat4(1.0f), glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
            // translate arm joint in the local x direction to shorten the arm
            glm::vec3 xforward = glm::normalize(glm::vec3(arm_rot[2][0], arm_rot[2][1], arm_rot[2][2])); // 3rd column of rotation matrix is local x
            xforward *= es.magic_arm_forward_offset;
            arm_translate = glm::translate(arm_translate, xforward);
            if (es.useFingerWidth)
            {
                glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), hand->arm.width * glm::vec3(es.leap_arm_local_scaler, es.leap_arm_local_scaler, es.leap_arm_local_scaler));
                bones_to_world.push_back(arm_translate * arm_rot * chirality * magic_leap_basis_fix * scalar * local_scaler);
            }
            else
            {
                bones_to_world.push_back(arm_translate * arm_rot * chirality * magic_leap_basis_fix * scalar);
            }
            /* end <using the arm bone (very inaccurate)>*/
        }
        else
        {
            /* <using the palm bone (accurate, but slightly too rigid)>*/
            palm_pos = palm_pos_raw + towards_hand_tips * es.magic_arm_offset;
            bones_to_world.push_back(glm::translate(glm::mat4(1.0f), palm_pos) * palm_orientation);
            /* <using the palm bone (accurate, but slightly too rigid)>*/
        }
        // fingers
        for (uint32_t f = 0; f < 5; f++)
        {
            LEAP_DIGIT finger = hand->digits[f];
            fingersExtended.push_back(finger.is_extended);
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
                if (es.useFingerWidth)
                {
                    glm::mat4 local_scaler = glm::scale(glm::mat4(1.0f), finger.bones[b].width * glm::vec3(es.leap_bone_local_scaler, es.leap_bone_local_scaler, es.leap_bone_local_scaler));
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
            rightFingersExtended = std::move(fingersExtended);
            // grab_angle_right = grab_angle;
        }
        else
        {
            bones_to_world_left = std::move(bones_to_world);
            joints_left = std::move(joints);
            leftFingersExtended = std::move(fingersExtended);
            // grab_angle_left = grab_angle;
        }
    }
    // t_profile1.stop();
    // t_profile1.start();
    // std::cout << "parse_frame: " << t_profile1.averageLapInMilliSec() << std::endl;
    // Free the allocated buffer when done.
    if (leap_poll_mode)
        free(frame->pHands);
    free(frame);
    // t_profile1.stop();
    // std::cout << "inside_get_leap: " << t_profile1.getElapsedTimeInMilliSec() << std::endl;
    return LEAP_STATUS::LEAP_NEWFRAME;
}

void create_virtual_cameras(GLCamera &gl_flycamera, GLCamera &gl_projector, GLCamera &gl_camera)
{
    Camera_Mode camera_mode = es.freecam_mode ? Camera_Mode::FREE_CAMERA : Camera_Mode::FIXED_CAMERA;
    glm::mat4 w2c;
    if (es.use_leap_calib_results == static_cast<int>(LeapCalibrationSettings::AUTO))
        w2c = w2c_auto;
    else
        w2c = w2c_user;
    glm::mat4 w2p = w2c;
    // std::cout << "Using calibration data for camera and projector settings" << std::endl;
    if (es.freecam_mode)
    {
        // gl_flycamera = GLCamera(w2vc, proj_project, Camera_Mode::FREE_CAMERA);
        // gl_flycamera = GLCamera(w2vc, proj_project, Camera_Mode::FREE_CAMERA, proj_width, proj_height, 10.0f);
        gl_flycamera = GLCamera(glm::vec3(20.0f, -160.0f, 190.0f),
                                glm::vec3(-50.0f, 200.0f, -30.0f),
                                glm::vec3(0.0f, 0.0f, -1.0f),
                                camera_mode,
                                es.proj_width,
                                es.proj_height,
                                1500.0f,
                                25.0f,
                                true);
        gl_projector = GLCamera(w2p, proj_project, camera_mode, es.proj_width, es.proj_height, 25.0f, true);
        gl_camera = GLCamera(w2c, cam_project, camera_mode, es.cam_width, es.cam_height, 25.0f, true);
    }
    else
    {
        gl_projector = GLCamera(w2p, proj_project, camera_mode, es.proj_width, es.proj_height);
        gl_camera = GLCamera(w2c, cam_project, camera_mode, es.cam_width, es.cam_height);
        gl_flycamera = GLCamera(w2c, proj_project, camera_mode, es.proj_width, es.proj_height);
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

void set_texture_shader(Shader *textureShader, bool flipVer, bool flipHor, bool isGray, bool binary, float threshold,
                        int src, glm::mat4 model, glm::mat4 projection, glm::mat4 view, bool gammaCorrection, glm::vec3 bgColor)
{
    textureShader->use();
    textureShader->setMat4("view", view);
    textureShader->setMat4("projection", projection);
    textureShader->setMat4("model", model);
    textureShader->setFloat("threshold", threshold);
    textureShader->setBool("flipHor", flipHor);
    textureShader->setBool("flipVer", flipVer);
    textureShader->setBool("binary", binary);
    textureShader->setBool("isGray", isGray);
    textureShader->setInt("src", src);
    textureShader->setBool("gammaCorrection", gammaCorrection);
    textureShader->setVec3("bgColor", bgColor);
}

void set_skinned_shader(SkinningShader *skinnedShader,
                        glm::mat4 transform,
                        bool flipVer, bool flipHor,
                        bool useGGX, bool renderUV,
                        bool bake, bool useProjector, bool projOnly, bool projIsSingleC,
                        glm::mat4 projTransform,
                        bool useMetric, std::vector<float> scalarPerBone, int src)
{
    skinnedShader->use();
    skinnedShader->setMat4("mvp", transform);
    skinnedShader->setBool("flipTexVertically", flipVer);
    skinnedShader->setBool("flipTexHorizontally", flipHor);
    skinnedShader->setBool("useGGX", useGGX);
    skinnedShader->setBool("renderUV", renderUV);
    skinnedShader->setBool("bake", bake);
    skinnedShader->setBool("useProjector", useProjector);
    skinnedShader->setBool("projectorOnly", projOnly);
    skinnedShader->setBool("projectorIsSingleChannel", projIsSingleC);
    skinnedShader->setMat4("projTransform", projTransform);
    skinnedShader->setBool("useMetric", useMetric);
    if (scalarPerBone.size() != 0)
        skinnedShader->setFloatArray("gBoneMetric", scalarPerBone, scalarPerBone.size());
    skinnedShader->setInt("src", src);
}

glm::vec3 triangulate(LeapCPP &leap, const glm::vec2 &leap1, const glm::vec2 &leap2)
{
    // leap image plane is x right, and y up like opengl...
    glm::vec2 l1_vert = Helpers::NDCtoScreen(leap1, es.leap_width, es.leap_height, false);
    LEAP_VECTOR l1_vert_leap = {l1_vert.x, l1_vert.y, 1.0f};
    glm::vec2 l2_vert = Helpers::NDCtoScreen(leap2, es.leap_width, es.leap_height, false);
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

bool mp_predict_single(cv::Mat origImage, std::vector<glm::vec2> &left, std::vector<glm::vec2> &right, bool &left_detected, bool &right_detected)
{
    cv::Mat image;
    // cv::flip(origImage, image, 1);
    cv::cvtColor(origImage, image, cv::COLOR_GRAY2RGB);
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};
    PyObject *image_object = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp *)&dimensions, NPY_UINT8, image.data);
    PyObject *myResult = PyObject_CallFunction(predict_single, "(O,O)", image_object, single_detector);
    if (!myResult)
    {
        std::cout << "Call failed!" << std::endl;
        PyErr_Print();
        return false;
        // exit(1);
    }
    PyObject *pyLeftLandmarks, *pyRightLandmarks;
    int left_, right_;
    if (!PyArg_ParseTuple(myResult, "OOii", &pyLeftLandmarks, &pyRightLandmarks, &left_, &right_))
    {
        std::cout << "Parse failed!" << std::endl;
        PyErr_Print();
        return false;
        // exit(1);
    }
    left_detected = static_cast<bool>(left_);
    right_detected = static_cast<bool>(right_);
    std::vector<glm::vec2> data_vec;
    if ((!left_detected) && (!right_detected))
        return false;
    // PyObject* myResult = PyObject_CallFunction(myprofile, "O", image_object);
    PyArrayObject *leftNumpyArray = reinterpret_cast<PyArrayObject *>(pyLeftLandmarks);
    glm::vec2 *leftData = (glm::vec2 *)PyArray_DATA(leftNumpyArray);
    std::vector<glm::vec2> raw_left(leftData, leftData + PyArray_SIZE(leftNumpyArray) / 2);
    left = std::move(raw_left);
    PyArrayObject *rightNumpyArray = reinterpret_cast<PyArrayObject *>(pyRightLandmarks);
    glm::vec2 *rightData = (glm::vec2 *)PyArray_DATA(rightNumpyArray);
    std::vector<glm::vec2> raw_right(rightData, rightData + PyArray_SIZE(rightNumpyArray) / 2);
    right = std::move(raw_right);
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
    return true;
}
bool mp_predict(cv::Mat origImage, int timestamp, std::vector<glm::vec2> &left, std::vector<glm::vec2> &right, bool &left_detected, bool &right_detected)
{
    cv::Mat image;
    cv::flip(origImage, image, 1);
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    std::lock_guard<std::mutex> lock(es.py_mutex);
    // cv::Mat image = cv::imread("../../resource/hand.png", cv::IMREAD_GRAYSCALE);
    // std::cout << "mp received timestamp: " << timestamp << std::endl;
    // cv::Mat image;
    // cv::Mat image = cv::imread("../../debug/ss/sg0o.0_raw_cam.png");
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
        return false;
        // exit(1);
    }
    PyObject *pyLeftLandmarks, *pyRightLandmarks;
    int left_, right_;
    if (!PyArg_ParseTuple(myResult, "OOii", &pyLeftLandmarks, &pyRightLandmarks, &left_, &right_))
    {
        std::cout << "Parse failed!" << std::endl;
        PyErr_Print();
        return false;
        // exit(1);
    }
    left_detected = static_cast<bool>(left_);
    right_detected = static_cast<bool>(right_);
    std::vector<glm::vec2> data_vec;
    if ((!left_detected) && (!right_detected))
        return false;
    // PyObject* myResult = PyObject_CallFunction(myprofile, "O", image_object);
    PyArrayObject *leftNumpyArray = reinterpret_cast<PyArrayObject *>(pyLeftLandmarks);
    glm::vec2 *leftData = (glm::vec2 *)PyArray_DATA(leftNumpyArray);
    std::vector<glm::vec2> raw_left(leftData, leftData + PyArray_SIZE(leftNumpyArray) / 2);
    left = std::move(raw_left);
    PyArrayObject *rightNumpyArray = reinterpret_cast<PyArrayObject *>(pyRightLandmarks);
    glm::vec2 *rightData = (glm::vec2 *)PyArray_DATA(rightNumpyArray);
    std::vector<glm::vec2> raw_right(rightData, rightData + PyArray_SIZE(rightNumpyArray) / 2);
    right = std::move(raw_right);
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
    return true;
}

void handleCameraInput(CGrabResultPtr ptrGrabResult, bool simulatedCam, cv::Mat simulatedImage)
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
    if (!simulatedCam)
    {
        bool sucess = camera.capture_single_image(ptrGrabResult);
        if (!sucess)
        {
            std::cout << "Failed to capture image" << std::endl;
            exit(1);
        }
        // camera_queue.wait_dequeue(ptrGrabResult);
        if (es.undistortCamera)
        {
            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
            cv::remap(camImageOrig, camImage, undistort_map1, undistort_map2, cv::INTER_LINEAR);
            camTexture.load(camImage.data, true, es.cam_buffer_format);
        }
        else
        {
            camTexture.load((uint8_t *)ptrGrabResult->GetBuffer(), true, es.cam_buffer_format);
            camImageOrig = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
        }
    }
    else
    {
        // camImageOrig = cv::imread("../../resource/hand.png", cv::IMREAD_GRAYSCALE);
        camImageOrig = simulatedImage;
        camTexture.load((uint8_t *)camImageOrig.data, true, es.cam_buffer_format);
    }
    // camera_queue.wait_dequeue(ptrGrabResult);
    // std::cout << "after: " << camera_queue.size_approx() << std::endl;
    // curCamImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer()).clone();
    // curCamBuf = std::vector<uint8_t>((uint8_t *)ptrGrabResult->GetBuffer(), (uint8_t *)ptrGrabResult->GetBuffer() + ptrGrabResult->GetImageSize());
}

bool loadGamePoses(std::string loadPath, std::vector<std::vector<glm::mat4>> &poses)
{
    for (const auto &entry : fs::directory_iterator(loadPath))
    {
        if (entry.path().extension() != ".npy")
            continue;
        // if (entry.path().string().find("bones_left") != std::string::npos)
        // {
        std::vector<glm::mat4> raw_game_bones_left;
        cnpy::NpyArray bones_left_npy;
        std::cout << "loading: " << entry.path().string() << std::endl;
        bones_left_npy = cnpy::npy_load(entry.path().string());
        std::vector<float> raw_data = bones_left_npy.as_vec<float>();
        for (int i = 0; i < raw_data.size(); i += 16)
        {
            raw_game_bones_left.push_back(glm::make_mat4(raw_data.data() + i));
        }
        poses.push_back(raw_game_bones_left);
        // }
    }
    return true;
}

void saveLeapData(LEAP_STATUS leap_status, uint64_t image_timestamp, bool record_images)
{
    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME || es.record_every_frame)
    {
        if (es.two_hand_recording)
        {
            if ((bones_to_world_right.size() > 0) && (joints_right.size() > 0) && (bones_to_world_left.size() > 0) && (joints_left.size() > 0))
            {
                savedLeapBonesLeft.push_back(bones_to_world_left);
                savedLeapJointsLeft.push_back(joints_left);
                savedLeapBonesRight.push_back(bones_to_world_right);
                savedLeapJointsRight.push_back(joints_right);
                savedLeapTimestamps.push_back(es.curFrameTimeStamp);
                if (record_images)
                {
                    recordedImages.push_back(camImageOrig);
                    savedCameraTimestamps.push_back(image_timestamp);
                }
            }
        }
        else
        {
            if (es.recordedHand == static_cast<int>(Hand::RIGHT))
            {
                if ((bones_to_world_right.size() > 0) && (joints_right.size() > 0))
                {
                    savedLeapBonesRight.push_back(bones_to_world_right);
                    savedLeapJointsRight.push_back(joints_right);
                    savedLeapTimestamps.push_back(es.curFrameTimeStamp);
                    if (record_images)
                    {
                        recordedImages.push_back(camImageOrig);
                        savedCameraTimestamps.push_back(image_timestamp);
                    }
                }
            }
            else
            {
                if ((bones_to_world_left.size() > 0) && (joints_left.size() > 0))
                {
                    savedLeapBonesLeft.push_back(bones_to_world_left);
                    savedLeapJointsLeft.push_back(joints_left);
                    savedLeapTimestamps.push_back(es.curFrameTimeStamp);
                    if (record_images)
                    {
                        recordedImages.push_back(camImageOrig);
                        savedCameraTimestamps.push_back(image_timestamp);
                    }
                }
            }
        }
    }
}

void saveSession(std::string savepath, bool record_images)
{
    fs::path savepath_path(savepath);
    if (!fs::exists(savepath_path))
    {
        fs::create_directory(savepath_path);
    }
    if (es.leap_global_scaler != 1.0f)
    {
        std::cout << "cannot record session with global scale transform" << std::endl;
        return;
    }
    fs::path bones_left_path = savepath_path / fs::path(std::string("bones_left.npy"));
    fs::path joints_left_path = savepath_path / fs::path(std::string("joints_left.npy"));
    fs::path timestamps_path = savepath_path / fs::path(std::string("timestamps.npy"));
    fs::path bones_right_path = savepath_path / fs::path(std::string("bones_right.npy"));
    fs::path joints_right_path = savepath_path / fs::path(std::string("joints_right.npy"));
    if (es.two_hand_recording)
    {
        if ((savedLeapBonesLeft.size() > 0) && (savedLeapBonesRight.size() > 0))
        {
            for (int i = 0; i < savedLeapBonesLeft.size(); i++)
            {
                cnpy::npy_save(bones_left_path.string().c_str(), &savedLeapBonesLeft[i][0][0].x, {1, savedLeapBonesLeft[i].size(), 4, 4}, "a");
                cnpy::npy_save(joints_left_path.string().c_str(), &savedLeapJointsLeft[i][0].x, {1, savedLeapJointsLeft[i].size(), 3}, "a");
                cnpy::npy_save(timestamps_path.string().c_str(), &savedLeapTimestamps[i], {1}, "a");
                cnpy::npy_save(bones_right_path.string().c_str(), &savedLeapBonesRight[i][0][0].x, {1, savedLeapBonesRight[i].size(), 4, 4}, "a");
                cnpy::npy_save(joints_right_path.string().c_str(), &savedLeapJointsRight[i][0].x, {1, savedLeapJointsRight[i].size(), 3}, "a");
            }
        }
    }
    else
    {
        if (es.recordedHand == static_cast<int>(Hand::RIGHT))
        {
            if (savedLeapBonesRight.size() > 0)
            {
                for (int i = 0; i < savedLeapBonesRight.size(); i++)
                {
                    cnpy::npy_save(bones_right_path.string().c_str(), &savedLeapBonesRight[i][0][0].x, {1, savedLeapBonesRight[i].size(), 4, 4}, "a");
                    cnpy::npy_save(joints_right_path.string().c_str(), &savedLeapJointsRight[i][0].x, {1, savedLeapJointsRight[i].size(), 3}, "a");
                    cnpy::npy_save(timestamps_path.string().c_str(), &savedLeapTimestamps[i], {1}, "a");
                }
            }
        }
        else
        {
            if (savedLeapBonesLeft.size() > 0)
            {
                for (int i = 0; i < savedLeapBonesLeft.size(); i++)
                {
                    cnpy::npy_save(bones_left_path.string().c_str(), &savedLeapBonesLeft[i][0][0].x, {1, savedLeapBonesLeft[i].size(), 4, 4}, "a");
                    cnpy::npy_save(joints_left_path.string().c_str(), &savedLeapJointsLeft[i][0].x, {1, savedLeapJointsLeft[i].size(), 3}, "a");
                    cnpy::npy_save(timestamps_path.string().c_str(), &savedLeapTimestamps[i], {1}, "a");
                }
            }
        }
    }
    if (record_images)
    {
        for (int i = 0; i < recordedImages.size(); i++)
        {
            fs::path image_path = savepath_path / fs::path(std::format("{:06d}", savedCameraTimestamps[i]) + std::string("_cam.png"));
            cv::Mat image;
            cv::flip(recordedImages[i], image, 1);
            cv::imwrite(image_path.string(), image);
        }
    }
}

void saveSession(std::string savepath, LEAP_STATUS leap_status, uint64_t image_timestamp, bool record_images)
{
    fs::path savepath_path(savepath);
    if (!fs::exists(savepath_path))
    {
        fs::create_directory(savepath_path);
    }
    if (es.leap_global_scaler != 1.0f)
    {
        std::cout << "cannot record session with global scale transform" << std::endl;
        return;
    }
    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME || es.record_every_frame)
    {
        if (bones_to_world_left.size() > 0)
        {
            // save leap frame
            fs::path bones_left_path = savepath_path / fs::path(std::string("bones_left.npy"));
            fs::path timestamps_path = savepath_path / fs::path(std::string("timestamps.npy"));
            // fs::path app_timestamps_path = savepath_path / fs::path(std::string("app_timestamps.npy"));
            fs::path joints_left_path = savepath_path / fs::path(std::string("joints_left.npy"));
            // fs::path bones_right(savepath + tmp_filename.filename().string() + std::string("_bones_right.npy"));
            // fs::path joints_left_path(savepath + tmp_filename.filename().string() + std::string("_joints_left.npy"));
            // fs::path timestamps_path(savepath + tmp_filename.filename().string() + std::string("_timestamps.npy"));
            // fs::path session(savepath + tmp_filename.filename().string() + std::string("_session.npz"));
            // cnpy::npy_save(session.string().c_str(), "joints", &skeleton_vertices[0].x, {skeleton_vertices.size(), 3}, "a");
            // cnpy::npy_save(session.string().c_str(), "bones_left", &bones_to_world_left[0][0].x, {bones_to_world_left.size(), 4, 4}, "a");
            // cnpy::npz_save(session.string().c_str(), "bones_right", &bones_to_world_right[0][0].x, {bones_to_world_right.size(), 4, 4}, "a");
            cnpy::npy_save(bones_left_path.string().c_str(), &bones_to_world_left[0][0].x, {1, bones_to_world_left.size(), 4, 4}, "a");
            cnpy::npy_save(timestamps_path.string().c_str(), &es.curFrameTimeStamp, {1}, "a");
            // cnpy::npy_save(app_timestamps_path.string().c_str(), &curAppFrameTimeStamp, {1}, "a");
            cnpy::npy_save(joints_left_path.string().c_str(), &joints_left[0].x, {1, joints_left.size(), 3}, "a");
            // cnpy::npy_save(bones_right.string().c_str(), &bones_to_world_left[0][0], {bones_to_world_right.size(), 4, 4}, "a");
            // cnpy::npy_save(joints.string().c_str(), &joints_left[0].x, {1, joints_left.size(), 3}, "a");
            if (record_images)
            {
                // also save the current camera image and finger joints
                fs::path image_path = savepath_path / fs::path(std::format("{:06d}", image_timestamp) + std::string("_cam.png"));
                cv::Mat image;
                cv::flip(camImageOrig, image, 1);
                cv::imwrite(image_path.string(), camImageOrig);
            }
        }
    }
}

LEAP_STATUS handleLeapInput(int num_frames)
{
    LEAP_STATUS leap_status;
    if (es.leap_poll_mode)
        num_frames = 1;
    std::vector<std::vector<glm::mat4>> tmp_bones_to_world_left, tmp_bones_to_world_right;
    std::vector<std::vector<glm::vec3>> tmp_joints_left, tmp_joints_right;
    std::vector<int32_t> time_delays = integer_linear_spacing(es.magic_leap_time_delay - es.leap_accumulate_spread, es.magic_leap_time_delay + es.leap_accumulate_spread, num_frames);
    // t_profile0.start();
    for (int i = 0; i < num_frames; i++)
    {
        leap_status = getLeapFrame(leap, time_delays[i], bones_to_world_left, bones_to_world_right, joints_left, joints_right, left_fingers_extended, right_fingers_extended, es.leap_poll_mode, es.curFrameID, es.curFrameTimeStamp);
        if (leap_status == LEAP_STATUS::LEAP_NEWFRAME)
        {
            if (joints_left.size() > 0)
            {
                tmp_joints_left.push_back(std::move(joints_left));
                tmp_bones_to_world_left.push_back(std::move(bones_to_world_left));
            }
            if (joints_right.size() > 0)
            {
                tmp_bones_to_world_right.push_back(std::move(bones_to_world_right));
                tmp_joints_right.push_back(std::move(joints_right));
            }
        }
        else
        {
            return leap_status;
        }
    }
    // t_profile0.stop();
    // std::cout << "outside_get_leap: " << t_profile0.getElapsedTimeInMilliSec() << std::endl;
    joints_left = Helpers::accumulate(tmp_joints_left);
    joints_right = Helpers::accumulate(tmp_joints_right);
    bones_to_world_left = Helpers::accumulate(tmp_bones_to_world_left);
    bones_to_world_right = Helpers::accumulate(tmp_bones_to_world_right);
    // for (int i = 0; i < bones_to_world_left.size(); i++)
    // {
    //     joints_left.push_back(glm::vec3(bones_to_world_left[i][3]));
    // }
    // for (int i = 0; i < bones_to_world_right.size(); i++)
    // {
    //     joints_right.push_back(glm::vec3(bones_to_world_right[i][3]));
    // }
    return leap_status;
}

LEAP_STATUS handleLeapInput()
{
    LEAP_STATUS leap_status;
    leap_status = getLeapFrame(leap, es.magic_leap_time_delay, bones_to_world_left, bones_to_world_right, joints_left, joints_right, left_fingers_extended, right_fingers_extended, es.leap_poll_mode, es.curFrameID, es.curFrameTimeStamp);
    if (leap_status == LEAP_STATUS::LEAP_NEWFRAME) // deal with user setting a global scale transform
    {
        glm::mat4 global_scale_transform = glm::scale(glm::mat4(1.0f), glm::vec3(es.leap_global_scaler));
        if (bones_to_world_right.size() > 0)
        {
            glm::mat4 right_translate = glm::translate(glm::mat4(1.0f), glm::vec3(bones_to_world_right[0][3][0], bones_to_world_right[0][3][1], bones_to_world_right[0][3][2]));
            es.global_scale_right = right_translate * global_scale_transform * glm::inverse(right_translate);
        }
        if (bones_to_world_left.size() > 0)
        {
            glm::mat4 left_translate = glm::translate(glm::mat4(1.0f), glm::vec3(bones_to_world_left[0][3][0], bones_to_world_left[0][3][1], bones_to_world_left[0][3][2]));
            es.global_scale_left = left_translate * global_scale_transform * glm::inverse(left_translate);
        }
    }
    return leap_status;
}

void handlePostProcess(SkinnedModel &leftHandModel,
                       SkinnedModel &rightHandModel,
                       Texture &camTexture,
                       std::unordered_map<std::string, Shader *> &shader_map)
{
    Shader *textureShader = shader_map["textureShader"];
    Shader *vcolorShader = shader_map["vcolorShader"];
    Shader *maskShader = shader_map["maskShader"];
    Shader *jfaInitShader = shader_map["jfaInitShader"];
    Shader *jfaShader = shader_map["jfaShader"];
    Shader *NNShader = shader_map["NNShader"];
    Shader *uv_NNShader = shader_map["uv_NNShader"];
    Shader *gridColorShader = shader_map["gridColorShader"];
    Shader *blurShader = shader_map["blurShader"];
    Texture *leftHandTexture;
    Texture *rightHandTexture;
    switch (es.texture_mode)
    {
    case static_cast<int>(TextureMode::ORIGINAL):
    {
        leftHandTexture = leftHandModel.GetMaterial().pDiffuse;
        rightHandTexture = rightHandModel.GetMaterial().pDiffuse;
        break;
    }
    case static_cast<int>(TextureMode::BAKED):
    {
        leftHandTexture = bake_fbo_left.getTexture();
        rightHandTexture = bake_fbo_right.getTexture();
        break;
    }
    case static_cast<int>(TextureMode::FROM_FILE):
    {
        leftHandTexture = dynamicTexture;
        rightHandTexture = dynamicTexture;
        break;
    }
    case static_cast<int>(TextureMode::DYNAMIC):
    {
        leftHandTexture = dynamic_fbo.getTexture();
        rightHandTexture = dynamic_fbo.getTexture();
        break;
    }
    default:
    {
        leftHandTexture = leftHandModel.GetMaterial().pDiffuse;
        rightHandTexture = rightHandModel.GetMaterial().pDiffuse;
        break;
    }
    }

    switch (es.postprocess_mode)
    {
    case static_cast<int>(PostProcessMode::NONE):
    {
        // bind fbo
        postprocess_fbo.bind();
        // bind texture
        if (es.use_mls || es.use_of)
            mls_fbo.getTexture()->bind();
        else
            hands_fbo.getTexture()->bind();
        // render
        set_texture_shader(textureShader, false, false, false, false, es.masking_threshold, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), false, es.mask_bg_color);
        fullScreenQuad.render(false, false, true);
        // if (skeleton_vertices.size() > 0)
        // {
        //     vcolorShader.use();
        //     vcolorShader.setMat4("mvp", glm::mat4(1.0f));
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
        set_texture_shader(textureShader, true, true, true, es.threshold_flag || es.threshold_flag2, es.masking_threshold);
        camTexture.bind();
        postprocess_fbo.bind();
        fullScreenQuad.render();
        postprocess_fbo.unbind();
        break;
    }
    case static_cast<int>(PostProcessMode::OVERLAY):
    {
        maskShader->use();
        maskShader->setVec3("bgColor", es.mask_bg_color);
        maskShader->setVec3("fgColor", es.mask_fg_color);
        maskShader->setFloat("alpha", es.mask_alpha);
        maskShader->setBool("missingColorIsCamera", es.mask_missing_color_is_camera);
        maskShader->setVec3("missingInfoColor", es.mask_missing_info_color);
        maskShader->setVec3("unusedInfoColor", es.mask_unused_info_color);
        maskShader->setBool("fgSingleColor", es.mask_fg_single_color);
        if (es.use_mls || es.use_of)
            postProcess.mask(maskShader, mls_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, es.masking_threshold);
        else
            postProcess.mask(maskShader, hands_fbo.getTexture()->getTexture(), camTexture.getTexture(), &postprocess_fbo, es.masking_threshold);
        break;
    }
    case static_cast<int>(PostProcessMode::JUMP_FLOOD):
    {
        if (es.auto_pilot)
        {
            int n_visible_hands = (projected_filtered_left.size() > 0) + (projected_filtered_right.size() > 0);
            if (n_visible_hands == 1)
                es.postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD_UV);
        }
        if (es.use_mls || es.use_of)
            postProcess.jump_flood(*jfaInitShader, *jfaShader, *NNShader,
                                   mls_fbo.getTexture()->getTexture(),
                                   camTexture.getTexture(),
                                   &postprocess_fbo, es.masking_threshold, es.jfa_distance_threshold, es.mask_bg_color);
        else
            postProcess.jump_flood(*jfaInitShader, *jfaShader, *NNShader,
                                   hands_fbo.getTexture()->getTexture(),
                                   camTexture.getTexture(),
                                   &postprocess_fbo, es.masking_threshold, es.jfa_distance_threshold, es.mask_bg_color);
        break;
    }
    case static_cast<int>(PostProcessMode::JUMP_FLOOD_UV):
    {
        if (es.auto_pilot)
        {
            int n_visible_hands = (projected_filtered_left.size() > 0) + (projected_filtered_right.size() > 0);
            if (n_visible_hands == 2)
                es.postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD);
        }
        // todo: assumes both hand use same texture, which is not the case generally
        if (es.use_mls || es.use_of)
            postProcess.jump_flood_uv(*jfaInitShader, *jfaShader, *uv_NNShader, mls_fbo.getTexture()->getTexture(),
                                      leftHandTexture->getTexture(),
                                      camTexture.getTexture(),
                                      &postprocess_fbo, es.masking_threshold, es.jfa_distance_threshold, es.jfa_seam_threshold, es.mask_bg_color);
        else
            postProcess.jump_flood_uv(*jfaInitShader, *jfaShader, *uv_NNShader, uv_fbo.getTexture()->getTexture(),
                                      leftHandTexture->getTexture(),
                                      camTexture.getTexture(),
                                      &postprocess_fbo, es.masking_threshold, es.jfa_distance_threshold, es.jfa_seam_threshold, es.mask_bg_color);
        break;
    }
    default:
        break;
    }
    // add blur to final render
    if (es.postprocess_blur)
    {
        postProcess.gaussian_blur(blurShader, &postprocess_fbo, &postprocess_fbo2, es.dst_width, es.dst_height);
    }
    // apply the homography between camera and projector
    c2p_fbo.bind();
    if (es.use_coaxial_calib)
        set_texture_shader(textureShader, false, false, false, false, es.masking_threshold, 0, glm::mat4(1.0f), c2p_homography);
    else
        set_texture_shader(textureShader, false, false, false, false, es.masking_threshold, 0, glm::mat4(1.0f), glm::mat4(1.0f));
    postprocess_fbo.getTexture()->bind();
    fullScreenQuad.render();
    // use this opportunity to perhaps render some debug info
    if (es.mls_show_grid)
    {
        gridColorShader->use();
        gridColorShader->setBool("flipVer", false);
        deformationGrid.renderGridLines();
    }
    if (es.show_of && es.use_of)
    {
        std::lock_guard<std::mutex> lock(es.mls_mutex);
        vcolorShader->use();
        if (ControlPointsQ.size() == es.of_debug.size())
        {
            for (int i = 0; i < es.of_debug.size(); i++)
            {
                glm::vec2 avg_flow = es.of_debug[i];
                cv::Point2f pointq = Helpers::NDCtoScreen(ControlPointsQ[i], es.of_downsize.width, es.of_downsize.height);
                std::vector<glm::vec2> square_endpoints =
                    {
                        glm::vec2(pointq.x - (es.of_roi / 2), pointq.y - (es.of_roi / 2)),
                        glm::vec2(pointq.x + (es.of_roi / 2), pointq.y - (es.of_roi / 2)),
                        glm::vec2(pointq.x + (es.of_roi / 2), pointq.y + (es.of_roi / 2)),
                        glm::vec2(pointq.x - (es.of_roi / 2), pointq.y + (es.of_roi / 2)),
                    };
                square_endpoints = Helpers::ScreenToNDC(square_endpoints, es.of_downsize.width, es.of_downsize.height);
                PointCloud square(square_endpoints, es.screen_verts_color_green);
                square.renderAsLineLoop();
                std::vector<glm::vec2> dv_endpoints =
                    {
                        glm::vec2(pointq.x, pointq.y),
                        glm::vec2(pointq.x + 5 * avg_flow[0], pointq.y + 5 * avg_flow[1]),
                    };
                dv_endpoints = Helpers::ScreenToNDC(dv_endpoints, es.of_downsize.width, es.of_downsize.height);
                PointCloud dv(dv_endpoints, es.screen_verts_color_red);
                dv.renderAsLines();
            }
        }
    }
    if (es.show_landmarks)
    {
        float landmarkSize = 8.0f;
        vcolorShader->use();
        if (es.use_coaxial_calib)
            vcolorShader->setMat4("mvp", c2p_homography);
        else
            vcolorShader->setMat4("mvp", glm::mat4(1.0f));
        std::vector<glm::vec2> pfl = Helpers::vec3to2(projected_filtered_left);
        std::vector<glm::vec2> pfr = Helpers::vec3to2(projected_filtered_right);
        PointCloud cloud_left(pfl, es.screen_verts_color_white);
        PointCloud cloud_right(pfr, es.screen_verts_color_white);
        cloud_left.render(landmarkSize);
        cloud_right.render(landmarkSize);
        if (es.use_mls || es.use_of)
        {
            std::lock_guard<std::mutex> lock(es.mls_mutex);
            // PointCloud cloud_src_input_left(ControlPointsP_input_left, es.screen_verts_color_red); // leap landmarks used as input for landmark thread
            // PointCloud cloud_src_input_right(ControlPointsP_input_right, es.screen_verts_color_red);
            std::vector<glm::vec2> pglm = Helpers::cv2glm(ControlPointsP);
            std::vector<glm::vec2> qglm = Helpers::cv2glm(ControlPointsQ);
            PointCloud cloud_src(pglm, es.screen_verts_color_cyan);    // leap landmarks used for mls
            PointCloud cloud_dst(qglm, es.screen_verts_color_magenta); // camera landmarks used for mls
            // cloud_src_input_left.render(landmarkSize);
            // cloud_src_input_right.render(landmarkSize);
            cloud_src.render(landmarkSize);
            cloud_dst.render(landmarkSize);
        }
    }
    c2p_fbo.unbind();
}

void getLightTransform(glm::mat4 &lightProjection,
                       glm::mat4 &lightView,
                       const std::vector<glm::mat4> &bones2world)
{
    if (es.light_is_projector)
    {
        lightProjection = gl_projector.getProjectionMatrix();
        lightView = gl_projector.getViewMatrix();
    }
    else
    {
        glm::vec3 at = es.light_at;
        glm::vec3 to = es.light_to;
        if (es.light_relative)
        {
            glm::vec3 palm_loc = glm::vec3(bones2world[0][3][0], bones2world[0][3][1], bones2world[0][3][2]);
            at += palm_loc;
            to += palm_loc;
        }
        lightProjection = glm::ortho(-100.0f, 100.0f, -100.0f, 100.0f, es.light_near, es.light_far);
        lightView = glm::lookAt(at,
                                to,
                                es.light_up);
    }
}

void handleSkinning(const std::vector<glm::mat4> &bones2world,
                    bool isRightHand,
                    bool isFirstHand,
                    std::unordered_map<std::string, Shader *> &shader_map,
                    SkinnedModel &handModel,
                    glm::mat4 cam_view_transform,
                    glm::mat4 cam_projection_transform)
{
    SkinningShader *skinnedShader = dynamic_cast<SkinningShader *>(shader_map["skinnedShader"]);
    Shader *vcolorShader = shader_map["vcolorShader"];
    Shader *textureShader = shader_map["textureShader"];
    // std::vector<glm::mat4> bones_to_world = isRightHand ? bones_to_world_right : bones_to_world_left;
    if (bones2world.size() > 0)
    {
        glm::mat4 global_scale = isRightHand ? es.global_scale_right : es.global_scale_left;
        if (isFirstHand)
            hands_fbo.bind(true);
        else
            hands_fbo.bind(false);
        glEnable(GL_DEPTH_TEST); // depth test on (todo: why is it off to begin with?)
        // a first pass for depth if shadow mapping is enabled
        if (es.use_shadow_mapping)
        {
            hands_fbo.unbind();
            // first pass for shadows
            // glCullFace(GL_FRONT); // solves peter panning
            shadowmap_fbo.bind();
            skinnedShader->use();
            glm::mat4 lightProjection, lightView;
            getLightTransform(lightProjection, lightView, bones2world);
            set_skinned_shader(skinnedShader, lightProjection * lightView * global_scale);
            handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
            // glCullFace(GL_BACK); // reset original culling face
            shadowmap_fbo.unbind();
            // debug pass to see depth map
            // hands_fbo.bind();
            // set_texture_shader(textureShader, false, false, true);
            // unsigned int depthMap = shadowmap_fbo.getDepthTexture();
            // glActiveTexture(GL_TEXTURE0);
            // glBindTexture(GL_TEXTURE_2D, depthMap);
            // fullScreenQuad.render();
            // hands_fbo.saveColorToFile("test.png");
            // second pass for rendering
            if (isFirstHand)
                hands_fbo.bind(true);
            else
                hands_fbo.bind(false);
            // dirLight.calcLocalDirection(bones2world[0]);
            // dirLight.setWorldDirection(debug_vec);
            // skinnedShader->SetDirectionalLight(dirLight);
            // glm::vec3 camWorldPos = glm::vec3(cam_view_transform[3][0], cam_view_transform[3][1], cam_view_transform[3][2]);
            // skinnedShader->SetCameraLocalPos(camWorldPos);
            skinnedShader->use();
            unsigned int depthMap = shadowmap_fbo.getDepthTexture();
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, depthMap);
            skinnedShader->setInt("shadowMap", 4);
            skinnedShader->setMat4("lightTransform", lightProjection * lightView);
            skinnedShader->setBool("useShadow", true);
            skinnedShader->setFloat("shadowBias", es.shadow_bias);
        }
        else
        {
            skinnedShader->use();
            skinnedShader->setBool("useShadow", false);
        }
        switch (es.material_mode)
        {
        case static_cast<int>(MaterialMode::DIFFUSE):
        {
            /* render skinned mesh to fbo, in camera space*/
            switch (es.texture_mode)
            {
            case static_cast<int>(TextureMode::ORIGINAL): // original texture loaded with mesh
            {
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
                break;
            }
            case static_cast<int>(TextureMode::BAKED): // a baked texture from a bake operation
            {
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                if (isRightHand)
                    handModel.Render(*skinnedShader, bones2world, es.rotx, false, bake_fbo_right.getTexture());
                else
                    handModel.Render(*skinnedShader, bones2world, es.rotx, false, bake_fbo_left.getTexture());
                break;
            }
            case static_cast<int>(TextureMode::PROJECTIVE): // a projective texture from the virtual cameras viewpoint
            {
                projectiveTexture = texturePack[es.curSelectedPTexture];
                dynamicTexture = texturePack[es.curSelectedTexture];
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale,
                                   false, false, false, false, false, true, false, false, cam_projection_transform * cam_view_transform);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, dynamicTexture, projectiveTexture);
                break;
            }
            case static_cast<int>(TextureMode::FROM_FILE): // a projective texture from the virtual cameras viewpoint
            {
                dynamicTexture = texturePack[es.curSelectedTexture];
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, dynamicTexture);
                break;
            }
            case static_cast<int>(TextureMode::CAMERA): // project camera input onto mesh
            {
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale,
                                   true, true, false, false, false, true, true, true, cam_projection_transform * cam_view_transform);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr, &camTexture);
                break;
            }
            case static_cast<int>(TextureMode::SHADER):
            {
                hands_fbo.unbind();
                game_fbo.bind();
                Shader *shaderToy = shader_map["shaderToySea"]; // shaderToySea, shaderToyCloud
                shaderToy->use();
                shaderToy->setFloat("iTime", t_app.getElapsedTimeInSec());
                shaderToy->setVec2("iResolution", glm::vec2(es.proj_width, es.proj_height));
                shaderToy->setVec2("iMouse", glm::vec2(0.5f, 0.5f));
                fullScreenQuad.render();
                game_fbo.unbind();
                if (isFirstHand)
                    hands_fbo.bind(true);
                else
                    hands_fbo.bind(false);
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, game_fbo.getTexture());
                break;
            }
            case static_cast<int>(TextureMode::MULTI_PASS_SHADER):
            {
                hands_fbo.unbind();
                Shader *shaderToyBufferA = shader_map["shaderToyGameBufferA"];
                Shader *shaderToyBufferB = shader_map["shaderToyGameBufferB"];
                Shader *shaderToyImage = shader_map["shaderToyGameImage"];
                // BufferA
                FBO *fbo_content;
                FBO *fbo_texture;
                if (es.gameFrameCount % 2 == 0)
                {
                    fbo_content = &game_fbo_aux1;
                    fbo_texture = &game_fbo_aux2;
                }
                else
                {
                    fbo_content = &game_fbo_aux2;
                    fbo_texture = &game_fbo_aux1;
                }
                if ((es.gameFrameCount == 0) || (es.prevGameFrameCount != es.gameFrameCount))
                {
                    fbo_content->bind();
                    shaderToyBufferA->use();
                    fbo_texture->getTexture()->bind(GL_TEXTURE0);
                    game_fbo_aux3.getTexture()->bind(GL_TEXTURE1);
                    // shaderToyBufferA->setFloat("iTime", time);
                    // shaderToyBufferA->setFloat("iTimeDelta", deltaTime);
                    shaderToyBufferA->setVec2("iResolution", glm::vec2(es.proj_width, es.proj_height));
                    // shaderToyBufferA->setVec2("iMouse", glm::vec2(250.0f, 300.0f));
                    shaderToyBufferA->setInt("iFrame", static_cast<int>(es.gameFrameCount * 1.0f));
                    shaderToyBufferA->setInt("iChannel0", 0);
                    shaderToyBufferA->setInt("iChannel1", 1);
                    // check if palm is rotated left
                    glm::vec3 palm_z = bones2world[0][2];
                    glm::vec3 cam_z = glm::vec3(cam_view_transform[2][0], cam_view_transform[2][1], cam_view_transform[2][2]);
                    float dot_prod = glm::dot(palm_z, cam_z);
                    // std::cout << dot_prod << std::endl;
                    shaderToyBufferA->setBool("left", dot_prod <= -0.5f);
                    shaderToyBufferA->setBool("right", dot_prod > 1.3f);
                    fullScreenQuad.render();
                    fbo_content->unbind();
                    // BufferB
                    game_fbo_aux3.bind();
                    fbo_content->getTexture()->bind();
                    shaderToyBufferB->use();
                    shaderToyBufferB->setVec2("iResolution", glm::vec2(es.proj_width, es.proj_height));
                    shaderToyBufferB->setInt("iChannel0", 0);
                    fullScreenQuad.render();
                    game_fbo_aux3.unbind();
                }
                // Image
                game_fbo.bind();
                shaderToyImage->use();
                shaderToyImage->setVec2("iResolution", glm::vec2(es.proj_width, es.proj_height));
                // shaderToyImage->setVec2("iMouse", glm::vec2(0.5f, 0.5f));
                shaderToyImage->setInt("iFrame", static_cast<int>(es.gameFrameCount * 1.0f));
                shaderToyImage->setInt("iChannel0", 0);
                shaderToyImage->setInt("iChannel1", 1);
                fbo_content->getTexture()->bind(GL_TEXTURE0);
                game_fbo_aux3.getTexture()->bind(GL_TEXTURE1);
                // fullScreenQuad.render();
                Quad handQuad(es.game_verts);
                handQuad.render();
                game_fbo.unbind();
                if (isFirstHand)
                    hands_fbo.bind(true);
                else
                    hands_fbo.bind(false);
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, game_fbo.getTexture());
                // dynamicTexture = texturePack[curSelectedTexture];
                // set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale,
                //                    false, false, false, false, false, true, false, false, cam_projection_transform * cam_view_transform);
                // handModel.Render(*skinnedShader, bones2world, rotx, false, dynamicTexture, game_fbo.getTexture());
                es.gameTime += es.deltaTime;
                if (es.gameTime >= es.gameSpeed)
                {
                    es.prevGameFrameCount = es.gameFrameCount;
                    es.gameFrameCount++;
                    es.gameTime = 0.0f;
                }
                break;
            }
            case static_cast<int>(TextureMode::DYNAMIC):
            {
                dynamicTexture = dynamic_fbo.getTexture();
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, dynamicTexture);
                break;
            }
            default: // original texture loaded with mesh
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
                break;
            }
            break;
        }
        case static_cast<int>(MaterialMode::GGX): // uses GGX-like material
        {
            skinnedShader->use();
            if (es.surround_light)
            {
                es.light_phi += es.surround_light_speed;
            }
            es.light_at = glm::vec3(es.light_radius * sin(es.light_theta) * cos(es.light_phi),
                                    es.light_radius * sin(es.light_theta) * sin(es.light_phi),
                                    es.light_radius * cos(es.light_theta));
            dirLight.setColor(es.light_color);
            dirLight.setAmbientIntensity(es.light_ambient_intensity);
            dirLight.setDiffuseIntensity(es.light_diffuse_intensity);
            dirLight.setWorldDirection(es.light_to - es.light_at);
            // glm::mat4 lightProjection, lightView;
            // getLightTransform(lightProjection, lightView, bones_to_world_right);
            // dirLight.setWorldDirection(debug_vec);
            // dirLight.calcLocalDirection(glm::inverse(lightView));
            skinnedShader->SetDirectionalLight(dirLight); // set direction in world space
            skinnedShader->setBool("useNormalMap", es.use_normal_mapping);
            skinnedShader->setBool("useArmMap", es.use_arm_mapping);
            // skinnedShader->setBool("useDispMap", use_disp_mapping);
            glm::mat4 cam2world = glm::inverse(cam_view_transform);
            glm::vec3 camPos = glm::vec3(cam2world[3][0], cam2world[3][1], cam2world[3][2]);
            skinnedShader->SetCameraLocalPos(camPos);
            switch (es.texture_mode)
            {
            case static_cast<int>(TextureMode::ORIGINAL):
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, true);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr, nullptr, normalMap, armMap, dispMap);
                break;
            case static_cast<int>(TextureMode::BAKED):
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, true);
                if (isRightHand)
                    handModel.Render(*skinnedShader, bones2world, es.rotx, false, bake_fbo_right.getTexture());
                else
                    handModel.Render(*skinnedShader, bones2world, es.rotx, false, bake_fbo_left.getTexture());
                break;
            case static_cast<int>(TextureMode::PROJECTIVE): // a projective texture from the virtual cameras viewpoint
                projectiveTexture = texturePack[es.curSelectedPTexture];
                dynamicTexture = texturePack[es.curSelectedTexture];
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, true, false,
                                   false, true, false, false, cam_projection_transform * cam_view_transform);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, dynamicTexture, projectiveTexture);
                break;
            case static_cast<int>(TextureMode::FROM_FILE): // a projective texture from the virtual cameras viewpoint
                dynamicTexture = texturePack[es.curSelectedTexture];
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, true);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, dynamicTexture, nullptr, normalMap, armMap, dispMap);
                break;
            case static_cast<int>(TextureMode::CAMERA):
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, true, true, true, false, false,
                                   true, true, true, cam_projection_transform * cam_view_transform);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr, &camTexture);
                break;
            default:
                set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, true);
                handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
                break;
            }
            break;
        }
        case static_cast<int>(MaterialMode::SKELETON): // mesh is rendered as a skeleton connecting the joints
        {
            if (es.skeleton_as_gizmos)
            {
                vcolorShader->use();
                vcolorShader->setBool("allWhite", false);
                std::vector<glm::mat4> BoneToLocalTransforms;
                handModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                glBindVertexArray(gizmoVAO);
                // in bind pose
                // for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                // {

                //     vcolorShader->setMat4("mvp", cam_projection_transform * cam_view_transform * rotx * BoneToLocalTransforms[i]);
                //     glDrawArrays(GL_LINES, 0, 6);
                // }
                // in leap motion pose
                for (unsigned int i = 0; i < bones2world.size(); i++)
                {
                    vcolorShader->setMat4("mvp", cam_projection_transform * cam_view_transform * bones2world[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
            }
            else
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
                vcolorShader->use();
                vcolorShader->setBool("allWhite", false);
                vcolorShader->setMat4("mvp", cam_projection_transform * cam_view_transform * global_scale);
                glBindVertexArray(skeletonVAO);
                glDrawArrays(GL_LINES, 0, static_cast<int>(skele_for_upload.size() / 2));
            }
            break;
        }
        case static_cast<int>(MaterialMode::PER_BONE_SCALAR):
        {
            std::vector<float> weights_leap, weights_mesh;
            if (required_pose_bones_to_world_left.size() > 0)
            {
                weights_leap = computeDistanceFromPose(bones2world, required_pose_bones_to_world_left);
                weights_mesh = handModel.scalarLeapBoneToMeshBone(weights_leap);
                for (int i = 0; i < weights_mesh.size(); i++)
                {
                    weights_mesh[i] = sqrt(weights_mesh[i]);
                    // if (weights_mesh[i] > 0.5f) // clamp bad weights to help user
                    //     weights_mesh[i] = 1.0f;
                }
            }
            else
            {
                weights_mesh = std::vector<float>(50, 1.0f);
            }
            set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale,
                               false, false, false, false, false, false, false, false, glm::mat4(1.0f), true, weights_mesh);
            handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
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
        // todo: currently only works for one hand. make it work for both
        // if ((isRightHand && es.jfauv_right_hand) || (!isRightHand && !es.jfauv_right_hand))
        // {
        if (es.postprocess_mode == static_cast<int>(PostProcessMode::JUMP_FLOOD_UV))
        {
            uv_fbo.bind();
            set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * global_scale, false, false, false, true);
            handModel.Render(*skinnedShader, bones2world, es.rotx, false, nullptr);
            uv_fbo.unbind();
        }
        // }
        glDisable(GL_DEPTH_TEST); // todo: why not keep it on ?
    }
}

void handleBaking(std::unordered_map<std::string, Shader *> &shader_map,
                  SkinnedModel &leftHandModel,
                  SkinnedModel &rightHandModel,
                  glm::mat4 cam_view_transform,
                  glm::mat4 cam_projection_transform)
{
    Shader *textureShader = shader_map["textureShader"];
    Shader *thresholdAlphaShader = shader_map["thresholdAlphaShader"];
    Shader *gridShader = shader_map["gridShader"];
    if ((bones_to_world_right.size() > 0) || (bones_to_world_left.size() > 0))
    {
        switch (es.bake_mode)
        {
        case static_cast<int>(BakeMode::CONTROL_NET):
        {
            if (es.bakeRequest)
            {
                if (!es.sd_running)
                {
                    es.sd_running = true;
                    bones_to_world_right_bake = bones_to_world_right;
                    bones_to_world_left_bake = bones_to_world_left;
                    // render a binary mask of the virtual hand
                    sd_fbo.bind(true);
                    hands_fbo.getTexture()->bind();
                    thresholdAlphaShader->use();
                    thresholdAlphaShader->setMat4("view", glm::mat4(1.0));
                    thresholdAlphaShader->setMat4("projection", glm::mat4(1.0));
                    thresholdAlphaShader->setMat4("model", glm::mat4(1.0));
                    thresholdAlphaShader->setFloat("threshold", 0.5);
                    thresholdAlphaShader->setBool("flipHor", false);
                    thresholdAlphaShader->setBool("flipVer", true);
                    thresholdAlphaShader->setBool("isGray", false);
                    thresholdAlphaShader->setBool("binary", true);
                    thresholdAlphaShader->setInt("src", 0);
                    fullScreenQuad.render();
                    sd_fbo.unbind();
                    // sd_fbo.saveColorToFile("test.png", false);
                    std::vector<uint8_t> buf_mask = sd_fbo.getBuffer(1);
                    if (sd_thread.joinable())
                        sd_thread.join();
                    sd_thread = std::thread([buf_mask]() { // thread to run controlnet inference
                        // std::vector<uint8_t> result_buffer;
                        bool success = controlNetClient.inference(buf_mask,
                                                                  es.img2img_data,
                                                                  es.controlnet_preset,
                                                                  512, 512, 1,
                                                                  es.diffuse_seed,
                                                                  es.cur_prompt,
                                                                  es.diffuse_fit_to_view,
                                                                  es.diffuse_pad_size,
                                                                  es.diffuse_select_top_animal,
                                                                  es.no_preprompt,
                                                                  es.cur_bake_file_stem,
                                                                  &es.py_mutex);
                        if (success)
                        {
                            std::cout << "ControlNet inference successful" << std::endl;
                            // if (es.save_byproducts)
                            // {
                            //     cv::Mat img2img_result = cv::Mat(512, 512, CV_8UC3, es.img2img_data.data()).clone();
                            //     cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
                            //     cv::imwrite("../../resource/sd.png", img2img_result);
                            // }
                            es.bake_preproc_succeed = true;
                        }
                        else
                        {
                            std::cout << "ControlNet inference failed" << std::endl;
                        }
                        es.sd_running = false;
                    });
                }
                es.bakeRequest = false;
            }
            if (es.bake_preproc_succeed)
            {
                Texture *tmp = new Texture(GL_TEXTURE_2D);
                tmp->init(512, 512, 3);
                tmp->load(es.img2img_data.data(), true, GL_RGB);
                handleBakingInternal(shader_map, *tmp, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, true, false, false, false);
                delete tmp;
                es.bake_preproc_succeed = false;
            }
            break;
        }
        case static_cast<int>(BakeMode::SD):
        {
            if (es.bakeRequest)
            {
                if (!es.sd_running)
                {
                    es.sd_running = true;
                    bones_to_world_right_bake = bones_to_world_right;
                    bones_to_world_left_bake = bones_to_world_left;
                    // launch thread etc.
                    // download camera image to cpu (resizing to 1024x1024 will occur)
                    sd_fbo.bind(true);
                    set_texture_shader(textureShader, false, true, true, false);
                    camTexture.bind();
                    fullScreenQuad.render();
                    sd_fbo.unbind();
                    // if (save_byproducts)
                    //     sd_fbo.saveColorToFile("../../resource/sd_image.png", false);
                    std::vector<uint8_t> buf = sd_fbo.getBuffer(1);
                    cv::Mat cam_cv = cv::Mat(512, 512, CV_8UC1, buf.data()).clone();
                    // download camera image thresholded to cpu (todo: consider doing all this on CPU)
                    sd_fbo.bind(true);
                    set_texture_shader(textureShader, false, true, true, true, es.masking_threshold);
                    camTexture.bind();
                    fullScreenQuad.render();
                    sd_fbo.unbind();
                    // if (save_byproducts)
                    //     sd_fbo.saveColorToFile("../../resource/sd_mask.png", false);
                    std::vector<uint8_t> buf_mask = sd_fbo.getBuffer(1);
                    if (sd_thread.joinable())
                        sd_thread.join();
                    sd_thread = std::thread([buf, buf_mask, cam_cv]() { // send camera image to stable diffusion
                        try
                        {
                            es.img2img_data = StableDiffusionClient::img2img(es.cur_prompt,
                                                                             es.sd_outwidth, es.sd_outheight,
                                                                             buf, buf_mask, es.diffuse_seed,
                                                                             512, 512, 1,
                                                                             512, 512, false, false, es.sd_mask_mode);
                            // if (es.save_byproducts)
                            // {
                            //     cv::Mat img2img_result = cv::Mat(512, 512, CV_8UC3, es.img2img_data.data()).clone();
                            //     cv::cvtColor(img2img_result, img2img_result, cv::COLOR_RGB2BGR);
                            //     cv::imwrite("../../resource/sd.png", img2img_result);
                            // }
                            es.bake_preproc_succeed = true;
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << e.what() << '\n';
                        }
                        es.sd_running = false;
                    });
                }
                es.bakeRequest = false;
            }
            if (es.bake_preproc_succeed)
            {
                if (es.img2img_data.size() > 0)
                {
                    Texture *tmp = new Texture(GL_TEXTURE_2D);
                    tmp->init(es.sd_outwidth, es.sd_outheight, 3);
                    tmp->load(es.img2img_data.data(), true, GL_RGB);
                    // bake dynamic texture
                    handleBakingInternal(shader_map, *tmp, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, true, false, false, false);
                    delete tmp;
                    // use_mls = true;
                }
                es.bake_preproc_succeed = false;
            }
            break;
        }
        case static_cast<int>(BakeMode::FILE):
        {
            if (es.bakeRequest)
            {
                bones_to_world_right_bake = bones_to_world_right;
                bones_to_world_left_bake = bones_to_world_left;
                Texture tmp(es.inputBakeFile.c_str(), GL_TEXTURE_2D);
                tmp.init_from_file(GL_LINEAR, GL_CLAMP_TO_BORDER);
                handleBakingInternal(shader_map, tmp, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, false, false, false, false);
                es.bakeRequest = false;
            }
            break;
        }
        case static_cast<int>(BakeMode::CAMERA):
        {
            if (es.bakeRequest)
            {
                bones_to_world_right_bake = bones_to_world_right;
                bones_to_world_left_bake = bones_to_world_left;
                handleBakingInternal(shader_map, camTexture, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, true, true, true, false);
                es.bakeRequest = false;
            }
            break;
        }
        case static_cast<int>(BakeMode::POSE):
        {
            if (es.bakeRequest)
            {
                bones_to_world_right_bake = bones_to_world_right;
                bones_to_world_left_bake = bones_to_world_left;
                handleBakingInternal(shader_map, *hands_fbo.getTexture(), leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform, false, false, false, true);
                es.bakeRequest = false;
            }
            break;
        }
        default:
        {
            break;
        }
        }
    }
}

void handleBakingInternal(std::unordered_map<std::string, Shader *> &shader_map,
                          Texture &texture,
                          SkinnedModel &leftHandModel,
                          SkinnedModel &rightHandModel,
                          glm::mat4 cam_view_transform,
                          glm::mat4 cam_projection_transform,
                          bool flipVertical,
                          bool flipHorizontal,
                          bool projSingleChannel,
                          bool ignoreGlobalScale)
{
    SkinningShader *skinnedShader = dynamic_cast<SkinningShader *>(shader_map["skinnedShader"]);
    Shader *uvDilateShader = shader_map["uvDilateShader"];
    /* right hand baking */
    pre_bake_fbo.bind();
    glDisable(GL_CULL_FACE); // todo: why disbale backface ? doesn't this mean bake will reach the back of the hand ?
    glEnable(GL_DEPTH_TEST);
    if (ignoreGlobalScale)
    {
        set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform,
                           flipVertical, flipHorizontal, false, false, true, true, true, projSingleChannel,
                           cam_projection_transform * cam_view_transform);
    }
    else
    {
        set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * es.global_scale_right,
                           flipVertical, flipHorizontal, false, false, true, true, true, projSingleChannel,
                           cam_projection_transform * cam_view_transform * es.global_scale_right);
    }
    rightHandModel.Render(*skinnedShader, bones_to_world_right_bake, es.rotx, false, nullptr, &texture);
    pre_bake_fbo.unbind();
    // dilate the baked texture
    bake_fbo_right.bind();
    uvDilateShader->use();
    uvDilateShader->setMat4("mvp", glm::mat4(1.0f));
    uvDilateShader->setVec2("resolution", glm::vec2(1024.0f, 1024.0f));
    uvDilateShader->setInt("src", 0);
    pre_bake_fbo.getTexture()->bind();
    fullScreenQuad.render();
    bake_fbo_right.unbind();
    /* left hand baking */
    pre_bake_fbo.bind();
    if (ignoreGlobalScale)
    {
        set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform,
                           flipVertical, flipHorizontal, false, false, true, true, true, projSingleChannel,
                           cam_projection_transform * cam_view_transform);
    }
    else
    {
        set_skinned_shader(skinnedShader, cam_projection_transform * cam_view_transform * es.global_scale_left,
                           flipVertical, flipHorizontal, false, false, true, true, true, projSingleChannel,
                           cam_projection_transform * cam_view_transform * es.global_scale_left);
    }
    leftHandModel.Render(*skinnedShader, bones_to_world_left_bake, es.rotx, false, nullptr, &texture);
    pre_bake_fbo.unbind();
    // dilate the baked texture
    bake_fbo_left.bind();
    uvDilateShader->use();
    uvDilateShader->setMat4("mvp", glm::mat4(1.0f));
    uvDilateShader->setVec2("resolution", glm::vec2(1024.0f, 1024.0f));
    uvDilateShader->setInt("src", 0);
    pre_bake_fbo.getTexture()->bind();
    fullScreenQuad.render();
    bake_fbo_left.unbind();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    if (es.save_byproducts)
    {
        std::string bake_right = es.cur_bake_file_stem + "_baked_right.png";
        std::string bake_left = es.cur_bake_file_stem + "_baked_left.png";
        bake_fbo_right.saveColorToFile(bake_right);
        bake_fbo_left.saveColorToFile(bake_left);
    }
}

cv::Mat computeGridDeformation(const std::vector<cv::Point2f> &P,
                               const std::vector<cv::Point2f> &Q,
                               int deformation_mode, float alpha,
                               Grid &grid)
{
    // todo: can refactor control points to avoid this part
    cv::Mat p = cv::Mat::zeros(2, P.size(), CV_32F);
    cv::Mat q = cv::Mat::zeros(2, Q.size(), CV_32F);
    for (int i = 0; i < P.size(); i++)
    {
        p.at<float>(0, i) = (P.at(i)).x;
        p.at<float>(1, i) = (P.at(i)).y;
    }
    for (int i = 0; i < Q.size(); i++)
    {
        q.at<float>(0, i) = (Q.at(i)).x;
        q.at<float>(1, i) = (Q.at(i)).y;
    }
    // compute deformation
    cv::Mat fv;
    cv::Mat w = MLSprecomputeWeights(p, grid.getM(), alpha);
    switch (deformation_mode)
    {
    case static_cast<int>(DeformationMode::AFFINE):
    {
        cv::Mat A = MLSprecomputeAffine(p, grid.getM(), w);
        fv = MLSPointsTransformAffine(w, A, q);
        break;
    }
    case static_cast<int>(DeformationMode::SIMILARITY):
    {
        std::vector<_typeA> A = MLSprecomputeSimilar(p, grid.getM(), w);
        fv = MLSPointsTransformSimilar(w, A, q);
        break;
    }
    case static_cast<int>(DeformationMode::RIGID):
    {
        typeRigid A = MLSprecomputeRigid(p, grid.getM(), w);
        fv = MLSPointsTransformRigid(w, A, q);
        break;
    }
    default:
    {
        cv::Mat A = MLSprecomputeAffine(p, grid.getM(), w);
        fv = MLSPointsTransformAffine(w, A, q);
        break;
    }
    }
    return fv;
}

void projectAndFilterJoints(const std::vector<glm::vec3> &raw_joints_left,
                            const std::vector<glm::vec3> &raw_joints_right,
                            std::vector<glm::vec3> &out_joints_left,
                            std::vector<glm::vec3> &out_joints_right)
{
    out_joints_left.clear();
    out_joints_right.clear();
    if (raw_joints_left.size() > 0)
    {
        for (int i = 0; i < es.leap_selection_vector.size(); i++)
        {
            glm::vec3 projected = Helpers::project_point_w_depth(raw_joints_left[es.leap_selection_vector[i]],
                                                                 glm::mat4(1.0f),
                                                                 gl_camera.getViewMatrix(),
                                                                 gl_camera.getProjectionMatrix());
            out_joints_left.push_back(projected);
        }
    }
    if (raw_joints_right.size() > 0)
    {
        for (int i = 0; i < es.leap_selection_vector.size(); i++)
        {
            glm::vec3 projected = Helpers::project_point_w_depth(raw_joints_right[es.leap_selection_vector[i]],
                                                                 glm::mat4(1.0f),
                                                                 gl_camera.getViewMatrix(),
                                                                 gl_camera.getProjectionMatrix());
            out_joints_right.push_back(projected);
        }
    }
}

void landmarkDetectionThread(std::vector<glm::vec3> projected_filtered_left,
                             std::vector<float> rendered_depths_left,
                             bool isLeftHandVis,
                             std::vector<glm::vec3> projected_filtered_right,
                             std::vector<float> rendered_depths_right,
                             bool isRightHandVis)
{
    t_mls_thread.start();
    try
    {
        bool useRightHand = isRightHandVis;
        bool useLeftHand = isLeftHandVis;
        std::vector<glm::vec2> mp_glm_left, mp_glm_right;
        std::vector<glm::vec2> cur_pred_glm_left, cur_pred_glm_right;
        std::vector<glm::vec2> pred_glm_left, pred_glm_right, kalman_forecast_left, kalman_forecast_right; // todo preallocate
        std::vector<glm::vec2> projected_diff_left(projected_filtered_left.size(), glm::vec2(0.0f, 0.0f));
        std::vector<glm::vec2> projected_diff_right(projected_filtered_right.size(), glm::vec2(0.0f, 0.0f));
        bool mp_detected_left, mp_detected_right;
        // launch the 2D landmark tracker (~17ms blocking operation)
        if (mp_predict(camImage, es.totalFrameCount, mp_glm_left, mp_glm_right, mp_detected_left, mp_detected_right))
        {
            if (useRightHand)
            {
                if (mp_detected_right)
                {
                    for (int i = 0; i < es.mp_selection_vector.size(); i++)
                    {
                        cur_pred_glm_right.push_back(mp_glm_right[es.mp_selection_vector[i]]);
                    }
                }
                else
                {
                    useRightHand = false;
                    // mls_running = false;
                    // return;
                }
            }
            if (useLeftHand)
            {
                if (mp_detected_left)
                {
                    for (int i = 0; i < es.mp_selection_vector.size(); i++)
                    {
                        cur_pred_glm_left.push_back(mp_glm_left[es.mp_selection_vector[i]]);
                    }
                }
                else
                {
                    useLeftHand = false;
                    // return;
                }
            }
            // perform smoothing over predicted control points
            if (es.mls_cp_smooth_window > 0)
            {
                int diff;
                prev_pred_glm_left.push_back(cur_pred_glm_left);
                pred_glm_left = Helpers::accumulate(prev_pred_glm_left);
                diff = prev_pred_glm_left.size() - es.mls_cp_smooth_window;
                if (diff > 0)
                {
                    for (int i = 0; i < diff; i++)
                        prev_pred_glm_left.erase(prev_pred_glm_left.begin());
                }
                prev_pred_glm_right.push_back(cur_pred_glm_right);
                pred_glm_right = Helpers::accumulate(prev_pred_glm_right);
                diff = prev_pred_glm_right.size() - es.mls_cp_smooth_window;
                if (diff > 0)
                {
                    for (int i = 0; i < diff; i++)
                        prev_pred_glm_right.erase(prev_pred_glm_right.begin());
                }
            }
            else
            {
                pred_glm_left = std::move(cur_pred_glm_left);
                pred_glm_right = std::move(cur_pred_glm_right);
            }
            std::vector<cv::Point2f> leap_keypoints_left, diff_keypoints_left, mp_keypoints_left,
                mp_keypoints_right, leap_keypoints_right, diff_keypoints_right;
            glm::vec2 global_shift_left = glm::vec2(0.0f, 0.0f);
            glm::vec2 global_shift_right = glm::vec2(0.0f, 0.0f);
            // possibly, use current leap info to move mp/leap landmarks
            // if (es.mls_use_latest_leap)
            // {
            //     std::vector<glm::mat4> cur_left_bones, cur_right_bones;
            //     std::vector<glm::vec3> cur_vertices_left, cur_vertices_right;
            //     std::vector<uint32_t> cur_left_fingers_extended, cur_right_fingers_extended;
            //     bool success;
            //     if (simulation)
            //     {
            //         success = handleGetMostRecentSkeleton(cur_left_bones, cur_vertices_left, cur_right_bones, cur_vertices_right);
            //     }
            //     else
            //     {
            //         LEAP_STATUS status = getLeapFrame(leap, es.magic_leap_time_delay_mls, cur_left_bones, cur_right_bones, cur_vertices_left, cur_vertices_right, cur_left_fingers_extended, cur_right_fingers_extended, es.leap_poll_mode, es.curFrameID, es.curFrameTimeStamp);
            //         success = status == LEAP_STATUS::LEAP_NEWFRAME;
            //     }
            //     if (success)
            //     {
            //         if ((cur_vertices_left.size() > 0) && (useLeftHand))
            //         {
            //             std::vector<glm::vec2> projected_new, projected_new_all;
            //             projected_new_all = Helpers::project_points(cur_vertices_left, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
            //             // use only selected control points
            //             for (int i = 0; i < es.leap_selection_vector.size(); i++)
            //             {
            //                 projected_new.push_back(projected_new_all[es.leap_selection_vector[i]]);
            //             }
            //             for (int i = 0; i < projected_new.size(); i++)
            //             {
            //                 projected_diff_left[i] += projected_new[i] - projected_left[i];
            //             }
            //             if (es.mls_global_forecast)
            //             {
            //                 global_shift_left = Helpers::average(projected_diff_left);
            //                 for (glm::vec2 &x : projected_left)
            //                 {
            //                     x += global_shift_left;
            //                 }
            //                 leap_keypoints_left = Helpers::glm2cv(projected_left);
            //             }
            //             else
            //                 leap_keypoints_left = Helpers::glm2cv(projected_new);
            //         }
            //         else
            //         {
            //             leap_keypoints_left = Helpers::glm2cv(projected_left);
            //         }
            //         if ((cur_vertices_right.size() > 0) && (useRightHand))
            //         {
            //             std::vector<glm::vec2> projected_new, projected_new_all;
            //             projected_new_all = Helpers::project_points(cur_vertices_right, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
            //             // use only selected control points
            //             for (int i = 0; i < es.leap_selection_vector.size(); i++)
            //             {
            //                 projected_new.push_back(projected_new_all[es.leap_selection_vector[i]]);
            //             }
            //             for (int i = 0; i < projected_new.size(); i++)
            //             {
            //                 projected_diff_right[i] += projected_new[i] - projected_right[i];
            //             }
            //             if (es.mls_global_forecast)
            //             {
            //                 global_shift_right = Helpers::average(projected_diff_right);
            //                 for (glm::vec2 &x : projected_right)
            //                 {
            //                     x += global_shift_right;
            //                 }
            //                 leap_keypoints_right = Helpers::glm2cv(projected_right);
            //             }
            //             else
            //                 leap_keypoints_right = Helpers::glm2cv(projected_new);
            //         }
            //         else
            //         {
            //             leap_keypoints_right = Helpers::glm2cv(projected_right);
            //         }
            //     }
            //     else
            //     {
            //         leap_keypoints_left = Helpers::glm2cv(projected_left);
            //         leap_keypoints_right = Helpers::glm2cv(projected_right);
            //     }
            // }
            // else
            // {
            //     leap_keypoints_left = Helpers::glm2cv(projected_left);
            //     leap_keypoints_right = Helpers::glm2cv(projected_right);
            // }
            leap_keypoints_left = Helpers::glm2cv(Helpers::vec3to2(projected_filtered_left));
            leap_keypoints_right = Helpers::glm2cv(Helpers::vec3to2(projected_filtered_right));

            diff_keypoints_left = Helpers::glm2cv(projected_diff_left);
            diff_keypoints_right = Helpers::glm2cv(projected_diff_right);
            mp_keypoints_left = Helpers::glm2cv(pred_glm_left);
            mp_keypoints_right = Helpers::glm2cv(pred_glm_right);
            std::vector<bool> visible_landmarks_left(leap_keypoints_left.size(), true);
            std::vector<bool> visible_landmarks_right(leap_keypoints_right.size(), true);
            // if mls_depth_test was on, we need to filter out control points that are occluded
            for (int i = 0; i < rendered_depths_left.size(); i++)
            {
                // see: https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
                float rendered_depth = rendered_depths_left[i];
                float projected_depth = projected_filtered_left[i].z;
                float cam_near = 1.0f;
                float cam_far = 1500.0f;
                rendered_depth = (2.0 * rendered_depth) - 1.0;                                                                // to NDC
                rendered_depth = (2.0 * cam_near * cam_far) / (cam_far + cam_near - (rendered_depth * (cam_far - cam_near))); // to linear
                if ((std::abs(rendered_depth - projected_depth) > es.mls_depth_threshold) && (i != 0))                        // always include wrist
                {
                    visible_landmarks_left[i] = false;
                }
            }
            for (int i = 0; i < rendered_depths_right.size(); i++)
            {
                // see: https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
                float rendered_depth = rendered_depths_right[i];
                float projected_depth = projected_filtered_right[i].z;
                float cam_near = 1.0f;
                float cam_far = 1500.0f;
                rendered_depth = (2.0 * rendered_depth) - 1.0; // logarithmic NDC
                rendered_depth = (2.0 * cam_near * cam_far) / (cam_far + cam_near - (rendered_depth * (cam_far - cam_near)));
                projected_depth = (2.0 * projected_depth) - 1.0; // logarithmic NDC
                projected_depth = (2.0 * cam_near * cam_far) / (cam_far + cam_near - (projected_depth * (cam_far - cam_near)));
                if ((std::abs(rendered_depth - projected_depth) <= es.mls_depth_threshold) && (i != 1)) // always include wrist
                {
                    visible_landmarks_right[i] = false;
                }
            }
            // final control points computation
            std::lock_guard<std::mutex> guard(es.mls_mutex);
            ControlPointsP.clear();
            ControlPointsQ.clear();
            if (useLeftHand)
                for (int i = 0; i < visible_landmarks_left.size(); i++)
                {
                    if (visible_landmarks_left[i])
                    {
                        ControlPointsP.push_back(leap_keypoints_left[i]);
                        if (es.mls_global_forecast)
                            ControlPointsQ.push_back(mp_keypoints_left[i] + cv::Point2f(global_shift_left.x, global_shift_left.y));
                        else
                            ControlPointsQ.push_back(mp_keypoints_left[i] + diff_keypoints_left[i]);
                    }
                }
            if (useRightHand)
                for (int i = 0; i < visible_landmarks_right.size(); i++)
                {
                    if (visible_landmarks_right[i])
                    {
                        ControlPointsP.push_back(leap_keypoints_right[i]);
                        if (es.mls_global_forecast)
                            ControlPointsQ.push_back(mp_keypoints_right[i] + cv::Point2f(global_shift_right.x, global_shift_right.y));
                        else
                            ControlPointsQ.push_back(mp_keypoints_right[i] + diff_keypoints_right[i]);
                    }
                }
            es.mls_landmark_thread_succeed = true;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    es.mls_running = false;
    t_mls_thread.stop();
}

void landmarkDetection(bool blocking)
{
    if (!es.mls_running && !es.mls_landmark_thread_succeed) // no mls thread is running and no mls thread results are waiting to be processed, so begin to launch a new mls thread
    {
        if (projected_filtered_left.size() > 0 || projected_filtered_right.size() > 0)
        {
            // the joints are extrapolations to compensate for full system latency.
            // todo: use joints that are extrapolations for the time the camera frame was taken, slightly more in the past than the full system latency
            bool isRightHandVis = projected_filtered_right.size() > 0;
            bool isLeftHandVis = projected_filtered_left.size() > 0;
            std::vector<float> rendered_depths_left, rendered_depths_right;
            std::vector<glm::vec2> screen_space;
            // t_profile.start();
            if (es.mls_depth_test)
            {
                // some control points are occluded according to leap, we don't want to use them for mls
                // we need to do this on main thread, since we need to sample the depth buffer
                screen_space = Helpers::NDCtoScreen(Helpers::vec3to2(projected_filtered_left), es.dst_width, es.dst_height, false);
                rendered_depths_left = hands_fbo.sampleDepthBuffer(screen_space); // todo: make async
                screen_space = Helpers::NDCtoScreen(Helpers::vec3to2(projected_filtered_right), es.dst_width, es.dst_height, false);
                rendered_depths_right = hands_fbo.sampleDepthBuffer(screen_space); // todo: make async
            }
            // t_profile.stop();
            ControlPointsP_input_left = projected_filtered_left;
            ControlPointsP_input_right = projected_filtered_right;
            camImage = camImageOrig.clone(); // use copy for thread, as the main thread will continue to modify the original
            if (mls_thread.joinable())       // make sure the previous mls thread is done
                mls_thread.join();
            es.mls_running = true;
            mls_thread = std::thread(landmarkDetectionThread,
                                     projected_filtered_left, rendered_depths_left, isLeftHandVis,
                                     projected_filtered_right, rendered_depths_right, isRightHandVis);
        } // if (joints.size() > 0)
    }     // if (!mls_running && !mls_landmark_thread_succeed)
    if (blocking)
    {
        if (mls_thread.joinable())
            mls_thread.join();
    }
}

void constructGrid(bool updateGLBuffers)
{
    std::lock_guard<std::mutex> guard(es.mls_mutex);
    /* <can be done by landmark detection thread or main thread> */
    if ((ControlPointsP.size() > 0) && (ControlPointsP.size() == ControlPointsQ.size()))
    {
        // compute deformation
        cv::Mat fv = computeGridDeformation(ControlPointsP, ControlPointsQ, es.deformation_mode, es.mls_alpha, deformationGrid);
        // update grid points for render
        deformationGrid.constructDeformedGridSmooth(fv, es.mls_grid_smooth_window);
    }
    else
    {
        // update grid points for render
        deformationGrid.constructGrid();
    }
    /* end of <can be done by landmark detection thread or main thread> */
    if (updateGLBuffers)
        deformationGrid.updateGLBuffers();
}

void renderGrid(Shader &gridShader)
{
    // render current image using the latest deformation grid
    mls_fbo.bind();
    glDisable(GL_CULL_FACE); // todo: why is this necessary? flip grid triangles...
    if (es.postprocess_mode == static_cast<int>(PostProcessMode::JUMP_FLOOD_UV))
        uv_fbo.getTexture()->bind();
    else
        hands_fbo.getTexture()->bind();
    gridShader.use();
    gridShader.setInt("src", 0);
    gridShader.setFloat("threshold", es.mls_grid_shader_threshold);
    gridShader.setBool("flipVer", false);
    deformationGrid.render();
    mls_fbo.unbind();
    glEnable(GL_CULL_FACE);
}

void handleMLS(Shader &gridShader, bool blocking, bool detect_landmarks, bool new_frame, bool simulation)
{
    if (es.auto_pilot)
    {
        int n_visible_hands = (projected_filtered_left.size() > 0) + (projected_filtered_right.size() > 0);
        if (n_visible_hands == 1)
            es.use_mls = true;
        else
            es.use_mls = false;
    }
    if (es.use_mls)
    {
        if (new_frame)
        {
            if (detect_landmarks)
                landmarkDetection(blocking);
            // get leap joints used to render this iteration
            if (simulation)
            {
                int32_t cur_timestamp = getCurrentSimulationIndex();
                int future_timestamp = cur_timestamp + es.mls_future_frame_offset;
                if (future_timestamp > session_timestamps.size() - 1)
                    future_timestamp = session_timestamps.size() - 1;
                // get the current leap frame
                std::vector<glm::mat4> bones2world_left, bones2world_right;
                handleGetSkeletonByTimestamp(future_timestamp, bones2world_left, joints_left, bones2world_right, joints_right);
                projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
            }
            std::vector<cv::Point2f> curP_left = Helpers::glm2cv(Helpers::vec3to2(projected_filtered_left));
            std::vector<cv::Point2f> curP_right = Helpers::glm2cv(Helpers::vec3to2(projected_filtered_right));
            std::vector<cv::Point2f> curP(curP_left.begin(), curP_left.end());
            curP.insert(curP.end(), curP_right.begin(), curP_right.end());
            // temporally filter landmarks
            if (es.mls_use_kalman)
            {
                std::lock_guard<std::mutex> guard(es.mls_mutex);
                bool do_correction_step = es.mls_landmark_thread_succeed;
                // get the time passed since the last frame
                float cur_mls_time;
                if (simulation)
                    cur_mls_time = es.simulationTime;
                else
                    cur_mls_time = t_app.getElapsedTimeInMilliSec();
                float dt = cur_mls_time - es.prev_mls_time;
                for (int i = 0; i < ControlPointsQ.size(); i++)
                {
                    cv::Mat pred = kalman_filters_left[i].predict(dt);
                    if (do_correction_step)
                    {
                        cv::Mat measurement(2, 1, CV_32F);
                        measurement.at<float>(0) = ControlPointsQ[i].x;
                        measurement.at<float>(1) = ControlPointsQ[i].y;
                        cv::Mat corr = kalman_filters_left[i].correct(measurement);
                    }
                    cv::Mat forecast = kalman_filters_left[i].forecast(es.kalman_lookahead);
                    ControlPointsQ[i] = cv::Point2f(forecast.at<float>(0), forecast.at<float>(1));
                }
                es.prev_mls_time = cur_mls_time;
                // pred_glm = kalman_forecast;
            }
            // move target points using the difference between the leap frame used to render, and the one matching the landmark detection timestamp
            {
                std::lock_guard<std::mutex> guard(es.mls_mutex);
                if ((curP.size() == ControlPointsP.size()) && (ControlPointsP.size() == ControlPointsQ.size()))
                {
                    cv::Point2f diff;
                    int n_landmarks = 6; // curP.size()
                    for (int i = 0; i < n_landmarks; i++)
                    {
                        // ControlPointsQ[i] += curP[i] - ControlPointsP[i];
                        // ControlPointsQ[i] += avg;
                        diff += curP[i] - ControlPointsP[i];
                    }
                    diff = diff / static_cast<float>(n_landmarks);
                    // if (es.auto_pilot)
                    // {
                    // if (cv::norm(diff) > es.auto_pilot_thr_extrapolate)
                    // {
                    //     es.auto_pilot_cnt_below_thr = 0;
                    //     es.auto_pilot_cnt_above_thr += 1;
                    // }
                    // else
                    // {
                    //     es.auto_pilot_cnt_above_thr = 0;
                    //     es.auto_pilot_cnt_below_thr += 1;
                    // }
                    // if (es.auto_pilot_cnt_above_thr > es.auto_pilot_count_thr)
                    //     es.mls_extrapolate = true;
                    // else if (es.auto_pilot_cnt_below_thr > es.auto_pilot_count_thr)
                    //     es.mls_extrapolate = false;
                    // es.auto_pilot_delta = std::min(static_cast<float>(cv::norm(diff)) / es.auto_pilot_thr_extrapolate, 1.0f);
                    // es.auto_pilot_alpha = sqrt(es.auto_pilot_delta);
                    // float alpha = std::min(static_cast<float>(cv::norm(diff)) / es.auto_pilot_thr_extrapolate, 1.0f);
                    // for (int i = 0; i < curP.size(); i++)
                    // {
                    // ControlPointsQ[i] += es.auto_pilot_alpha * diff;
                    // }
                    // }
                    // else
                    // {
                    if (es.mls_extrapolate)
                    {
                        for (int i = 0; i < curP.size(); i++)
                        {
                            ControlPointsQ[i] += diff;
                        }
                    }
                    // }
                }
            }
            // use the leap frame used to render as source points
            if (es.mls_use_latest_leap)
                ControlPointsP = curP;

            // solve mls
            if (es.mls_solve_every_frame)
            {
                constructGrid(true);
            }
            if (es.mls_landmark_thread_succeed)
            {
                es.mls_landmark_thread_succeed = false;
                es.mls_succeeded_this_frame = true;
                if (!es.mls_solve_every_frame)
                {
                    constructGrid(true);
                }
            }
        }
        // in any case, if mls is on we use the latest grid to deform the current rendered frame
        renderGrid(gridShader);
    }
}

void updateOFParams()
{
    camImagePrev = cv::Mat::zeros(es.of_downsize.height, es.of_downsize.width, CV_8UC1);
    flow = cv::Mat(es.of_downsize, CV_32FC2);
    rawFlow = cv::Mat(es.of_downsize, CV_32FC2);
    switch (es.of_mode)
    {
    case static_cast<int>(OFMode::FB_GPU):
    {
        gprev.upload(camImagePrev);
        gflow.upload(flow);
#ifdef OPENCV_WITH_CUDA
        fbof = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
#else
        fbof = nullptr;
#endif
        break;
    }
    case static_cast<int>(OFMode::NV_GPU):
    {
// flow = cv::Mat(es.of_downsize, CV_32FC2);
#ifdef OPENCV_WITH_CUDA
        nvof = cv::cuda::NvidiaOpticalFlow_2_0::create(es.of_downsize,
                                                       cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_SLOW,
                                                       cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
                                                       cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_1,
                                                       true, // set to true !
                                                       false);
#else
        nvof = nullptr;
#endif
        // int gridSize = nvof->getGridSize();
        // std::cout << "gridSize: " << gridSize << std::endl;
        flow = cv::Mat(es.of_downsize, CV_16FC2);
        rawFlow = cv::Mat(es.of_downsize, CV_32FC2);
        // gprev.upload(camImagePrev);
        // gflow.upload(flow);
        break;
    }
    default:
        break;
    }
    if (OFTexture != nullptr)
    {
        delete OFTexture;
        OFTexture = new Texture();
        OFTexture->init(es.of_downsize.width, es.of_downsize.height, 3);
    }
    es.totalFrameCountOF = 0;
}

void handleOF(Shader *gridShader)
{
    // apply OF on camera image
    if (es.use_of)
    {
        cv::Mat image;
        cv::flip(camImageOrig, image, 1); // flip horizontally
        if (es.of_resize_factor != 1)
        {
            cv::resize(image, image, es.of_downsize);
        }
        switch (es.of_mode)
        {
        case static_cast<int>(OFMode::NAIVE_BLOB):
        {
            cv::Mat thr;
            cv::threshold(image, thr, static_cast<int>(es.masking_threshold * 255), 255, cv::THRESH_BINARY);
            // cv::Scalar tmp = cv::mean(image, thr)[0];
            // float image_mean = tmp[0];
            // std::cout << "image mean: " << image_mean << std::endl;
            // if (image_mean > 1.1 * es.prev_image_mean)
            // {
            //     // the image is too bright, reduce the brightness of the image
            //     cv::multiply(image, cv::Mat::ones(image.size(), image.type()), image, es.prev_image_mean / image_mean);
            //     cv::threshold(image, thr, static_cast<int>(es.masking_threshold * 255), 255, cv::THRESH_BINARY);
            //     tmp = cv::mean(image, thr)[0];
            //     image_mean = tmp[0];
            //     std::cout << "new image mean: " << image_mean << std::endl;
            // }
            // es.prev_image_mean = image_mean;
            cv::Moments m = cv::moments(thr, true);
            cv::Point2f p(m.m10 / m.m00, m.m01 / m.m00);
            flow = cv::Mat(es.of_downsize, CV_32FC2, cv::Scalar(p.x - es.prev_blob.x, p.y - es.prev_blob.y));
            es.prev_blob = p;
            break;
        }
        case static_cast<int>(OFMode::FB_CPU):
        {
            cv::calcOpticalFlowFarneback(camImagePrev, image, flow, 0.5, 3, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
            camImagePrev = image.clone();
            break;
        }
        case static_cast<int>(OFMode::FB_GPU):
        {
            if (es.totalFrameCountOF % 2 == 0)
            {
                gcur.upload(image);
                fbof->calc(gprev, gcur, gflow);
            }
            else
            {
                gprev.upload(image);
                fbof->calc(gcur, gprev, gflow);
            }
            gflow.download(flow);
            break;
        }
        case static_cast<int>(OFMode::NV_GPU):
        {
            nvof->calc(image, camImagePrev, rawFlow);
            nvof->convertToFloat(rawFlow, flow);
            camImagePrev = image.clone();
            break;
        }
        default:
            break;
        }
        es.totalFrameCountOF += 1;
        // move control points using the flow
        // std::vector<glm::vec2> cp = Helpers::NDCtoScreen(ControlPointsP_glm, es.of_downsize.width, es.of_downsize.height);
        std::lock_guard<std::mutex> guard(es.mls_mutex);
        std::vector<cv::Point2f> cq = Helpers::NDCtoScreen(ControlPointsQ, es.of_downsize.width, es.of_downsize.height, true);
        if (cq.size() > 0)
        {
            es.of_debug.clear();
            // float max_magnitude = 0.0f;
            // cv::Scalar max_flow;
            for (int i = 0; i < cq.size(); i++)
            {
                // glm::vec2 p = cp[i];
                cv::Point2f q = cq[i];
                // clamp to image boundaries
                int x = static_cast<int>(std::round(q.x - (es.of_roi / 2)));
                x = x < 0 ? 0 : x;
                x = x > flow.cols - es.of_roi ? flow.cols - es.of_roi : x;
                int y = static_cast<int>(std::round(q.y - (es.of_roi / 2)));
                y = y < 0 ? 0 : y;
                y = y > flow.rows - es.of_roi ? flow.rows - es.of_roi : y;
                // get average flow in roi. note flow is in (of_downsize) pixel units
                cv::Rect flowrect(x,
                                  y,
                                  es.of_roi,
                                  es.of_roi);
                cv::Mat roi_flow = flow(flowrect).clone();
                cv::Scalar avg_flow = cv::mean(roi_flow);
                // if (cv::norm(avg_flow) > max_magnitude)
                // {
                //     max_magnitude = cv::norm(avg_flow);
                //     max_flow = avg_flow;
                // }
                cv::Point2f dxdy(avg_flow[0], avg_flow[1]);
                // cp[i] = p + dxdy;
                cq[i] = q + dxdy;
                es.of_debug.push_back(glm::vec2(avg_flow[0], -avg_flow[1]));
            }
            // for (int i = 0; i < cq.size(); i++)
            // {
            //     // glm::vec2 p = cp[i];
            //     cq[i] += cv::Point2f(max_flow[0], max_flow[1]);
            //     es.of_debug.push_back(glm::vec2(max_flow[0], -max_flow[1]));
            // }
            // ControlPointsP_glm = Helpers::ScreenToNDC(cp, es.of_downsize.width, es.of_downsize.height);
            ControlPointsQ = Helpers::ScreenToNDC(cq, es.of_downsize.width, es.of_downsize.height, true);
            // ControlPointsP = Helpers::glm2cv(Helpers::vec3to2(projected_filtered_left));
            // constructGrid(true);
            // renderGrid(*gridShader);
        }
    }
}

bool handleGetSkeletonByTimestamp(uint32_t timestamp,
                                  std::vector<glm::mat4> &bones2world_left,
                                  std::vector<glm::vec3> &joints_left,
                                  std::vector<glm::mat4> &bones2world_right,
                                  std::vector<glm::vec3> &joints_right)
{
    // deal with lagged timestamp (will be used for projection)
    if (!getSkeletonByTimestamp(timestamp, bones2world_left, joints_left, session_bones_left, session_joints_left, false))
        return false;
    if (!getSkeletonByTimestamp(timestamp, bones2world_right, joints_right, session_bones_right, session_joints_right, true))
        return false;
    return true;
}

bool getSkeletonByTimestamp(uint32_t timestamp, std::vector<glm::mat4> &bones_out,
                            std::vector<glm::vec3> &joints_out,
                            const std::vector<glm::mat4> &bones_session,
                            const std::vector<glm::vec3> &joints_session,
                            bool isRightHand)
{
    std::vector<glm::mat4> bones2world;
    std::vector<glm::vec3> joints;
    bones_out.clear();
    joints_out.clear();
    LEAP_STATUS leap_status = getLeapFramePreRecorded(bones_out, joints_out, timestamp, es.total_session_time_stamps, bones_session, joints_session);
    if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
        return false;
    return true;
}
bool interpolateBones(float time, std::vector<glm::mat4> &bones_out, const std::vector<glm::mat4> &session, bool isRightHand)
{
    auto upper_iter = std::upper_bound(session_timestamps.begin(), session_timestamps.end(), time);
    int interp_index = upper_iter - session_timestamps.begin();
    if (interp_index >= session_timestamps.size())
        return false;
    float interp1 = session_timestamps[interp_index - 1];
    float interp2 = session_timestamps[interp_index];
    float interp_factor = (time - interp1) / (interp2 - interp1);
    std::vector<glm::mat4> bones2world_interp1, bones2world_interp2;

    std::vector<glm::vec3> dummy_joints;
    std::vector<glm::vec3> dummy_session_joints;
    LEAP_STATUS leap_status = getLeapFramePreRecorded(bones2world_interp1, dummy_joints, interp_index - 1, es.total_session_time_stamps, session, dummy_session_joints);
    if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
        return false;
    leap_status = getLeapFramePreRecorded(bones2world_interp2, dummy_joints, interp_index, es.total_session_time_stamps, session, dummy_session_joints);
    if (leap_status != LEAP_STATUS::LEAP_NEWFRAME)
        return false;
    bones_out.clear();
    for (int i = 0; i < bones2world_interp2.size(); i++)
    {
        glm::mat4 interpolated = Helpers::interpolate(bones2world_interp1[i], bones2world_interp2[i], interp_factor, true, isRightHand);
        bones_out.push_back(interpolated);
    }
    return true;
}

bool handleInterpolateFrames(std::vector<glm::mat4> &bones2world_left_cur,
                             std::vector<glm::mat4> &bones2world_right_cur,
                             std::vector<glm::mat4> &bones2world_left_lag,
                             std::vector<glm::mat4> &bones2world_right_lag)
{
    float required_time_lag = es.simulationTime;
    float required_time_cur = (es.vid_simulated_latency_ms * es.vid_playback_speed) + required_time_lag;
    if (required_time_cur >= session_timestamps.back())
        return false;
    // deal with lagged timestamp (will be used for projection)
    if (!interpolateBones(required_time_lag, bones2world_left_lag, session_bones_left, false))
        return false;
    if (!interpolateBones(required_time_lag, bones2world_right_lag, session_bones_right, true))
        return false;
    // deal with current timestamp (will be used as camera image)
    if (!interpolateBones(required_time_cur, bones2world_left_cur, session_bones_left, false))
        return false;
    if (!interpolateBones(required_time_cur, bones2world_right_cur, session_bones_right, true))
        return false;
    return true;
}

void handleGuessAnimalGame(std::unordered_map<std::string, Shader *> &shaderMap,
                           SkinnedModel &leftHandModel,
                           SkinnedModel &rightHandModel,
                           TextModel &textModel,
                           glm::mat4 &cam_view_transform,
                           glm::mat4 &cam_projection_transform)
{
    if (!guessAnimalGame.isInitialized())
        return;
    int state = guessAnimalGame.getState();
    Shader *gridShader = shaderMap["gridShader"];
    Shader *textureShader = shaderMap["textureShader"];
    /* deal with camera input */
    t_camera.start();
    CGrabResultPtr ptrGrabResult;
    if (es.simulated_camera)
    {
        cv::Mat sim = cv::Mat(es.cam_height, es.cam_width, CV_8UC1, 255);
        handleCameraInput(ptrGrabResult, true, sim);
    }
    else
    {
        handleCameraInput(ptrGrabResult, false, cv::Mat());
    }
    t_camera.stop();

    /* deal with leap input */
    t_leap.start();
    LEAP_STATUS leap_status = handleLeapInput();
    projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
    t_leap.stop();
    /* skin hand meshes */
    t_skin.start();
    handleSkinning(bones_to_world_right, true, true, shaderMap, rightHandModel, cam_view_transform, cam_projection_transform);
    handleSkinning(bones_to_world_left, false, bones_to_world_right.size() == 0, shaderMap, leftHandModel, cam_view_transform, cam_projection_transform);
    t_skin.stop();

    /* deal with bake request */
    t_bake.start();
    if (state == static_cast<int>(GuessAnimalGameState::BAKE))
    {
        guessAnimalGame.resetState();
        handleBakeConfig();
        es.bakeRequest = true;
    }
    handleBaking(shaderMap, leftHandModel, rightHandModel, cam_view_transform, cam_projection_transform);
    t_bake.stop();

    /* run MLS on MP prediction to reduce bias */
    t_mls.start();
    handleMLS(*gridShader);
    t_mls.stop();

    /* post process fbo using camera input */
    t_pp.start();
    handlePostProcess(leftHandModel, rightHandModel, camTexture, shaderMap);
    /* render final output to screen */
    if (!es.debug_mode)
    {
        glViewport(0, 0, es.proj_width, es.proj_height); // set viewport
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        set_texture_shader(textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
        c2p_fbo.getTexture()->bind();
        fullScreenQuad.render();
    }
    t_pp.stop();
}

void handleGuessCharGame(std::unordered_map<std::string, Shader *> &shaderMap,
                         SkinnedModel &leftHandModel,
                         SkinnedModel &rightHandModel,
                         TextModel &textModel,
                         glm::mat4 &cam_view_transform,
                         glm::mat4 &cam_projection_transform)
{
    if (!guessCharGame.isInitialized())
        return;
    Shader *textShader = shaderMap["textShader"];
    Shader *gridShader = shaderMap["gridShader"];
    Shader *textureShader = shaderMap["textureShader"];
    auto bones = es.gameUseRightHand ? bones_to_world_right : bones_to_world_left;
    bool frontView = false;
    if (bones.size() > 0)
        frontView = Helpers::isPalmFacingCamera(bones[0], cam_view_transform);
    // std::cout << frontView << std::endl;
    int state = guessCharGame.getState();
    guessCharGame.setBonesVisible(bones.size() > 0);
    glm::vec2 palm_ndc = Helpers::NDCtoScreen(glm::vec2(-0.66f, -0.683f), es.proj_width, es.proj_height);
    switch (state)
    {
    // wait for user to place their hand infront of screen
    case static_cast<int>(GuessCharGameState::WAIT_FOR_USER):
    {
        es.texture_mode = static_cast<int>(TextureMode::DYNAMIC);
        dynamic_fbo.bind(true, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        float palm_scale = 2.0f;
        if (frontView)
        {
            palm_ndc = Helpers::NDCtoScreen(glm::vec2(-0.589f, -0.679), es.proj_width, es.proj_height);
            palm_scale = 1.134f;
        }
        textModel.Render(*textShader, "Ready?", palm_ndc.x, palm_ndc.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
        dynamic_fbo.unbind();
        break;
    }
    // countdown to start game
    case static_cast<int>(GuessCharGameState::COUNTDOWN):
    {
        es.material_mode = static_cast<int>(MaterialMode::DIFFUSE);
        // texture_mode = static_cast<int>(TextureMode::FROM_FILE);
        es.texture_mode = static_cast<int>(TextureMode::DYNAMIC);
        int cd_time = guessCharGame.getCountdownTime();
        dynamic_fbo.bind(true, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        float palm_scale = 3.0f;
        if (frontView)
        {
            palm_scale = 2.5f;
            palm_ndc = Helpers::NDCtoScreen(glm::vec2(-0.554f, -0.679f), es.proj_width, es.proj_height);
        }
        switch (cd_time)
        {
        case 0:
        {
            textModel.Render(*textShader, "3!", palm_ndc.x, palm_ndc.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
            break;
        }
        case 1:
        {
            textModel.Render(*textShader, "2!", palm_ndc.x, palm_ndc.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
            break;
        }
        case 2:
        {
            textModel.Render(*textShader, "1!", palm_ndc.x, palm_ndc.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
            break;
        }
        default:
            break;
        }
        dynamic_fbo.unbind();
        break;
    }
    case static_cast<int>(GuessCharGameState::PLAY):
    {
        es.texture_mode = static_cast<int>(TextureMode::DYNAMIC);
        std::string chars;
        int correctIndex = guessCharGame.getRandomChars(chars);
        int selectedIndex = -1;
        bool allExtended = true;
        bool moreThanOneDown = false;
        for (int i = 1; i < left_fingers_extended.size(); i++)
        {
            if (left_fingers_extended[i] == 0)
            {
                allExtended = false;
                if (selectedIndex != -1)
                    moreThanOneDown = true;
                selectedIndex = i - 1;
            }
        }
        if (allExtended)
            guessCharGame.setAllExtended(true);
        else
            guessCharGame.setAllExtended(false);
        if (!allExtended && !moreThanOneDown)
        {
            if (selectedIndex == correctIndex)
            {
                guessCharGame.setResponse(true);
            }
            else
            {
                guessCharGame.setResponse(false);
            }
        }
        dynamic_fbo.bind(true, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        float scale = 0.758f;
        float palm_scale = 0.758f * 4.0f;
        if (frontView)
        {
            scale = 0.5f;
            palm_scale = 2.5f;
        }
        glm::vec2 ndc_cords = Helpers::NDCtoScreen(glm::vec2(es.debug_vec), es.proj_width, es.proj_height);
        // textModel.Render(*textShader, debug_text, ndc_cords.x, ndc_cords.y, debug_scalar, glm::vec3(1.0f, 0.0f, 1.0f));
        auto fingerMap = guessCharGame.getNumberLocations(frontView);
        glm::vec2 palm_screen = Helpers::NDCtoScreen(fingerMap["palm"], es.proj_width, es.proj_height);
        glm::vec2 index_screen = Helpers::NDCtoScreen(fingerMap["index"], es.proj_width, es.proj_height);
        glm::vec2 middle_screen = Helpers::NDCtoScreen(fingerMap["middle"], es.proj_width, es.proj_height);
        glm::vec2 ring_screen = Helpers::NDCtoScreen(fingerMap["ring"], es.proj_width, es.proj_height);
        glm::vec2 pinky_screen = Helpers::NDCtoScreen(fingerMap["pinky"], es.proj_width, es.proj_height);
        textModel.Render(*textShader, chars.substr(correctIndex, 1), palm_screen.x, palm_screen.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
        textModel.Render(*textShader, chars.substr(0, 1), index_screen.x, index_screen.y, scale, glm::vec3(0.0f, 0.0f, 0.0f));
        textModel.Render(*textShader, chars.substr(1, 1), middle_screen.x, middle_screen.y, scale, glm::vec3(0.0f, 0.0f, 0.0f));
        textModel.Render(*textShader, chars.substr(2, 1), ring_screen.x, ring_screen.y, scale, glm::vec3(0.0f, 0.0f, 0.0f));
        textModel.Render(*textShader, chars.substr(3, 1), pinky_screen.x, pinky_screen.y, scale, glm::vec3(0.0f, 0.0f, 0.0f));
        // // glm::vec2 thumb_screen = Helpers::NDCtoScreen(glm::vec2(0.829f, -0.741f), proj_width, proj_height);
        // // textModel.Render(textShader, "T", thumb_ndc.x, thumb_ndc.y, scale, glm::vec3(1.0f, 0.0f, 1.0f));
        dynamic_fbo.unbind();
        break;
    }
    case static_cast<int>(GuessCharGameState::WAIT):
    {
        int selectedIndex = -1;
        bool allExtended = true;
        bool moreThanOneDown = false;
        for (int i = 1; i < left_fingers_extended.size(); i++)
        {
            if (left_fingers_extended[i] == 0)
            {
                allExtended = false;
                if (selectedIndex != -1)
                    moreThanOneDown = true;
                selectedIndex = i - 1;
            }
        }
        if (allExtended)
            guessCharGame.setAllExtended(true);
        else
            guessCharGame.setAllExtended(false);
        es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
        es.curSelectedTexture = "XGA_rand";
        break;
    }
    case static_cast<int>(GuessCharGameState::END):
    {
        float palm_scale = 3.0f;
        if (frontView)
        {
            palm_scale = 2.5f;
            palm_ndc = Helpers::NDCtoScreen(glm::vec2(-0.554f, -0.679f), es.proj_width, es.proj_height);
        }
        es.texture_mode = static_cast<int>(TextureMode::DYNAMIC);
        dynamic_fbo.bind(true, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        textModel.Render(*textShader, ":)", palm_ndc.x, palm_ndc.y, palm_scale, glm::vec3(0.0f, 0.0f, 0.0f));
        dynamic_fbo.unbind();
        break;
    }
    default:
        break;
    }
    /* deal with camera input */
    t_camera.start();
    CGrabResultPtr ptrGrabResult;
    handleCameraInput(ptrGrabResult, false, cv::Mat());
    t_camera.stop();

    /* deal with leap input */
    t_leap.start();
    LEAP_STATUS leap_status = handleLeapInput();
    t_leap.stop();

    /* skin hand meshes */
    t_skin.start();
    if (es.gameUseRightHand)
    {
        handleSkinning(bones, es.gameUseRightHand, true, shaderMap, rightHandModel, cam_view_transform, cam_projection_transform);
    }
    else
    {
        if (frontView)
            handleSkinning(bones, es.gameUseRightHand, true, shaderMap, *extraLeftHandModel, cam_view_transform, cam_projection_transform);
        else
            handleSkinning(bones, es.gameUseRightHand, true, shaderMap, leftHandModel, cam_view_transform, cam_projection_transform);
    }
    t_skin.stop();

    /* run MLS on MP prediction to reduce bias */
    t_mls.start();
    handleMLS(*gridShader);
    t_mls.stop();

    /* post process fbo using camera input */
    t_pp.start();
    if (frontView)
        handlePostProcess(*extraLeftHandModel, rightHandModel, camTexture, shaderMap);
    else
        handlePostProcess(leftHandModel, rightHandModel, camTexture, shaderMap);
    t_pp.stop();

    /* render final output to screen */
    glViewport(0, 0, es.proj_width, es.proj_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    set_texture_shader(textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
    c2p_fbo.getTexture()->bind();
    fullScreenQuad.render();
}

void handleGuessPoseGame(std::unordered_map<std::string, Shader *> &shaderMap,
                         SkinnedModel &leftHandModel,
                         SkinnedModel &rightHandModel,
                         Quad &topRightQuad,
                         glm::mat4 &cam_view_transform,
                         glm::mat4 &cam_projection_transform)
{
    Shader *gridShader = shaderMap["gridShader"];
    Shader *textureShader = shaderMap["textureShader"];
    int state = guessPoseGame.getState();
    switch (state)
    {
    case static_cast<int>(GuessPoseGameState::WAIT_FOR_USER):
    {
        guessPoseGame.setBonesVisible(bones_to_world_left.size() > 0);
        break;
    }
    case static_cast<int>(GuessPoseGameState::COUNTDOWN):
    {
        es.material_mode = static_cast<int>(MaterialMode::DIFFUSE);
        es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
        int cd_time = guessPoseGame.getCountdownTime();
        switch (cd_time)
        {
        case 0:
        {
            es.curSelectedTexture = "3";
            break;
        }
        case 1:
        {
            es.curSelectedTexture = "2";
            break;
        }
        case 2:
        {
            es.curSelectedTexture = "1";
            break;
        }
        default:
            break;
        }
        break;
    }
    case static_cast<int>(GuessPoseGameState::PLAY):
    {
        es.material_mode = static_cast<int>(MaterialMode::PER_BONE_SCALAR);
        guessPoseGame.setBonesVisible(bones_to_world_left.size() > 0);
        required_pose_bones_to_world_left = guessPoseGame.getPose();
        if (bones_to_world_left.size() > 0)
        {
            std::vector<float> weights_leap = computeDistanceFromPose(bones_to_world_left, required_pose_bones_to_world_left);
            std::vector<float> scores = leftHandModel.scalarLeapBoneToMeshBone(weights_leap);
            float minScore = 1.0f;
            for (int i = 0; i < scores.size(); i++)
            {
                if ((scores[i] < minScore) && (scores[i] > 0.0f))
                    minScore = scores[i];
            }
            guessPoseGame.setScore(minScore);
            // float avgScore = 0.0f;
            // for (int i = 0; i < scores.size(); i++)
            //     avgScore += scores[i];
            // avgScore /= scores.size();
            // guessPosegame.setScore(avgScore);
        }
        break;
    }
    case static_cast<int>(GuessPoseGameState::END):
    {
        break;
    }
    default:
        break;
    }

    /* deal with camera input */
    t_camera.start();
    CGrabResultPtr ptrGrabResult;
    handleCameraInput(ptrGrabResult, false, cv::Mat());
    t_camera.stop();

    /* deal with leap input */
    t_leap.start();
    LEAP_STATUS leap_status = handleLeapInput();
    t_leap.stop();

    /* skin hand meshes */
    t_skin.start();
    handleSkinning(bones_to_world_left, false, true, shaderMap, leftHandModel, cam_view_transform, cam_projection_transform);
    t_skin.stop();

    /* run MLS on MP prediction to reduce bias */
    t_mls.start();
    handleMLS(*gridShader);
    t_mls.stop();

    /* post process fbo using camera input */
    t_pp.start();
    handlePostProcess(leftHandModel, rightHandModel, camTexture, shaderMap);
    t_pp.stop();

    /* render final output to screen */
    glViewport(0, 0, es.proj_width, es.proj_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    set_texture_shader(textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
    c2p_fbo.getTexture()->bind();
    fullScreenQuad.render();
    // material_mode = static_cast<int>(MaterialMode::DIFFUSE);
    // texture_mode = static_cast<int>(TextureMode::ORIGINAL);
    if (es.showGameHint)
    {
        handleSkinning(required_pose_bones_to_world_left, false, true, shaderMap, leftHandModel, cam_view_transform, cam_projection_transform);
        set_texture_shader(textureShader, false, false, false);
        hands_fbo.getTexture()->bind();
        topRightQuad.render();
    }
}

void handleSimulation(std::unordered_map<std::string, Shader *> &shaderMap,
                      SkinnedModel &leftHandModel,
                      SkinnedModel &rightHandModel,
                      TextModel &textModel,
                      glm::mat4 &cam_view_transform,
                      glm::mat4 &cam_projection_transform)
{
    if (es.pre_recorded_session_loaded) // a video was loaded
    {
        if (es.debug_playback)
        {
            if (es.canUseRecordedImages)
            {
                if (!playVideoReal(shaderMap,
                                   leftHandModel,
                                   rightHandModel,
                                   textModel,
                                   cam_view_transform,
                                   cam_projection_transform))
                {
                    es.simulationTime = 0.0f; // continuous video playback
                }
            }
        }
    }
}

void handleUserStudy(std::unordered_map<std::string, Shader *> &shaderMap,
                     GLFWwindow *window,
                     SkinnedModel &leftHandModel,
                     SkinnedModel &rightHandModel,
                     TextModel &textModel,
                     glm::mat4 &cam_view_transform,
                     glm::mat4 &cam_projection_transform)
{
    Shader *textShader = shaderMap["textShader"];
    if (es.pre_recorded_session_loaded) // a video was loaded
    {
        if (es.run_user_study) // user marked the checkbox
        {
            if (es.video_reached_end) // we reached the video end, or we just started
            {
                int attempts = user_study.getAttempts();
                if (attempts % 2 == 0) // if this is an even attempt, we should get human response and start a new trial
                {
                    if (attempts != 0)
                    {
                        // display "1 / 2" to subject indicating they need to choose
                        std::vector<std::string> texts_to_render = {std::string("1 / 2")};
                        for (int i = 0; i < texts_to_render.size(); ++i)
                        {
                            textModel.Render(*textShader, texts_to_render[i], -150 + es.proj_width / 2, es.proj_height / 2, 3.0f, glm::vec3(1.0f, 1.0f, 1.0f));
                        }
                        bool button_values[9] = {false, false, false, false, false, false, false, false, false};
                        GetButtonStates(button_values); // get input from subject
                        es.humanChoice = 0;
                        bool one_pressed = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
                        bool two_pressed = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;
                        if (button_values[0] || one_pressed)
                            es.humanChoice = 1;
                        if (button_values[1] || two_pressed)
                            es.humanChoice = 2;
                        if (es.humanChoice == 0 || (one_pressed && two_pressed))
                        {
                            return;
                        }
                    }
                }
                es.vid_simulated_latency_ms = user_study.randomTrial(es.humanChoice); // get required latency given subject choice
                es.simulationTime = 0.0f;                                             // reset video playback
                es.video_reached_end = false;
                if (user_study.getTrialFinished()) // experiment is finished
                {
                    user_study.printStats();
                    CloseMidiController();
                    es.run_user_study = false;
                    return;
                }
                // std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // let subject rest a bit
            }
            else
            {
                playVideo(shaderMap,
                          leftHandModel,
                          rightHandModel,
                          textModel,
                          cam_view_transform,
                          cam_projection_transform);
            }
        }
        else
        {
            if (es.debug_playback)
            {

                if (!playVideo(shaderMap,
                               leftHandModel,
                               rightHandModel,
                               textModel,
                               cam_view_transform,
                               cam_projection_transform))
                {
                    t_profile0.stop();
                    t_profile0.stop();
                    std::cout << "video playback time: " << t_profile0.getElapsedTimeInMilliSec() << " ms" << std::endl;
                    es.simulationTime = 0.0f;
                    es.is_first_in_video_pair = true;
                    es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
                    t_profile0.start();
                }
            }
        }
    }
}

void handleDebugMode(std::unordered_map<std::string, Shader *> &shader_map,
                     SkinnedModel &rightHandModel,
                     SkinnedModel &leftHandModel,
                     SkinnedModel &otherObject,
                     TextModel &text)
{
    SkinningShader *skinnedShader = dynamic_cast<SkinningShader *>(shader_map["skinnedShader"]);
    Shader *textureShader = shader_map["textureShader"];
    Shader *vcolorShader = shader_map["vcolorShader"];
    Shader *textShader = shader_map["textShader"];
    Shader *lineShader = shader_map["lineShader"];
    if (es.debug_mode && es.operation_mode == static_cast<int>(OperationMode::SANDBOX))
    {
        glm::mat4 proj_view_transform = gl_projector.getViewMatrix();
        glm::mat4 proj_projection_transform = gl_projector.getProjectionMatrix();
        glm::mat4 flycam_view_transform = gl_flycamera.getViewMatrix();
        glm::mat4 flycam_projection_transform = gl_flycamera.getProjectionMatrix();
        glm::mat4 cam_view_transform = gl_camera.getViewMatrix();
        glm::mat4 cam_projection_transform = gl_camera.getProjectionMatrix();
        // very redundant, but redraw hand meshes
        {
            handleSkinning(bones_to_world_right, true, true, shader_map, rightHandModel, flycam_view_transform, flycam_projection_transform);
            handleSkinning(bones_to_world_left, false, bones_to_world_right.size() == 0, shader_map, leftHandModel, flycam_view_transform, flycam_projection_transform);
            glViewport(0, 0, es.proj_width, es.proj_height); // set viewport
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            set_texture_shader(textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
            hands_fbo.getTexture()->bind();
            fullScreenQuad.render();
        }
        // draws some mesh (lit by camera input)
        {
            vcolorShader->use();
            vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform);
            vcolorShader->setBool("allWhite", false);
            otherObject.Render(*vcolorShader, camTexture.getTexture(), false);
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
        // draws unit cube at world origin
        {
            vcolorShader->use();
            glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
            vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * model);
            vcolorShader->setBool("allWhite", false);
            glBindVertexArray(cubeVAO);
            glDrawElements(GL_TRIANGLES,    // primitive type
                           36,              // # of indices
                           GL_UNSIGNED_INT, // data type
                           (void *)0);
            model = glm::scale(glm::mat4(1.0f), glm::vec3(15.0f, 15.0f, 15.0f));
            vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * model);
            glBindVertexArray(gizmoVAO);
            glDrawArrays(GL_LINES, 0, 6);
        }
        if (bones_to_world_right.size() > 0)
        {
            // draw light location as a cube, and gizmo for light orientation. todo: slightly broken
            {
                vcolorShader->use();
                glm::mat4 lightProjection, lightView;
                getLightTransform(lightProjection, lightView, bones_to_world_right);
                // vcolorShader->setMat4("projection", flycam_projection_transform);
                // vcolorShader->setMat4("view", flycam_view_transform);
                // vcolorShader->setMat4("model", glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 20.0f, 20.0f)));
                glm::mat4 light2world = glm::inverse(lightView);
                // glm::vec3 at = glm::vec3(light2world[3][0], light2world[3][1], light2world[3][2]);
                // glm::mat4 model = glm::translate(glm::mat4(1.0f), at);
                glm::mat4 model = glm::scale(light2world, glm::vec3(10.0f, 10.0f, 10.0f));
                vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * model);
                vcolorShader->setBool("allWhite", false);
                glBindVertexArray(cubeVAO);
                glDrawElements(GL_TRIANGLES,    // primitive type
                               36,              // # of indices
                               GL_UNSIGNED_INT, // data type
                               (void *)0);
                model = glm::scale(light2world, glm::vec3(15.0f, 15.0f, 15.0f));
                vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * model);
                glBindVertexArray(gizmoVAO);
                glDrawArrays(GL_LINES, 0, 6);
            }
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
            // draw gizmo for palm orientation
            {
                // vcolorShader.use();
                // vcolorShader.setMat4("projection", flycam_projection_transform);
                // vcolorShader.setMat4("view", flycam_view_transform);
                // vcolorShader.setMat4("model", bones_to_world_right[0]);
                // glBindVertexArray(gizmoVAO);
                // glDrawArrays(GL_LINES, 0, 6);
            }
        }
        if (bones_to_world_left.size() > 0)
        {
            // draw bones local coordinates as gizmos
            {
                // vcolorShader->use();
                // std::vector<glm::mat4> BoneToLocalTransforms;
                // leftHandModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                // glBindVertexArray(gizmoVAO);
                // // glm::mat4 scaler = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f, 10.0f, 10.0f));
                // for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                // {
                //     // in bind pose
                //     vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * rotx * BoneToLocalTransforms[i]);
                //     glDrawArrays(GL_LINES, 0, 6);
                // }
                // for (unsigned int i = 0; i < bones_to_world_left.size(); i++)
                // {
                //     // in leap motion pose
                //     vcolorShader->setMat4("mvp", flycam_projection_transform * flycam_view_transform * bones_to_world_left[i]);
                //     glDrawArrays(GL_LINES, 0, 6);
                // }
            }
        }
        // draws frustrum of camera (=vproj)
        {
            if (es.showCamera)
            {
                std::vector<glm::vec3> vprojFrustumVerticesData(28);
                lineShader->use();
                lineShader->setMat4("projection", flycam_projection_transform);
                lineShader->setMat4("view", flycam_view_transform);
                lineShader->setMat4("model", glm::mat4(1.0f));
                lineShader->setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
                glm::mat4 camUnprojectionMat = glm::inverse(cam_projection_transform * cam_view_transform);
                for (unsigned int i = 0; i < es.frustumCornerVertices.size(); i++)
                {
                    glm::vec4 unprojected = camUnprojectionMat * glm::vec4(es.frustumCornerVertices[i], 1.0f);
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
            if (es.showProjector)
            {
                std::vector<glm::vec3> vcamFrustumVerticesData(28);
                lineShader->use();
                lineShader->setMat4("projection", flycam_projection_transform);
                lineShader->setMat4("view", flycam_view_transform);
                lineShader->setMat4("model", glm::mat4(1.0f));
                lineShader->setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
                glm::mat4 projUnprojectionMat = glm::inverse(proj_projection_transform * proj_view_transform);
                for (int i = 0; i < es.frustumCornerVertices.size(); ++i)
                {
                    glm::vec4 unprojected = projUnprojectionMat * glm::vec4(es.frustumCornerVertices[i], 1.0f);
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
            if (es.showCamera)
            {
                std::vector<glm::vec3> camNearVerts(4);
                std::vector<glm::vec3> camMidVerts(4);
                // unproject points
                glm::mat4 camUnprojectionMat = glm::inverse(cam_projection_transform * cam_view_transform);
                for (int i = 0; i < es.mid_frustrum.size(); i++)
                {
                    glm::vec4 unprojected = camUnprojectionMat * glm::vec4(es.mid_frustrum[i], 1.0f);
                    camMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                    unprojected = camUnprojectionMat * glm::vec4(es.near_frustrum[i], 1.0f);
                    camNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
                }
                Quad camNearQuad(camNearVerts);
                // Quad camMidQuad(camMidVerts);
                // directly render camera input or any other texture
                set_texture_shader(textureShader, false, false, false, false, es.masking_threshold, 0, glm::mat4(1.0f), flycam_projection_transform, flycam_view_transform);
                // camTexture.bind();
                // glBindTexture(GL_TEXTURE_2D, resTexture);
                postprocess_fbo.getTexture()->bind();
                // hands_fbo.getTexture()->bind();
                camNearQuad.render();
            }
        }
        // draw warped output to near plane of projector
        {
            if (es.showProjector)
            {
                std::vector<glm::vec3> projNearVerts(4);
                // std::vector<glm::vec3> projMidVerts(4);
                std::vector<glm::vec3> projFarVerts(4);
                glm::mat4 projUnprojectionMat = glm::inverse(proj_projection_transform * proj_view_transform);
                for (int i = 0; i < es.mid_frustrum.size(); i++)
                {
                    // glm::vec4 unprojected = projUnprojectionMat * glm::vec4(mid_frustrum[i], 1.0f);
                    // projMidVerts[i] = glm::vec3(unprojected) / unprojected.w;
                    glm::vec4 unprojected = projUnprojectionMat * glm::vec4(es.near_frustrum[i], 1.0f);
                    projNearVerts[i] = glm::vec3(unprojected) / unprojected.w;
                    unprojected = projUnprojectionMat * glm::vec4(es.far_frustrum[i], 1.0f);
                    projFarVerts[i] = glm::vec3(unprojected) / unprojected.w;
                }
                Quad projNearQuad(projNearVerts);
                // Quad projMidQuad(projMidVerts);
                Quad projFarQuad(projFarVerts);
                set_texture_shader(textureShader, false, false, false, false, es.masking_threshold, 0, glm::mat4(1.0f), flycam_projection_transform, flycam_view_transform);
                c2p_fbo.getTexture()->bind();
                projNearQuad.render();
            }
        }
        // draws debug text
        {
            float text_spacing = 10.0f;
            glm::vec3 cur_cam_pos = gl_flycamera.getPos();
            glm::vec3 cur_cam_front = gl_flycamera.getFront();
            glm::vec3 cam_pos = gl_camera.getPos();
            std::vector<std::string> texts_to_render = {
                std::format("debug_vector: {:.02f}, {:.02f}, {:.02f}", es.debug_vec.x, es.debug_vec.y, es.debug_vec.z),
                std::format("ms_per_frame: {:.02f}, fps: {}", es.ms_per_frame, es.fps),
                std::format("cur_camera pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cur_cam_pos.x, cur_cam_pos.y, cur_cam_pos.z, gl_flycamera.Zoom),
                std::format("cur_camera front: {:.02f}, {:.02f}, {:.02f}", cur_cam_front.x, cur_cam_front.y, cur_cam_front.z),
                std::format("cam pos: {:.02f}, {:.02f}, {:.02f}, cam fov: {:.02f}", cam_pos.x, cam_pos.y, cam_pos.z, gl_camera.Zoom),
                std::format("Rhand visible? {}", bones_to_world_right.size() > 0 ? "yes" : "no"),
                std::format("Lhand visible? {}", bones_to_world_left.size() > 0 ? "yes" : "no"),
                std::format("modifiers : shift: {}, ctrl: {}, space: {}", es.shift_modifier ? "on" : "off", es.ctrl_modifier ? "on" : "off", es.space_modifier ? "on" : "off")};
            for (int i = 0; i < texts_to_render.size(); ++i)
            {
                text.Render(*textShader, texts_to_render[i], 25.0f, texts_to_render.size() * text_spacing - text_spacing * i, 0.25f, glm::vec3(1.0f, 1.0f, 1.0f));
            }
        }
    }
}

uint32_t getCurrentSimulationIndex()
{
    auto upper_iter = std::upper_bound(session_timestamps.begin(), session_timestamps.end(), es.simulationTime);
    return upper_iter - session_timestamps.begin() - 1;
}

bool playVideoReal(std::unordered_map<std::string, Shader *> &shader_map,
                   SkinnedModel &leftHandModel,
                   SkinnedModel &rightHandModel,
                   TextModel &textModel,
                   glm::mat4 &cam_view_transform,
                   glm::mat4 &cam_projection_transform)
{
    es.project_this_frame = false;
    Shader *textureShader = shader_map["textureShader"];
    Shader *vcolorShader = shader_map["vcolorShader"];
    Shader *gridShader = shader_map["gridShader"];
    int32_t cur_timestamp = getCurrentSimulationIndex();
    int discrete_timestep_diff = cur_timestamp - es.playback_prev_frame;
    if (discrete_timestep_diff > 1)
    {
        // we need to reduce simulation speed...
        es.pseudo_vid_playback_speed *= 0.9f;
    }
    bool new_frame = discrete_timestep_diff != 0;
    // make sure we are not at the end of the video
    if (cur_timestamp >= session_timestamps.size() || es.simulationTime >= session_timestamps.back())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return false;
    }
    // interpolate hand pose to the required latency
    t_leap.start();
    std::vector<glm::mat4> bones2world_left, bones2world_right;
    // std::vector<glm::vec3> jointsLeft, jointsRight;
    bool success = handleGetSkeletonByTimestamp(cur_timestamp + es.mls_future_frame_offset,
                                                bones2world_left, joints_left,
                                                bones2world_right, joints_right);
    if (!success)
    {
        std::cout << "what just happened?" << std::endl;
        exit(1);
    }
    projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
    t_leap.stop();
    // produce fake camera image (left hand only)
    t_camera.start();
    camImageOrig = recordedImages[cur_timestamp];
    camTexture.load((uint8_t *)camImageOrig.data, true, es.cam_buffer_format);
    t_camera.stop();

    // skin the mesh
    t_skin.start();
    handleSkinning(bones2world_right, true, true, shader_map, rightHandModel, cam_view_transform, cam_projection_transform);
    handleSkinning(bones2world_left, false, bones2world_right.size() == 0, shader_map, leftHandModel, cam_view_transform, cam_projection_transform);
    t_skin.stop();

    /* run Optical Flow */
    t_of.start();
    if (new_frame)
        handleOF(gridShader);
    t_of.stop();

    /* run MLS on MP prediction to reduce bias */
    t_mls.start();
    // check if this is a new camera frame, and mls should run this frame
    bool mls_landmark_detect_condition = (cur_timestamp % es.mls_every == 0) && (new_frame);
    // if simulated latency enbaled, and mls should run we need to extract info from the past
    if ((es.mls_n_latency_frames != 0) && mls_landmark_detect_condition)
    {
        int past_timestamp = cur_timestamp - es.mls_n_latency_frames;
        if (past_timestamp < 0)
            past_timestamp = 0;
        // update camera image to the one from the past
        camImageOrig = recordedImages[past_timestamp];
        // update the current leap frame to the past
        handleGetSkeletonByTimestamp(past_timestamp, bones2world_left, joints_left, bones2world_right, joints_right);
        projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
    }
    handleMLS(*gridShader, es.mls_blocking, mls_landmark_detect_condition, new_frame, true);
    t_mls.stop();

    /* post process fbo using camera input */
    t_pp.start();
    // get the current frame
    handleGetSkeletonByTimestamp(cur_timestamp + es.mls_future_frame_offset, bones2world_left, joints_left, bones2world_right, joints_right);
    projectAndFilterJoints(joints_left, joints_right, projected_filtered_left, projected_filtered_right);
    handlePostProcess(leftHandModel, rightHandModel, camTexture, shader_map);
    /* render final output to screen */
    glViewport(0, 0, es.proj_width, es.proj_height); // set viewport
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    set_texture_shader(textureShader, false, false, false, false, 0.035f, 0, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), es.gamma_correct);
    c2p_fbo.getTexture()->bind();
    fullScreenQuad.render();
    t_pp.stop();
    // add a red dot to the screen to indicate if landmark detectrion occured
    t_debug.start();
    if ((cur_timestamp % es.mls_every == 0) && es.use_mls)
    {
        std::vector<glm::vec2> tmp;
        tmp.push_back(glm::vec2(-0.9f, 0.9f));
        PointCloud cloud_src(tmp, es.screen_verts_color_red);
        vcolorShader->use();
        vcolorShader->setMat4("mvp", glm::mat4(1.0f));
        cloud_src.render(10.0f);
    }
    if (es.mls_succeeded_this_frame)
    {
        es.mls_succeed_counter += 1;
        es.mls_succeeded_this_frame = false;
    }
    if (new_frame)
    {
        // collect some stats
        es.mls_succeed_counters.push_back(static_cast<float>(es.mls_succeed_counter));
        es.mls_succeed_counter = 0;
        // also project the result
        es.project_this_frame = true;
    }
    es.playback_prev_frame = cur_timestamp;
    es.simulationTime += es.deltaTime * 1000.0f * es.vid_playback_speed * es.pseudo_vid_playback_speed;
    t_debug.stop();
    return true;
}

bool playVideo(std::unordered_map<std::string, Shader *> &shader_map,
               SkinnedModel &leftHandModel,
               SkinnedModel &rightHandModel,
               TextModel &textModel,
               glm::mat4 &cam_view_transform,
               glm::mat4 &cam_projection_transform)
{
    Shader *thresholdAlphaShader = shader_map["thresholdAlphaShader"];
    Shader *textureShader = shader_map["textureShader"];
    Shader *overlayShader = shader_map["overlayShader"];
    Shader *textShader = shader_map["textShader"];
    // interpolate hand pose to the required latency
    t_leap.start();
    std::vector<glm::mat4> bones2world_left_cur, bones2world_left_lag;
    std::vector<glm::mat4> bones2world_right_cur, bones2world_right_lag;
    bool success = handleInterpolateFrames(bones2world_left_cur, bones2world_right_cur, bones2world_left_lag, bones2world_right_lag);
    if (!success)
    {
        es.video_reached_end = true;
        es.is_first_in_video_pair = !es.is_first_in_video_pair;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return false;
    }
    t_leap.stop();
    // produce fake camera image (left hand only)
    t_camera.start();
    // first render the mesh normally with a skin texture
    es.curSelectedTexture = "realistic_skin3";
    handleSkinning(bones2world_left_cur, false, true, shader_map, leftHandModel, cam_view_transform, cam_projection_transform);
    handleSkinning(bones2world_right_cur, false, bones2world_left_cur.size() == 0, shader_map, rightHandModel, cam_view_transform, cam_projection_transform);
    fake_cam_fbo.bind();
    set_texture_shader(textureShader, true, true, false);
    hands_fbo.getTexture()->bind();
    fullScreenQuad.render();
    fake_cam_fbo.unbind();
    // now create a binary version of the same image (1 channel, either 1's or 0's)
    fake_cam_binary_fbo.bind();
    thresholdAlphaShader->use();
    thresholdAlphaShader->setMat4("view", glm::mat4(1.0));
    thresholdAlphaShader->setMat4("projection", glm::mat4(1.0));
    thresholdAlphaShader->setMat4("model", glm::mat4(1.0));
    thresholdAlphaShader->setFloat("threshold", 0.5);
    thresholdAlphaShader->setBool("flipHor", true);
    thresholdAlphaShader->setBool("flipVer", true);
    thresholdAlphaShader->setBool("isGray", false);
    thresholdAlphaShader->setBool("binary", true);
    thresholdAlphaShader->setInt("src", 0);
    hands_fbo.getTexture()->bind();
    fullScreenQuad.render();
    fake_cam_binary_fbo.unbind();
    t_camera.stop();
    // now render the projection image
    t_skin.start();
    es.curSelectedTexture = es.userStudySelectedTexture;
    handleSkinning(bones2world_left_lag, false, true, shader_map, leftHandModel, cam_view_transform, cam_projection_transform);
    handleSkinning(bones2world_right_lag, true, bones2world_left_lag.size() == 0, shader_map, rightHandModel, cam_view_transform, cam_projection_transform);
    t_skin.stop();
    t_pp.start();
    // post process the projection with any required effect
    handlePostProcess(leftHandModel, rightHandModel, *fake_cam_binary_fbo.getTexture(), shader_map);
    // }
    // overlay final render and camera image, and render to screen
    // this is needed to simulate how a projection is mixed with the skin color
    postprocess_fbo2.bind();
    overlayShader->use();
    overlayShader->setMat4("mvp", glm::mat4(1.0));
    overlayShader->setInt("projectiveTexture", 0);
    overlayShader->setFloat("mixRatio", es.projection_mix_ratio);
    overlayShader->setFloat("skinBrightness", es.skin_brightness);
    c2p_fbo.getTexture()->bind(GL_TEXTURE0);
    overlayShader->setInt("objectTexture", 1);
    overlayShader->setBool("gammaCorrect", es.gamma_correct);
    fake_cam_fbo.getTexture()->bind(GL_TEXTURE1);
    // set_texture_shader(textureShader, false, false, false);
    fullScreenQuad.render();
    postprocess_fbo2.unbind();
    // render to screen
    glViewport(0, 0, es.proj_width, es.proj_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    postprocess_fbo2.getTexture()->bind();
    set_texture_shader(textureShader, false, false, false);
    fullScreenQuad.render();
    t_pp.stop();
    // add a "number" to the screen to indicate which video is being played
    t_debug.start();
    float text_spacing = 10.0f;
    std::vector<std::string> texts_to_render = {std::format("{}", es.is_first_in_video_pair ? "1" : "2")};
    for (int i = 0; i < texts_to_render.size(); ++i)
    {
        textModel.Render(*textShader, texts_to_render[i], 25.0f, texts_to_render.size() * text_spacing - text_spacing * i, 3.0f, glm::vec3(1.0f, 1.0f, 1.0f));
    }
    t_debug.stop();
    es.simulationTime += es.deltaTime * 1000.0f * es.vid_playback_speed * es.pseudo_vid_playback_speed;
    return true;
}

void handleBakeConfig()
{
    switch (es.prompt_mode)
    {
    case static_cast<int>(PromptMode::MANUAL_PROMPT):
    {
        es.cur_prompt = es.manual_prompt;
        break;
    }
    case static_cast<int>(PromptMode::AUTO_PROMPT):
    {
        es.cur_prompt = "";
        break;
    }
    case static_cast<int>(PromptMode::SELECTED):
    {
        es.cur_prompt = es.selected_listed_prompt;
        break;
    }
    case static_cast<int>(PromptMode::RANDOM):
    {
        std::vector<std::string> random_animal;
        std::sample(es.listedPrompts.begin(),
                    es.listedPrompts.end(),
                    std::back_inserter(random_animal),
                    1,
                    std::mt19937{std::random_device{}()});
        es.cur_prompt = random_animal[0];
        break;
    }
    default:
        es.cur_prompt = es.manual_prompt;
        break;
    }
    if (es.save_byproducts)
    {
        fs::path bakeOutputPath{es.bake_folder};
        uint32_t count = 0;
        if (!fs::exists(bakeOutputPath))
        {
            fs::create_directories(bakeOutputPath);
        }
        else
        {
            std::set<std::string> unique_names;
            for (const auto &entry : fs::directory_iterator(bakeOutputPath))
            {
                std::string stem = entry.path().filename().stem().string(); // get stem
                std::string token = stem.substr(0, stem.find("_"));         // get number part
                if (unique_names.find(token) == unique_names.end())
                    unique_names.insert(token);
            }
            while (true)
            {
                if (unique_names.find(std::format("{:05d}", count)) == unique_names.end())
                {
                    break;
                }
                count++;
            }
        }
        std::string tmp_name = std::format("{:05d}", count);
        es.cur_bake_file_stem = fs::path(bakeOutputPath / fs::path(tmp_name)).string();
    }
    else
    {
        es.cur_bake_file_stem = "";
    }
}
// ---------------------------------------------------------------------------------------------
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
    ImGui::Begin("Augmented Hands", NULL, window_flags);
    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::TreeNode("General"))
        {
            if (ImGui::Checkbox("Use Projector", &es.use_projector))
            {
                if (es.use_projector)
                {
                    if (!projector->init())
                    {
                        std::cerr << "Failed to initialize projector\n";
                        es.use_projector = false;
                    }
                    else
                    {
                        if (!es.simulated_projector)
                        {
                            c2p_homography = PostProcess::findHomography(es.cur_screen_verts);
                            es.use_coaxial_calib = true;
                            es.gamma_correct = true;
                        }
                        else
                        {
                            std::string string_path = std::format("../../debug/{}", es.output_recording_name);
                            fs::path mypath(string_path);
                            if (!fs::exists(mypath))
                            {
                                fs::create_directory(mypath);
                            }
                            SaveToDisk *save_to_disk_projector = dynamic_cast<SaveToDisk *>(projector);
                            save_to_disk_projector->setDestination(string_path);
                        }
                    }
                }
                else
                {
                    if (!es.simulated_projector)
                    {
                        projector->kill();
                        es.use_coaxial_calib = false;
                        es.gamma_correct = false;
                    }
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Gamma Correction", &es.gamma_correct);
            ImGui::SameLine();
            ImGui::Checkbox("Command Line Stats", &es.cmd_line_stats);
            ImGui::Checkbox("Debug Mode", &es.debug_mode);
            ImGui::SameLine();
            if (ImGui::Checkbox("Freecam Mode", &es.freecam_mode))
            {
                create_virtual_cameras(gl_flycamera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            ImGui::Checkbox("PBO", &es.use_pbo);
            ImGui::Checkbox("Double PBO", &es.double_pbo);
            ImGui::SeparatorText("Operation Mode");
            if (ImGui::RadioButton("Normal", &es.operation_mode, static_cast<int>(OperationMode::SANDBOX)))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                es.exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(es.exposure);
                es.simulated_camera = false;
                if (es.simulated_projector)
                {
                    es.simulated_projector = false;
                    es.proj_channel_order = es.simulated_projector ? GL_RGB : GL_BGR;
                    delete projector;
                    projector = new DynaFlashProjector(true, false);
                }
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Simulation", &es.operation_mode, static_cast<int>(OperationMode::SIMULATION)))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                es.use_mls = true;
                es.debug_mode = false;
                es.postprocess_mode = static_cast<int>(PostProcessMode::OVERLAY);
                es.mask_missing_color_is_camera = true;
                es.mask_unused_info_color = es.mask_bg_color;
                es.mls_blocking = true;
                es.deformation_mode = static_cast<int>(DeformationMode::SIMILARITY);
                if (!es.simulated_projector)
                {
                    es.simulated_projector = true;
                    es.proj_channel_order = es.simulated_projector ? GL_RGB : GL_BGR;
                    delete projector;
                    projector = new SaveToDisk("../../debug/video/", es.proj_height, es.proj_width);
                }
                es.simulated_camera = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("User Study", &es.operation_mode, static_cast<int>(OperationMode::USER_STUDY)))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                es.use_mls = false;
                es.debug_mode = false;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Camera", &es.operation_mode, static_cast<int>(OperationMode::CAMERA)))
            {
                leap.setImageMode(false);
                es.exposure = 10000.0f;
                camera.set_exposure_time(es.exposure);
            }
            if (ImGui::RadioButton("Coaxial Calibration", &es.operation_mode, static_cast<int>(OperationMode::COAXIAL)))
            {
                es.debug_mode = false;
                leap.setImageMode(false);
                es.exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(es.exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Leap Calibration", &es.operation_mode, static_cast<int>(OperationMode::LEAP)))
            {
                projector->kill();
                es.use_projector = false;
                leap.setImageMode(true);
                leap.setPollMode(true);
                std::vector<uint8_t> buffer1, buffer2;
                while (!leap.getImage(buffer1, buffer2, es.leap_width, es.leap_height))
                    continue;
                // throttle down producer speed to allow smooth display
                // see https://docs.baslerweb.com/pylonapi/cpp/pylon_advanced_topics#grab-strategies
                es.exposure = 10000.0f;
                camera.set_exposure_time(es.exposure);
                es.debug_mode = false;
            }
            if (ImGui::RadioButton("Guess Pose Game", &es.operation_mode, static_cast<int>(OperationMode::GUESS_POSE_GAME)))
            {
                es.recording_name = "game";
                std::vector<std::vector<glm::mat4>> poses;
                if (!loadGamePoses(std::format("../../resource/recordings/{}", es.recording_name), poses))
                {
                    std::cout << "Failed to load recording: " << es.recording_name << std::endl;
                }
                guessPoseGame.setPoses(poses);
                guessPoseGame.reset(false);
                leap.setImageMode(false);
                leap.setPollMode(false);
                // mls_grid_shader_threshold = 0.8f; // allows for alpha blending mls results in game mode...
                es.material_mode = static_cast<int>(MaterialMode::PER_BONE_SCALAR);
                es.exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(es.exposure);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Guess Char Game", &es.operation_mode, static_cast<int>(OperationMode::GUESS_CHAR_GAME)))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                // mls_grid_shader_threshold = 0.8f; // allows for alpha blending mls results in game mode...
                es.material_mode = static_cast<int>(MaterialMode::DIFFUSE);
                es.exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(es.exposure);
                if (extraLeftHandModel == nullptr)
                {
                    extraLeftHandModel = new SkinnedModel(es.extraMeshFile,
                                                          es.userTextureFile,
                                                          es.proj_width, es.proj_height,
                                                          es.cam_width, es.cam_height);
                }
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Guess Animal Game", &es.operation_mode, static_cast<int>(OperationMode::GUESS_ANIMAL_GAME)))
            {
                leap.setImageMode(false);
                leap.setPollMode(false);
                // mls_grid_shader_threshold = 0.8f; // allows for alpha blending mls results in game mode...
                es.postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD_UV);
                es.texture_mode = static_cast<int>(TextureMode::BAKED);
                es.material_mode = static_cast<int>(MaterialMode::DIFFUSE);
                es.bake_mode = static_cast<int>(BakeMode::CONTROL_NET);
                es.prompt_mode = static_cast<int>(PromptMode::AUTO_PROMPT);
                es.no_preprompt = false;
                es.exposure = 1850.0f; // max exposure allowing for max fps
                camera.set_exposure_time(es.exposure);
            }
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Calibration"))
        {
            switch (es.operation_mode)
            {
            case static_cast<int>(OperationMode::SANDBOX):
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
                if (ImGui::Checkbox("Ready To Collect", &es.ready_to_collect))
                {
                    if (es.ready_to_collect)
                    {
                        imgpoints.clear();
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                    }
                }
                float calib_progress = imgpoints.size() / (float)es.n_points_cam_calib;
                char buf[32];
                sprintf(buf, "%d/%d points", (int)(calib_progress * es.n_points_cam_calib), es.n_points_cam_calib);
                ImGui::ProgressBar(calib_progress, ImVec2(-1.0f, 0.0f), buf);
                if (ImGui::Button("Calibrate Camera"))
                {
                    if (imgpoints.size() >= es.n_points_cam_calib)
                    {
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::SOLVE);
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
                    cnpy::npy_save("../../resource/calibrations/coaxial_calibration/coax_user.npy", es.cur_screen_verts.data(), {4, 2}, "w");
                }
                if (ImGui::BeginTable("Cam2Proj Vertices", 2))
                {
                    std::vector<glm::vec2> tmpVerts;
                    if (es.use_coaxial_calib)
                        tmpVerts = es.cur_screen_verts;
                    else
                        tmpVerts = es.screen_verts;
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
                ImGui::SliderInt("# Points to Collect", &es.n_points_leap_calib, 1000, 100000);
                ImGui::Checkbox("Use RANSAC", &es.leap_calib_use_ransac);
                ImGui::Text("Collection Procedure");
                ImGui::RadioButton("Auto Raw", &es.leap_collection_setting, static_cast<int>(LeapCollectionSettings::AUTO_RAW));
                ImGui::SameLine();
                ImGui::RadioButton("Auto Finger", &es.leap_collection_setting, static_cast<int>(LeapCollectionSettings::AUTO_FINGER));
                ImGui::SliderFloat("Binary Threshold", &es.leap_binary_threshold, 0.0f, 1.0f);
                if (ImGui::IsItemActive())
                {
                    es.leap_threshold_flag = true;
                }
                else
                {
                    es.leap_threshold_flag = false;
                }
                if (ImGui::Checkbox("Ready To Collect", &es.ready_to_collect))
                {
                    if (es.ready_to_collect)
                    {
                        points_2d.clear();
                        points_3d.clear();
                        points_2d_inliners.clear();
                        points_2d_reprojected.clear();
                        points_2d_inliers_reprojected.clear();
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
                    }
                }
                float calib_progress = points_2d.size() / (float)es.n_points_leap_calib;
                char buf[32];
                sprintf(buf, "%d/%d points", (int)(calib_progress * es.n_points_leap_calib), es.n_points_leap_calib);
                ImGui::ProgressBar(calib_progress, ImVec2(-1.0f, 0.0f), buf);
                ImGui::Text("cur. triangulated: %05.1f, %05.1f, %05.1f", es.triangulated.x, es.triangulated.y, es.triangulated.z);
                if (joints_right.size() > 0)
                {
                    ImGui::Text("cur. skeleton: %05.1f, %05.1f, %05.1f", joints_right[es.mark_bone_index].x, joints_right[es.mark_bone_index].y, joints_right[es.mark_bone_index].z);
                    float distance = glm::l2Norm(joints_right[es.mark_bone_index] - es.triangulated);
                    ImGui::Text("diff: %05.2f", distance);
                }
                ImGui::SliderInt("Selected Bone Index", &es.mark_bone_index, 0, 30);
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputInt("Iters", &es.pnp_iters);
                ImGui::SameLine();
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputFloat("Err.", &es.pnp_rep_error);
                ImGui::SameLine();
                ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.2f);
                ImGui::InputFloat("Conf.", &es.pnp_confidence);
                if (ImGui::Button("Calibrate"))
                {
                    if (points_2d.size() >= es.n_points_leap_calib)
                    {
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::SOLVE);
                    }
                }
                if (es.calibrationSuccess)
                {
                    if (ImGui::Checkbox("Show Calib Reprojections", &es.showReprojections))
                    {
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::SHOW);
                        es.showTestPoints = false;
                    }
                    if (ImGui::Checkbox("Show Test Points", &es.showTestPoints))
                    {
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::MARK);
                        es.showReprojections = false;
                    }
                    if (es.showReprojections)
                    {
                        ImGui::Checkbox("Show only inliers", &es.showInliersOnly);
                    }
                    if (es.showTestPoints)
                    {
                        ImGui::RadioButton("Stream", &es.leap_mark_setting, 0);
                        ImGui::SameLine();
                        ImGui::RadioButton("Point by Point", &es.leap_mark_setting, 1);
                        ImGui::SameLine();
                        ImGui::RadioButton("Whole Hand", &es.leap_mark_setting, 2);
                        ImGui::SameLine();
                        ImGui::RadioButton("Single Bone", &es.leap_mark_setting, 3);
                        // ImGui::ListBox("listbox", &item_current, items, IM_ARRAYSIZE(items), 4);
                        if (es.leap_mark_setting == static_cast<int>(LeapMarkSettings::POINT_BY_POINT))
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
                    if (!es.showReprojections && !es.showTestPoints)
                        es.calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
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
            if (ImGui::RadioButton("Calibration", &es.use_leap_calib_results, 0))
            {
                GLCamera dummy_camera;
                create_virtual_cameras(dummy_camera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Manual", &es.use_leap_calib_results, 1))
            {
                GLCamera dummy_camera;
                create_virtual_cameras(dummy_camera, gl_projector, gl_camera);
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Use Coaxial Calib", &es.use_coaxial_calib))
            {
                if (es.use_coaxial_calib)
                    c2p_homography = PostProcess::findHomography(es.cur_screen_verts);
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
            if (ImGui::SliderFloat("Camera Exposure [us]", &es.exposure, 300.0f, 10000.0f))
            {
                // std::cout << "current exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
                camera.set_exposure_time(es.exposure);
                // std::cout << "new exposure: " << camera.get_exposure_time() << " [us]" << std::endl;
            }
            ImGui::SliderFloat("Masking Threshold", &es.masking_threshold, 0.0f, 1.0f);
            if (ImGui::IsItemActive())
            {
                es.threshold_flag = true;
            }
            else
            {
                es.threshold_flag = false;
            }
            if (ImGui::Button("Screen Shot"))
            {
                std::string name = std::tmpnam(nullptr);
                fs::path filename(name);
                std::string savepath(std::string("../../debug/"));
                // std::cout << "unique file name: " << filename.filename().string() << std::endl;
                fs::path raw_image(savepath + filename.filename().string() + std::string("_raw_cam.png"));
                fs::path game_image(savepath + filename.filename().string() + std::string("_game.png"));
                fs::path mask_path(savepath + filename.filename().string() + std::string("_mask.png"));
                fs::path render_color(savepath + filename.filename().string() + std::string("_render_color.png"));
                fs::path render_depth(savepath + filename.filename().string() + std::string("_render_depth.png"));
                fs::path uv(savepath + filename.filename().string() + std::string("_uv.png"));
                fs::path mls(savepath + filename.filename().string() + std::string("_mls_fbo.png"));
                fs::path pp(savepath + filename.filename().string() + std::string("_pp.png"));
                fs::path pp2(savepath + filename.filename().string() + std::string("_pp2.png"));
                fs::path coax(savepath + filename.filename().string() + std::string("_coax.png"));
                fs::path joints_left_path(savepath + filename.filename().string() + std::string("_joints_left.npy"));
                fs::path joints_left_projected_path(savepath + filename.filename().string() + std::string("_joints_2d_left.npy"));
                fs::path bones_left_path(savepath + filename.filename().string() + std::string("_bones_left.npy"));
                fs::path joints_right_path(savepath + filename.filename().string() + std::string("_joints_right.npy"));
                fs::path joints_right_projected_path(savepath + filename.filename().string() + std::string("_joints_2d_right.npy"));
                fs::path bones_right_path(savepath + filename.filename().string() + std::string("_bones_right.npy"));
                if (es.operation_mode == static_cast<int>(OperationMode::USER_STUDY))
                {
                    fake_cam_fbo.saveColorToFile(raw_image.string(), false);
                    fake_cam_binary_fbo.saveColorToFile(mask_path.string(), false);
                }
                else
                {
                    cv::Mat tmp, mask;
                    cv::flip(camImageOrig, tmp, 1); // flip horizontally
                    cv::resize(tmp, tmp, cv::Size(es.proj_width, es.proj_height));
                    cv::threshold(tmp, mask, static_cast<int>(es.masking_threshold * 255), 255, cv::THRESH_BINARY);
                    cv::imwrite(raw_image.string(), tmp);
                    cv::imwrite(mask_path.string(), mask);
                }
                game_fbo.saveColorToFile(game_image.string());
                hands_fbo.saveColorToFile(render_color.string());
                hands_fbo.saveDepthToFile(render_depth.string(), true, 1.0f, 1500.0f);
                uv_fbo.saveColorToFile(uv.string());
                mls_fbo.saveColorToFile(mls.string());
                postprocess_fbo.saveColorToFile(pp.string());
                postprocess_fbo2.saveColorToFile(pp2.string());
                c2p_fbo.saveColorToFile(coax.string());
                if (joints_right.size() > 0)
                {
                    cnpy::npy_save(joints_right_path.string().c_str(), &joints_right[0].x, {joints_right.size(), 3}, "w");
                    cnpy::npy_save(bones_right_path.string().c_str(), &bones_to_world_right[0][0].x, {bones_to_world_right.size(), 4, 4}, "w");
                    std::vector<glm::vec2> projected = Helpers::project_points(joints_right, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                    cnpy::npy_save(joints_right_projected_path.string().c_str(), &projected[0].x, {projected.size(), 2}, "w");
                }
                if (joints_left.size() > 0)
                {
                    cnpy::npy_save(joints_left_path.string().c_str(), &joints_left[0].x, {joints_left.size(), 3}, "w");
                    cnpy::npy_save(bones_left_path.string().c_str(), &bones_to_world_left[0][0].x, {bones_to_world_left.size(), 4, 4}, "w");
                    std::vector<glm::vec2> projected = Helpers::project_points(joints_left, glm::mat4(1.0f), gl_camera.getViewMatrix(), gl_camera.getProjectionMatrix());
                    cnpy::npy_save(joints_left_projected_path.string().c_str(), &projected[0].x, {projected.size(), 2}, "w");
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Undistort Camera Input", &es.undistortCamera);
            ImGui::SeparatorText("Debug Mode Controls");
            ImGui::Checkbox("Show Camera", &es.showCamera);
            ImGui::SameLine();
            ImGui::Checkbox("Show Projector", &es.showProjector);
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
        if (ImGui::TreeNode("Light Controls"))
        {
            ImGui::Checkbox("Hard Shadows", &es.use_shadow_mapping);
            ImGui::SliderFloat("Shadow Bias", &es.shadow_bias, 0.001f, 0.1f);
            ImGui::SeparatorText("Light Mode");
            ImGui::RadioButton("Point", &es.light_mode, static_cast<int>(LightMode::POINT));
            ImGui::SameLine();
            ImGui::RadioButton("Directional", &es.light_mode, static_cast<int>(LightMode::DIRECTIONAL));
            ImGui::SameLine();
            ImGui::RadioButton("Projector", &es.light_mode, static_cast<int>(LightMode::PROJECTOR));
            switch (es.light_mode)
            {
            case static_cast<int>(LightMode::DIRECTIONAL):
            {
                ImGui::Checkbox("Light Is Projector", &es.light_is_projector);
                ImGui::SameLine();
                ImGui::Checkbox("Light Relative to Palm", &es.light_relative);
                if (ImGui::Button("LightPos 2 CamPos"))
                {
                    glm::mat4 camView = gl_camera.getViewMatrix();
                    es.light_at = glm::vec3(camView[3][0], camView[3][1], camView[3][2]);
                }
                ImGui::SliderFloat("Light Ambient Intensity", &es.light_ambient_intensity, 0.0f, 1.0f);
                ImGui::SliderFloat("Light Diffuse Intensity", &es.light_diffuse_intensity, 0.0f, 1.0f);
                ImGui::ColorEdit3("Light Color", &es.light_color.x, ImGuiColorEditFlags_NoOptions);
                ImGui::SliderFloat("Light Rad", &es.light_radius, 0.0f, 500.0f);
                ImGui::SliderFloat("Light Theta", &es.light_theta, -3.14f, 3.14f);
                ImGui::SliderFloat("Light Phi", &es.light_phi, 0.0f, 2 * 3.14f);
                // ImGui::SliderFloat3("Light At", &light_at.x, -1000.0f, 1000.0f);
                // ImGui::SliderFloat3("Light To", &light_to.x, -1000.0f, 1000.0f);
                // ImGui::SliderFloat3("Light Up", &light_up.x, -1.0f, 1.0f);
                ImGui::SliderFloat("Light Near", &es.light_near, 0.1f, 1000.0f);
                ImGui::SliderFloat("Light Far", &es.light_far, 0.1f, 1000.0f);
                break;
            }
            default:
                break;
            }
            ImGui::Checkbox("Surround Light", &es.surround_light);
            ImGui::SliderFloat("Surround Light Speed", &es.surround_light_speed, 0.001f, 0.01f);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Material & Effects"))
        {
            ImGui::SeparatorText("Material Type");
            ImGui::RadioButton("Diffuse", &es.material_mode, static_cast<int>(MaterialMode::DIFFUSE));
            ImGui::SameLine();
            ImGui::RadioButton("GGX", &es.material_mode, static_cast<int>(MaterialMode::GGX));
            ImGui::SameLine();
            ImGui::RadioButton("Skeleton", &es.material_mode, static_cast<int>(MaterialMode::SKELETON));
            ImGui::SameLine();
            ImGui::RadioButton("Per Bone Scalar", &es.material_mode, static_cast<int>(MaterialMode::PER_BONE_SCALAR));
            switch (es.material_mode)
            {
            case static_cast<int>(MaterialMode::SKELETON):
            {
                ImGui::Checkbox("Skeleton as Gizmos", &es.skeleton_as_gizmos);
                break;
            }
            default:
                break;
            }
            ImGui::SeparatorText("Diffuse Texture Type");
            ImGui::RadioButton("Original", &es.texture_mode, static_cast<int>(TextureMode::ORIGINAL));
            ImGui::SameLine();
            ImGui::RadioButton("From File", &es.texture_mode, static_cast<int>(TextureMode::FROM_FILE));
            ImGui::SameLine();
            ImGui::RadioButton("Projective Texture", &es.texture_mode, static_cast<int>(TextureMode::PROJECTIVE));
            ImGui::RadioButton("Baked", &es.texture_mode, static_cast<int>(TextureMode::BAKED));
            ImGui::SameLine();
            ImGui::RadioButton("Camera", &es.texture_mode, static_cast<int>(TextureMode::CAMERA));
            ImGui::SameLine();
            ImGui::RadioButton("Shader", &es.texture_mode, static_cast<int>(TextureMode::SHADER));
            if (ImGui::RadioButton("Multi-Shader", &es.texture_mode, static_cast<int>(TextureMode::MULTI_PASS_SHADER)))
            {
                es.gameFrameCount = 0;
                es.prevGameFrameCount = 0;
            }
            ImGui::SeparatorText("GGX Effects");
            ImGui::Checkbox("Diffuse Mapping", &es.use_diffuse_mapping);
            ImGui::SameLine();
            ImGui::Checkbox("Normal Mapping", &es.use_normal_mapping);
            // ImGui::Checkbox("Displacement Mapping", &use_disp_mapping);
            ImGui::SameLine();
            ImGui::Checkbox("AO/Roughness/Metallic Mapping", &es.use_arm_mapping);
            if (ImGui::BeginCombo("Mesh Texture", es.curSelectedTexture.c_str(), 0))
            {
                std::vector<std::string> keys;
                for (auto &it : texturePack)
                {
                    keys.push_back(it.first);
                }
                std::sort(keys.begin(), keys.end());
                for (auto &it : keys)
                {
                    const bool is_selected = (es.curSelectedTexture == it);
                    if (ImGui::Selectable(it.c_str(), is_selected))
                    {
                        es.curSelectedTexture = it;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::BeginCombo("Proj. Texture", es.curSelectedPTexture.c_str(), 0))
            {
                std::vector<std::string> keys;
                for (auto &it : texturePack)
                {
                    keys.push_back(it.first);
                }
                std::sort(keys.begin(), keys.end());
                for (auto &it : keys)
                {
                    const bool is_selected = (es.curSelectedPTexture == it);
                    if (ImGui::Selectable(it.c_str(), is_selected))
                    {
                        es.curSelectedPTexture = it;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::BeginCombo("User Study Texture", es.userStudySelectedTexture.c_str(), 0))
            {
                std::vector<std::string> keys;
                for (auto &it : texturePack)
                {
                    keys.push_back(it.first);
                }
                std::sort(keys.begin(), keys.end());
                for (auto &it : keys)
                {
                    const bool is_selected = (es.userStudySelectedTexture == it);
                    if (ImGui::Selectable(it.c_str(), is_selected))
                    {
                        es.userStudySelectedTexture = it;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SeparatorText("Load Texture");
            if (ImGui::Button("Load Texture"))
            {
                fs::path userTextureFilePath{es.userTextureFile};
                if (texturePack.find(userTextureFilePath.stem().string()) == texturePack.end())
                {
                    if (fs::exists(userTextureFilePath))
                    {
                        if (fs::is_regular_file(userTextureFilePath))
                        {
                            Texture *tmp = new Texture(es.userTextureFile.c_str(), GL_TEXTURE_2D);
                            tmp->init_from_file();
                            texturePack.insert({userTextureFilePath.stem().string(), tmp});
                        }
                    }
                }
            }
            ImGui::SameLine();
            ImGui::InputText("Texture File", &es.userTextureFile);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Bake"))
        {
            ImGui::SeparatorText("Bake Mode");
            ImGui::RadioButton("Control Net", &es.bake_mode, static_cast<int>(BakeMode::CONTROL_NET));
            ImGui::SameLine();
            ImGui::RadioButton("Stable Diffusion", &es.bake_mode, static_cast<int>(BakeMode::SD));
            ImGui::SameLine();
            ImGui::RadioButton("From File", &es.bake_mode, static_cast<int>(BakeMode::FILE));
            ImGui::RadioButton("Camera", &es.bake_mode, static_cast<int>(BakeMode::CAMERA));
            ImGui::SameLine();
            ImGui::RadioButton("Pose", &es.bake_mode, static_cast<int>(BakeMode::POSE));
            if (es.bake_mode == static_cast<int>(BakeMode::FILE))
            {
                ImGui::InputText("Input file", &es.inputBakeFile);
            }
            ImGui::InputText("Bake Output Path", &es.bake_folder);
            // ImGui::InputText("Bake texture file (right)", &es.bakeFileRight);
            // ImGui::InputText("Bake texture file (left)", &es.bakeFileLeft);
            if (ImGui::Button("Bake"))
            {
                handleBakeConfig();
                switch (es.bake_mode)
                {
                case static_cast<int>(BakeMode::FILE):
                {
                    fs::path inputBakeFilePath{es.inputBakeFile};
                    if (fs::exists(inputBakeFilePath))
                    {
                        if (fs::is_regular_file(inputBakeFilePath))
                        {
                            es.bakeRequest = true;
                        }
                    }
                    break;
                }
                default:
                {
                    es.bakeRequest = true;
                    break;
                }
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Save Intermediate Outputs", &es.save_byproducts);
            ImGui::SeparatorText("Stable Diffusion / Control Net");
            ImGui::SliderInt("ControlNet Present", &es.controlnet_preset, 0, 22);
            ImGui::Text("Prompts Mode");
            ImGui::SameLine();
            if (ImGui::RadioButton("Manual", &es.prompt_mode, 0))
            {
                es.no_preprompt = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Auto", &es.prompt_mode, static_cast<int>(PromptMode::AUTO_PROMPT)))
            {
                es.no_preprompt = false;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("List", &es.prompt_mode, static_cast<int>(PromptMode::SELECTED)))
            {
                es.no_preprompt = false;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Random", &es.prompt_mode, static_cast<int>(PromptMode::RANDOM)))
            {
                es.no_preprompt = false;
            }
            ImGui::Checkbox("Fit Mask to Viewport", &es.diffuse_fit_to_view);
            ImGui::SameLine();
            ImGui::InputInt("Pad Size", &es.diffuse_pad_size);
            ImGui::Checkbox("Select Top Animal", &es.diffuse_select_top_animal);
            ImGui::Checkbox("No preprompt", &es.no_preprompt);
            ImGui::InputInt("Random Seed", &es.diffuse_seed);
            ImGui::InputText("Manual Prompt", &es.manual_prompt);
            if (ImGui::BeginCombo("Listed Prompts", es.selected_listed_prompt.c_str(), 0))
            {
                std::vector<std::string> keys;
                for (auto &it : es.listedPrompts)
                {
                    keys.push_back(it);
                }
                std::sort(keys.begin(), keys.end());
                for (auto &it : keys)
                {
                    const bool is_selected = (es.selected_listed_prompt == it);
                    if (ImGui::Selectable(it.c_str(), is_selected))
                    {
                        es.selected_listed_prompt = it;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Text("SD Mask Mode");
            ImGui::SameLine();
            ImGui::RadioButton("Fill", &es.sd_mask_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Original", &es.sd_mask_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Latent Noise", &es.sd_mask_mode, 2);
            ImGui::SameLine();
            ImGui::RadioButton("Latent Nothing", &es.sd_mask_mode, 3);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("MLS"))
        {
            ImGui::SeparatorText("MLS");
            if (ImGui::Checkbox("MLS", &es.use_mls))
            {
                if (!es.use_mls)
                {
                    ControlPointsP_input_left.clear();
                    ControlPointsP_input_right.clear();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Auto Pilot", &es.auto_pilot);
            ImGui::SameLine();
            ImGui::Checkbox("Landmark Thread Blocking", &es.mls_blocking);
            ImGui::SliderFloat("Auto Pilot Thr", &es.auto_pilot_thr_extrapolate, 0.0f, 0.1f);
            // ImGui::ProgressBar(es.auto_pilot_delta, ImVec2(-1.0f, 0.0f), "delta");
            // ImGui::ProgressBar(es.auto_pilot_alpha, ImVec2(-1.0f, 0.0f), "alpha");
            ImGui::SliderInt("Auto Pilot Cnt Thr", &es.auto_pilot_count_thr, 0, 20);
            if (ImGui::Checkbox("Extrapolate Q using Leap", &es.mls_extrapolate))
            {
                if (es.mls_extrapolate)
                {
                    es.use_of = false;
                }
            }
            ImGui::Checkbox("Use Latest Leap as P", &es.mls_use_latest_leap);
            ImGui::Checkbox("Solve Grid Every Frame", &es.mls_solve_every_frame);
            if (ImGui::Checkbox("Kalman Filter Q", &es.mls_use_kalman))
            {
                if (es.mls_use_kalman)
                {
                    initKalmanFilters();
                }
            }
            ImGui::SliderInt("Sim: MLS Every N Frames", &es.mls_every, 1, 20);
            ImGui::SliderInt("Sim: MLS N Latency", &es.mls_n_latency_frames, 0, 20);
            ImGui::SliderInt("Sim: MLS N Future", &es.mls_future_frame_offset, 0, 20);
            // ImGui::RadioButton("CP1", &es.mls_mode, static_cast<int>(MLSMode::CONTROL_POINTS1));
            // ImGui::SameLine();
            // ImGui::RadioButton("CP2", &es.mls_mode, static_cast<int>(MLSMode::CONTROL_POINTS2));
            // ImGui::SameLine();
            // ImGui::RadioButton("GRID", &es.mls_mode, static_cast<int>(MLSMode::GRID));
            ImGui::RadioButton("Rigid", &es.deformation_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Similarity", &es.deformation_mode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Affine", &es.deformation_mode, 2);
            ImGui::Checkbox("Show MLS grid", &es.mls_show_grid);
            ImGui::SameLine();
            ImGui::Checkbox("Show Landmarks", &es.show_landmarks);
            ImGui::Checkbox("Global Forecast", &es.mls_global_forecast);
            ImGui::SameLine();
            // ImGui::Checkbox("Probe Recent Leap Frame", &es.mls_probe_recent_leap);
            ImGui::SliderInt("MLS CP Smooth window", &es.mls_cp_smooth_window, 0, 10);
            ImGui::SliderFloat("MLS grid shader thresh.", &es.mls_grid_shader_threshold, 0.0f, 1.0f);
            ImGui::SliderInt("MLS Grid Smooth window", &es.mls_grid_smooth_window, 0, 10);
            ImGui::SliderInt("Leap dt [us]", &es.magic_leap_time_delay, -50000, 50000);
            ImGui::SliderInt("Leap dt (mls) [us]", &es.magic_leap_time_delay_mls, -50000, 50000);

            ImGui::SliderFloat("Kalman Lookahead [ms]", &es.kalman_lookahead, 0.0f, 200.0f);
            // if (ImGui::IsItemDeactivatedAfterEdit())
            // {
            //     initKalmanFilters();
            // }
            ImGui::SliderFloat("Kalman Pnoise", &es.kalman_process_noise, 0.00001f, 1.0f, "%.6f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                initKalmanFilters();
            }
            ImGui::SliderFloat("Kalman Mnoise", &es.kalman_measurement_noise, 0.00001f, 1.0f, "%.6f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                initKalmanFilters();
            }

            ImGui::SliderFloat("MLS Alpha", &es.mls_alpha, 0.01f, 5.0f);
            // ImGui::SliderFloat("MLS grab threshold", &mls_grab_threshold, -1.0f, 5.0f);
            ImGui::Checkbox("MLS Depth Test", &es.mls_depth_test);
            ImGui::SameLine();
            ImGui::SliderFloat("D. thre", &es.mls_depth_threshold, 1.0f, 1500.0f, "%.6f");
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Post Process"))
        {
            ImGui::SeparatorText("Post Processing Mode");
            ImGui::RadioButton("Render Only", &es.postprocess_mode, static_cast<int>(PostProcessMode::NONE));
            ImGui::SameLine();
            ImGui::RadioButton("Camera Feed", &es.postprocess_mode, static_cast<int>(PostProcessMode::CAM_FEED));
            ImGui::SameLine();
            ImGui::RadioButton("Overlay", &es.postprocess_mode, static_cast<int>(PostProcessMode::OVERLAY));
            ImGui::RadioButton("JFA", &es.postprocess_mode, static_cast<int>(PostProcessMode::JUMP_FLOOD));
            ImGui::SameLine();
            ImGui::RadioButton("JFA UV", &es.postprocess_mode, static_cast<int>(PostProcessMode::JUMP_FLOOD_UV));
            ImGui::Checkbox("JF-UV for Right Hand", &es.jfauv_right_hand);
            ImGui::SameLine();
            ImGui::Checkbox("Blur", &es.postprocess_blur);
            ImGui::SliderFloat("Masking Threshold", &es.masking_threshold, 0.0f, 1.0f);
            if (ImGui::IsItemActive())
            {
                es.threshold_flag = true;
            }
            else
            {
                es.threshold_flag = false;
            }
            ImGui::SliderFloat("JFA Distance Threshold", &es.jfa_distance_threshold, 0.0f, 100.0f);
            ImGui::SliderFloat("JFA Seam Threshold", &es.jfa_seam_threshold, 0.0f, 2.0f);
            ImGui::ColorEdit3("FG Color", &es.mask_fg_color.x, ImGuiColorEditFlags_NoOptions);
            ImGui::ColorEdit3("BG Color", &es.mask_bg_color.x, ImGuiColorEditFlags_NoOptions);
            ImGui::ColorEdit3("Missing Info Color", &es.mask_missing_info_color.x, ImGuiColorEditFlags_NoOptions);
            ImGui::ColorEdit3("Unused Info Color", &es.mask_unused_info_color.x, ImGuiColorEditFlags_NoOptions);
            ImGui::Checkbox("FG single color", &es.mask_fg_single_color);
            ImGui::SameLine();
            ImGui::Checkbox("Missing Info Is Camera", &es.mask_missing_color_is_camera);
            ImGui::SameLine();
            // ImGui::SliderFloat("Alpha Blending", &es.mask_alpha, 0.0f, 1.0f);
            ImGui::Checkbox("Threshold Camera", &es.threshold_flag2);
            ImGui::TreePop();
        }
        /////////////////////////////////////////////////////////////////////////////
        if (ImGui::TreeNode("Leap Controls"))
        {
            ImGui::SeparatorText("General Controls");
            if (ImGui::Checkbox("Leap Polling Mode", &es.leap_poll_mode))
            {
                leap.setPollMode(es.leap_poll_mode);
            }
            ImGui::SliderInt("Leap Accumulate Frames", &es.leap_accumulate_frames, 1, 50);
            ImGui::SliderInt("Leap Accumulate Spread", &es.leap_accumulate_spread, 10, 5000);
            ImGui::SliderInt("Leap dt [us]", &es.magic_leap_time_delay, -50000, 50000);
            ImGui::SliderInt("Leap dt (mls) [us]", &es.magic_leap_time_delay_mls, -50000, 50000);
            if (ImGui::RadioButton("Desktop", &es.leap_tracking_mode, 0))
            {
                leap.setTrackingMode(eLeapTrackingMode_Desktop);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("HMD", &es.leap_tracking_mode, 1))
            {
                leap.setTrackingMode(eLeapTrackingMode_ScreenTop);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Screentop", &es.leap_tracking_mode, 2))
            {
                leap.setTrackingMode(eLeapTrackingMode_HMD);
            }
            ImGui::Checkbox("Use Finger Width", &es.useFingerWidth);
            ImGui::SameLine();
            ImGui::Checkbox("Use Arm Bone", &es.leap_use_arm);
            ImGui::SliderFloat("Leap Global Scale", &es.leap_global_scaler, 0.1f, 10.0f);
            ImGui::SliderFloat("Leap Bone Scale", &es.magic_leap_scale_factor, 1.0f, 20.0f);
            ImGui::SliderFloat("Leap Wrist Offset", &es.magic_wrist_offset, -100.0f, 100.0f);
            ImGui::SliderFloat("Leap Arm Offset (from palm)", &es.magic_arm_offset, -200.0f, 200.0f);
            ImGui::SliderFloat("Leap Arm Offset (from elbow)", &es.magic_arm_forward_offset, -300.0f, 200.0f);
            ImGui::SliderFloat("Leap Local Bone Scale", &es.leap_bone_local_scaler, 0.001f, 0.1f);
            ImGui::SliderFloat("Leap Palm Scale", &es.leap_palm_local_scaler, 0.001f, 0.1f);
            ImGui::SliderFloat("Leap Arm Scale", &es.leap_arm_local_scaler, 0.001f, 0.1f);
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Record / Playback"))
        {
            ImGui::SeparatorText("Record");
            if (ImGui::Checkbox("Record Session", &es.record_session))
            {
                std::cout << "Recording started" << std::endl;
                es.recordStartTime = t_app.getElapsedTimeInSec();
                savedLeapBonesLeft.clear();
                savedLeapJointsLeft.clear();
                savedLeapBonesRight.clear();
                savedLeapJointsRight.clear();
                savedLeapTimestamps.clear();
                recordedImages.clear();
                savedCameraTimestamps.clear();
            }
            ImGui::RadioButton("Record Right Hand", &es.recordedHand, static_cast<int>(Hand::RIGHT));
            ImGui::SameLine();
            ImGui::RadioButton("Record Left Hand", &es.recordedHand, static_cast<int>(Hand::LEFT));
            ImGui::Checkbox("Two Hands Recording", &es.two_hand_recording);
            ImGui::SameLine();
            ImGui::Checkbox("Record Images", &es.recordImages);
            ImGui::SameLine();
            ImGui::Checkbox("Record Every Frame", &es.record_every_frame);
            ImGui::SliderFloat("Recording Duration [s]", &es.recordDuration, 0.0f, 20.0f);
            if (ImGui::Button("Record Single Pose"))
            {
                es.record_single_pose = true;
            }
            ImGui::InputText("Raw Recording Name", &es.recording_name);
            ImGui::InputText("Output Recording Name", &es.output_recording_name);
            ImGui::SeparatorText("Playback");
            // ImGui::Checkbox("Playback w Images", &es.playback_with_images);
            ImGui::SliderFloat("Desired Latency [ms]", &es.vid_simulated_latency_ms, 0.0f, 50.0f);
            ImGui::SliderFloat("Initial Latency [ms]", &es.initial_simulated_latency_ms, 0.0f, 50.0f);
            ImGui::SliderFloat("Video Playback Speed", &es.vid_playback_speed, 0.1f, 10.0f);         // will take into account the simulated latency
            ImGui::SliderFloat("Video Playback Limiter", &es.pseudo_vid_playback_speed, 0.0f, 1.5f); // will just speed things up, without taking into account the simulated latency
            if (ImGui::Button("Step 1 Frame"))
            {
                auto upper_iter = std::upper_bound(session_timestamps.begin(), session_timestamps.end(), es.simulationTime);
                int32_t most_recent_index = upper_iter - session_timestamps.begin() - 1;
                es.simulationTime = session_timestamps[most_recent_index + 1];
            }
            ImGui::SliderFloat("Mixer Ratio", &es.projection_mix_ratio, 0.0f, 1.0f);
            ImGui::SliderFloat("Skin Brightness", &es.skin_brightness, 0.0f, 1.0f);
            if (ImGui::Checkbox("Debug Playback", &es.debug_playback))
            {
                if (es.debug_playback)
                {
                    if (es.recording_name != es.loaded_session_name)
                    {
                        if (!loadSession())
                            std::cout << "Failed to load recording: " << es.recording_name << std::endl;
                    }
                    es.simulationTime = 0.0f;
                    es.video_reached_end = true;
                    es.is_first_in_video_pair = true;
                    es.use_coaxial_calib = false;
                    t_profile0.start();
                    es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
                }
            }

            ImGui::TreePop();
        }
        if (ImGui::TreeNode("User Study"))
        {
            ImGui::InputText("Subject Name", &es.subject_name);
            if (ImGui::Checkbox("Run JND User Study", &es.run_user_study))
            {
                if (es.run_user_study)
                {
                    if (!OpenMidiController())
                        std::cout << "Midi Controller is not available!" << std::endl;
                    if (es.recording_name != es.loaded_session_name)
                    {
                        if (!loadSession())
                            std::cout << "Failed to load recording: " << es.recording_name << std::endl;
                    }
                    es.jfa_distance_threshold = 100.0f;
                    es.simulationTime = 0.0f;
                    es.video_reached_end = true;
                    es.is_first_in_video_pair = true;
                    es.use_coaxial_calib = false;
                    es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
                    user_study.reset(es.initial_simulated_latency_ms);
                }
                else
                {
                    CloseMidiController();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Forced Latency", &es.force_latency);
            ImGui::SameLine();
            if (ImGui::Button("Refresh MIDI Controller"))
            {
                CloseMidiController();
                if (!OpenMidiController())
                    std::cout << "Midi Controller is not available!" << std::endl;
            }
            ImGui::SliderFloat("Desired Latency [ms]", &es.vid_simulated_latency_ms, 0.0f, 50.0f);
            ImGui::SliderFloat("Initial Latency [ms]", &es.initial_simulated_latency_ms, 0.0f, 50.0f);
            ImGui::SliderFloat("Video Playback Speed", &es.vid_playback_speed, 0.1f, 10.0f);
            ImGui::SliderFloat("Video Playback Limiter", &es.pseudo_vid_playback_speed, 0.0f, 1.5f);
            ImGui::SliderFloat("Mixer Ratio", &es.projection_mix_ratio, 0.0f, 1.0f);
            ImGui::SliderFloat("Skin Brightness", &es.skin_brightness, 0.0f, 1.0f);
            if (ImGui::Checkbox("Debug Playback", &es.debug_playback))
            {
                if (es.debug_playback)
                {
                    if (es.recording_name != es.loaded_session_name)
                    {
                        if (!loadSession())
                            std::cout << "Failed to load recording: " << es.recording_name << std::endl;
                    }
                    es.simulationTime = 0.0f;
                    es.video_reached_end = true;
                    es.is_first_in_video_pair = true;
                    es.texture_mode = static_cast<int>(TextureMode::FROM_FILE);
                }
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Game Controls"))
        {
            if (ImGui::Button("Reset Game"))
            {
                int sessionNumber = guessCharGame.getTotalSessionCounter();
                bool randomizeSession;
                if (sessionNumber == 0)
                {
                    if (es.gameSessionType == static_cast<int>(GameSessionType::A))
                        es.curSessionType = static_cast<int>(GameSessionType::A);
                    else
                        es.curSessionType = static_cast<int>(GameSessionType::B);
                    randomizeSession = true;
                }
                else
                {
                    if (sessionNumber % 2 == 0)
                    {
                        if (es.curSessionType == static_cast<int>(GameSessionType::A))
                            es.curSessionType = static_cast<int>(GameSessionType::B);
                        else
                            es.curSessionType = static_cast<int>(GameSessionType::A);
                    }
                    if (sessionNumber % 4 == 0)
                        randomizeSession = true;
                    else
                        randomizeSession = false;
                }
                if (es.curSessionType == static_cast<int>(GameSessionType::A))
                {
                    es.postprocess_mode = static_cast<int>(PostProcessMode::NONE);
                    es.use_mls = false;
                    guessCharGame.reset(randomizeSession, "Baseline");
                }
                else
                {
                    es.postprocess_mode = static_cast<int>(PostProcessMode::JUMP_FLOOD_UV);
                    es.use_mls = true;
                    guessCharGame.reset(randomizeSession, "Ours");
                }
            }
            // ImGui::SameLine();
            // if (ImGui::Button("Skip Session"))
            // {
            //     guessCharGame.setResponse(true);
            // }
            if (ImGui::RadioButton("Session Type A", &es.gameSessionType, static_cast<int>(GameSessionType::A)))
            {
                guessCharGame.hardReset();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Session Type B", &es.gameSessionType, static_cast<int>(GameSessionType::B)))
            {
                guessCharGame.hardReset();
            }
            ImGui::Checkbox("Use Right Hand", &es.gameUseRightHand);
            ImGui::SameLine();
            ImGui::Checkbox("Show Game Hint", &es.showGameHint);
            ImGui::SliderFloat3("Debug Vector", &es.debug_vec.x, -1.0f, 1.0f);
            ImGui::SliderFloat("Debug Scalar", &es.debug_scalar, 0.0f, 5.0f);
            ImGui::InputText("Debug Text", &es.debug_text);
            ImGui::SeparatorText("Game Quad");
            ImWidgets::RangeSelect2D("Game Quad", &es.game_min.x, &es.game_min.y, &es.game_max.x, &es.game_max.y, -1.0f, -1.0f, 1.0f, 1.0f, 0.5f);
            // ImGui::SliderFloat("Corner", &game_corner_loc, -1.0f, 1.0f);
            if (ImGui::IsItemActive())
            {
            }
            else
            {
                std::vector<glm::vec2> new_game_verts = {glm::vec2(es.game_min.x, es.game_max.y),
                                                         glm::vec2(es.game_min.x, es.game_min.y),
                                                         glm::vec2(es.game_max.x, es.game_min.y),
                                                         glm::vec2(es.game_max.x, es.game_max.y)};
                es.game_verts = new_game_verts;
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Optical Flow"))
        {
            ImGui::SeparatorText("Optical Flow");
            if (ImGui::Checkbox("Use OF", &es.use_of))
            {
                if (es.use_of)
                {
                    updateOFParams();
                    es.mls_extrapolate = false;
                }
            }
            if (ImGui::RadioButton("Naive", &es.of_mode, static_cast<int>(OFMode::NAIVE_BLOB)))
            {
                updateOFParams();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Farneback CPU", &es.of_mode, static_cast<int>(OFMode::FB_CPU)))
            {
                updateOFParams();
            }
#ifdef OPENCV_WITH_CUDA
            if (ImGui::RadioButton("Farneback GPU", &es.of_mode, static_cast<int>(OFMode::FB_GPU)))
            {
                updateOFParams();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Nvidia GPU", &es.of_mode, static_cast<int>(OFMode::NV_GPU)))
            {
                updateOFParams();
            }
#endif
            if (ImGui::SliderInt("Resize Factor", &es.of_resize_factor_exp, 0, 3))
            {
                es.of_resize_factor = std::pow(2, es.of_resize_factor_exp);
                es.of_downsize = cv::Size(es.cam_width / es.of_resize_factor, es.cam_height / es.of_resize_factor);
                updateOFParams();
            }
            ImGui::Checkbox("Show OF", &es.show_of);
            ImGui::SliderInt("ROI size", &es.of_roi, 10, 100);
            ImGui::TreePop();
        }
    }
    ImGui::End();
}