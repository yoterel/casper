#ifndef ENGINE_STATE_H
#define ENGINE_STATE_H
#include <string>
#include "leapCPP.h"
#include "helpers.h"

class EngineState // dumb dumb engine state container
{
public:
    EngineState() {}
    const unsigned int proj_width = 1024;
    const unsigned int proj_height = 768;
    const unsigned int cam_width = 720;
    const unsigned int cam_height = 540;
    const unsigned int num_texels = proj_width * proj_height;
    const unsigned int projected_image_size = num_texels * 3 * sizeof(uint8_t);
    bool debug_mode = false;
    bool cam_space = false;
    bool cmd_line_stats = false;
    bool use_cuda = false;
    bool simulated_camera = false;
    bool simulated_projector = false;
    int proj_channel_order = simulated_projector ? GL_RGB : GL_BGR;
    bool freecam_mode = false;
    bool use_pbo = true;
    bool double_pbo = false;
    bool use_projector = false;
    bool project_this_frame = true;
    bool gamma_correct = false;
    bool use_screen = true;
    bool cam_color_mode = false;
    bool icp_apply_transform = true;
    bool showCamera = true;
    bool showProjector = true;
    bool undistortCamera = false;
    bool jfauv_right_hand = false;
    bool mask_fg_single_color = false;
    bool mask_missing_color_is_camera = false;
    bool threshold_flag = false;
    bool threshold_flag2 = false;
    int postprocess_mode = static_cast<int>(PostProcessMode::OVERLAY);
    int texture_mode = static_cast<int>(TextureMode::ORIGINAL);
    int material_mode = static_cast<int>(MaterialMode::DIFFUSE);
    int light_mode = static_cast<int>(LightMode::DIRECTIONAL);
    glm::vec3 light_color = glm::vec3(255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f);
    glm::vec3 light_at = glm::vec3(-45.0f, -131.0f, -61.0f);
    glm::vec3 light_to = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 light_up = glm::vec3(0.0f, 0.0f, 1.0f);
    // cv::Mat white_image = cv::Mat(cam_height, cam_width, CV_8UC1, cv::Scalar(255));
    float light_theta = 2.0f;
    float light_phi = 4.3f;
    float light_radius = 150.0f;
    float light_near = 1.0f;
    float light_far = 1000.0f;
    float light_ambient_intensity = 0.2f;
    float light_diffuse_intensity = 1.0f;
    bool light_is_projector = false;
    bool surround_light = false;
    float surround_light_speed = 0.001f;
    bool light_relative = true;
    bool use_shadow_mapping = false;
    bool use_diffuse_mapping = true;
    bool use_normal_mapping = true;
    // bool use_disp_mapping = false;
    bool use_arm_mapping = true;
    float shadow_bias = 0.005f;
    float deltaTime = 0.0f;
    float masking_threshold = 0.035f;
    float jfa_distance_threshold = 10.0f;
    float jfa_seam_threshold = 0.5f;
    glm::vec3 mask_bg_color = glm::vec3(0.0f, 0.0f, 0.0f);
    float mask_alpha = 1.0f;
    glm::vec3 mask_fg_color = glm::vec3(64.0f / 255.0f, 176.0f / 255.0f, 166.0f / 255.0f);
    glm::vec3 mask_missing_info_color = glm::vec3(47.0f / 255.0f, 103.0f / 255.0f, 177.0f / 255.0f);
    glm::vec3 mask_unused_info_color = glm::vec3(191.0f / 255.0f, 44.0f / 255.0f, 35.0f / 255.0f);
    glm::vec3 debug_vec = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 triangulated = glm::vec3(0.0f, 0.0f, 0.0f);
    std::string debug_text = "X";
    float debug_scalar = 1.0f;
    unsigned int fps = 0;
    float ms_per_frame = 0;
    unsigned int displayBoneIndex = 0;
    int64_t totalFrameCount = 0;
    int64_t maxVideoFrameCount = 0;
    bool canUseRecordedImages = false;
    int64_t curFrameID = 0;
    int64_t curFrameTimeStamp = 0;
    bool space_modifier = false;
    bool shift_modifier = false;
    bool ctrl_modifier = false;
    bool activateGUI = false;
    bool tab_pressed = false;
    bool rmb_pressed = false;
    std::string meshFile;
    std::string extraMeshFile;
    // bake/sd controls
    bool bakeRequest = false;
    bool deformedBaking = false;
    bool bake_preproc_succeed = false;
    bool sd_running = false;
    int sd_mode = static_cast<int>(SDMode::PROMPT);
    int bake_mode = static_cast<int>(BakeMode::SD);
    int sd_mask_mode = 2;
    bool saveIntermed = false;
    int sd_outwidth, sd_outheight;
    std::vector<uint8_t> img2img_data;
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
    // game controls
    bool showGameHint = false;
    int gameSessionType = static_cast<int>(GameSessionType::A);
    int curSessionType = static_cast<int>(GameSessionType::A);
    glm::vec2 game_min = glm::vec2(-1.0f, -1.0f);
    glm::vec2 game_max = glm::vec2(1.0f, 1.0f);
    std::vector<glm::vec2> game_verts = {glm::vec2(game_min.x, game_max.y),
                                         glm::vec2(game_min.x, game_min.y),
                                         glm::vec2(game_max.x, game_min.y),
                                         glm::vec2(game_max.x, game_max.y)};
    float gameTime = 0.0f;
    float gameSpeed = 0.01f; // the lower the faster
    int64_t gameFrameCount = 0;
    int64_t prevGameFrameCount = 0;
    bool gameUseRightHand = false;
    // user study controls
    bool run_user_study = false;
    bool force_latency = false;
    int humanChoice = 1;
    bool video_reached_end = true;
    bool is_first_in_video_pair = true;
    double prev_vid_time;
    double cur_vid_time;
    std::pair<int, int> video_pair;
    float vid_simulated_latency_ms = 1.0f;
    float initial_simulated_latency_ms = 20.0f;
    // record & playback controls
    bool debug_playback = false;
    int32_t playback_prev_frame = -1;
    // bool playback_with_images = false;
    float pseudo_vid_playback_speed = 1.1f;
    float vid_playback_speed = 1.0f;
    float projection_mix_ratio = 0.4f;
    float skin_brightness = 0.5f;
    float videoFrameCountCont = 0.0f;
    bool recordImages = false;
    std::string recording_name = "test"; // all_10s
    std::string output_recording_name = "video";
    std::string subject_name = "subject1";
    std::string loaded_session_name = "";
    bool pre_recorded_session_loaded = false;
    bool playback_video_loaded = false;
    int total_session_time_stamps = 0;
    bool record_session = false;
    bool two_hand_recording = false;
    int recordedHand = static_cast<int>(Hand::LEFT);
    float recordStartTime = 0.0;
    float projectStartTime = 0.0;
    float recordDuration = 5.0;
    bool record_single_pose = false;
    bool record_every_frame = false;
    // leap controls
    bool leap_poll_mode = false;
    bool leap_use_arm = false;
    int leap_tracking_mode = eLeapTrackingMode_HMD;
    uint64_t leap_cur_frame_id = 0;
    uint32_t leap_width = 640;
    uint32_t leap_height = 240;
    bool useFingerWidth = false;
    int32_t leap_accumulate_frames = 3;
    int32_t leap_accumulate_spread = 1000;     // us
    int32_t magic_leap_time_delay = 10000;     // us
    int32_t magic_leap_time_delay_mls = 10000; // us
    float leap_global_scaler = 1.0f;
    float magic_leap_scale_factor = 10.0f;
    float leap_arm_local_scaler = 0.019f;
    float leap_palm_local_scaler = 0.011f;
    float leap_bone_local_scaler = 0.05f;
    float magic_wrist_offset = -65.0f;
    float magic_arm_offset = -140.0f;
    float magic_arm_forward_offset = -170.0f;
    float leap_binary_threshold = 0.3f;
    bool leap_threshold_flag = false;
    double whole = 0.0;
    glm::mat4 global_scale_right = glm::mat4(1.0f);
    glm::mat4 global_scale_left = glm::mat4(1.0f);
    bool ready_to_collect = false;
    bool use_coaxial_calib = false;
    int calibration_state = static_cast<int>(CalibrationStateMachine::COLLECT);
    int checkerboard_width = 10;
    int checkerboard_height = 7;
    int leap_collection_setting = static_cast<int>(LeapCollectionSettings::AUTO_FINGER);
    int leap_mark_setting = static_cast<int>(LeapMarkSettings::STREAM);
    bool leap_calib_use_ransac = false;
    int mark_bone_index = 17;
    int leap_calibration_mark_state = 0;
    int use_leap_calib_results = static_cast<int>(LeapCalibrationSettings::MANUAL);
    int operation_mode = static_cast<int>(OperationMode::NORMAL);
    int n_points_leap_calib = 2000;
    int n_points_cam_calib = 30;
    float lastX = proj_width / 2.0f;
    float lastY = proj_height / 2.0f;
    bool firstMouse = true;
    bool dragging = false;
    int dragging_vert = 0;
    int closest_vert = 0;
    float min_dist = 100000.0f;
    int pnp_iters = 500;
    float pnp_rep_error = 2.0f;
    float pnp_confidence = 0.95f;
    bool showInliersOnly = true;
    bool showReprojections = false;
    bool showTestPoints = false;
    bool calibrationSuccess = false;
    std::vector<glm::vec2> screen_verts = {{-1.0f, 1.0f},
                                           {-1.0f, -1.0f},
                                           {1.0f, -1.0f},
                                           {1.0f, 1.0f}};
    std::vector<glm::vec2> cur_screen_verts = {{-1.0f, 1.0f},
                                               {-1.0f, -1.0f},
                                               {1.0f, -1.0f},
                                               {1.0f, 1.0f}};
    glm::vec2 cur_screen_vert = {0.0f, 0.0f};
    std::vector<glm::vec3> screen_verts_color_red = {{1.0f, 0.0f, 0.0f}};
    std::vector<glm::vec3> screen_verts_color_green = {{0.0f, 1.0f, 0.0f}};
    std::vector<glm::vec3> screen_verts_color_blue = {{0.0f, 0.0f, 1.0f}};
    std::vector<glm::vec3> screen_verts_color_magenta = {{1.0f, 0.0f, 1.0f}};
    std::vector<glm::vec3> screen_verts_color_cyan = {{0.0f, 1.0f, 1.0f}};
    std::vector<glm::vec3> screen_verts_color_white = {{1.0f, 1.0f, 1.0f}};
    std::vector<glm::vec3> near_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, -1.0f},
                                                                  {-1.0f, -1.0f, -1.0f},
                                                                  {1.0f, -1.0f, -1.0f},
                                                                  {1.0f, 1.0f, -1.0f}};
    std::vector<glm::vec3> mid_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, 0.7f},
                                                                 {-1.0f, -1.0f, 0.7f},
                                                                 {1.0f, -1.0f, 0.7f},
                                                                 {1.0f, 1.0f, 0.7f}};
    std::vector<glm::vec3> far_frustrum = std::vector<glm::vec3>{{-1.0f, 1.0f, 1.0f},
                                                                 {-1.0f, -1.0f, 1.0f},
                                                                 {1.0f, -1.0f, 1.0f},
                                                                 {1.0f, 1.0f, 1.0f}};
    std::vector<glm::vec3> frustumCornerVertices = std::vector<glm::vec3>{
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
    // camera controls
    unsigned int n_cam_channels = cam_color_mode ? 4 : 1;
    unsigned int cam_buffer_format = cam_color_mode ? GL_RGBA : GL_RED;
    float exposure = 1850.0f; // 1850.0f;
    int dst_width = cam_space ? cam_width : proj_width;
    int dst_height = cam_space ? cam_height : proj_height;

    // GL controls
    std::string inputBakeFile = "../../resource/images/butterfly.png";
    std::string bakeFileLeft = "../../resource/baked_textures/baked_left.png";
    std::string bakeFileRight = "../../resource/baked_textures/baked_right.png";
    std::string userTextureFile = "../../resource/images/uv.png";
    std::string sd_prompt = "A natural skinned human hand with a colorful dragon tattoo, photorealistic skin";
    std::vector<std::string> texturePaths{
        "../../resource",
        "../../resource/images",
        "../../resource/baked_textures",
        "../../resource/pbr/wood",
    };
    std::string curSelectedTexture = "uv";
    std::string curSelectedPTexture = "uv";
    std::string userStudySelectedTexture = "uv";
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    bool skeleton_as_gizmos = true;
    // mls controls
    const int grid_x_point_count = 21;
    const int grid_y_point_count = 21;
    const float grid_x_spacing = 2.0f / static_cast<float>(grid_x_point_count - 1);
    const float grid_y_spacing = 2.0f / static_cast<float>(grid_y_point_count - 1);
    float mls_alpha = 0.5f; // emperically best: 0.8f for rigid, 0.5f for affine
    std::vector<int> leap_selection_vector{1, 5, 11, 19, 27, 35, 9, 17, 25, 33, 41, 7, 15, 23, 31, 39};
    std::vector<int> mp_selection_vector{0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20, 3, 7, 11, 15, 19};
    bool mls_succeed = false;
    bool mls_succeeded_this_frame = false;
    int mls_succeed_counter = 0;
    int mls_every = 1;
    int mls_n_latency_frames = 0;
    bool mls_blocking = false;
    std::vector<float> mls_succeed_counters;
    bool mls_running = false;
    bool mls_probe_recent_leap = false;
    bool use_mls = true;
    bool postprocess_blur = false;
    int mls_mode = static_cast<int>(MLSMode::CONTROL_POINTS1);
    bool show_landmarks = false;
    bool show_mls_grid = false;
    int mls_cp_smooth_window = 0;
    int mls_grid_smooth_window = 0;
    bool use_mp_kalman = false;
    float prev_mls_time = 0.0f;
    bool mls_forecast = false;
    bool mls_global_forecast = false;
    float mls_grid_shader_threshold = 1.0f;
    glm::vec2 mls_shift = glm::vec2(0.0f, 0.0f);
    glm::vec2 prev_com = glm::vec2(0.0f, 0.0f);
    float kalman_process_noise = 0.01f;
    float kalman_measurement_noise = 0.0001f;
    float mls_depth_threshold = 30.0f;
    bool mls_depth_test = true;
    float kalman_lookahead = 17.0f;
    int deformation_mode = static_cast<int>(DeformationMode::RIGID);
    // of controls
    bool use_of = false;
    bool show_of = false;
    int of_mode = static_cast<int>(OFMode::FB_GPU);
    uint64_t totalFrameCountOF = 0;
    int of_resize_factor = 2;
    int of_resize_factor_exp = 1;
    cv::Size of_downsize = cv::Size(cam_width / of_resize_factor, cam_height / of_resize_factor);
    int of_roi = 10;
    std::vector<cv::Mat> of_debug;
};
#endif // ENGINE_STATE_H