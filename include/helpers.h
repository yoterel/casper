#ifndef HELPERS_H
#define HELPERS_H
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include <opencv2/opencv.hpp>

class Helpers
{
public:
    static std::vector<glm::vec2> vec3to2(std::vector<glm::vec3> vec);
    static glm::vec2 ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y = false);
    static std::vector<glm::vec2> ScreenToNDC(const std::vector<glm::vec2> &pixels, int width, int height, bool flip_y = false);
    static cv::Point2f ScreenToNDC(const cv::Point2f &pixel, int width, int height, bool flip_y = false);
    static std::vector<cv::Point2f> ScreenToNDC(const std::vector<cv::Point2f> &pixels, int width, int height, bool flip_y = false);
    static glm::vec2 NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y = false);
    static cv::Point2f NDCtoScreen(const cv::Point2f &NDC, int width, int height, bool flip_y = false);
    static std::vector<glm::vec2> NDCtoScreen(const std::vector<glm::vec2> &NDCs, int width, int height, bool flip_y = false);
    static std::vector<cv::Point2f> NDCtoScreen(const std::vector<cv::Point2f> &NDCs, int width, int height, bool flip_y = false);
    static void UV2NDC(std::vector<glm::vec2> &uv);
    static void saveTexture(std::string filepath,
                            unsigned int texture,
                            unsigned int width,
                            unsigned int height,
                            bool flipVertically = false,
                            bool threshold = false);
    static std::vector<float> flatten_glm(std::vector<glm::vec2> vec);
    static std::vector<float> flatten_glm(std::vector<glm::vec3> vec);
    static std::vector<double> flatten_cv(std::vector<cv::Point> vec);
    static std::vector<glm::vec2> flatten_2dgrid(cv::Mat grid);
    static cv::Point2f glm2cv(glm::vec2 glm_vec);
    static std::vector<cv::Point2f> glm2cv(std::vector<glm::vec2> glm_vec);
    static std::vector<glm::vec2> cv2glm(std::vector<cv::Point2f> vec);
    static std::vector<glm::vec2> cv2glm(std::vector<cv::Point> vec);
    static std::vector<glm::vec3> cv2glm(std::vector<cv::Point3f> vec);
    static glm::vec2 project_point(glm::vec3 point, glm::mat4 mvp);
    static glm::vec2 project_point(glm::vec3 point, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static std::vector<glm::vec2> project_points(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static glm::vec3 project_point_w_depth(glm::vec3 point, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static std::vector<glm::vec3> project_points_w_depth(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static void setupGizmoBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupFrustrumBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeBuffers(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO);
    static void setupCubeTexturedBuffers(unsigned int &VAO, unsigned int &VBO1, unsigned int &VBO2);
    static void setupSkeletonBuffers(unsigned int &VAO, unsigned int &VBO);
    static float MSE(const std::vector<glm::vec2> &a, const std::vector<glm::vec2> &b /*, std::vector<float> &mse*/);
    static std::vector<glm::vec2> accumulate(const std::vector<std::vector<glm::vec2>> &a, bool normalize = true);
    static std::vector<glm::vec3> accumulate(const std::vector<std::vector<glm::vec3>> &a, bool normalize = true);
    static std::vector<glm::mat4> accumulate(const std::vector<std::vector<glm::mat4>> &a, bool normalize = true);
    static float average(std::vector<float> &v);
    static glm::vec2 average(std::vector<glm::vec2> &v);
    static cv::Point2f average(std::vector<cv::Point2f> &v);
    static glm::mat4 interpolate(const glm::mat4 &_mat1, const glm::mat4 &_mat2, float _time, bool prescale = false, bool isRightHand = false);
    static bool isPalmFacingCamera(glm::mat4 palm_bone, glm::mat4 cam_view_transform);
    static void visualize_flow(cv::Mat flow, cv::Mat bg, std::string dst, float threshold = -1.0f);

private:
    Helpers();
};

enum class Hand
{
    LEFT = 0,
    RIGHT = 1,
};

enum class TextureMode
{
    ORIGINAL = 0,
    FROM_FILE = 1,
    PROJECTIVE = 2,
    BAKED = 3,
    CAMERA = 4,
    SHADER = 5,
    MULTI_PASS_SHADER = 6,
    DYNAMIC = 7,
};
enum class PromptMode
{
    MANUAL_PROMPT = 0,
    AUTO_PROMPT = 1,
    SELECTED = 2,
    RANDOM = 3,
};
enum class BakeMode
{
    SD = 0,
    FILE = 1,
    CAMERA = 2,
    POSE = 3,
    CONTROL_NET = 4,
};
enum class MaterialMode
{
    DIFFUSE = 0,
    GGX = 1,
    SKELETON = 2,
    PER_BONE_SCALAR = 3,
};
enum class LightMode
{
    AMBIENT = 0,
    POINT = 1,
    DIRECTIONAL = 2,
    PROJECTOR = 3,
    CUBEMAP = 4,
    ENVMAP = 5,
};
enum class PostProcessMode
{
    NONE = 0,
    CAM_FEED = 1,
    OVERLAY = 2,
    JUMP_FLOOD = 3,
    JUMP_FLOOD_UV = 4,
};
enum class LeapCalibrationSettings
{
    AUTO = 0,
    MANUAL = 1,
};
enum class CalibrationStateMachine
{
    COLLECT = 0,
    SOLVE = 1,
    SHOW = 2,
    MARK = 3,
};
enum class LeapCollectionSettings
{
    AUTO_RAW = 0,
    AUTO_FINGER = 1,
};
enum class LeapMarkSettings
{
    STREAM = 0,
    POINT_BY_POINT = 1,
    WHOLE_HAND = 2,
    ONE_BONE = 3,
};
enum class OperationMode
{
    SANDBOX = 0,
    USER_STUDY = 1,
    CAMERA = 2,
    COAXIAL = 3,
    LEAP = 4,
    GUESS_POSE_GAME = 5,
    GUESS_CHAR_GAME = 6,
    SIMULATION = 7,
    GUESS_ANIMAL_GAME = 8,
};
enum class DeformationMode
{
    RIGID = 0,
    SIMILARITY = 1,
    AFFINE = 2,
    NONE = 3,
};

enum class MLSMode
{
    CONTROL_POINTS1 = 0,
    CONTROL_POINTS2 = 1,
    GRID = 2,
};

enum class GameSessionType
{
    A = 0,
    B = 1,
};

enum class OFMode
{
    FB_CPU = 0,
    FB_GPU = 1,
    NV_GPU = 2,
    NAIVE_BLOB = 3,
};

enum class DynamicShaderMode
{
    SEA = 0,
    FIREWORKS = 1,
    CLOUDS = 2,
    FRACTAL = 3,
    NEON = 4,
};

template <typename T>
std::vector<float> linear_spacing(T start, T end, int num)
{
    std::vector<float> linspaced;
    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    float delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}

template <typename T>
std::vector<int32_t> integer_linear_spacing(T start, T end, int num)
{
    std::vector<int32_t> linspaced;
    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(static_cast<int32_t>(start));
        return linspaced;
    }

    float delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(static_cast<int32_t>(start + delta * i));
    }
    linspaced.push_back(static_cast<int32_t>(end));
    return linspaced;
}

#endif // HELPERS_H