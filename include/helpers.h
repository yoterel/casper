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
    static std::vector<glm::vec2> ScreenToNDC(const std::vector<glm::vec2> &pixels, int width, int height, bool flip_y);
    static glm::vec2 NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y = false);
    static std::vector<glm::vec2> NDCtoScreen(const std::vector<glm::vec2> &NDCs, int width, int height, bool flip_y);
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
    static std::vector<cv::Point2f> glm2opencv(std::vector<glm::vec2> glm_vec);
    static std::vector<glm::vec2> opencv2glm(std::vector<cv::Point2f> vec);
    static std::vector<glm::vec2> opencv2glm(std::vector<cv::Point> vec);
    static std::vector<glm::vec3> opencv2glm(std::vector<cv::Point3f> vec);
    static std::vector<glm::vec2> project_points(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static std::vector<glm::vec3> project_points_w_depth(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection);
    static void setupGizmoBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupFrustrumBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeTexturedBuffers(unsigned int &VAO, unsigned int &VBO1, unsigned int &VBO2);
    static void setupSkeletonBuffers(unsigned int &VAO, unsigned int &VBO);
    static float MSE(const std::vector<glm::vec2> &a, const std::vector<glm::vec2> &b /*, std::vector<float> &mse*/);
    static std::vector<glm::vec2> accumulate(const std::vector<std::vector<glm::vec2>> &a, bool normalize = true);
    static std::vector<glm::vec3> accumulate(const std::vector<std::vector<glm::vec3>> &a, bool normalize = true);
    static glm::mat4 interpolate(const glm::mat4 &_mat1, const glm::mat4 &_mat2, float _time, bool prescale = false);

private:
    Helpers();
};

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
    SKELETON = 2,
};
enum class PostProcessMode
{
    NONE = 0,
    CAM_FEED = 1,
    OVERLAY = 2,
    JUMP_FLOOD = 3,
    JUMP_FLOOD_UV = 4,
    ICP = 5,
};
enum class LeapCalibrationSettings
{
    AUTO = 0,
    MANUAL = 1,
};
enum class CalibrationStateMachine
{
    COLLECT = 0,
    CALIBRATE = 1,
    SHOW = 2,
    MARK = 3,
};
enum class LeapCollectionSettings
{
    MANUAL_RAW = 0,
    MANUAL_FINGER = 1,
    AUTO_RAW = 2,
    AUTO_FINGER = 3,
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
    NORMAL = 0,
    VIDEO = 1,
    CAMERA = 2,
    COAXIAL = 3,
    LEAP = 4,
};

#endif // HELPERS_H