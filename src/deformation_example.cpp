#include "timer.h"
#include "grid.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "moving_least_squares.h"
#include "shader.h"
#include "point_cloud.h"
#include "helpers.h"
#include "texture.h"
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// globals

std::vector<cv::Point2f> ControlPointsP = {cv::Point2f(-0.95, -0.95), cv::Point2f(-0.95, 0.95), cv::Point2f(0.95, 0.95),
                                           cv::Point2f(0.95, -0.95), cv::Point2f(0.51, 0.0), cv::Point2f(-0.51, -0.0)};
std::vector<cv::Point2f> ControlPointsQ = {cv::Point2f(-0.95, -0.95), cv::Point2f(-0.95, 0.95), cv::Point2f(0.95, 0.95),
                                           cv::Point2f(0.95, -0.95), cv::Point2f(0.6, 0.0), cv::Point2f(-0.6, -0.0)};
const int grid_x_point_count = 21;
const int grid_y_point_count = 21;
const float grid_x_spacing = 2.0f / static_cast<float>(grid_x_point_count - 1);
const float grid_y_spacing = 2.0f / static_cast<float>(grid_y_point_count - 1);
const unsigned int SCR_WIDTH = 720;
const unsigned int SCR_HEIGHT = 540;
bool objChanged = true;
int deformation_mode = static_cast<int>(DeformationMode::RIGID);
float mls_alpha = 1.0;
float mls_grid_shader_threshold = 1.0;
int mls_grid_smooth_window = 1;
bool dragging = false;
int dragging_vert = 0;
int closest_vert = 0;
float min_dist = 100000.0f;
// forward declarations
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
cv::Mat computeGridDeformation(std::vector<cv::Point2f> &P,
                               std::vector<cv::Point2f> &Q,
                               int deformation_mode, float alpha,
                               Grid &grid);

int main(int argc, char *argv[])
{
    // glfw/glad boiler plate
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "MLS Deformation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    // ---------------------------------------
    Grid deformationGrid(grid_x_point_count, grid_y_point_count, grid_x_spacing, grid_y_spacing);
    deformationGrid.initGLBuffers();
    Shader *gridShader = new Shader("../../src/shaders/grid_texture.vs", "../../src/shaders/grid_texture.fs");
    Shader *vColorShader = new Shader("../../src/shaders/color_by_vertex.vs", "../../src/shaders/color_by_vertex.fs");
    Texture *mytexture = new Texture("../../resource/images/birds.jpg");
    mytexture->init_from_file();

    // -------------------------
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        if (objChanged)
        {
            if (deformation_mode == static_cast<int>(DeformationMode::NONE))
            {
                deformationGrid.constructGrid();
            }
            else
            {
                cv::Mat fv = computeGridDeformation(ControlPointsP, ControlPointsQ, deformation_mode, mls_alpha, deformationGrid);
                deformationGrid.constructDeformedGridSmooth(fv, mls_grid_smooth_window);
            }
            deformationGrid.updateGLBuffers();
            objChanged = false;
        }
        processInput(window);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mytexture->bind();
        gridShader->use();

        glDisable(GL_CULL_FACE); // todo: why is this necessary? flip grid triangles...
        glDisable(GL_DEPTH_TEST);
        gridShader->setInt("src", 0);
        gridShader->setFloat("threshold", mls_grid_shader_threshold);
        gridShader->setBool("flipVer", false);
        // gridShader.setVec2("shift", es.mls_shift);
        deformationGrid.render();
        std::vector<glm::vec3> screen_verts_color_red = {{1.0f, 0.0f, 0.0f}};
        std::vector<glm::vec3> screen_verts_color_green = {{0.0f, 1.0f, 0.0f}};
        std::vector<glm::vec2> ControlPointsP_glm = Helpers::cv2glm(ControlPointsP);
        std::vector<glm::vec2> ControlPointsQ_glm = Helpers::cv2glm(ControlPointsQ);
        PointCloud cloud_src(ControlPointsP_glm, screen_verts_color_red);
        PointCloud cloud_dst(ControlPointsQ_glm, screen_verts_color_green);
        vColorShader->use();
        vColorShader->setMat4("mvp", glm::mat4(1.0f));
        cloud_src.render();
        cloud_dst.render();

        // glDisable(GL_CULL_FACE);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        deformation_mode = static_cast<int>(DeformationMode::RIGID);
        objChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        deformation_mode = static_cast<int>(DeformationMode::SIMILARITY);
        objChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        deformation_mode = static_cast<int>(DeformationMode::AFFINE);
        objChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        deformation_mode = static_cast<int>(DeformationMode::NONE);
        objChanged = true;
    }
}
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    // glm::vec2 mouse_pos = glm::vec2((2.0f * xpos / proj_width) - 1.0f, -1.0f * ((2.0f * ypos / proj_height) - 1.0f));
    glm::vec2 mouse_pos = Helpers::ScreenToNDC(glm::vec2(xpos, ypos), SCR_WIDTH, SCR_HEIGHT, true);
    if (dragging)
    {
        ControlPointsQ[dragging_vert] = cv::Point2f(mouse_pos.x, mouse_pos.y);
        objChanged = true;
    }
    else
    {
        float cur_min_dist = 100.0f;
        for (int i = 0; i < ControlPointsQ.size(); i++)
        {
            glm::vec2 v = glm::vec2(ControlPointsQ[i].x, ControlPointsQ[i].y);
            float dist = glm::distance(v, mouse_pos);
            if (dist < cur_min_dist)
            {
                cur_min_dist = dist;
                closest_vert = i;
            }
        }
        min_dist = cur_min_dist;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    // if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    // {
    //     rmb_pressed = true;
    // }
    // if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
    // {
    //     if (rmb_pressed)
    //         es.activateGUI = !es.activateGUI;
    //     es.rmb_pressed = false;
    // }
    // if (es.activateGUI) // dont allow moving cameras when GUI active
    //     return;

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
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

cv::Mat computeGridDeformation(std::vector<cv::Point2f> &P,
                               std::vector<cv::Point2f> &Q,
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