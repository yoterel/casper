#include "post_process.h"
#include "stb_image_write.h"
#include "GLMhelpers.h"
#include <opencv2/opencv.hpp>
#include "helpers.h"
#include "timer.h"

PostProcess::PostProcess(unsigned int srcWidth, unsigned int srcHeight,
                         unsigned int dstWidth, unsigned int dstHeight) : m_srcWidth(srcWidth),
                                                                          m_srcHeight(srcHeight),
                                                                          m_dstWidth(dstWidth),
                                                                          m_dstHeight(dstHeight),
                                                                          m_quad(0.0f, false)
{
    m_num_texels = m_srcWidth * m_srcHeight;
    m_num_values = m_num_texels * 4;
    m_size_tex_data = sizeof(GLubyte) * m_num_values;
}

glm::mat4 PostProcess::findHomography(std::vector<glm::vec2> screen_verts)
{
    std::vector<glm::vec2> orig_screen_verts = {{-1.0f, 1.0f},
                                                {-1.0f, -1.0f},
                                                {1.0f, -1.0f},
                                                {1.0f, 1.0f}};
    std::vector<cv::Point2f> origpts, newpts;
    for (int i = 0; i < 4; ++i)
    {
        origpts.push_back(cv::Point2f(orig_screen_verts[i].x, orig_screen_verts[i].y));
        newpts.push_back(cv::Point2f(screen_verts[i].x, screen_verts[i].y));
    }
    cv::Mat1f hom = cv::getPerspectiveTransform(origpts, newpts);
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
        cv::Vec4f cord = cv::Vec4f(orig_screen_verts[i].x, orig_screen_verts[i].y, 0.0f, 1.0f);
        cv::Mat tmp = perspective * cv::Mat(cord);
        screen_verts[i].x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);
        screen_verts[i].y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);
    }
    // cv::Mat hom4x4 = cv::Mat::eye(4, 4, CV_32FC1);
    // hom.copyTo(hom4x4(cv::Rect(0, 0, 3, 3)));
    glm::mat4 projection;
    GLMHelpers::CV2GLM(perspective, &projection);
    return projection;
}

cv::Mat PostProcess::findFingers(cv::Mat gray, float threshold,
                                 std::vector<cv::Point> &fingers,
                                 std::vector<cv::Point> &valleys)
{
    // see: https://github.com/PierfrancescoSoffritti/handy
#define LIMIT_ANGLE_SUP 60
#define LIMIT_ANGLE_INF 5
#define BOUNDING_RECT_FINGER_SIZE_SCALING 0.3
#define BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING 0.05
    cv::Scalar color_white = cv::Scalar(255, 255, 255);
    cv::Mat contours_image = cv::Mat::zeros(gray.size(), CV_8UC1);
    cv::Mat flipped;
    cv::flip(gray, flipped, 1);
    cv::Mat thr;
    cv::threshold(flipped, thr, static_cast<int>(threshold * 255), 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::erode(thr, thr, cv::Mat());
    cv::dilate(thr, thr, cv::Mat());
    cv::findContours(thr, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.size() <= 0)
        return flipped;
    int biggest_contour_index = -1;
    double biggest_area = 0.0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i], false);
        if (area > biggest_area)
        {
            biggest_area = area;
            biggest_contour_index = i;
        }
    }
    if (biggest_contour_index < 0)
        return flipped;
    std::vector<cv::Point> hull_points;
    std::vector<int> hull_ints;
    // for drawing the convex hull and for finding the bounding rectangle
    cv::convexHull(cv::Mat(contours[biggest_contour_index]), hull_points, true, true);
    // for finding the defects
    cv::convexHull(cv::Mat(contours[biggest_contour_index]), hull_ints, true, false);
    std::vector<cv::Vec4i> defects;
    if (hull_ints.size() > 3)
        cv::convexityDefects(cv::Mat(contours[biggest_contour_index]), hull_ints, defects);
    else
        return flipped;

    // we bound the convex hull
    cv::Rect bounding_rectangle = cv::boundingRect(cv::Mat(hull_points));

    // we find the center of the bounding rectangle, this should approximately also be the center of the hand
    cv::Point center_bounding_rect(
        (bounding_rectangle.tl().x + bounding_rectangle.br().x) / 2,
        (bounding_rectangle.tl().y + bounding_rectangle.br().y) / 2);

    // we separate the defects keeping only the ones of intrest
    std::vector<cv::Point> start_points;
    std::vector<cv::Point> far_points;

    for (int i = 0; i < defects.size(); i++)
    {
        start_points.push_back(contours[biggest_contour_index][defects[i].val[0]]);

        // filtering the far point based on the distance from the center of the bounding rectangle
        if (findPointsDistance(contours[biggest_contour_index][defects[i].val[2]], center_bounding_rect) < bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING)
            far_points.push_back(contours[biggest_contour_index][defects[i].val[2]]);
    }

    // we compact them on their medians
    std::vector<cv::Point> filtered_start_points = compactOnNeighborhoodMedian(start_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);
    std::vector<cv::Point> filtered_far_points = compactOnNeighborhoodMedian(far_points, bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING);

    // now we try to find the fingers
    std::vector<cv::Point> filtered_finger_points;

    if (filtered_far_points.size() > 1)
    {
        std::vector<cv::Point> finger_points;

        for (int i = 0; i < filtered_start_points.size(); i++)
        {
            std::vector<cv::Point> closest_points = findClosestOnX(filtered_far_points, filtered_start_points[i]);

            if (isFinger(closest_points[0], filtered_start_points[i], closest_points[1], LIMIT_ANGLE_INF, LIMIT_ANGLE_SUP, center_bounding_rect, bounding_rectangle.height * BOUNDING_RECT_FINGER_SIZE_SCALING))
                finger_points.push_back(filtered_start_points[i]);
        }

        if (finger_points.size() > 0)
        {

            // we have at most five fingers usually :)
            while (finger_points.size() > 5)
                finger_points.pop_back();

            // filter out the points too close to each other
            for (int i = 0; i < finger_points.size() - 1; i++)
            {
                if (findPointsDistanceOnX(finger_points[i], finger_points[i + 1]) > bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 1.5)
                    filtered_finger_points.push_back(finger_points[i]);
            }

            if (finger_points.size() > 2)
            {
                if (findPointsDistanceOnX(finger_points[0], finger_points[finger_points.size() - 1]) > bounding_rectangle.height * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 1.5)
                    filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
            }
            else
                filtered_finger_points.push_back(finger_points[finger_points.size() - 1]);
        }
    }

    // we draw what found on the returned image
    // cv::drawContours(contours_image, contours, biggest_contour_index, color_green, 2, 8, hierarchy);
    // cv::polylines(contours_image, hull_points, true, color_blue);
    // cv::rectangle(contours_image, bounding_rectangle.tl(), bounding_rectangle.br(), color_red, 2, 8, 0);
    // cv::circle(contours_image, center_bounding_rect, 5, color_purple, 2, 8);
    // drawVectorPoints(contours_image, filtered_start_points, color_blue, true);
    // drawVectorPoints(contours_image, filtered_far_points, color_red, true);
    // drawVectorPoints(contours_image, filtered_finger_points, color_yellow, false);
    // cv::putText(contours_image, to_string(filtered_finger_points.size()), center_bounding_rect, FONT_HERSHEY_PLAIN, 3, color_purple);

    // and on the starting frame
    // cv::drawContours(flipped, contours, biggest_contour_index, color_white, 2, 8, hierarchy);
    // cv::circle(flipped, center_bounding_rect, 5, color_white, 2, 8);
    // drawVectorPoints(flipped, filtered_finger_points, color_white, false);
    // cv::putText(frame, to_string(filtered_finger_points.size()), center_bounding_rect, FONT_HERSHEY_PLAIN, 3, color_purple);
    fingers = filtered_finger_points;
    valleys = filtered_far_points;
    return flipped;
}

std::vector<cv::Point> PostProcess::compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance)
{
    std::vector<cv::Point> median_points;

    if (points.size() == 0)
        return median_points;

    if (max_neighbor_distance <= 0)
        return median_points;

    // we start with the first point
    cv::Point reference = points[0];
    cv::Point median = points[0];

    for (int i = 1; i < points.size(); i++)
    {
        if (findPointsDistance(reference, points[i]) > max_neighbor_distance)
        {

            // the point is not in range, we save the median
            median_points.push_back(median);

            // we swap the reference
            reference = points[i];
            median = points[i];
        }
        else
            median = (points[i] + median) / 2;
    }

    // last median
    median_points.push_back(median);

    return median_points;
}

double PostProcess::findPointsDistance(cv::Point a, cv::Point b)
{
    cv::Point difference = a - b;
    return sqrt(difference.ddot(difference));
}

double PostProcess::findPointsDistanceOnX(cv::Point a, cv::Point b)
{
    double to_return = 0.0;

    if (a.x > b.x)
        to_return = a.x - b.x;
    else
        to_return = b.x - a.x;

    return to_return;
}

std::vector<cv::Point> PostProcess::findClosestOnX(std::vector<cv::Point> points, cv::Point pivot)
{
    std::vector<cv::Point> to_return(2);

    if (points.size() == 0)
        return to_return;

    double distance_x_1 = DBL_MAX;
    double distance_1 = DBL_MAX;
    double distance_x_2 = DBL_MAX;
    double distance_2 = DBL_MAX;
    int index_found = 0;

    for (int i = 0; i < points.size(); i++)
    {
        double distance_x = findPointsDistanceOnX(pivot, points[i]);
        double distance = findPointsDistance(pivot, points[i]);

        if (distance_x < distance_x_1 && distance_x != 0 && distance <= distance_1)
        {
            distance_x_1 = distance_x;
            distance_1 = distance;
            index_found = i;
        }
    }

    to_return[0] = points[index_found];

    for (int i = 0; i < points.size(); i++)
    {
        double distance_x = findPointsDistanceOnX(pivot, points[i]);
        double distance = findPointsDistance(pivot, points[i]);

        if (distance_x < distance_x_2 && distance_x != 0 && distance <= distance_2 && distance_x != distance_x_1)
        {
            distance_x_2 = distance_x;
            distance_2 = distance;
            index_found = i;
        }
    }

    to_return[1] = points[index_found];

    return to_return;
}

double PostProcess::findAngle(cv::Point a, cv::Point b, cv::Point c)
{
    double ab = findPointsDistance(a, b);
    double bc = findPointsDistance(b, c);
    double ac = findPointsDistance(a, c);
    return acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc)) * 180 / CV_PI;
}

bool PostProcess::isFinger(cv::Point a, cv::Point b, cv::Point c, double limit_angle_inf, double limit_angle_sup, cv::Point palm_center, double min_distance_from_palm)
{
    double angle = findAngle(a, b, c);
    if (angle > limit_angle_sup || angle < limit_angle_inf)
        return false;

    // the finger point sohould not be under the two far points
    int delta_y_1 = b.y - a.y;
    int delta_y_2 = b.y - c.y;
    if (delta_y_1 > 0 && delta_y_2 > 0)
        return false;

    // the two far points should not be both under the center of the hand
    int delta_y_3 = palm_center.y - a.y;
    int delta_y_4 = palm_center.y - c.y;
    if (delta_y_3 < 0 && delta_y_4 < 0)
        return false;

    double distance_from_palm = findPointsDistance(b, palm_center);
    if (distance_from_palm < min_distance_from_palm)
        return false;

    // this should be the case when no fingers are up
    double distance_from_palm_far_1 = findPointsDistance(a, palm_center);
    double distance_from_palm_far_2 = findPointsDistance(c, palm_center);
    if (distance_from_palm_far_1 < min_distance_from_palm / 4 || distance_from_palm_far_2 < min_distance_from_palm / 4)
        return false;

    return true;
}

void PostProcess::drawVectorPoints(cv::Mat image, std::vector<cv::Point> points, cv::Scalar color, bool with_numbers)
{
    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(image, points[i], 5, color, 2, 8);
        // if (with_numbers)
        //     cv::putText(image, to_string(i), points[i], FONT_HERSHEY_PLAIN, 3, color);
    }
}

void PostProcess::bake(Shader &uvShader, unsigned int textureToBake, unsigned int TextureUV, const std::string &filepath)
{
    // render models uvcoordinates into an offscreen texture
    FBO fbo(1024, 1024);
    fbo.bind();
    uvShader.use();
    uvShader.setInt("toBake", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureToBake);
    uvShader.setInt("uvTexture", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, TextureUV);
    m_quad.render();
    fbo.unbind();
    fbo.saveColorToFile(filepath.c_str());
    // model.Render(uvShader, 0, false);
    // render screen sized quad and look up texture coordinates from the baked texture
}

void PostProcess::mask(Shader *mask_shader, unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo, const float threshold)
{
    // bind fbo
    target_fbo->bind();
    // bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderedSceneTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, camTexture);
    // render
    mask_shader->use();
    mask_shader->setInt("src", 0);
    mask_shader->setInt("mask", 1);
    mask_shader->setFloat("threshold", threshold);
    mask_shader->setBool("maskIsGray", true);
    mask_shader->setBool("flipVer", false);
    mask_shader->setBool("flipMaskVer", true);
    mask_shader->setBool("flipMaskHor", true);
    m_quad.render(false, false, true);
    // unbind fbo
    target_fbo->unbind();
}

void PostProcess::jump_flood_uv(Shader &jfaInit, Shader &jfa, Shader &uv_NN_shader,
                                unsigned int uvTexture, unsigned int uvUnwrappedTexture, unsigned int camTexture, FBO *target_fbo,
                                const float threshold, const float distance_threshold, const float seam_threshold)
{
    // jfaInit determines seeds for jump flood (everywhere there is information will be a seed)
    // distances / nearest neighbors / etc will be calculated from these seeds (but not for the seeds themselves)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, uvTexture);
    // the pingpong framebuffers m_FBO[0] and m_FBO[1] will be used to store the result of the jump flood
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[0]);
    jfaInit.use();
    jfaInit.setInt("src", 0);
    jfaInit.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_quad.render(false, false, true);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[1]);
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // jump flood
    int numPasses = 11; // generally log(max(width, height)) + 1 is a good number of passes, paper recommends + 2 for robustness
    // see "Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform"
    for (int i = 0; i < numPasses; i++)
    {
        jfa.use();
        jfa.setInt("src", 0);
        jfa.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
        jfa.setInt("pass", i);
        jfa.setInt("numPasses", numPasses);
        m_quad.render();
        // bind texture from current iter, bind next fbo
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[(i + 1) % 2]);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i % 2]);
        // glViewport(0, 0, m_srcWidth, m_srcHeight);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    if (target_fbo != NULL)
    {
        target_fbo->bind();
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, m_dstWidth, m_dstHeight);
    }
    // flood fill result will be contained in the second ping pong buffer texture
    // mask result using the camera texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, uvTexture); // the uvs of the mesh in camspace
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[1]); // the jumpflood nearest neighbor result
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, camTexture); // the camera image, used as a mask
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, uvUnwrappedTexture); // the unwrapped texture of the mesh, that can be used with the uvs to get the intended color
    uv_NN_shader.use();
    uv_NN_shader.setInt("uv", 0);
    uv_NN_shader.setInt("jfa", 1);
    uv_NN_shader.setInt("mask", 2);
    uv_NN_shader.setInt("unwrapped", 3);
    uv_NN_shader.setFloat("threshold", threshold);
    uv_NN_shader.setFloat("distThreshold", distance_threshold);
    uv_NN_shader.setFloat("seamThreshold", seam_threshold);
    uv_NN_shader.setBool("flipVer", false);
    uv_NN_shader.setBool("maskIsGray", true);
    uv_NN_shader.setBool("flipMaskVer", true);
    uv_NN_shader.setBool("flipMaskHor", true);
    uv_NN_shader.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    m_quad.render();
    if (target_fbo != NULL)
    {
        target_fbo->unbind();
    }
}

void PostProcess::jump_flood(Shader &jfaInit, Shader &jfa, Shader &NN_shader,
                             unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo,
                             const float threshold, const float distance_threshold)
{
    // init jump flood seeds
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderedSceneTexture);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[0]);
    jfaInit.use();
    jfaInit.setInt("src", 0);
    jfaInit.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_quad.render(false, false, true);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[1]);
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // jump flood
    int numPasses = 11;
    for (int i = 0; i < numPasses; i++)
    {
        jfa.use();
        jfa.setInt("src", 0);
        jfa.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
        jfa.setInt("pass", i);
        jfa.setInt("numPasses", numPasses);
        m_quad.render();
        // bind texture from current iter, bind next fbo
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[(i + 1) % 2]);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i % 2]);
        // glViewport(0, 0, m_srcWidth, m_srcHeight);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    if (target_fbo != NULL)
    {
        target_fbo->bind();
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, m_dstWidth, m_dstHeight);
    }
    // flood fill result will be contained in the second ping pong buffer texture
    // mask result using the camera texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderedSceneTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, camTexture);
    // glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[0]);
    // unbind FBO and return viewport to default
    NN_shader.use();
    NN_shader.setInt("src", 0);
    NN_shader.setInt("jfa", 1);
    NN_shader.setInt("mask", 2);
    NN_shader.setFloat("threshold", threshold);
    NN_shader.setFloat("distThreshold", distance_threshold);
    NN_shader.setBool("flipVer", false);
    NN_shader.setBool("maskIsGray", true);
    NN_shader.setBool("flipMaskVer", true);
    NN_shader.setBool("flipMaskHor", true);
    NN_shader.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    m_quad.render();
    // result will be contained in the first ping pong buffer texture
    if (target_fbo != NULL)
    {
        target_fbo->unbind();
    }
}

void PostProcess::saveColorToFile(std::string filepath, unsigned int fbo_id)
{
    unsigned int nrChannels = 4;
    std::vector<char> buffer(m_num_values);
    GLsizei stride = nrChannels * m_dstWidth;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_dstWidth, m_dstHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath.c_str(), m_dstWidth, m_dstHeight, 4, buffer.data(), stride);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PostProcess::initGLBuffers()
{
    m_quad.init();
    // todo: replace with FBO class
    // create fbos
    glGenTextures(2, m_pingpong_textures);
    glGenRenderbuffers(2, m_depth_buffer);
    glGenFramebuffers(2, m_FBO);
    for (int i = 0; i < 2; i++)
    {
        // glCheckError();
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[i]);
        // glCheckError();
        //  set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
        // glCheckError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // glCheckError();
        //  set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        // glCheckError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // glCheckError();
        //  #ifdef USE_TEXSUBIMAGE2D
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_dstWidth, m_dstHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        // glCheckError();
        glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer[i]);
        // glCheckError();
        //  allocate storage
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_dstWidth, m_dstHeight);
        // glCheckError();
        //  clean up
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        // glCheckError();
        // glCheckError();
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i]);
        // glCheckError();
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pingpong_textures[i], 0);
        // glCheckError();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer[i]);
        // glCheckError();
        GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "ERROR: Incomplete framebuffer status." << std::endl;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    // glCheckError();
}