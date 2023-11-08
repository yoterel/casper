#include <chrono>
#include <stdio.h>
#include <iostream>
#include "timer.h"
#include "opencv2/opencv.hpp"
#include <opencv2/video/tracking.hpp>
#include <filesystem>
#include <numeric>
#include <opencv2/opencv.hpp>
// #include "opencv2/cudaarithm.hpp"
// #include "opencv2/features2d.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
// #include "opencv2/core/cuda.hpp"
// #include "opencv2/cudaimgproc.hpp"

namespace fs = std::filesystem;
// using namespace std;
// using namespace cv;
// using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

inline bool isFlowCorrect(cv::Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
};
// this function is taken from opencv/samples/gpu/optical_flow.cpp
cv::Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

// this function is taken from opencv/samples/gpu/optical_flow.cpp
static void drawOpticalFlow(const cv::Mat_<float> &flowx, const cv::Mat_<float> &flowy, cv::Mat &dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                cv::Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = cv::max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            cv::Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

void alignImages(cv::Mat &im1, cv::Mat &im2, cv::Mat &im1Reg, cv::Mat &h)
{
    // Convert images to grayscale
    cv::Mat im1Gray, im2Gray;
    cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

    // Variables to store keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    orb->detectAndCompute(im1Gray, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(im2Gray, cv::Mat(), keypoints2, descriptors2);

    // Match features.
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, cv::Mat());

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Draw top matches
    cv::Mat imMatches;
    cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
    cv::imwrite("matches.jpg", imMatches);

    // Extract location of good matches
    std::vector<cv::Point2f> points1, points2;

    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Find homography
    h = findHomography(points1, points2, cv::RANSAC);

    // Use homography to warp image
    warpPerspective(im1, im1Reg, h, im2.size());
};

std::vector<cv::Mat> load_images(std::string path)
{
    std::vector<cv::Mat> images;
    std::vector<std::string> paths;
    for (const auto &entry : fs::directory_iterator(path))
        // std::cout << entry.path() << std::endl;
        paths.push_back(entry.path().string());
    for (int i = 2300; i < 2500; i++)
    {
        // std::format("../../resource/image1.png", debug_vec.x, debug_vec.y, debug_vec.z),
        cv::Mat im1 = cv::imread(paths[i]);
        images.push_back(im1);
    }
    std::cout << std::format("loaded {} images.", images.size()) << std::endl;
    return images;
};

cv::Mat warpImage(cv::Mat inputIm, cv::Mat homography)
{
    cv::Mat outputIm;
    cv::warpPerspective(inputIm, outputIm, homography, inputIm.size());
    return outputIm;
};

cv::Mat procFlow(cv::Mat flow, cv::Size size)
{
    cv::Mat magnitude, normalized_magnitude, angle;
    cv::Mat hsv[3], merged_hsv, hsv_8u, bgr;
    hsv[1] = cv::Mat::ones(size, CV_32F);
    cv::Mat flow_xy[2], flow_x, flow_y;
    split(flow, flow_xy);

    // get the result
    flow_x = flow_xy[0];
    flow_y = flow_xy[1];

    // convert from cartesian to polar coordinates
    cartToPolar(flow_x, flow_y, magnitude, angle, true);

    // normalize magnitude from 0 to 1
    normalize(magnitude, normalized_magnitude, 0.0, 1.0, cv::NORM_MINMAX);

    // get angle of optical flow
    angle *= ((1 / 360.0) * (180 / 255.0));

    // build hsv image
    hsv[0] = angle;
    hsv[2] = normalized_magnitude;
    cv::merge(hsv, 3, merged_hsv);

    // multiply each pixel value to 255
    merged_hsv.convertTo(hsv_8u, CV_8U, 255);

    // convert hsv to bgr
    cv::cvtColor(hsv_8u, bgr, cv::COLOR_HSV2BGR);
    return bgr;
};

int main(int argc, char **argv)
{
    int downscale_factor = 2;
    int stride = 1;
    int slow_tracker_every = 6;
    Timer t1, t2, t3, t4, t5, t6, t_cuda;
    // Read the images to be aligned
    std::string path = "C:/Users/sens/Desktop/augmented_hands/session2";
    std::vector<cv::Mat> images = load_images(path);
    std::vector<std::string> algorithms = {"OF_nv_GPU", "OF_fb_GPU", "OF_fb_CPU", "OF_sparse_CPU"}; // , "ORB", "CC", "OF_GPU", "FAST"
    // todo:
    // cornerSubPix, OF with parameter tuning, OF GPU, FastFeature
    // references:
    // https://github.com/opencv/opencv/blob/3.2.0/samples/python/lk_homography.py  demo for homography finding with sparse of in python
    // https://github.com/opencv/opencv/blob/3.2.0/samples/python/lk_track.py
    // https://github.com/opencv/opencv/blob/3.2.0/samples/cpp/lkdemo.cpp // demo for sparse of in c++, no homography finding
    // https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html  // demo for fast feature detection
    for (std::string alg : algorithms)
    {
        if (alg == "OF_nv_GPU")
        {
            cv::Size down_size = cv::Size(images[0].cols / downscale_factor, images[0].rows / downscale_factor);
            cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> nvof = cv::cuda::NvidiaOpticalFlow_2_0::create(down_size,
                                                                                                    cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_SLOW,
                                                                                                    cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
                                                                                                    cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_1,
                                                                                                    false,
                                                                                                    false);
            cv::Mat flowx, flowy, flowxy, floatFlow, image;
            cv::Mat im1, im1_gray, im1_gray_down;
            cv::Mat im2, im2_gray, im2_gray_down;
            im1 = images[0];
            cv::cvtColor(im1, im1_gray, cv::COLOR_BGR2GRAY);
            cv::resize(im1_gray, im1_gray_down, down_size);
            for (int i = 1; i < images.size() - 1; i++)
            {
                std::cout << '\r' << std::format("{:04d} / {}", i, images.size()) << std::flush;
                cv::Mat im2 = images[i];
                t1.start();
                cv::cvtColor(im2, im2_gray, cv::COLOR_BGR2GRAY);
                t1.stop();
                t2.start();
                cv::resize(im2_gray, im2_gray_down, down_size);
                t2.stop();
                t_cuda.start();
                nvof->calc(im1_gray_down, im2_gray_down, flowxy);
                nvof->convertToFloat(flowxy, floatFlow);
                // float test = floatFlow.at<cv::Vec2f>(0, 0)[0];
                // std::cout << test << std::endl;
                t_cuda.stop();
                cv::imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i - 1).c_str(), procFlow(floatFlow, down_size));
                // cv::writeOpticalFlow(std::format("../../benchmark/{}/{:04d}.flo", 0).c_str(), floatFlow);
                im2.copyTo(im1);
                im2_gray.copyTo(im1_gray);
                im2_gray_down.copyTo(im1_gray_down);
                // nvof->collectGarbage();
            }
            std::cout << "GPU report:" << std::endl;
            std::cout << "cvtColor avg: " << t1.averageLapInMilliSec() << std::endl;
            std::cout << "resize avg: " << t2.averageLapInMilliSec() << std::endl;
            std::cout << "NvidiaOpticalFlow_2_0 avg: " << t_cuda.averageLapInMilliSec() << std::endl;
            std::cout << "total avg: " << t_cuda.averageLapInMilliSec() + t1.averageLapInMilliSec() + t2.averageLapInMilliSec() << std::endl
                      << std::endl;
            t1.reset();
            t2.reset();
            t_cuda.reset();
        }
        if (alg == "OF_fb_GPU")
        {
            cv::Size down_size = cv::Size(images[0].cols / downscale_factor, images[0].rows / downscale_factor);
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> fbof = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
            cv::cuda::GpuMat gpu_im1, gpu_im2, gpu_im1_gray, gpu_im2_gray, gpu_im1_down, gpu_im2_down;
            cv::Mat flow = cv::Mat::zeros(down_size, CV_32FC2);
            cv::cuda::GpuMat gpu_flow;
            gpu_flow.upload(flow);
            cv::Mat im1, im1_gray, im1_gray_down;
            cv::Mat im2, im2_gray, im2_gray_down;
            im1 = images[0];
            cv::cvtColor(im1, im1_gray, cv::COLOR_BGR2GRAY);
            cv::resize(im1_gray, im1_gray_down, down_size);
            gpu_im1.upload(im1_gray_down);
            // cv::cuda::resize(gpu_im1, gpu_im1_down, down_size, 0, 0, cv::INTER_LINEAR);
            // cv::cuda::cvtColor(gpu_im1_down, gpu_im1_gray, cv::COLOR_BGR2GRAY);
            for (int i = 1; i < images.size() - 1; i++)
            {
                std::cout << '\r' << std::format("{:04d} / {}", i, images.size()) << std::flush;
                cv::Mat im2 = images[i];
                t1.start();
                cv::cvtColor(im2, im2_gray, cv::COLOR_BGR2GRAY);
                t1.stop();
                t2.start();
                cv::resize(im2_gray, im2_gray_down, down_size);
                t2.stop();
                t3.start();
                gpu_im2.upload(im2_gray_down);
                t3.stop();
                // upload pre-processed frame to GPU
                // t1.start();
                // gpu_im2.upload(images[i]);
                // t1.stop();
                // std::cout << "upload GPU: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // resize
                // t2.start();
                // cv::cuda::resize(gpu_im2, gpu_im2_down, down_size, 0, 0, cv::INTER_LINEAR);
                // t2.stop();
                // std::cout << "resize GPU: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // convert to gray
                // t3.start();
                // cv::cuda::cvtColor(gpu_im2_down, gpu_im2_gray, cv::COLOR_BGR2GRAY);
                // t3.stop();
                // std::cout << "cvtColor GPU: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // OF
                t4.start();
                fbof->calc(gpu_im1, gpu_im2, gpu_flow);
                t4.stop();

                // t_cuda.start();
                // nvof->calc(gpu_im1_gray, gpu_im2_gray, gpu_flow);
                // nvof->upSampler(flowxy, frameL.size().width, frameL.size().height,
                //                 nvof->getGridSize(), upsampledFlowXY);
                // t_cuda.stop();
                // std::cout << "calcOpticalFlowFarneback GPU: " << t1.getElapsedTimeInMilliSec() << std::endl;

                t5.start();
                gpu_flow.download(flow);
                t5.stop();
                // std::cout << "download GPU: " << t1.getElapsedTimeInMilliSec() << std::endl  << std::endl;
                imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i - 1).c_str(), procFlow(flow, down_size));
                gpu_im2.copyTo(gpu_im1);
                // gpu_im2_down.copyTo(gpu_im1_down);
                // gpu_im2_gray.copyTo(gpu_im1_gray);

                // cv::cuda::GpuMat gpu_flow_xy[2];
                // cv::cuda::split(gpu_flow, gpu_flow_xy);

                // // convert from cartesian to polar coordinates
                // cv::cuda::cartToPolar(gpu_flow_xy[0], gpu_flow_xy[1], gpu_magnitude, gpu_angle, true);

                // // normalize magnitude from 0 to 1
                // cv::cuda::normalize(gpu_magnitude, gpu_normalized_magnitude, 0.0, 1.0, cv::NORM_MINMAX, -1);

                // // get angle of optical flow
                // gpu_angle.download(angle);
                // angle *= ((1 / 360.0) * (180 / 255.0));

                // // build hsv image
                // gpu_hsv[0].upload(angle);
                // gpu_hsv[2] = gpu_normalized_magnitude;
                // cv::cuda::merge(gpu_hsv, 3, gpu_merged_hsv);

                // // multiply each pixel value to 255
                // gpu_merged_hsv.cuda::GpuMat::convertTo(gpu_hsv_8u, CV_8U, 255.0);

                // // convert hsv to bgr
                // cv::cuda::cvtColor(gpu_hsv_8u, gpu_bgr, cv::COLOR_HSV2BGR);

                // // send result from GPU back to CPU
                // gpu_bgr.download(bgr);
                // imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i).c_str(), bgr);
            }
            std::cout << "GPU report:" << std::endl;
            std::cout << "cvtColor avg: " << t1.averageLapInMilliSec() << std::endl;
            std::cout << "resize avg: " << t2.averageLapInMilliSec() << std::endl;
            std::cout << "upload avg: " << t3.averageLapInMilliSec() << std::endl;
            std::cout << "calcOpticalFlowFarneback avg: " << t4.averageLapInMilliSec() << std::endl;
            std::cout << "download avg: " << t5.averageLapInMilliSec() << std::endl
                      << std::endl;
            std::cout << "total avg: " << t1.averageLapInMilliSec() + t2.averageLapInMilliSec() + t3.averageLapInMilliSec() + t4.averageLapInMilliSec() + t5.averageLapInMilliSec() << std::endl
                      << std::endl;
            t1.reset();
            t2.reset();
            t3.reset();
            t4.reset();
            t5.reset();
        }
        if (alg == "OF_fb_CPU")
        {
            cv::Size down_size = cv::Size(images[0].cols / downscale_factor, images[0].rows / downscale_factor);
            cv::Mat flow = cv::Mat::zeros(down_size, CV_32FC2);
            cv::Mat im1, im2;
            cv::Mat im1_gray, im2_gray;
            cv::Mat im1_gray_down, im2_gray_down;
            im1 = images[0];
            cv::cvtColor(im1, im1_gray, cv::COLOR_BGR2GRAY);
            cv::resize(im1_gray, im1_gray_down, down_size);
            for (int i = 1; i < images.size() - 1; i++)
            {
                std::cout << '\r' << std::format("{:04d} / {}", i, images.size()) << std::flush;
                im2 = images[i + 1];
                t1.start();
                cv::cvtColor(im2, im2_gray, cv::COLOR_BGR2GRAY);
                t1.stop();
                // std::cout << "cvtColor CPU: " << t1.getElapsedTimeInMilliSec() << std::endl;
                t2.start();
                cv::resize(im2_gray, im2_gray_down, down_size);
                t2.stop();
                // std::cout << "resize CPU: " << t1.getElapsedTimeInMilliSec() << std::endl;
                t3.start();
                calcOpticalFlowFarneback(im1_gray_down, im2_gray_down, flow, 0.5, 5, 15, 3, 5, 1.2, cv::OPTFLOW_USE_INITIAL_FLOW);
                t3.stop();
                // std::cout << "calcOpticalFlowFarneback CPU: " << t1.getElapsedTimeInMilliSec() << std::endl << std::endl;
                imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i).c_str(), procFlow(flow, im1_gray_down.size()));
                im2.copyTo(im1);
                im2_gray.copyTo(im1_gray);
                im2_gray_down.copyTo(im1_gray_down);
            }
            std::cout << "CPU report:" << std::endl;
            std::cout << "cvtColor avg: " << t1.averageLapInMilliSec() << std::endl;
            std::cout << "resize avg: " << t2.averageLapInMilliSec() << std::endl;
            std::cout << "calcOpticalFlowFarneback avg: " << t3.averageLapInMilliSec() << std::endl
                      << std::endl;
            std::cout << "total avg: " << t1.averageLapInMilliSec() + t2.averageLapInMilliSec() + t3.averageLapInMilliSec() << std::endl
                      << std::endl;
            t1.reset();
            t2.reset();
            t3.reset();
        }
        if (alg == "OF_sparse_CPU")
        {
            // downscale_factor = 1;
            cv::Size down_size = cv::Size(images[0].cols / downscale_factor, images[0].rows / downscale_factor);
            std::vector<cv::Point2f> p0, p1;
            std::vector<cv::Scalar> colors;
            cv::RNG rng;

            for (int i = 0; i < 100; i++)
            {
                int r = rng.uniform(0, 256);
                int g = rng.uniform(0, 256);
                int b = rng.uniform(0, 256);
                colors.push_back(cv::Scalar(r, g, b));
            }
            // vector<Point2f> good_old, good_new;
            cv::Mat im1, im2;
            cv::Mat im1_gray, im2_gray;
            cv::Mat im1_gray_down, im2_gray_down;
            // cv::Mat im1_bin, im2_bin;
            im1 = images[0];
            cvtColor(im1, im1_gray, cv::COLOR_BGR2GRAY);
            // threshold(im1_gray, im1_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            resize(im1_gray, im1_gray_down, down_size);
            std::vector<int> feature_num;
            int fails = 0;
            std::vector<int> good_feature_num;
            for (int i = 1; i < images.size(); i++)
            {
                std::cout << '\r' << std::format("{:04d} / {}", i, images.size()) << std::flush;
                p0.clear();
                p1.clear();
                cv::Mat mask = cv::Mat::zeros(images[0].size(), images[0].type());
                cv::Mat im2 = images[i];
                t1.start();
                // convert images to gray scale;
                cvtColor(im2, im2_gray, cv::COLOR_BGR2GRAY);
                t1.stop();
                // std::cout << "t bgr2gray: " << t1.getElapsedTimeInMilliSec() << std::endl;

                // create pyramid from gray scale images
                // t1.start();
                // std::vector<Mat> im1_pyr, im2_pyr;
                // buildPyramid(im1_gray, im1_pyr, 3);
                // buildPyramid(im2_gray, im2_pyr, 3);
                // t1.stop();
                // std::cout << "t buildPyramid: " << t1.getElapsedTimeInMilliSec() << std::endl;

                // threshold images
                // t2.start();
                // threshold(im2_gray, im2_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                // t2.stop();
                // std::cout << "t threshold: " << t1.getElapsedTimeInMilliSec() << std::endl;

                // adaptive threshold
                // t1.start();
                // adaptiveThreshold(im1_gray, im1_bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 4);
                // adaptiveThreshold(im2_gray, im2_bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 4);
                // t1.stop();
                // std::cout << "t adaptiveThreshold: " << t1.getElapsedTimeInMilliSec() << std::endl;

                // downscale images for faster processing
                t2.start();
                resize(im2_gray, im2_gray_down, down_size);
                t2.stop();
                // t1.stop();
                // std::cout << "t resize: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // std::cout << "resized to: " << im1_gray.rows << "x" << im1_gray.cols << std::endl;

                // cv::Mat im1_aligned;
                // if ((i % slow_tracker_every == 0) && (i > 0))
                // {
                //     // for(uint j = 0; j < p0.size(); j++)
                //     // circle(mask, p0[j], 5, colors[j], -1);
                //     cv::Mat img;
                //     add(im1, mask, img);
                //     cv::imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i).c_str(), img);
                //     p0.clear();
                //     // good_new.clear();
                //     // good_old.clear();
                //     mask = cv::Mat::zeros(images[0].size(), images[0].type());
                // }
                t3.start();
                float quality = 0.1;
                goodFeaturesToTrack(im1_gray_down, p0, 100, quality, 3, cv::Mat(), 7, false, 0.04);
                t3.stop();
                feature_num.push_back(p0.size());
                std::vector<uchar> status;
                std::vector<float> err;
                cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
                t4.start();
                cv::calcOpticalFlowPyrLK(im1_gray_down, im2_gray_down, p0, p1, status, err, cv::Size(15, 15), 2, criteria);
                t4.stop();
                // std::cout << "calcOpticalFlowPyrLK: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // t1.start();
                int counter = 0;
                for (uint i = 0; i < p0.size(); i++)
                {
                    // Select good points
                    if (status[i] == 1)
                    {
                        counter += 1;
                        // good_old.push_back(p0[i]);
                        // good_new.push_back(p1[i]);
                    }
                }
                good_feature_num.push_back(counter);
                // t1.stop();
                // std::cout << "select good points: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // t1.start();
                // Find homography
                // Mat h = findHomography( good_old, good_new, RANSAC );
                // t1.stop();
                // std::cout << "findHomography: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // t1.start();
                // Use homography to warp image
                // warpPerspective(im1, im1_aligned, h, im2.size());
                // t1.stop();
                // std::cout << "warpPerspective: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // find affine transform
                // if (p1.size() < 3)
                //     continue;
                if ((p1.size() < 3) || p0.size() < 3)
                {
                    fails += 1;
                    continue;
                }
                t5.start();
                cv::Mat f = cv::estimateAffine2D(p0, p1);
                // f.at<float>(0, 2) *= downscale_factor;
                // f.at<float>(1, 2) *= downscale_factor;
                t5.stop();
                // std::cout << "estimateAffine2D: " << t1.getElapsedTimeInMilliSec() << std::endl;
                t6.start();
                cv::Mat im1_aligned2;
                cv::warpAffine(im1, im1_aligned2, f, im1.size());
                t6.stop();
                // visualize
                // std::cout << "warpAffine: " << t1.getElapsedTimeInMilliSec() << std::endl;
                // imwrite(std::format("../../benchmark/{}/{:04d}_image1.png", alg, i).c_str(), im1);
                // imwrite(std::format("../../benchmark/{}/{:04d}_image1_aligned.png", alg, i).c_str(), im1_aligned2);
                // imwrite(std::format("../../benchmark/{}/{:04d}_image2.png", alg, i).c_str(), im2);
                for (uint j = 0; j < p0.size(); j++)
                {
                    // cv::Point2f debug_vec = p0[i]*downscale_factor;
                    circle(mask, p1[i] * downscale_factor, 5, colors[i], -1);
                    cv::line(mask, p1[j] * downscale_factor, p0[j] * downscale_factor, colors[j], 2);
                    cv::Mat img;
                    add(im1, mask, img);
                    cv::imwrite(std::format("../../benchmark/{}/{:04d}.png", alg, i).c_str(), img);
                }
                im2.copyTo(im1);
                im2_gray.copyTo(im1_gray);
                im2_gray_down.copyTo(im1_gray_down);
                // imwrite(std::format("../../benchmark/{}/{:04d}_of.png", alg, i).c_str(), img);
            }
            std::cout << "Sprase CPU report:" << std::endl;
            std::cout << "cvtColor avg: " << t1.averageLapInMilliSec() << std::endl;
            std::cout << "resize avg: " << t2.averageLapInMilliSec() << std::endl;
            std::cout << "features extraction avg: " << t3.averageLapInMilliSec() << std::endl;
            std::cout << "calcOpticalFlowPyrLK avg: " << t4.averageLapInMilliSec() << std::endl;
            std::cout << "estimateAffine2D avg: " << t5.averageLapInMilliSec() << std::endl;
            std::cout << "warpAffine avg: " << t6.averageLapInMilliSec() << std::endl;
            std::cout << "fails: " << fails << std::format("/{}", images.size()) << std::endl;
            std::cout << "feature_num avg: " << std::accumulate(feature_num.begin(), feature_num.end(), 0.0) / feature_num.size() << std::endl;
            std::cout << "good_feature_num avg: " << std::accumulate(good_feature_num.begin(), good_feature_num.end(), 0.0) / good_feature_num.size() << std::endl
                      << std::endl;
            std::cout << "total avg: " << t1.averageLapInMilliSec() + t2.averageLapInMilliSec() + t3.averageLapInMilliSec() + t4.averageLapInMilliSec() + t5.averageLapInMilliSec() << std::endl
                      << std::endl;
            t1.reset();
            t2.reset();
            t3.reset();
            t4.reset();
            t5.reset();
            // if (alg == "FAST")
            // {
            //     const int kMaxMatchingSize = 50;
            //     std::vector<KeyPoint> kpts1;
            //     std::vector<KeyPoint> kpts2;
            //     Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create(80, true);
            //     BFMatcher matcher(cv::NORM_L2, true);
            //     fastDetector->detect(im1, kpts1);
            //     fastDetector->detect(im2, kpts2);
            //     // desc_matcher.match(im1_gray, im2_gray, matches, Mat());
            //     // match(match_type, desc1, desc2, matches);
            //     // std::sort(matches.begin(), matches.end());
            //     // while (matches.front().distance * kDistanceCoef < matches.back().distance) {
            //     //     matches.pop_back();
            //     // }
            //     // while (matches.size() > kMaxMatchingSize) {
            //     //     matches.pop_back();
            //     // }
            //     // Mat flow(prvs.size(), CV_32FC2);
            //     // calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            // }
            // if (alg == "ORB")
            // {
            //     const int warp_mode = MOTION_AFFINE;
            //     Mat warp_matrix;
            //     t1.start();
            //     // Initialize the matrix to identity
            //     warp_matrix = Mat::eye(2, 3, CV_32F);

            //     // Specify the number of iterations.
            //     int number_of_iterations = 500;

            //     // Specify the threshold of the increment
            //     // in the correlation coefficient between two iterations
            //     double termination_eps = 1e-10;

            //     // Define termination criteria
            //     TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);

            //     // Run the ECC algorithm. The results are stored in warp_matrix.
            //     findTransformECC(
            //         im2_gray_down,
            //         im1_gray_down,
            //         warp_matrix,
            //         warp_mode,
            //         criteria
            //     );
            //     warp_matrix.at<float>(0,2) *= downscale_factor;
            //     warp_matrix.at<float>(1,2) *= downscale_factor;
            //     // Storage for warped image.
            //     Mat im1_aligned;

            //     // if (warp_mode != MOTION_HOMOGRAPHY)
            //         // Use warpAffine for Translation, Euclidean and Affine
            //     warpAffine(im1, im1_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
            //     // else
            //     //     // Use warpPerspective for Homography
            //     //     warpPerspective (im1, im1_aligned, warp_matrix, im1.size(),INTER_LINEAR + WARP_INVERSE_MAP);
            //     t1.stop();
            //     std::cout << "findTransformECC: " << t1.getElapsedTimeInMilliSec() << std::endl;
            //     // Show final result
            //     std::string test = std::format("../../benchmark/{:04d}_image1.png", i);
            //     std::filesystem::path p = test;
            //     // std::cout << "full path is : " << std::filesystem::absolute(p) << std::endl;
            //     // imwrite(p.string(), im1);
            //     imwrite(std::format("../../benchmark/{}/{:04d}_image1.png", alg, i).c_str(), im1);
            //     imwrite(std::format("../../benchmark/{}/{:04d}_image2.png", alg, i).c_str(), im2);
            //     imwrite(std::format("../../benchmark/{}/{:04d}_image1_aligned.png", alg, i).c_str(), im1_aligned);
            //     // imshow("Image 1", im1);
            //     // imshow("Image 2", im2);
            //     // imshow("Image 2 Aligned", im2_aligned);
            //     // waitKey(0);

            // }
        }
    }
}