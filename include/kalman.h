#ifndef KALMAN_H
#define KALMAN_H
#include <opencv2/opencv.hpp>

class Kalman
{
public:
    Kalman(int stateDim, int measDim, int contrDim = 0);
    cv::Mat predict();
    cv::Mat correct(cv::Mat measurement);
    cv::Mat forecast(int steps);
    cv::Mat getProcNoiseCoV()
    {
        return KF.processNoiseCov;
    };
    cv::Mat getMeasNoiseCoV()
    {
        return KF.measurementNoiseCov;
    };
    cv::Mat getStatePost()
    {
        return KF.statePost;
    };
    cv::Mat getMeasurementMatrix() // maps bewteen state and measurement
    {
        return KF.measurementMatrix;
    };
    cv::Mat getTransitionMatrix()
    {
        return KF.transitionMatrix;
    };

protected:
    cv::KalmanFilter KF;
};

class Kalman1D : public Kalman
{
public:
    Kalman1D(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};

class Kalman2D : public Kalman
{
public:
    Kalman2D(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};

// class Kalman2DAcc : public Kalman
// {
// public:
//     Kalman2DAcc(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f);
// };
#endif // KALMAN_H