#ifndef KALMAN_H
#define KALMAN_H
#include <opencv2/opencv.hpp>

class Kalman
{
public:
    Kalman(int stateDim, int measDim, int contrDim = 0);
    void setInitialState(float error);
    cv::Mat predict();
    cv::Mat correct(cv::Mat measurement, bool saveMeasurement = false);
    cv::Mat forecast(int steps);
    void rewindToCheckpoint(bool verbose = false);
    cv::Mat fastforward(cv::Mat measurement, bool verbose = false);
    cv::Mat getCheckpointMeasurement();
    void saveCheckpoint();
    cv::Mat getProcNoiseCoV()
    {
        return KF.processNoiseCov;
    };
    cv::Mat getMeasNoiseCoV()
    {
        return KF.measurementNoiseCov;
    };
    void setMeasNoiseCoV(cv::Mat measNoiseCov)
    {
        KF.measurementNoiseCov = measNoiseCov;
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
    std::vector<cv::Mat> measurements;
    std::vector<cv::Mat> measCovs;
    cv::Mat checkpoint_state_post;
    cv::Mat checkpoint_cov_post;
};

class Kalman1D : public Kalman
{
public:
    Kalman1D(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};

class Kalman2DXY : public Kalman
{
public:
    Kalman2DXY(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};

class Kalman2D : public Kalman
{
public:
    Kalman2D(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
    void setInitialState(cv::Mat state);
};

class Kalman2DAcc : public Kalman
{
public:
    Kalman2DAcc(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};
#endif // KALMAN_H