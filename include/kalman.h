#ifndef KALMAN_H
#define KALMAN_H
#include <opencv2/opencv.hpp>

class Kalman
{
public:
    // see https://github.com/opencv/opencv/blob/4.x/modules/video/src/kalman.cpp
    Kalman(int stateDim, int measDim, int contrDim = 0);
    virtual void setState(cv::Mat state = cv::Mat::zeros(4, 1, CV_32F));
    virtual cv::Mat predict(float dt = 1.0f);
    virtual cv::Mat forecast(float dt = 1.0f);
    cv::Mat correct(cv::Mat measurement, bool saveMeasurement = false);
    void rewindToCheckpoint(bool verbose = false);
    cv::Mat fastforward(cv::Mat measurement, bool verbose = false);
    bool getCheckpointMeasurement(cv::Mat &measurement);
    void saveCheckpoint();
    cv::Mat getProcNoiseCoV()
    {
        return KF.processNoiseCov.clone();
    };
    cv::Mat getMeasNoiseCoV()
    {
        return KF.measurementNoiseCov.clone();
    };
    void setMeasNoiseCoV(cv::Mat measNoiseCov)
    {
        KF.measurementNoiseCov = measNoiseCov.clone();
    };
    cv::Mat getStatePost()
    {
        return KF.statePost.clone();
    };
    cv::Mat getMeasurementMatrix() // maps bewteen state and measurement
    {
        return KF.measurementMatrix.clone();
    };
    cv::Mat getTransitionMatrix()
    {
        return KF.transitionMatrix.clone();
    };

protected:
    cv::KalmanFilter KF;
    std::vector<cv::Mat> measurements;
    std::vector<cv::Mat> measCovs;
    std::vector<float> dts;
    cv::Mat checkpoint_state_post;
    cv::Mat checkpoint_cov_post;
    float cur_dt;
};

class Kalman1D : public Kalman
{
public:
    Kalman1D(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};

class Kalman2D_ConstantV : public Kalman
{
public:
    Kalman2D_ConstantV(float processNoise = 1e-5, float measurementNoise = 1e-1);
    virtual cv::Mat predict(float dt) override;
    virtual cv::Mat forecast(float dt) override;

private:
    float m_velocity;
};

class Kalman2D_ConstantV2 : public Kalman
{
public:
    Kalman2D_ConstantV2(float processNoise = 1e-5, float measurementNoise = 1e-1);
    virtual void setState(cv::Mat state = cv::Mat::zeros(4, 1, CV_32F)) override;
    virtual cv::Mat predict(float dt) override;
    virtual cv::Mat forecast(float dt) override;

private:
    float m_velocity;
};

class Kalman2D_ConstantA : public Kalman
{
public:
    Kalman2D_ConstantA(float processNoise = 1e-5, float measurementNoise = 1e-1, float error = 1.0f, float dt = 0.01f);
};
#endif // KALMAN_H