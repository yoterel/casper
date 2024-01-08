#include "kalman.h"

Kalman1D::Kalman1D(float processNoise, float measurementNoise, float error) : KF(2, 1, 0)
{
    KF.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
    cv::setIdentity(KF.measurementMatrix); // [1, 0]
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}

cv::Mat Kalman1D::predict()
{
    cv::Mat prediction = KF.predict();
    return prediction;
}

cv::Mat Kalman1D::correct(cv::Mat measurement)
{
    KF.correct(measurement);
    return KF.statePost;
}

cv::Mat Kalman1D::forecast(int steps)
{
    cv::Mat new_state = cv::Mat(KF.transitionMatrix * KF.statePost);
    for (int i = 1; i < steps; i++)
    {
        new_state = cv::Mat(KF.transitionMatrix * new_state);
    }
    return new_state;
}

Kalman2D::Kalman2D(float processNoise, float measurementNoise, float error) : KF(4, 2, 0)
{
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    setIdentity(KF.measurementMatrix); // [1, 0, 0, 0; 0, 1, 0, 0] 2x4
    // std::cout << "KF.measurementMatrix = " << KF.measurementMatrix << std::endl;
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}

cv::Mat Kalman2D::predict()
{
    cv::Mat prediction = KF.predict();
    return prediction;
}

cv::Mat Kalman2D::correct(cv::Mat measurement)
{
    KF.correct(measurement);
    return KF.statePost;
}

cv::Mat Kalman2D::forecast(int steps)
{
    if (steps < 1)
    {
        return KF.statePost;
    }
    else
    {
        cv::Mat new_state = cv::Mat(KF.transitionMatrix * KF.statePost);
        for (int i = 1; i < steps; i++)
        {
            new_state = cv::Mat(KF.transitionMatrix * new_state);
        }
        return new_state;
    }
}