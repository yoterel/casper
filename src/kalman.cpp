#include "kalman.h"

Kalman::Kalman(int stateDim, int measDim, int contrDim) : KF(stateDim, measDim, contrDim)
{
}

cv::Mat Kalman::predict()
{
    cv::Mat prediction = KF.predict();
    return prediction;
}

cv::Mat Kalman::correct(cv::Mat measurement)
{
    KF.correct(measurement);
    return KF.statePost;
}

cv::Mat Kalman::forecast(int steps)
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

Kalman1D::Kalman1D(float processNoise, float measurementNoise, float error, float dt) : Kalman(2, 1, 0)
{
    KF.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, dt, 0, 1);
    cv::setIdentity(KF.measurementMatrix); // [1, 0]
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}

Kalman2D::Kalman2D(float processNoise, float measurementNoise, float error, float dt) : Kalman(4, 2, 0)
{
    // transition Matrix (F) = [1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where state is 4x1 vector [x, y, dx, dy] (xn+1 = F * xn + process noise)
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    // initial measurement matrix (H) = [1, 0, 0, 0; 0, 1, 0, 0] 2x4
    // where measurement is 2x1 vector [x, y] (zn = H * xn + measurement noise)
    setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    // measurement noise covariance matrix (R) = 2x2
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}

Kalman2DAcc::Kalman2DAcc(float processNoise, float measurementNoise, float error, float dt) : Kalman(6, 2, 0)
{
    // transition Matrix (F) = [1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where state is 4x1 vector [x, y, dx, dy] (xn+1 = F * xn + process noise)
    KF.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, dt, 0, dt * dt / 2, 0,
                           0, 1, 0, dt, 0, dt * dt / 2,
                           0, 0, 1, 0, dt, 0,
                           0, 0, 0, 1, 0, dt,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 1);
    // initial measurement matrix (H) = [1, 0, 0, 0; 0, 1, 0, 0] 2x4
    // where measurement is 2x1 vector [x, y] (zn = H * xn + measurement noise)
    setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    // measurement noise covariance matrix (R) = 2x2
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}