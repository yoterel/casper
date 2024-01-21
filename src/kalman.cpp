#include "kalman.h"

Kalman::Kalman(int stateDim, int measDim, int contrDim) : KF(stateDim, measDim, contrDim)
{
}

void Kalman::setInitialState(float error)
{
    // set the initial state
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    // set the initial state covariance
    setIdentity(KF.errorCovPost, cv::Scalar::all(error));
}

cv::Mat Kalman::predict()
{
    cv::Mat prediction = KF.predict(); // predict the state (same dims as state)
    return prediction;
}

cv::Mat Kalman::correct(cv::Mat measurement, bool saveMeasurement)
{
    if (saveMeasurement)
    {
        measurements.push_back(measurement.clone());
        measCovs.push_back(KF.measurementNoiseCov.clone());
        // std::cout << "measureCov: " << KF.measurementNoiseCov << std::endl;
    }
    KF.correct(measurement); // update using measurement
    return KF.statePost;     // return the corrected state (same dims as state)
}

void Kalman::rewindToCheckpoint(bool verbose)
{
    if (verbose)
    {
        std::cout << "before rewind state: " << KF.statePost << std::endl;
        std::cout << "before rewind state cov : " << KF.errorCovPost << std::endl;
    }
    KF.statePost = checkpoint_state_post.clone();
    KF.errorCovPost = checkpoint_cov_post.clone();
    if (verbose)
    {
        std::cout << "after rewind state: " << KF.statePost << std::endl;
        std::cout << "after rewind state cov : " << KF.errorCovPost << std::endl;
    }
    // std::cout << "rewinded state: " << KF.statePost << std::endl;
    // std::cout << "rewinded state cov : " << KF.errorCovPost << std::endl;
}
cv::Mat Kalman::fastforward(cv::Mat measurement, bool verbose)
{
    if (verbose)
    {
        std::cout << "state before ff: " << KF.statePost << std::endl;
        std::cout << "measurement before ff: " << measurement << std::endl;
        std::cout << "measureCov before ff: " << KF.measurementNoiseCov << std::endl;
    }
    predict();
    cv::Mat test = correct(measurement, false);
    if (verbose)
        std::cout << "ff: state after iter 0: " << test << std::endl;
    for (int i = 1; i < measurements.size(); i++)
    {
        KF.measurementNoiseCov = measCovs[i];
        if (verbose)
        {
            std::cout << "ff measurement: " << measurements[i] << std::endl;
            std::cout << "ff measureCov: " << measCovs[i] << std::endl;
        }
        predict();
        test = correct(measurements[i], false);
        if (verbose)
            std::cout << "ff: state after iter " << i << ": " << test << std::endl;
    }
    measurements.clear();
    measCovs.clear();
    if (verbose)
        std::cout << "state after ff: " << KF.statePost << std::endl;
    return KF.statePost;
}

cv::Mat Kalman::getCheckpointMeasurement()
{
    return measurements[0];
}

void Kalman::saveCheckpoint()
{
    checkpoint_state_post = KF.statePost.clone();
    checkpoint_cov_post = KF.errorCovPost.clone();
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

Kalman2DXY::Kalman2DXY(float processNoise, float measurementNoise, float error, float dt) : Kalman(4, 2, 0)
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
    // todo: do we need to set errorCovPost and statePost?
    setInitialState(error);
    // setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    // randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}

Kalman2D::Kalman2D(float processNoise, float measurementNoise, float error, float dt) : Kalman(4, 4, 0)
{
    // transition Matrix (F) = [1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where state is 4x1 vector [x, y, dx, dy] (xn+1 = F * xn + process noise)
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    // initial measurement matrix (H) = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where measurement is 4x1 vector [x, y, dx, dy] (zn = H * xn + measurement noise)
    setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    // measurement noise covariance matrix (R) = 4x4
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // todo: do we need to set errorCovPost and statePost?
    Kalman::setInitialState(error);
    saveCheckpoint();
}

void Kalman2D::setInitialState(cv::Mat state)
{
    // set the initial state
    KF.statePost = state.clone();
    // set the initial state covariance ?
    // ?
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