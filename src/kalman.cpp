#include "kalman.h"
#include "helpersCV.h"

Kalman::Kalman(int stateDim, int measDim, int contrDim) : KF(stateDim, measDim, contrDim)
{
}

void Kalman::setState(cv::Mat state)
{
    // set the initial state
    KF.statePost = state.clone();
    // cv::randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    // cv::setIdentity(KF.errorCovPre);
    // cv::setIdentity(KF.errorCovPost);
}

cv::Mat Kalman::predict(float dt)
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
    // update using measurement, and return the corrected state
    return KF.correct(measurement);
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

cv::Mat Kalman::forecast(float dt)
{
    cv::Mat new_state = cv::Mat(KF.transitionMatrix * KF.statePost);
    return new_state;
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

Kalman2D_ConstantV::Kalman2D_ConstantV(float processNoise, float measurementNoise) : Kalman(4, 2, 0)
{
    // transition Matrix (F) = [1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where state is 4x1 vector [x, y, dx, dy] (xn+1 = F * xn + process noise)
    // KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    cv::setIdentity(KF.transitionMatrix);
    // initial measurement matrix (H) = [1, 0, 0, 0; 0, 1, 0, 0] 2x4
    // where measurement is 2x1 vector [x, y] (zn = H * xn + measurement noise)
    setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    KF.processNoiseCov.at<float>(0, 0) = 0.0;
    KF.processNoiseCov.at<float>(1, 1) = 0.0;
    KF.processNoiseCov.at<float>(2, 2) = processNoise; //* 100.0f;
    KF.processNoiseCov.at<float>(3, 3) = processNoise; //* 100.0f;
    // measurement noise covariance matrix (R) = 2x2
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // cv::Mat initialState = cv::Mat::zeros(4, 1, CV_32F);
    // initialState.at<float>(2) = velocity;
    // initialState.at<float>(3) = velocity;
    // setInitialState(initialState);
}

cv::Mat Kalman2D_ConstantV::predict(float dt)
{
    KF.transitionMatrix.at<float>(2) = dt;
    KF.transitionMatrix.at<float>(7) = dt;
    cv::Mat prediction = KF.predict(); // predict the state (same dims as state)
    return prediction;
}

cv::Mat Kalman2D_ConstantV::forecast(float dt)
{
    KF.transitionMatrix.at<float>(2) = dt;
    KF.transitionMatrix.at<float>(7) = dt;
    cv::Mat new_state = cv::Mat(KF.transitionMatrix * KF.statePost);
    // std::vector<float> before = HelpersCV::flatten_cv(KF.statePost);
    // std::vector<float> after = HelpersCV::flatten_cv(new_state);
    // std::vector<float> transMat = HelpersCV::flatten_cv(KF.transitionMatrix);
    return new_state;
}

Kalman2D_ConstantV2::Kalman2D_ConstantV2(float processNoise, float measurementNoise) : Kalman(4, 4, 0)
{
    // transition Matrix (F) = [1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where state is 4x1 vector [x, y, dx, dy] (xn+1 = F * xn + process noise)
    // KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    cv::setIdentity(KF.transitionMatrix);
    // initial measurement matrix (H) = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] 4x4
    // where measurement is 4x1 vector [x, y, dx, dy] (zn = H * xn + measurement noise)
    cv::setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    cv::setIdentity(KF.processNoiseCov);
    KF.processNoiseCov.at<float>(0, 0) = processNoise;
    KF.processNoiseCov.at<float>(1, 1) = processNoise;
    KF.processNoiseCov.at<float>(2, 2) = processNoise * 10;
    KF.processNoiseCov.at<float>(3, 3) = processNoise * 10;
    // measurement noise covariance matrix (R) = 4x4
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // cv::Mat initialState = cv::Mat::zeros(4, 1, CV_32F);
    // initialState.at<float>(2) = velocity;
    // initialState.at<float>(3) = velocity;
    // setInitialState(initialState);
    saveCheckpoint();
}

cv::Mat Kalman2D_ConstantV2::predict(float dt)
{
    KF.transitionMatrix.at<float>(2) = dt;
    KF.transitionMatrix.at<float>(7) = dt;
    cv::Mat prediction = KF.predict(); // predict the state (same dims as state)
    return prediction;
}

cv::Mat Kalman2D_ConstantV2::forecast(float dt)
{
    KF.transitionMatrix.at<float>(2) = dt;
    KF.transitionMatrix.at<float>(7) = dt;
    cv::Mat new_state = cv::Mat(KF.transitionMatrix * KF.statePost);
    return new_state;
}

void Kalman2D_ConstantV2::setState(cv::Mat state)
{
    // set the initial state
    KF.statePost = state.clone();
    // randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
    // cv::setIdentity(KF.errorCovPre);
    // cv::setIdentity(KF.errorCovPost);
}

Kalman2D_ConstantA::Kalman2D_ConstantA(float processNoise, float measurementNoise, float error, float dt) : Kalman(6, 2, 0)
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
    cv::setIdentity(KF.measurementMatrix);
    // process noise covariance matrix (Q) = 4x4
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(processNoise));
    // measurement noise covariance matrix (R) = 2x2
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(measurementNoise));
    // setIdentity(KF.errorCovPost, cv::Scalar::all(error));
    cv::randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
}