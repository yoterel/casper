#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "kalman.h"

std::vector<int> x_mouse_pos;
std::vector<int> y_mouse_pos;

cv::Point calcPoint(cv::Point2f center, double R, double angle);
void circle_example();
void mouse_example();
void CallBackFunc(int event, int x, int y, int flags, void *userdata);

cv::Point calcPoint(cv::Point2f center, double R, double angle)
{
    return center + cv::Point2f((float)cos(angle), (float)-sin(angle)) * (float)R;
}

void circle_example()
{
    cv::Mat img(500, 500, CV_8UC3);
    Kalman1D kalman = Kalman1D();
    cv::Mat state(2, 1, CV_32F);                        /* actual state: (phi, delta_phi) */
    cv::Mat processNoise(2, 1, CV_32F);                 /* actual process noise: (phi_noise, delti_phi_noise) */
    cv::Mat measurement = cv::Mat::zeros(1, 1, CV_32F); /* actual measurements: (phi) */
    char code = (char)-1;
    for (;;)
    {
        img = cv::Scalar::all(0);
        state.at<float>(0) = 0.0f;                   // initial angle
        state.at<float>(1) = 2.f * (float)CV_PI / 6; // initial angular velocity
        for (;;)
        {
            // compute physical location based on state
            cv::Point2f center(img.cols * 0.5f, img.rows * 0.5f);
            float R = img.cols / 3.f;
            double stateAngle = state.at<float>(0);
            cv::Point statePt = calcPoint(center, R, stateAngle);
            // perform prediction step of Kalman filter (just by taking into account the dynamic model)
            cv::Mat prediction = kalman.predict();
            double predictAngle = prediction.at<float>(0);
            cv::Point predictPt = calcPoint(center, R, predictAngle);
            // generate measurement, assumes the kalman measurement noise cov is exactly the g.t. measurement noise cov (rarely true in practice)
            randn(measurement, cv::Scalar::all(0), cv::Scalar::all(kalman.getMeasNoiseCoV().at<float>(0)));
            measurement += kalman.getMeasurementMatrix() * state; // z_k = H * x_k + v_k
            double measAngle = measurement.at<float>(0);
            cv::Point measPt = calcPoint(center, R, measAngle);
            // correct the state estimates based on measurements
            // updates statePost & errorCovPost
            cv::Mat statePost = kalman.correct(measurement);
            // get corrected estimation of phi
            double improvedAngle = statePost.at<float>(0);
            cv::Point improvedPt = calcPoint(center, R, improvedAngle);
            // plot points
            img = img * 0.2;
            drawMarker(img, measPt, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, predictPt, cv::Scalar(0, 255, 255), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, improvedPt, cv::Scalar(0, 255, 0), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, statePt, cv::Scalar(255, 255, 255), cv::MARKER_STAR, 10, 1);
            // forecast some steps
            cv::Mat test = kalman.forecast(3);
            drawMarker(img, calcPoint(center, R, test.at<float>(0)),
                       cv::Scalar(255, 255, 0), cv::MARKER_SQUARE, 12, 1);
            line(img, statePt, measPt, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, 0);
            line(img, statePt, predictPt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0);
            line(img, statePt, improvedPt, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);
            // evolve true state, assumes the kalman process noise cov is exactly the g.t. process noise cov (rarely true in practice)
            randn(processNoise, cv::Scalar(0), cv::Scalar::all(sqrt(kalman.getProcNoiseCoV().at<float>(0, 0))));
            state = kalman.getTransitionMatrix() * state + processNoise;
            imshow("Kalman", img);
            code = (char)cv::waitKey(1000);
            if (code > 0)
                break;
        }
        if (code == 27 || code == 'q' || code == 'Q')
            break;
    }
}

void CallBackFunc(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_MOUSEMOVE)
    {
        x_mouse_pos.push_back(x);
        y_mouse_pos.push_back(y);
    }
}
void mouse_example()
{
    cv::Mat img(500, 500, CV_8UC3);
    cv::namedWindow("Kalman", 1);
    cv::setMouseCallback("Kalman", CallBackFunc, NULL);
    int past_state_memory = 5;
    Kalman2D kalman = Kalman2D(1e-3, 1e-1, 1.0f);
    // cv::Mat state(4, 1, CV_32F);                        /* actual state: (phi, delta_phi) */
    cv::Mat past_state(4, 1, CV_32F);                   /* actual state: (phi, delta_phi) */
    cv::Mat processNoise(2, 1, CV_32F);                 /* actual process noise: (phi_noise, delti_phi_noise) */
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F); /* actual measurements: (phi) */
    char code = (char)-1;
    for (;;)
    {
        if (x_mouse_pos.size() >= past_state_memory)
        {

            img = cv::Scalar::all(0);
            // state.at<float>(0) = x_mouse_pos[x_mouse_pos.size() - 1];
            // state.at<float>(1) = y_mouse_pos[x_mouse_pos.size() - 1];
            // state.at<float>(2) = 0.0f;
            // state.at<float>(3) = 0.0f;
            past_state.at<float>(0) = x_mouse_pos[x_mouse_pos.size() - past_state_memory];
            past_state.at<float>(1) = y_mouse_pos[x_mouse_pos.size() - past_state_memory];
            past_state.at<float>(2) = 0.0f;
            past_state.at<float>(3) = 0.0f;
            // compute physical location based on state
            // cv::Point2f center(img.cols * 0.5f, img.rows * 0.5f);
            // float R = img.cols / 3.f;
            float statex = past_state.at<float>(0);
            float statey = past_state.at<float>(1);
            cv::Point statePt = cv::Point(statex, statey);
            // perform prediction step of Kalman filter (just by taking into account the dynamic model)
            cv::Mat prediction = kalman.predict();
            float predictx = prediction.at<float>(0);
            float predicty = prediction.at<float>(1);
            cv::Point predictPt = cv::Point(predictx, predicty);
            // generate measurement, assumes the kalman measurement noise cov is exactly the g.t. measurement noise cov (rarely true in practice)
            // randn(measurement, cv::Scalar::all(0), cv::Scalar::all(kalman.getMeasNoiseCoV().at<float>(0)));
            measurement = kalman.getMeasurementMatrix() * past_state; // 2x4 * 4x1 = 2x1
            // double measAngle = measurement.at<float>(0);
            // cv::Point measPt = calcPoint(center, R, measAngle);
            // correct the state estimates based on measurements
            // updates statePost & errorCovPost
            cv::Mat statePost = kalman.correct(measurement);
            // get corrected estimation of phi
            float improved_predictx = statePost.at<float>(0);
            float improved_predicty = statePost.at<float>(1);
            cv::Point improvedPt = cv::Point(improved_predictx, improved_predicty);
            // plot points
            // img = img * 0.2;
            // drawMarker(img, measPt, cv::Scalar(0, 0, 255), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, predictPt, cv::Scalar(255, 0, 0), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, improvedPt, cv::Scalar(0, 255, 0), cv::MARKER_SQUARE, 5, 2);
            drawMarker(img, statePt, cv::Scalar(255, 255, 255), cv::MARKER_STAR, 10, 1);
            // forecast some steps
            cv::Mat forecast = kalman.forecast(4);
            float forecastx = forecast.at<float>(0);
            float forecasty = forecast.at<float>(1);
            cv::Point forecastPt = cv::Point(forecastx, forecasty);
            drawMarker(img, forecastPt, cv::Scalar(255, 255, 0), cv::MARKER_SQUARE, 12, 1);
            // line(img, statePt, measPt, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, 0);
            // line(img, statePt, predictPt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0);
            // line(img, statePt, improvedPt, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);
            // evolve true state, assumes the kalman process noise cov is exactly the g.t. process noise cov (rarely true in practice)
            // randn(processNoise, cv::Scalar(0), cv::Scalar::all(sqrt(kalman.getProcNoiseCoV().at<float>(0, 0))));
            // state = kalman.getTransitionMatrix() * state;
        }
        else
        {
            std::cout << "Not enough points to start kalman filter" << std::endl;
        }
        imshow("Kalman", img);
        code = (char)cv::waitKey(1);
        if (code == 27 || code == 'q' || code == 'Q')
            break;
    }
}

int main(int, char **)
{
    // circle_example();
    mouse_example();
    return 0;
}