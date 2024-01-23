#ifndef HELPERSCV_H
#define HELPERSCV_H

#include <opencv2/opencv.hpp>

class HelpersCV
{
public:
    static std::vector<double> flatten_cv(std::vector<cv::Point> vec)
    {
        std::vector<double> flat_vec;
        for (int i = 0; i < vec.size(); i++)
        {
            flat_vec.push_back(static_cast<double>(vec[i].x));
            flat_vec.push_back(static_cast<double>(vec[i].y));
        }
        return flat_vec;
    }
    static std::vector<float> flatten_cv(cv::Mat mat)
    {
        std::vector<float> flat_vec;
        for (int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                flat_vec.push_back(mat.at<float>(i, j));
            }
        }
        return flat_vec;
    }

private:
    HelpersCV();
};

#endif // HELPERSCV_H