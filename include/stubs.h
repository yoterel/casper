#ifndef STUBS_H
#define STUBS_H

#include <opencv2/opencv.hpp>

class GPUMATStub
{
public:
    void upload(cv::Mat mat){};
    void download(cv::Mat &mat){};
};

class OFStub
{
public:
    void calc(GPUMATStub &gprev, GPUMATStub &gcur, GPUMATStub &gflow){};
    void calc(cv::Mat gprev, cv::Mat gcur, cv::Mat gflow){};
    void convertToFloat(cv::Mat flowin, cv::Mat flowout){};
};

#endif // STUBS_H