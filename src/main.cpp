#include <chrono>
#include <thread>
#include "queue.h"
#include "camera.h"
#include "display.h"
#include "SerialPort.h"
#include <GLFW/glfw3.h>

using namespace std::literals::chrono_literals;

int main( int /*argc*/, char* /*argv*/[] )
{
    // auto start = std::chrono::system_clock::now();
    // main1();
    // auto runtime = std::chrono::system_clock::now() - start;
    // std::cout << "consumer producer took "
    //    << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()
    //    << " usec\n";
    int proj_width = 1024;
    int proj_height = 768;
    DynaFlashProjector projector(proj_width, proj_height);
    bool success = projector.init();
    if (!success) {
        std::cerr << "Failed to initialize projector\n";
        return 1;
    }
    char* portName = "\\\\.\\COM4";
    bool close_signal = false;
    #define DATA_LENGTH 255
    SerialPort *arduino = new SerialPort(portName);
    std::cout << "Arduino is connected: " << arduino->isConnected() << std::endl;
    const char *sendString = "trigger\n"; 
    if (arduino->isConnected()){
        bool hasWritten = arduino->writeSerialPort(sendString, DATA_LENGTH);
        if (hasWritten) std::cout << "Data Written Successfully" << std::endl;
        else std::cerr << "Data was not written" << std::endl;
    }
    blocking_queue<cv::Mat> camera_queue;
    auto consumer = std::thread([&camera_queue, &close_signal, &projector]() {  //, &projector
            bool flag = true;
            cv::Mat white_image(projector.width, projector.height, CV_8UC3, cv::Scalar(255, 255, 255));
            while (!close_signal) {
                
                if (flag == true || camera_queue.size() == 0) {
                    // cv::Mat image;
                    // bool success = camera_queue.pop_with_timeout(1, image);
                    projector.show(white_image);
                }else{
                    auto start = std::chrono::system_clock::now();
                    std::cout << camera_queue.size() << "\n";
                    cv::Mat image = camera_queue.pop();
                    cv::resize(image, image, cv::Size(projector.width, projector.height));
                    projector.show(image);
                    auto runtime = std::chrono::system_clock::now() - start;
                    std::cout << "ms: "
                    << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
                    << "\n";
                }
                //     continue;
                // }
                // else
                // {
                //     if (flag == true || camera_queue.size() == 0)
                //     {
                        // image = cv::Mat::ones(cv::Size(1024, 768), CV_8UC3);
                    // }
                    // else
                    // {
                    //     image = camera_queue.pop();
                    //     // cv::namedWindow("image", cv::WINDOW_AUTOSIZE );
                    //     // cv::imshow("image", image);
                    //     // cv::waitKey(1);
                    // }
                flag = !flag;
                // }
                
            }
            std::cout << "Consumer finish" << std::endl;
        });
    
    BaslerCamera camera(camera_queue, close_signal);
    camera.acquire();
    while (!close_signal)
    {
        std::string userInput;
        std::getline(std::cin, userInput);
        for (size_t i = 0; i < userInput.size(); ++i)
        {
            char key = userInput[i];
            if (key == 'q')
            {
                close_signal = true;
                break;
            }
        }
    }
    consumer.join();
    return 0;
}

