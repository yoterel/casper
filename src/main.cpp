#include <chrono>
#include <thread>
#include "queue.h"
#include "camera.h"
#include "display.h"
#include "SerialPort.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

using namespace std::literals::chrono_literals;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void keycallback(GLFWwindow *window, int key, int scancode, int action, int mods);

int main( int /*argc*/, char* /*argv*/[] )
{
    int proj_width = 1024;
    int proj_height = 768;
    // render output window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow* window = glfwCreateWindow(proj_width, proj_height, "project", NULL, NULL);  //monitors[0], NULL for full screen
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    std::cout << "Succeeded to create GL window." << std::endl;
    std::cout << "  GL Version   : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "  GL Vendor    : " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "  GL Renderer  : " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "  GLSL Version : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << std::endl;
    
    glfwSwapInterval(0);  // do not sync to monitor
    glViewport(0, 0, proj_width, proj_height);
    glClearColor(0.5, 0.5, 0.5, 0); // glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, keycallback);
    double previousTime = glfwGetTime();
    int frameCount = 0;
    while(!glfwWindowShouldClose(window))
    {
        // Measure speed
        double currentTime = glfwGetTime();
        frameCount++;
        // If a second has passed.
        if ( currentTime - previousTime >= 1.0 )
        {
            // Display the frame count here any way you want.
            std::cout << "FPS: " << frameCount << std::endl;;

            frameCount = 0;
            previousTime = currentTime;
        }
        // render
        // ------
        glClear(GL_COLOR_BUFFER_BIT);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();    
    }
    glfwTerminate();
    // auto start = std::chrono::system_clock::now();
    // main1();
    // auto runtime = std::chrono::system_clock::now() - start;
    // std::cout << "consumer producer took "
    //    << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()
    //    << " usec\n";
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

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void keycallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
}
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}