#include <chrono>
#include <thread>
#include "queue.h"
#include "camera.h"
#include "display.h"
#include "SerialPort.h"
#include "shader.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "timer.h"
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std::literals::chrono_literals;

/* settings */

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void keycallback(GLFWwindow *window, int key, int scancode, int action, int mods);
// unsigned int compile_shaders();
unsigned int setup_buffers();
void saveImage(char* filepath, GLFWwindow* w);

int main( int /*argc*/, char* /*argv*/[] )
{
    const int proj_width = 1024;
    const int proj_height = 768;
    const int image_size = proj_width * proj_height * 3;
    Timer t0, t1, t2, t3, t4;  // t1.start(); t1.stop(); t1.getElapsedTimeInMilliSec();
    // render output window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow* window = glfwCreateWindow(proj_width, proj_height, "ahands", NULL, NULL);  //monitors[0], NULL for full screen
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
    Shader ourShader("C:/src/augmented_hands/src/texture.vs", "C:/src/augmented_hands/src/texture.fs");
    // int params;
    // glGetInternalformativ(GL_TEXTURE_2D, GL_RGB, GL_INTERNALFORMAT_PREFERRED, 1, &params);
    // std::cout << "  Preffered Internal Format : " << std::hex << params << std::endl;
    // glGetInternalformativ(GL_TEXTURE_2D, GL_RGB, GL_READ_PIXELS_FORMAT, 1, &params);
    // std::cout << "  Preffered Read Pixels Format : " << std::hex << params << std::endl;
    // glGetInternalformativ(GL_TEXTURE_2D, GL_RGB, GL_TEXTURE_IMAGE_FORMAT, 1, &params);
    // std::cout << "  Preffered Texture Image Format : " << std::hex << params << std::endl;
    // unsigned int shaderProgram = compile_shaders();
    unsigned int VAO = setup_buffers();

    // load and create a texture 
    // -------------------------
    unsigned int texture1;
    // texture 1
    // ---------
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    // int width, height, nrChannels;
    // stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    // unsigned char *data = stbi_load("C:/src/augmented_hands/resource/container.jpg", &width, &height, &nrChannels, 0);
    // if (data)
    // {
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    //     // glGenerateMipmap(GL_TEXTURE_2D);
    // }
    // else
    // {
    //     std::cout << "Failed to load texture" << std::endl;
    // }
    

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    ourShader.use(); // don't forget to activate/use the shader before setting uniforms!
    ourShader.setInt("texture1", 0);
    ourShader.setFloat("threshold", 0.15f);

    glfwSwapInterval(0);  // do not sync to monitor
    glViewport(0, 0, proj_width, proj_height);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f); // glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, keycallback);
    double previousTime = glfwGetTime();
    int frameCount = 0;
    // bool save_flag = true;
    bool close_signal = false;
    bool use_pbo = false;
    bool producer_is_fake = false;
    uint8_t* colorBuffer = new uint8_t[image_size];
    uint32_t cam_height = 0;
    uint32_t cam_width = 0;
    blocking_queue<CPylonImage> camera_queue;
    BaslerCamera camera;
    DynaFlashProjector projector(proj_width, proj_height);
    if (!projector.init()) {
        std::cerr << "Failed to initialize projector\n";
    }
    std::thread producer;
    if (producer_is_fake) {
        /* fake producer */
        cam_height = 540;
        cam_width = 720;
        producer = std::thread([&camera_queue, &close_signal, &cam_height, &cam_width]() {  //, &projector
            CPylonImage image = CPylonImage::Create( PixelType_RGB8packed, cam_width, cam_height);
            while (!close_signal) {
                camera_queue.push(image);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "Producer finish" << std::endl;
        });
    }
    else
    {
        /* camera producer */
        camera.init(camera_queue, close_signal, cam_height, cam_width);
        camera.acquire();
    }
    while(!glfwWindowShouldClose(window))  // todo: change to PBO http://www.songho.ca/opengl/gl_pbo.html
    {
        // Measure speed
        double currentTime = glfwGetTime();
        frameCount++;
        // If a second has passed.
        if ( currentTime - previousTime >= 1.0 )
        {
            // Display the frame count here any way you want.
            std::cout << "avg ms: " << 1000.0f/frameCount<<" FPS: " << frameCount << std::endl;
            std::cout << "last wait_for_cam time: " << t0.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last Cam->GPU time: " << t1.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last GPU Process time: " << t2.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last GPU->CPU time: " << t3.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last project time: " << t4.getElapsedTimeInMilliSec() << std::endl;
            frameCount = 0;
            previousTime = currentTime;
        }
        // render
        // ------
        glClear(GL_COLOR_BUFFER_BIT);
        t0.start();
        CPylonImage pylonImage = camera_queue.pop();
        uint8_t* buffer = ( uint8_t*) pylonImage.GetBuffer();
        t0.stop();
        t1.start();
        // load texture to GPU (large overhead)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_width, cam_height, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        t1.stop();
        t2.start();
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        // glActiveTexture(GL_TEXTURE1);
        // glBindTexture(GL_TEXTURE_2D, texture2);
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        t2.stop();
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        if (use_pbo) {
            // saveImage("test.png", window);
            // save_flag = false;
        }else{
            t3.start();
            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, proj_width, proj_height, GL_RGB, GL_UNSIGNED_BYTE, colorBuffer);
            t3.stop();
            t4.start();
            projector.show_buffer(colorBuffer);
            t4.stop();
            // stbi_flip_vertically_on_write(true);
            // int stride = 3 * proj_width;
            // stride += (stride % 4) ? (4 - stride % 4) : 0;
            // stbi_write_png("test.png", proj_width, proj_height, 3, colorBuffer, stride);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // stbi_image_free(data);
    glfwTerminate();
    close_signal = true;
    delete[] colorBuffer;
    if (producer_is_fake)
    {
        producer.join();
    }
    /* setup projector */
    // DynaFlashProjector projector(proj_width, proj_height);
    // bool success = projector.init();
    // if (!success) {
    //     std::cerr << "Failed to initialize projector\n";
    //     return 1;
    // }
    /* setup trigger */
    // char* portName = "\\\\.\\COM4";
    // #define DATA_LENGTH 255
    // SerialPort *arduino = new SerialPort(portName);
    // std::cout << "Arduino is connected: " << arduino->isConnected() << std::endl;
    // const char *sendString = "trigger\n"; 
    // if (arduino->isConnected()){
    //     bool hasWritten = arduino->writeSerialPort(sendString, DATA_LENGTH);
    //     if (hasWritten) std::cout << "Data Written Successfully" << std::endl;
    //     else std::cerr << "Data was not written" << std::endl;
    // }
    /* end setup trigger */
    
    /* CPU consumer */
    // auto consumer = std::thread([&camera_queue, &close_signal, &projector, &cam_height, &cam_width]() {  //, &projector
    //     bool flag = false;
    //     // uint8_t cam_height = projector.height;
    //     // uint8_t cam_width = projector.width;
    //     // projector.show(white_image);
    //     while (!close_signal) {
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //         if (flag == true || camera_queue.size() == 0) {
    //             // cv::Mat image;
    //             // bool success = camera_queue.pop_with_timeout(1, image);
    //             // auto start = std::chrono::system_clock::now();
    //             // projector.show();
    //             // auto runtime = std::chrono::system_clock::now() - start;
    //             // std::cout << "ms: "
    //             // << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
    //             // << "\n";
    //         }else{
    //             auto start = std::chrono::system_clock::now();
    //             // std::cout << "queue size: " << camera_queue.size() << "\n";
    //             CPylonImage pylonImage = camera_queue.pop();
    //             uint8_t* buffer = ( uint8_t*) pylonImage.GetBuffer();
    //             // std::cout << "Image popped !!! " << std::endl;
    //             cv::Mat myimage = cv::Mat(cam_height, cam_width, CV_8UC3, buffer);
    //             // std::cout << myimage.empty() << std::endl;
    //             // cv::imwrite("test1.png", myimage);
    //             cv::cvtColor(myimage, myimage, cv::COLOR_RGB2GRAY);
    //             cv::threshold(myimage, myimage, 50, 255, cv::THRESH_BINARY);
    //             cv::cvtColor(myimage, myimage, cv::COLOR_GRAY2RGB);
    //             cv::resize(myimage, myimage, cv::Size(projector.width, projector.height));
    //             projector.show_buffer(myimage.data);
    //             // projector.show();
    //             // free(buffer);
    //             auto runtime = std::chrono::system_clock::now() - start;
    //             std::cout << "ms: "
    //             << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
    //             << "\n";
    //         }
    //         //     continue;
    //         // }
    //         // else
    //         // {
    //         //     if (flag == true || camera_queue.size() == 0)
    //         //     {
    //                 // image = cv::Mat::ones(cv::Size(1024, 768), CV_8UC3);
    //             // }
    //             // else
    //             // {
    //             //     image = camera_queue.pop();
    //             //     // cv::namedWindow("image", cv::WINDOW_AUTOSIZE );
    //             //     // cv::imshow("image", image);
    //             //     // cv::waitKey(1);
    //             // }
    //         // flag = !flag;
    //         // }
            
    //     }
    //     std::cout << "Consumer finish" << std::endl;
    // });
    /* main thread */
    // while (!close_signal)
    // {
    //     std::string userInput;
    //     std::getline(std::cin, userInput);
    //     for (size_t i = 0; i < userInput.size(); ++i)
    //     {
    //         char key = userInput[i];
    //         if (key == 'q')
    //         {
    //             close_signal = true;
    //             break;
    //         }
    //     }
    // }
    // consumer.join();
    return 0;
}

unsigned int setup_buffers()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // glGenBuffers(1, &PBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    return VAO;
}

void saveImage(char* filepath, GLFWwindow* w) {
    int width, height;
    glfwGetFramebufferSize(w, &width, &height);
    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    // stbi_flip_vertically_on_write(true);
    // stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
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