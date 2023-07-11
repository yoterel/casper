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

// const char *vertexShaderSource = "#version 330 core\n"
//     "layout (location = 0) in vec3 aPos;\n"
//     "void main()\n"
//     "{\n"
//     "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
//     "}\0";
// const char *fragmentShaderSource = "#version 330 core\n"
//     "out vec4 FragColor;\n"
//     "void main()\n"
//     "{\n"
//     "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
//     "}\n\0";

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
    Shader ourShader("../../src/texture.vs", "../../src/texture.fs");
    // unsigned int shaderProgram = compile_shaders();
    unsigned int VAO = setup_buffers();

    // load and create a texture 
    // -------------------------
    unsigned int texture1, texture2;
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
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    unsigned char *data = stbi_load("../../resource/container.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    // texture 2
    // ---------
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    data = stbi_load("../../resource/awesomeface.png", &width, &height, &nrChannels, 0);
    if (data)
    {
        // note that the awesomeface.png has transparency and thus an alpha channel, so make sure to tell OpenGL the data type is of GL_RGBA
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    ourShader.use(); // don't forget to activate/use the shader before setting uniforms!
    ourShader.setInt("texture1", 0);
    ourShader.setInt("texture2", 1);

    glfwSwapInterval(0);  // do not sync to monitor
    glViewport(0, 0, proj_width, proj_height);
    glClearColor(0.5, 0.5, 0.5, 0); // glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, keycallback);
    double previousTime = glfwGetTime();
    int frameCount = 0;
    bool save_flag = true;
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
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
        if (save_flag) {
            saveImage("test.png", window);
            save_flag = false;
        }
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
    /* setup trigger */
    // char* portName = "\\\\.\\COM4";
    bool close_signal = false;
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
    blocking_queue<cv::Mat> camera_queue;
    /* consumer */
    auto consumer = std::thread([&camera_queue, &close_signal, &projector]() {  //, &projector
        bool flag = false;
        // projector.show(white_image);
        while (!close_signal) {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            if (flag == true || camera_queue.size() == 0) {
                // cv::Mat image;
                // bool success = camera_queue.pop_with_timeout(1, image);
                // auto start = std::chrono::system_clock::now();
                // projector.show();
                // auto runtime = std::chrono::system_clock::now() - start;
                // std::cout << "ms: "
                // << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
                // << "\n";
            }else{
                auto start = std::chrono::system_clock::now();
                // std::cout << "queue size: " << camera_queue.size() << "\n";
                cv::Mat image = camera_queue.pop();
                // cv::resize(image, image, cv::Size(projector.width, projector.height));
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
            // flag = !flag;
            // }
            
        }
        std::cout << "Consumer finish" << std::endl;
    });
    /* camera producer */
    // BaslerCamera camera(camera_queue, close_signal);
    // camera.acquire();
    /* fake producer */
    auto producer = std::thread([&camera_queue, &close_signal, &projector]() {  //, &projector
        cv::Mat white_image(projector.width, projector.height, CV_8UC3, cv::Scalar(255, 255, 255));
        while (!close_signal) {
            camera_queue.push(white_image);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout << "Producer finish" << std::endl;
    });
    /* main thread */
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
    producer.join();
    return 0;
}
// unsigned int compile_shaders()
// {
//     // vertex shader
//     unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
//     glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
//     glCompileShader(vertexShader);
//     // check for shader compile errors
//     int success;
//     char infoLog[512];
//     glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
//     if (!success)
//     {
//         glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
//         std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
//     }
//     // fragment shader
//     unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
//     glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
//     glCompileShader(fragmentShader);
//     // check for shader compile errors
//     glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
//     if (!success)
//     {
//         glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
//         std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
//     }
//     // link shaders
//     unsigned int shaderProgram = glCreateProgram();
//     glAttachShader(shaderProgram, vertexShader);
//     glAttachShader(shaderProgram, fragmentShader);
//     glLinkProgram(shaderProgram);
//     // check for linking errors
//     glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
//     if (!success) {
//         glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
//         std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
//     }
//     glDeleteShader(vertexShader);
//     glDeleteShader(fragmentShader);
//     return shaderProgram;
// }

unsigned int setup_buffers()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

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
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
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