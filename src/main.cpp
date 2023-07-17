#include <chrono>
#include <thread>
#include "queue.h"
#include "camera.h"
#include "gl_camera.h"
#include "display.h"
#include "SerialPort.h"
#include "shader.h"
#include "model.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "timer.h"
// #include "stb_image.h"
// #include "stb_image_write.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std::literals::chrono_literals;

/* settings */

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
// unsigned int compile_shaders();
unsigned int setup_cam_buffers();
unsigned int setup_model_buffers();
void saveImage(char* filepath, GLFWwindow* w);
// settings
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int image_size = proj_width * proj_height * 3;
// camera
GLCamera gl_camera(glm::vec3(0.0f, 0.0f, 2.41f));
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main( int /*argc*/, char* /*argv*/[] )
{
    Timer t0, t1, t2, t3, t4;  // t1.start(); t1.stop(); t1.getElapsedTimeInMilliSec();
    // render output window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    int num_of_monitors;
    GLFWmonitor **monitors = glfwGetMonitors(&num_of_monitors);
    GLFWwindow* window = glfwCreateWindow(proj_width, proj_height, "augmented_hands", NULL, NULL);  //monitors[0], NULL for full screen
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
    glViewport(0, 0, proj_width, proj_height);  // set viewport
    glEnable(GL_DEPTH_TEST);  // enable depth testing
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f); // glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  // callback for resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // setup buffers
    unsigned int camVAO = setup_cam_buffers();
    // setup 3d objects
    Model ourModel("C:/src/augmented_hands/resource/backpack/backpack.obj");
    // setup shaders
    Shader camShader("C:/src/augmented_hands/src/cam.vs", "C:/src/augmented_hands/src/cam.fs");
    Shader modelShader("C:/src/augmented_hands/src/model.vs", "C:/src/augmented_hands/src/model.fs");
    // unsigned int modelVAO = setup_model_buffers();
    //setup textures

    // create a texture 
    // -------------------------
    unsigned int camera_texture;
    // texture 1
    // ---------
    glGenTextures(1, &camera_texture);
    glBindTexture(GL_TEXTURE_2D, camera_texture); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // create transformation matrices
    glm::mat4 mesh_model_mat = glm::mat4(1.0f);
    // model_mat = glm::rotate(model_mat, glm::radians(-55.0f), glm::vec3(0.5f, 1.0f, 0.0f));
    mesh_model_mat = glm::scale(mesh_model_mat, glm::vec3(0.5f, 0.5f, 0.5f));
    glm::mat4 canvas_model_mat = glm::mat4(1.0f);
    // glm::mat4 canvas_projection_mat = glm::ortho(0.0f, (float)proj_width, 0.0f, (float)proj_height, 0.1f, 100.0f);
    // canvas_model_mat = glm::scale(canvas_model_mat, glm::vec3(0.75f, 0.75f, 1.0f));  // 2.0f, 2.0f, 2.0f
    glm::mat4 view_mat = glm::mat4(1.0f);
    // glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  10.0f);
    // glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    // glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    // view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    // view_mat = glm::translate(view_mat, glm::vec3(0.0f, 0.0f, -3.0f));
    glm::mat4 projection_mat = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    // glm::mat4 projection_mat = glm::frustum(-(float)proj_width*0.5f, (float)proj_width*0.5f, -(float)proj_height*0.5f, (float)proj_height*0.5f, 0.1f, 100.0f);
    // setup shader inputs
    float bg_thresh = 0.08f;
    camShader.use();
    camShader.setInt("camera_texture", 0);
    camShader.setFloat("threshold", bg_thresh);
    camShader.setMat4("model", canvas_model_mat);
    camShader.setMat4("view", view_mat);
    camShader.setMat4("projection", projection_mat);
    modelShader.use();
    modelShader.setMat4("model", mesh_model_mat);
    modelShader.setMat4("view", view_mat);
    modelShader.setMat4("projection", projection_mat);
    
    // initialize
    float previousTime = static_cast<float>(glfwGetTime());
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

    // actual thread loops
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
    // main loop
    while(!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        frameCount++;
        // If a second has passed.
        if ( currentFrame - previousTime >= 1.0 )
        {
            // Display the frame count here any way you want.
            std::cout << "avg ms: " << 1000.0f/frameCount<<" FPS: " << frameCount << std::endl;
            std::cout << "last wait_for_cam time: " << t0.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last Cam->GPU time: " << t1.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last GPU Process time: " << t2.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last GPU->CPU time: " << t3.getElapsedTimeInMilliSec() << std::endl;
            std::cout << "last project time: " << t4.getElapsedTimeInMilliSec() << std::endl;
            frameCount = 0;
            previousTime = currentFrame;
        }
        // input
        processInput(window);
        // render
        // ------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t0.start();
        CPylonImage pylonImage = camera_queue.pop();
        uint8_t* buffer = ( uint8_t*) pylonImage.GetBuffer();
        t0.stop();
        
        t2.start();
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, camera_texture);
        // load texture to GPU (large overhead)
        t1.start();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_width, cam_height, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        t1.stop();
        // view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(gl_camera.Zoom), (float)proj_width / (float)proj_height, 0.1f, 100.0f);
        glm::mat4 view = gl_camera.GetViewMatrix();
        camShader.use();
        camShader.setMat4("projection", projection);
        camShader.setMat4("view", view);
        // camShader.setMat4("view", view_mat);
        // camShader.setInt("camera_texture", 0);
        glBindVertexArray(camVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        modelShader.use();
        modelShader.setMat4("projection", projection);
        modelShader.setMat4("view", view);
        ourModel.Draw(modelShader);
        // render the loaded model
        // glm::mat4 model = glm::mat4(1.0f);
        // mesh_model_mat = glm::scale(mesh_model_mat, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
        // modelShader.setMat4("model", mesh_model_mat);
        // mesh_model_mat = glm::rotate(mesh_model_mat, glm::radians(0.1f), glm::vec3(0.5f, 1.0f, 0.0f));
        // modelShader.setMat4("model", mesh_model_mat);
        // modelShader.setMat4("view", view_mat);
        // glBindVertexArray(modelVAO);
        // glDrawArrays(GL_TRIANGLES, 0, 36);
        t2.stop();
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        if (use_pbo) {  // todo: change to PBO http://www.songho.ca/opengl/gl_pbo.html
            // saveImage("test.png", window);
            // save_flag = false;
        }else{
            t3.start();
            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
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

unsigned int setup_cam_buffers()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float scale = 1.0f;
    float vertices[] = {
        // positions          // colors           // texture coords
         scale,  scale, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 1.0f, // top right
         scale, -scale, 0.0f,   1.0f, 1.0f, 1.0f,   1.0f, 0.0f, // bottom right
        -scale, -scale, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -scale,  scale, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 1.0f  // top left 
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

unsigned int setup_model_buffers()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
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

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    gl_camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    gl_camera.ProcessMouseScroll(static_cast<float>(yoffset));
}