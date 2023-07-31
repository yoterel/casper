#include <iostream>
#include <thread>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "queue.h"
#include "camera.h"
#include "gl_camera.h"
#include "display.h"
#include "SerialPort.h"
#include "shader.h"
#include "skinned_shader.h"
#include "skinned_model.h"
#include "timer.h"
#include "leap.h"
#include "text.h"
#include "canvas.h"
#include "utils.h"

// forward declarations
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void getLeapFrame(LeapConnect& leap, const int64_t& targetFrameTime, std::vector<glm::mat4>& bones_to_world, std::vector<glm::vec3>& skeleton_vertices, bool debug);
void setup_skeleton_hand_buffers(unsigned int& VAO, unsigned int& VBO);
void setup_gizmo_buffers(unsigned int& VAO, unsigned int& VBO);
// unsigned int setup_cube_buffers();
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO);

// settings
bool debug_mode = false;
bool use_cuda = true;
const unsigned int proj_width = 1024;
const unsigned int proj_height = 768;
const unsigned int cam_height = 540;
const unsigned int cam_width = 720;
// "fixed" camera
GLCamera gl_camera(glm::vec3(41.64f, 26.92f, -2.48f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f,0.0f,0.0f));
// "orbit" camera
// GLCamera gl_camera(glm::vec3(0.0f, -20.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));

// init
float lastX = proj_width / 2.0f;
float lastY = proj_height / 2.0f;
bool firstMouse = true;
double deltaTime = 0.0;
double lastFrame = 0.0;
unsigned int fps = 0;
float ms_per_frame = 0;
unsigned int displayBoneIndex = 0;
bool space_pressed_flag = false;
unsigned int n_bones = 0;
glm::mat4 cur_palm_orientation = glm::mat4(1.0f);
bool hand_in_frame = false;
const unsigned int image_size = proj_width * proj_height * 3;

int main( int /*argc*/, char* /*argv*/[] )
{
    Timer t0, t1, t2, t3, t4, t_app;
    t_app.start();
    // init GLFW
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
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glPointSize(10.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  // callback for resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // setup buffers
    unsigned int skeletonVAO, skeletonVBO;
    setup_skeleton_hand_buffers(skeletonVAO, skeletonVBO);
    unsigned int gizmoVAO, gizmoVBO;
    setup_gizmo_buffers(gizmoVAO, gizmoVBO);
    // unsigned int circleVAO, circleVBO;
    // setup_circle_buffers(circleVAO, circleVBO);
    SkinnedModel skinnedModel("C:/src/augmented_hands/resource/GenericHand.fbx", "C:/src/augmented_hands/resource/uv.png");
    Canvas canvas(cam_width, cam_height, proj_width, proj_height, use_cuda);
    n_bones = skinnedModel.NumBones();
    glm::vec3 coa = skinnedModel.getCenterOfMass();
    glm::mat4 coa_transform = glm::translate(glm::mat4(1.0f), -coa);
    glm::mat4 mm_to_cm = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    glm::mat4 cm_to_mm = glm::inverse(mm_to_cm);
    glm::mat4 timesTwenty = glm::scale(glm::mat4(1.0f), glm::vec3(20.0f, 20.0f, 20.0f));
    glm::mat4 rotx = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f,0.0f,0.0f));
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f,1.0f,0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    Text text("C:/src/augmented_hands/resource/arial.ttf");
    // setup shaders
    Shader canvasShader;
    if (use_cuda)
        canvasShader = Shader("C:/src/augmented_hands/src/shaders/canvas.vs", "C:/src/augmented_hands/src/shaders/canvas_cuda.fs");
    else
        canvasShader = Shader("C:/src/augmented_hands/src/shaders/canvas.vs", "C:/src/augmented_hands/src/shaders/canvas.fs");
    Shader vcolorShader("C:/src/augmented_hands/src/shaders/color_by_vertex.vs", "C:/src/augmented_hands/src/shaders/color_by_vertex.fs");
    Shader textShader("C:/src/augmented_hands/src/shaders/text.vs", "C:/src/augmented_hands/src/shaders/text.fs");
    textShader.use();
    glm::mat4 orth_projection_transform = glm::ortho(0.0f, static_cast<float>(proj_width), 0.0f, static_cast<float>(proj_height));
    textShader.setMat4("projection", orth_projection_transform);
    SkinningShader skinnedShader("C:/src/augmented_hands/src/shaders/skin_hand.vs", "C:/src/augmented_hands/src/shaders/skin_hand.fs");  
    // more inits
    double previousTime = glfwGetTime();
    double currentFrame = glfwGetTime();
    double whole = 0.0;
    int frameCount = 0;
    int64_t targetFrameTime = 0;
    uint64_t targetFrameSize = 0;
    std::vector<glm::vec3> skeleton_vertices;
    std::vector<glm::mat4> bones_to_world;
    size_t n_skeleton_primitives = 0;
    bool close_signal = false;
    bool use_pbo = false;
    int leap_time_delay = 50000;  // us
    bool producer_is_fake = false;
    uint8_t* colorBuffer = new uint8_t[image_size];
    uint32_t cam_height = 0;
    uint32_t cam_width = 0;
    blocking_queue<CPylonImage> camera_queue;
    blocking_queue<uint8_t*> projector_queue;
    BaslerCamera camera;
    DynaFlashProjector projector(proj_width, proj_height);
    if (!projector.init()) {
        std::cerr << "Failed to initialize projector\n";
    }
    LeapConnect leap;
    LEAP_CLOCK_REBASER clockSynchronizer;
    LeapCreateClockRebaser(&clockSynchronizer);
    std::thread producer, consumer;
    // actual thread loops
    // image producer
    if (producer_is_fake) {
        /* fake producer */
        cam_height = 540;
        cam_width = 720;
        producer = std::thread([&camera_queue, &close_signal, &cam_height, &cam_width]() {  //, &projector
            CPylonImage image = CPylonImage::Create( PixelType_RGB8packed, cam_width, cam_height);
            Timer t_block;
            t_block.start();
            while (!close_signal) {
                camera_queue.push(image);
                while (t_block.getElapsedTimeInMicroSec() < 300.0)
                {
                }
                t_block.stop();
                t_block.start();
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
    // image consumer
    consumer = std::thread([&projector_queue, &projector, &close_signal]() {  //, &projector
        uint8_t* buffer;
        bool sucess;
        while (!close_signal) {
            sucess = projector_queue.pop_with_timeout(100, buffer);
            if (sucess)
                projector.show_buffer(buffer);
            // else
            //     std::cout << "Consumer timeout" << std::endl;
        }
        std::cout << "Consumer finish" << std::endl;
    });
    // main loop
    while(!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        currentFrame = glfwGetTime();
        std::modf(currentFrame, &whole);
        LeapUpdateRebase(clockSynchronizer, static_cast<int64_t>(whole), leap.LeapGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        frameCount++;
        // stats display
        if ( currentFrame - previousTime >= 1.0 )
        {
            // Display the frame count here any way you want.
            fps = frameCount;
            ms_per_frame = 1000.0f/frameCount;
            // std::cout << "avg ms: " << 1000.0f/frameCount<<" FPS: " << frameCount << std::endl;
            std::cout << "last wait_for_cam time: " << t0.averageLap() << std::endl;
            std::cout << "last Cam->GPU time: " << t1.averageLap() << std::endl;
            std::cout << "last Processing time: " << t2.averageLap() << std::endl;
            std::cout << "last GPU->CPU time: " << t3.averageLap() << std::endl;
            std::cout << "last project time: " << t4.averageLap() << std::endl;
            std::cout << "cam q size: " << camera_queue.size() << std::endl;
            std::cout << "proj q size: " << projector_queue.size() << std::endl;
            frameCount = 0;
            previousTime = currentFrame;
            t0.reset();
            t1.reset();
            t2.reset();
            t3.reset();
            t4.reset();
        }
        // input
        processInput(window);
        // render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        t0.start();
        CPylonImage pylonImage = camera_queue.pop();
        uint8_t* buffer = ( uint8_t*) pylonImage.GetBuffer();
        t0.stop();
        t1.start();
        canvas.Render(canvasShader, buffer);
        t1.stop();
        t2.start();
        std::modf(glfwGetTime(), &whole);
        LeapRebaseClock(clockSynchronizer, static_cast<int64_t>(whole), &targetFrameTime);
        getLeapFrame(leap, targetFrameTime, bones_to_world, skeleton_vertices, debug_mode);
        glm::mat4 view_transform = gl_camera.GetViewMatrix();
        glm::mat4 projection_transform = glm::perspective(glm::radians(gl_camera.Zoom), 1.0f, 1.0f, 500.0f); // (float)proj_width / (float)proj_height
        if (bones_to_world.size() > 0)
        {
            glm::mat4 LocalToWorld = bones_to_world[0] * rotx * coa_transform;
            if (debug_mode)
            {
                // draw skeleton vertices
                glBindBuffer(GL_ARRAY_BUFFER, skeletonVBO);
                glBufferData(GL_ARRAY_BUFFER,  sizeof(float)*skeleton_vertices.size(), skeleton_vertices.data(), GL_STATIC_DRAW);
                n_skeleton_primitives = skeleton_vertices.size();
                vcolorShader.use();
                vcolorShader.setMat4("projection", projection_transform);
                vcolorShader.setMat4("view", view_transform);
                vcolorShader.setMat4("model", mm_to_cm);
                glBindVertexArray(skeletonVAO);
                glDrawArrays(GL_LINES, 0, static_cast<int>(n_skeleton_primitives));
                // draws global coordinate system gizmo
                vcolorShader.setMat4("model", glm::mat4(1.0f));
                glBindVertexArray(gizmoVAO);
                glDrawArrays(GL_LINES, 0, 6);
                // draw circle oriented like hand palm from leap motion
                // glBindVertexArray(circleVAO);
                // vcolorShader.setMat4("model", bones_to_world[0]);
                // glDrawArrays(GL_TRIANGLE_FAN, 0, 52);
                // draw skeleton bones (as gizmos representing their local coordinate system)
                std::vector<glm::mat4> BoneToLocalTransforms;
                skinnedModel.GetLocalToBoneTransforms(BoneToLocalTransforms, true, true);
                // in bind pose
                for (unsigned int i = 0; i < BoneToLocalTransforms.size(); i++)
                {
                    vcolorShader.setMat4("model", LocalToWorld * BoneToLocalTransforms[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                // in leap motion pose
                for (unsigned int i = 0; i < bones_to_world.size(); i++)
                {
                    vcolorShader.setMat4("model", bones_to_world[i]);
                    glDrawArrays(GL_LINES, 0, 6);
                }
                // draw debug info
                glm::vec4 palm_normal_hom = cur_palm_orientation * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                glm::vec3 palm_normal(palm_normal_hom);
                palm_normal = glm::normalize(palm_normal);
                text.Render(textShader, std::format("palm normal: {:.02f}, {:.02f}, {:.02f}, {:.02f}", palm_normal.x, palm_normal.y, palm_normal.z, glm::l2Norm(palm_normal)), 25.0f, 25.0f, 0.25f, glm::vec3(1.0f, 1.0f, 1.0f));
            }
            // draw skinned mesh
            skinnedShader.use();
            skinnedShader.SetDisplayBoneIndex(displayBoneIndex);
            skinnedShader.SetWorldTransform(projection_transform * view_transform);
            skinnedModel.Render(skinnedShader, bones_to_world, LocalToWorld, (float)t_app.getElapsedTimeInSec());
        }
        if (debug_mode)
        {
            glm::vec3 cam_pos = gl_camera.GetPos();
            glm::vec3 cam_forward = glm::normalize(-cam_pos);
            text.Render(textShader, std::format("camera pos: {:.02f}, {:.02f}, {:.02f}", cam_pos.x, cam_pos.y, cam_pos.z), 25.0f, 50.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("camera forward: {:.02f}, {:.02f}, {:.02f}", cam_forward.x, cam_forward.y, cam_forward.z), 25.0f, 75.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
            text.Render(textShader, std::format("bone index: {}, id: {}", displayBoneIndex, skinnedModel.getBoneName(displayBoneIndex)), 25.0f, 100.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        text.Render(textShader, std::format("ms_per_frame: {:.02f}, fps: {}", ms_per_frame, fps), 25.0f, 125.0f, 0.25f, glm::vec3(1.0f, 0.0f, 0.0f));
        t2.stop();
        
        // send result to projector queue
        if (use_pbo) {  // todo: change to asynchronous read-back http://www.songho.ca/opengl/gl_pbo.html
            std::cout << "pbo not implemented yet" << std::endl;
        }else{
            t3.start();
            glReadBuffer(GL_FRONT);
            // glCheckError();
            glReadPixels(0, 0, proj_width, proj_height, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
            // glCheckError();
            t3.stop();
            t4.start();
            // auto projector_thread = std::thread([&projector, &colorBuffer]() {  //, &projector
            projector_queue.push(colorBuffer);
            // projector.show_buffer(colorBuffer);
            // });
            t4.stop();
            // stbi_flip_vertically_on_write(true);
            // int stride = 3 * proj_width;
            // stride += (stride % 4) ? (4 - stride % 4) : 0;
            // stbi_write_png("test.png", proj_width, proj_height, 3, colorBuffer, stride);
        }
        // swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // cleanup
    close_signal = true;
    consumer.join();
    projector.kill();
    camera.kill();
    glfwTerminate();
    delete[] colorBuffer;
    if (producer_is_fake)
    {
        producer.join();
    }
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
    return 0;
}

void setup_skeleton_hand_buffers(unsigned int& VAO, unsigned int& VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
    // positions         // colors
     0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // bottom right
    -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // bottom left
     0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // top 
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void setup_gizmo_buffers(unsigned int& VAO, unsigned int& VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
    // positions         // colors
     0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,   // X
     1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,   // X
     0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,   // Y
     0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,   // Y
     0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,   // Z
     0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,   // Z
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
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
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        gl_camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        space_pressed_flag = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE)
    {
        if (space_pressed_flag)
        {
            space_pressed_flag = false;
            displayBoneIndex = (displayBoneIndex + 1) % n_bones;
        }
    }
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

void getLeapFrame(LeapConnect& leap, const int64_t& targetFrameTime, std::vector<glm::mat4>& bones_to_world, std::vector<glm::vec3>& skeleton_vertices, bool debug)
{
    skeleton_vertices.clear();
    bones_to_world.clear();
    uint64_t targetFrameSize = 0;
    int leap_time_delay = 40000;  // us
    glm::mat4 roty = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f,1.0f,0.0f));
    glm::mat4 flip_y = glm::mat4(1.0f);
    flip_y[1][1] = -1.0f;
    glm::mat4 flip_z = glm::mat4(1.0f);
    flip_z[2][2] = -1.0f;
    glm::mat4 mm_to_cm = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    glm::mat4 cm_to_mm = glm::inverse(mm_to_cm);
    //Get the buffer size needed to hold the tracking data
    if(LeapGetFrameSize(*leap.getConnectionHandle(), targetFrameTime+leap_time_delay, &targetFrameSize) == eLeapRS_Success)
    {
        //Allocate enough memory
        LEAP_TRACKING_EVENT* interpolatedFrame = (LEAP_TRACKING_EVENT*)malloc((size_t)targetFrameSize);
        //Get the frame
        if(LeapInterpolateFrame(*leap.getConnectionHandle(), targetFrameTime+leap_time_delay, interpolatedFrame, targetFrameSize) == eLeapRS_Success)
        {
            //Use the data...
            // std::cout << "frame id: " << interpolatedFrame->tracking_frame_id << std::endl;
            // std::cout << "frame delay (us): " << (long long int)LeapGetNow() - interpolatedFrame->info.timestamp << std::endl;
            // std::cout << "frame hands: " << interpolatedFrame->nHands << std::endl;
            if (debug)
            {
                if (interpolatedFrame->nHands > 0)
                {
                    if (!hand_in_frame)
                    {
                        std::cout << "hand in frame" << std::endl;
                    }
                    hand_in_frame = true;
                }
                else
                {
                    if (hand_in_frame)
                    {
                        std::cout << "no hand in frame" << std::endl;
                    }
                    hand_in_frame = false;
                }
            }
            glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
            for(uint32_t h = 0; h < interpolatedFrame->nHands; h++)
            {
                LEAP_HAND* hand = &interpolatedFrame->pHands[h];
                if (hand->type == eLeapHandType_Right)
                    continue;
                glm::vec3 palm_pos = glm::vec3(hand->palm.position.x,
                                                hand->palm.position.y,
                                                hand->palm.position.z);
                glm::mat4 palm_orientation = glm::toMat4(glm::quat(hand->palm.orientation.w,
                                                                    hand->palm.orientation.x,
                                                                    hand->palm.orientation.y,
                                                                    hand->palm.orientation.z));
                
                palm_orientation = palm_orientation * flip_z * flip_y;
                cur_palm_orientation = palm_orientation;
                glm::mat4 palm_trans = glm::translate(glm::mat4(1.0f), palm_pos);
                // if (debug)
                // {
                //     bones_to_world.push_back(palm_trans*roty*palm_orientation);
                // }
                // else
                // {
                bones_to_world.push_back(mm_to_cm*palm_trans*palm_orientation*cm_to_mm);
                // }
                LEAP_VECTOR arm_j1 = hand->arm.prev_joint;
                LEAP_VECTOR arm_j2 = hand->arm.next_joint;
                skeleton_vertices.push_back(glm::vec3(arm_j1.x, arm_j1.y, arm_j1.z));
                skeleton_vertices.push_back(red);
                skeleton_vertices.push_back(glm::vec3(arm_j2.x, arm_j2.y, arm_j2.z));
                skeleton_vertices.push_back(red);
                glm::mat4 rot = glm::toMat4(glm::quat(hand->arm.rotation.w,
                                                    hand->arm.rotation.x,
                                                    hand->arm.rotation.y,
                                                    hand->arm.rotation.z
                                                    ));
                // rot = palm_orientation * rot;
                glm::vec3 translate = glm::vec3(arm_j1.x, arm_j1.y,arm_j1.z);
                glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
                bones_to_world.push_back(mm_to_cm*trans*rot*roty*flip_z*flip_y*cm_to_mm);
                for(uint32_t f = 0; f < 5; f++)
                {
                    LEAP_DIGIT finger = hand->digits[f];
                    for(uint32_t b = 0; b < 4; b++)
                    {
                        LEAP_VECTOR joint1 = finger.bones[b].prev_joint;
                        LEAP_VECTOR joint2 = finger.bones[b].next_joint;
                        skeleton_vertices.push_back(glm::vec3(joint1.x, joint1.y, joint1.z));
                        skeleton_vertices.push_back(red);
                        skeleton_vertices.push_back(glm::vec3(joint2.x, joint2.y, joint2.z));
                        skeleton_vertices.push_back(red);
                        glm::mat4 rot = glm::toMat4(glm::quat(finger.bones[b].rotation.w,
                                                    finger.bones[b].rotation.x,
                                                    finger.bones[b].rotation.y,
                                                    finger.bones[b].rotation.z));
                        glm::vec3 translate = glm::vec3(joint1.x, joint1.y, joint1.z);
                        glm::mat4 trans = glm::translate(glm::mat4(1.0f), translate);
                        bones_to_world.push_back(mm_to_cm*trans*rot*roty*flip_z*flip_y*cm_to_mm);
                    }
                }
            }
            //Free the allocated buffer when done.
            free(interpolatedFrame);
        }
    }
}
// void saveImage(char* filepath, GLFWwindow* w) {
//     int width, height;
//     glfwGetFramebufferSize(w, &width, &height);
//     GLsizei nrChannels = 3;
//     GLsizei stride = nrChannels * width;
//     stride += (stride % 4) ? (4 - stride % 4) : 0;
//     GLsizei bufferSize = stride * height;
//     std::vector<char> buffer(bufferSize);
//     glPixelStorei(GL_PACK_ALIGNMENT, 4);
//     glReadBuffer(GL_FRONT);
//     glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
//     // stbi_flip_vertically_on_write(true);
//     // stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
// }
    // // create transformation matrices
    // glm::mat4 canvas_model_mat = glm::mat4(1.0f);
    // glm::mat4 skeleton_model_mat = glm::mat4(1.0f);
    // glm::mat4 mesh_model_mat = glm::mat4(1.0f);
    // // model_mat = glm::rotate(model_mat, glm::radians(-55.0f), glm::vec3(0.5f, 1.0f, 0.0f));
    // mesh_model_mat = glm::scale(mesh_model_mat, glm::vec3(0.5f, 0.5f, 0.5f));
    // // glm::mat4 canvas_projection_mat = glm::ortho(0.0f, (float)proj_width, 0.0f, (float)proj_height, 0.1f, 100.0f);
    // // canvas_model_mat = glm::scale(canvas_model_mat, glm::vec3(0.75f, 0.75f, 1.0f));  // 2.0f, 2.0f, 2.0f
    // glm::mat4 view_mat = gl_camera.GetViewMatrix();
    // // glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  10.0f);
    // // glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    // // glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    // // view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    // // view_mat = glm::translate(view_mat, glm::vec3(0.0f, 0.0f, -3.0f));
    // // glm::mat4 perspective_projection_mat = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    // // glm::mat4 projection_mat = glm::ortho(0.0f, (float)proj_width, 0.0f, (float)proj_height, 0.1f, 100.0f);
    // // glm::mat4 projection_mat = glm::frustum(-(float)proj_width*0.5f, (float)proj_width*0.5f, -(float)proj_height*0.5f, (float)proj_height*0.5f, 0.1f, 100.0f);
    // // setup shader inputs
    // // float bg_thresh = 0.05f;
    // // canvasShader.use();
    // // canvasShader.setInt("camera_texture", 0);
    // // canvasShader.setFloat("threshold", bg_thresh);
    // glm::mat4 canvas_projection_mat = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

//     unsigned int setup_cube_buffers()
// {
//     // set up vertex data (and buffer(s)) and configure vertex attributes
//     // ------------------------------------------------------------------
//     // ------------------------------------------------------------------
//     float vertices[] = {
//     -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
//      0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
//      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//     -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
//     -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

//     -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
//      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
//      0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
//      0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
//     -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
//     -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

//     -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
//     -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//     -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
//     -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
//     -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
//     -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

//      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
//      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//      0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
//      0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
//      0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
//      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

//     -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
//      0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
//      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
//      0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
//     -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
//     -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

//     -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
//      0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
//      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
//      0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
//     -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
//     -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
//     };
//     unsigned int VBO, VAO;
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glBindVertexArray(VAO);

//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

//     // position attribute
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
//     // texture coord attribute
//     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
//     glEnableVertexAttribArray(1);
//     return VAO;
// }

// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO)
// {
//     std::vector<glm::vec3> vertices;
//     // std::vector<glm::vec3> colors;
//     float radius = 0.5f;
//     glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
//     int n_vertices = 50;
//     vertices.push_back(center);
//     vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     for (int i = 0; i <= n_vertices; i++)   {
//         float twicePI = 2*glm::pi<float>();
//         vertices.push_back(glm::vec3(center.x + (radius * cos(i * twicePI / n_vertices)),
//                                      center.y,
//                                      center.z + (radius * sin(i * twicePI / n_vertices))));
//         vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     }
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glBindVertexArray(VAO);

//     auto test = vertices.data();
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, 6*(n_vertices+2) * sizeof(float), vertices.data(), GL_STATIC_DRAW);

//     // position attribute
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
//     // color attribute
//     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
//     glEnableVertexAttribArray(1);
// }