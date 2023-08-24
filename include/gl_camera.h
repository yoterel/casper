#ifndef Gl_CAMERA_H
#define Gl_CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

enum Camera_Mode
{
    FREE_CAMERA,
    ORBIT_CAMERA,
    FIXED_CAMERA
};
// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 10.0f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class GLCamera
{
public:
    GLCamera(glm::vec3 eye, glm::vec3 at, glm::vec3 up, Camera_Mode mode, float width, float height);
    GLCamera(glm::mat4 world2local, glm::mat4 projection, Camera_Mode mode, float width, float height);
    GLCamera(glm::vec3 position, glm::vec3 up, glm::vec3 front, float width, float height);
    GLCamera(){};
    GLCamera(GLCamera &s)
    {
        Position = s.Position;
        Front = s.Front;
        Up = s.Up;
        Right = s.Right;
        WorldUp = s.WorldUp;
        Yaw = s.Yaw;
        Pitch = s.Pitch;
        MovementSpeed = s.MovementSpeed;
        MouseSensitivity = s.MouseSensitivity;
        Zoom = s.Zoom;
        m_mode = s.m_mode;
        viewMatrix = s.viewMatrix;
        projectionMatrix = s.projectionMatrix;
    }
    GLCamera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);
    glm::mat4 getViewMatrix();
    glm::mat4 getLocal2WorldMatrix();
    glm::mat4 getProjectionMatrix();
    glm::vec3 getPos();
    glm::vec3 getFront();
    void processKeyboard(Camera_Movement direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);
    void processMouseScroll(float yoffset);
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;
    float m_width;
    float m_height;

private:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    Camera_Mode m_mode;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    void updateCameraVectors();
};

#endif