#ifndef Gl_CAMERA_H
#define Gl_CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
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
const float DEF_YAW = -90.0f;
const float DEF_PITCH = 0.0f;
const float DEF_SPEED = 2.0f;
const float DEF_SENSITIVITY = 0.1f;
const float DEF_ZOOM = 45.0f;
const float DEF_NEAR = 1.0f;
const float DEF_FAR = 2000.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class GLCamera
{
public:
    GLCamera(glm::vec3 eye, glm::vec3 at, glm::vec3 up, Camera_Mode mode, float width, float height, float far = DEF_FAR, float speed = DEF_SPEED, bool inverted = false);
    GLCamera(glm::mat4 world2local, glm::mat4 projection, Camera_Mode mode, float width, float height, float speed = DEF_SPEED, bool inverted = false);
    GLCamera(glm::vec3 position, glm::vec3 up, glm::vec3 front, float width, float height, float far = DEF_FAR, float speed = DEF_SPEED, bool inverted = false);
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
        m_width = s.m_width;
        m_height = s.m_height;
        m_inverted = s.m_inverted;
    }
    glm::mat4 getViewMatrix();
    void setViewMatrix(glm::mat4 newViewMatrix);
    glm::mat4 getLocal2WorldMatrix();
    glm::mat4 getProjectionMatrix();
    void setProjectionMatrix(glm::mat4 newProjectionMatrix);
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
    bool m_inverted;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    void calcCameraVectorsFromViewMatrix();
    void calcViewMatrixFromCameraVectors();
    void updateCameraVectors();
};

#endif