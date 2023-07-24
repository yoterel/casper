#include "gl_camera.h"


GLCamera::GLCamera(glm::vec3 position, glm::vec3 up) :
MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    Front = -glm::normalize(position);
    // Pitch = asin(Front.y);
    // Yaw = atan2(Front.x, Front.z);
    Position = position;
    WorldUp = up;
    updateCameraVectors();
}
GLCamera::GLCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) :
Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}
// constructor with scalar values
GLCamera::GLCamera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) :
Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    Position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

// returns the view matrix calculated using Euler Angles and the LookAt Matrix
glm::mat4 GLCamera::GetViewMatrix()
{
    return glm::lookAt(Position, Position + Front, Up);
}

glm::vec3 GLCamera::GetPos()
{
    return Position;
}
// processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
void GLCamera::ProcessKeyboard(Camera_Movement direction, float deltaTime)
{
    float velocity = MovementSpeed * deltaTime;
    if (direction == FORWARD)
        Position += Front * velocity;
    if (direction == BACKWARD)
        Position -= Front * velocity;
    if (direction == LEFT)
        Position -= Right * velocity;
    if (direction == RIGHT)
        Position += Right * velocity;
    if (direction == UP)
        Position += Up * velocity;
    if (direction == DOWN)
        Position -= Up * velocity;
    Front = -glm::normalize(Position);
    updateCameraVectors();
    // std::cout << "Cam pos: " << Position.x << ", " << Position.y << ", " << Position.z << std::endl;
    // std::cout << "Cam forward: " << Front.x << ", " << Front.y << ", " << Front.z << std::endl;
}

// processes input received from a mouse input system. Expects the offset value in both the x and y direction.
void GLCamera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
    // xoffset *= MouseSensitivity;
    // yoffset *= MouseSensitivity;

    // Yaw   += xoffset;
    // Pitch += yoffset;

    // // make sure that when pitch is out of bounds, screen doesn't get flipped
    // if (constrainPitch)
    // {
    //     if (Pitch > 89.0f)
    //         Pitch = 89.0f;
    //     if (Pitch < -89.0f)
    //         Pitch = -89.0f;
    // }

    // // update Front, Right and Up Vectors using the updated Euler angles
    // updateCameraVectors();
}

// processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
void GLCamera::ProcessMouseScroll(float yoffset)
{
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 45.0f)
        Zoom = 45.0f;
}
// calculates the front vector from the Camera's (updated) Euler Angles
void GLCamera::updateCameraVectors()
{
    // calculate the new Front vector
    // glm::vec3 front;
    // front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    // front.y = sin(glm::radians(Pitch));
    // front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    // Front = glm::normalize(front);
    // also re-calculate the Right and Up vector
    Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up    = glm::normalize(glm::cross(Right, Front));
}
