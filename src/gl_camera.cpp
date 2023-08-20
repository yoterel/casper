#include "gl_camera.h"

GLCamera::GLCamera(glm::vec3 position, glm::vec3 up) : MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), mode(Camera_Mode::ORBIT_CAMERA)
{

    Front = -glm::normalize(position);
    // Pitch = asin(Front.y);
    // Yaw = atan2(Front.x, Front.z);
    Position = position;
    WorldUp = up;
    projectionMatrix = glm::perspective(glm::radians(Zoom), 1.0f, 1.0f, 500.0f);
    updateCameraVectors();
}
GLCamera::GLCamera(glm::vec3 position, glm::vec3 up, glm::vec3 front) : MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), mode(Camera_Mode::FIXED_CAMERA)
{

    Front = glm::normalize(front);
    // Pitch = asin(Front.y);
    // Yaw = atan2(Front.x, Front.z);
    Position = position;
    WorldUp = up;
    Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));
    updateCameraVectors();
    viewMatrix = glm::lookAt(Position, Position + Front, Up);
    projectionMatrix = glm::perspective(glm::radians(this->Zoom), 1.0f, 1.0f, 500.0f);
}
GLCamera::GLCamera(glm::mat4 world2local, glm::mat4 projection) : MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), mode(Camera_Mode::FIXED_CAMERA)
{
    // glm::mat4 flipYZ = glm::mat4(1.0f);
    // flipYZ[1][1] = -1.0f;
    // flipYZ[2][2] = -1.0f;
    // glm::mat4 openglMatrix = flipYZ * world2local; // negate Y and Z columns (glm uses column major so need to left multiply)
    world2local[3][0] *= 0.1f;
    world2local[3][1] *= 0.1f;
    world2local[3][2] *= 0.1f;
    // viewMatrix = glm::transpose(openglMatrix);
    viewMatrix = world2local;
    projectionMatrix = projection;
    // std::cout << "OpenCV matrix: " << std::endl;
    // Position = glm::vec3(openglMatrix[0][3] / 10, openglMatrix[1][3] / 10, openglMatrix[2][3] / 10);
    // Right = glm::vec3(openglMatrix[0][0], openglMatrix[1][0], openglMatrix[2][0]);
    // Up = glm::vec3(-openglMatrix[0][1], -openglMatrix[1][1], -openglMatrix[2][1]);
    // Front = glm::vec3(-openglMatrix[0][2], -openglMatrix[1][2], -openglMatrix[2][2]);
}
GLCamera::GLCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), mode(Camera_Mode::FREE_CAMERA)
{
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    projectionMatrix = glm::perspective(glm::radians(Zoom), 1.0f, 1.0f, 500.0f);
    updateCameraVectors();
}
// constructor with scalar values
GLCamera::GLCamera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM), mode(Camera_Mode::FREE_CAMERA)
{
    Position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    projectionMatrix = glm::perspective(glm::radians(Zoom), 1.0f, 1.0f, 500.0f);
    updateCameraVectors();
}

// returns the view matrix calculated using Euler Angles and the LookAt Matrix
glm::mat4 GLCamera::getViewMatrix()
{
    if (mode == Camera_Mode::FIXED_CAMERA)
    {
        return viewMatrix;
    }
    else
    {
        return glm::lookAt(Position, Position + Front, Up);
    }
}

glm::mat4 GLCamera::getProjectionMatrix()
{
    return projectionMatrix;
}

glm::vec3 GLCamera::getPos()
{
    if (mode == Camera_Mode::FIXED_CAMERA)
    {
        glm::mat4 local2world = glm::inverse(viewMatrix);
        return glm::vec3(local2world[3][0], local2world[3][1], local2world[3][2]);
    }
    else
    {
        return Position;
    }
}
// processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
void GLCamera::processKeyboard(Camera_Movement direction, float deltaTime)
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

    updateCameraVectors();
    // std::cout << "Cam pos: " << Position.x << ", " << Position.y << ", " << Position.z << std::endl;
    // std::cout << "Cam forward: " << Front.x << ", " << Front.y << ", " << Front.z << std::endl;
}

// processes input received from a mouse input system. Expects the offset value in both the x and y direction.
void GLCamera::processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
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
void GLCamera::processMouseScroll(float yoffset)
{
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 60.0f)
        Zoom = 60.0f;
    projectionMatrix = glm::perspective(glm::radians(Zoom), 1.0f, 1.0f, 500.0f);
}
// calculates the front vector from the Camera's (updated) Euler Angles
void GLCamera::updateCameraVectors()
{
    switch (mode)
    {
    case Camera_Mode::ORBIT_CAMERA:
    {
        Front = -glm::normalize(Position);
        Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
        break;
    }
    case Camera_Mode::FREE_CAMERA:
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
        break;
    }
    case Camera_Mode::FIXED_CAMERA:
    {
        break;
    }
    default:
        break;
    }
}
