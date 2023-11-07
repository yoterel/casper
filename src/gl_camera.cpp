#include "gl_camera.h"

GLCamera::GLCamera(glm::vec3 eye, glm::vec3 at, glm::vec3 up, Camera_Mode mode,
                   float width, float height, float far, float speed, bool inverted) : MovementSpeed(speed),
                                                                                       MouseSensitivity(DEF_SENSITIVITY),
                                                                                       Zoom(DEF_ZOOM),
                                                                                       Yaw(DEF_YAW),
                                                                                       Pitch(DEF_PITCH),
                                                                                       m_width(width),
                                                                                       m_height(height),
                                                                                       m_inverted(inverted)
{
    projectionMatrix = glm::perspective(glm::radians(Zoom), m_width / m_height, DEF_NEAR, far);
    m_mode = mode;
    if (m_mode == Camera_Mode::FIXED_CAMERA)
    {
        glm::vec3 front = glm::normalize(at - eye);
        Front = glm::normalize(front);
        Position = eye;
        WorldUp = up;
        Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
        viewMatrix = glm::lookAt(Position, Position + Front, Up);
    }
    else
    {
        glm::vec3 front = glm::normalize(at - eye);
        Pitch = glm::degrees(asin(front.y));
        Yaw = glm::degrees(atan2(front.z, front.x));
        Position = eye;
        WorldUp = up;
        updateCameraVectors();
    }
}
GLCamera::GLCamera(glm::vec3 position, glm::vec3 up, glm::vec3 front,
                   float width, float height, float far, float speed, bool inverted) : MovementSpeed(speed),
                                                                                       MouseSensitivity(DEF_SENSITIVITY),
                                                                                       Zoom(DEF_ZOOM),
                                                                                       m_mode(Camera_Mode::FIXED_CAMERA),
                                                                                       m_width(width),
                                                                                       m_height(height),
                                                                                       m_inverted(inverted)
{
    projectionMatrix = glm::perspective(glm::radians(Zoom), m_width / m_height, DEF_NEAR, far);
    Front = glm::normalize(front);
    // Pitch = asin(Front.y);
    // Yaw = atan2(Front.x, Front.z);
    Position = position;
    WorldUp = up;
    Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));
    viewMatrix = glm::lookAt(Position, Position + Front, Up);
}
GLCamera::GLCamera(glm::mat4 world2local, glm::mat4 projection, Camera_Mode mode,
                   float width, float height, float speed, bool inverted) : MovementSpeed(speed),
                                                                            MouseSensitivity(DEF_SENSITIVITY),
                                                                            m_mode(mode),
                                                                            m_width(width),
                                                                            m_height(height),
                                                                            m_inverted(inverted)
{
    projectionMatrix = projection;
    Zoom = glm::degrees(2 * atan(1.0f / projection[1][1]));
    // glm::mat4 flipYZ = glm::mat4(1.0f);
    // flipYZ[1][1] = -1.0f;
    // flipYZ[2][2] = -1.0f;
    // glm::mat4 openglMatrix = flipYZ * world2local; // negate Y and Z columns (glm uses column major so need to left multiply)
    // world2local[3][0] *= 0.1f;
    // world2local[3][1] *= 0.1f;
    // world2local[3][2] *= 0.1f;
    // viewMatrix = glm::transpose(openglMatrix);
    viewMatrix = world2local;
    glm::mat4 local2world = getLocal2WorldMatrix();
    glm::vec3 front = glm::vec3(-local2world[2][0], -local2world[2][1], -local2world[2][2]);
    Pitch = glm::degrees(asin(front.y));
    Yaw = glm::degrees(atan2(front.z, front.x));
    Position = glm::vec3(local2world[3][0], local2world[3][1], local2world[3][2]);
    WorldUp = glm::vec3(local2world[1][0], local2world[1][1], local2world[1][2]);
    updateCameraVectors();
    // std::cout << "OpenCV matrix: " << std::endl;
    // Position = glm::vec3(openglMatrix[0][3] / 10, openglMatrix[1][3] / 10, openglMatrix[2][3] / 10);
    // Right = glm::vec3(openglMatrix[0][0], openglMatrix[1][0], openglMatrix[2][0]);
    // Up = glm::vec3(-openglMatrix[0][1], -openglMatrix[1][1], -openglMatrix[2][1]);
    // Front = glm::vec3(-openglMatrix[0][2], -openglMatrix[1][2], -openglMatrix[2][2]);
}

// returns the view matrix calculated using Euler Angles and the LookAt Matrix
glm::mat4 GLCamera::getViewMatrix() // world to local
{
    return viewMatrix;
}

void GLCamera::setViewMatrix(glm::mat4 newViewMatrix) // world to local
{
    viewMatrix = newViewMatrix;
}

glm::mat4 GLCamera::getProjectionMatrix() // world to local
{
    return projectionMatrix;
}

void GLCamera::setProjectionMatrix(glm::mat4 newwProjectionMatrix) // world to local
{
    projectionMatrix = newwProjectionMatrix;
}

glm::mat4 GLCamera::getLocal2WorldMatrix()
{
    return glm::inverse(viewMatrix);
}

glm::vec3 GLCamera::getPos()
{
    glm::mat4 local2world = getLocal2WorldMatrix();
    return glm::vec3(local2world[3][0], local2world[3][1], local2world[3][2]);
}

glm::vec3 GLCamera::getFront()
{
    glm::mat4 local2world = getLocal2WorldMatrix();
    return glm::vec3(local2world[2][0], local2world[2][1], local2world[2][2]);
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
    if (m_mode == Camera_Mode::FREE_CAMERA)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        if (m_inverted)
        {
            xoffset *= -1;
            yoffset *= -1;
        }
        Yaw += xoffset;
        Pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }
}

// processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
void GLCamera::processMouseScroll(float yoffset)
{
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 60.0f)
        Zoom = 60.0f;
    projectionMatrix = glm::perspective(glm::radians(Zoom), m_width / m_height, DEF_NEAR, DEF_FAR);
}
// calculates the front vector from the Camera's (updated) Euler Angles
void GLCamera::updateCameraVectors()
{
    switch (m_mode)
    {
    case Camera_Mode::ORBIT_CAMERA:
    {
        Front = -glm::normalize(Position);
        Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
        viewMatrix = glm::lookAt(Position, Position + Front, Up);
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
        viewMatrix = glm::lookAt(Position, Position + Front, Up);
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
