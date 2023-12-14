#include "leapCPP.h"

LeapCPP::LeapCPP(bool pollMode, bool with_images)
{
    OpenConnection();
    while (!IsConnected)
    {
        std::cout << "Leap: waiting for connection..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (with_images)
    {
        LeapSetPolicyFlags(connectionHandle, eLeapPolicyFlag_Images, 0);
    }
    else
    {
        LeapSetPolicyFlags(connectionHandle, 0, 0);
    }
    // LeapSetPolicyFlags(connectionHandle,
    //                    eLeapPolicyFlag_Images & eLeapPolicyFlag_MapPoints, 0);
    // LeapSetPolicyFlags(connectionHandle,
    //                    eLeapPolicyFlag_BackgroundFrames & eLeapPolicyFlag_Images, 0);
    LeapSetTrackingMode(connectionHandle, eLeapTrackingMode_ScreenTop); // eLeapTrackingMode_Desktop, eLeapTrackingMode_HMD, eLeapTrackingMode_ScreenTop
    // LeapRequestConfigValue();
    // LeapSaveConfigValue();
    m_poll = pollMode;
}

LeapCPP::~LeapCPP()
{
    kill();
}

void LeapCPP::setPollMode(bool pollMode)
{
    m_poll = pollMode;
}

void LeapCPP::setImageMode(bool imageMode)
{
    if (imageMode)
        LeapSetPolicyFlags(connectionHandle, eLeapPolicyFlag_Images, 0);
    else
        LeapSetPolicyFlags(connectionHandle, 0, eLeapPolicyFlag_Images);
}

/** Called by serviceMessageLoop() when a connection event is returned by LeapPollConnection(). */
void LeapCPP::handleConnectionEvent(const LEAP_CONNECTION_EVENT *connection_event)
{
    IsConnected = true;
    std::cout << "Leap: Connected." << std::endl;
}

/** Called by serviceMessageLoop() when a connection lost event is returned by LeapPollConnection(). */
void LeapCPP::handleConnectionLostEvent(const LEAP_CONNECTION_LOST_EVENT *connection_lost_event)
{
    IsConnected = false;
    std::cout << "leap: Disconnected." << std::endl;
}

void LeapCPP::handleDeviceEvent(const LEAP_DEVICE_EVENT *device_event)
{
    LEAP_DEVICE deviceHandle;
    // Open device using LEAP_DEVICE_REF from event struct.
    eLeapRS result = LeapOpenDevice(device_event->device, &deviceHandle);
    if (result != eLeapRS_Success)
    {
        std::cout << "Leap: Failed to open device " << ResultString(result) << std::endl;
        return;
    }

    // Create a struct to hold the device properties, we have to provide a buffer for the serial string
    LEAP_DEVICE_INFO deviceProperties = {sizeof(deviceProperties)};
    // Start with a length of 1 (pretending we don't know a priori what the length is).
    // Currently device serial numbers are all the same length, but that could change in the future
    deviceProperties.serial_length = 1;
    deviceProperties.serial = (char *)malloc(deviceProperties.serial_length);
    // This will fail since the serial buffer is only 1 character long
    //  But deviceProperties is updated to contain the required buffer length
    result = LeapGetDeviceInfo(deviceHandle, &deviceProperties);
    if (result == eLeapRS_InsufficientBuffer)
    {
        // try again with correct buffer size
        deviceProperties.serial = (char *)realloc(deviceProperties.serial, deviceProperties.serial_length);
        result = LeapGetDeviceInfo(deviceHandle, &deviceProperties);
        if (result != eLeapRS_Success)
        {
            std::cout << "Leap: Failed to get device info " << ResultString(result) << std::endl;
            free(deviceProperties.serial);
            return;
        }
    }
    setDevice(&deviceProperties);
    free(deviceProperties.serial);
    LeapCloseDevice(deviceHandle);
}

void LeapCPP::handlePolicyEvent(const LEAP_POLICY_EVENT *policy_event)
{
    std::cout << "Leap: Policy: " << policy_event->current_policy << std::endl;
}

void LeapCPP::handleConfigChangeEvent(const LEAP_CONFIG_CHANGE_EVENT *config_change_event)
{
    if (config_change_event->status)
    {
        std::cout << "Leap: Setting config request id: " << config_change_event->requestID << " was successfull." << std::endl;
    }
    else
    {
        std::cout << "Leap: Setting config request id: " << config_change_event->requestID << " failed." << std::endl;
    }
}
void LeapCPP::handleConfigResponseEvent(const LEAP_CONFIG_RESPONSE_EVENT *config_response_event)
{
    std::cout << "Leap: The config for request id: " << config_response_event->requestID << " is: " << config_response_event->value.strValue << std::endl;
}

void LeapCPP::handleTrackingEvent(const LEAP_TRACKING_EVENT *tracking_event)
{
    if (m_poll)
        setFrame(tracking_event); // support polling tracking data from different thread
}

void LeapCPP::handleTrackingModeEvent(const LEAP_TRACKING_MODE_EVENT *tracking_mode_event)
{
    std::cout << "Leap: Tracking mode is: " << tracking_mode_event->current_tracking_mode << std::endl;
}
void LeapCPP::handlePointMappingChangeEvent(const LEAP_POINT_MAPPING_CHANGE_EVENT *point_mapping_change_event)
{
    std::cout << "Leap: Point mapping change event received." << std::endl;
}

void LeapCPP::handleImageEvent(const LEAP_IMAGE_EVENT *imageEvent)
{
    setImage(imageEvent);
}

void LeapCPP::serviceMessageLoop()
{
    eLeapRS result;
    LEAP_CONNECTION_MESSAGE msg;
    while (_isRunning)
    {
        unsigned int timeout = 1000;
        result = LeapPollConnection(connectionHandle, timeout, &msg);

        if (result != eLeapRS_Success)
        {
            std::cout << "LeapC PollConnection call was:" << ResultString(result) << std::endl;
            continue;
        }
        // std::cout << "Leap: Message received: " << msg.type << std::endl;
        switch (msg.type)
        {
        case eLeapEventType_Connection:
            handleConnectionEvent(msg.connection_event);
            break;
        case eLeapEventType_ConnectionLost:
            handleConnectionLostEvent(msg.connection_lost_event);
            break;
        case eLeapEventType_Device:
            handleDeviceEvent(msg.device_event);
            break;
        case eLeapEventType_DeviceLost:
            // handleDeviceLostEvent(msg.device_event);
            break;
        case eLeapEventType_DeviceFailure:
            // handleDeviceFailureEvent(msg.device_failure_event);
            break;
        case eLeapEventType_Tracking:
            handleTrackingEvent(msg.tracking_event);
            break;
        case eLeapEventType_ImageComplete:
            // Ignore
            break;
        case eLeapEventType_ImageRequestError:
            // Ignore
            break;
        case eLeapEventType_LogEvent:
            // handleLogEvent(msg.log_event);
            break;
        case eLeapEventType_Policy:
            handlePolicyEvent(msg.policy_event);
            break;
        case eLeapEventType_ConfigChange:
            handleConfigChangeEvent(msg.config_change_event);
            break;
        case eLeapEventType_ConfigResponse:
            handleConfigResponseEvent(msg.config_response_event);
            break;
        case eLeapEventType_Image:
            handleImageEvent(msg.image_event);
            break;
        case eLeapEventType_PointMappingChange:
            handlePointMappingChangeEvent(msg.point_mapping_change_event);
            break;
        case eLeapEventType_TrackingMode:
            handleTrackingModeEvent(msg.tracking_mode_event);
            break;
        case eLeapEventType_LogEvents:
            // handleLogEvents(msg.log_events);
            break;
        case eLeapEventType_HeadPose:
            // handleHeadPoseEvent(msg.head_pose_event);
            break;
        case eLeapEventType_IMU:
            // handleImuEvent(msg.imu_event);
            break;
        default:
            // discard unknown message types
            std::cout << "Leap: Unhandled message type: " << msg.type << std::endl;
        } // switch on msg.type
    }
    // std::cout << "leap service loop finished." << std::endl;
}
void LeapCPP::OpenConnection(void)
{
    if (_isRunning)
    {
        return;
    }
    if (connectionHandle || LeapCreateConnection(NULL, &connectionHandle) == eLeapRS_Success)
    {
        eLeapRS result = LeapOpenConnection(connectionHandle);
        if (result == eLeapRS_Success)
        {
            _isRunning = true;
            // InitializeCriticalSection(&dataLock);
            pollingThread = std::thread(&LeapCPP::serviceMessageLoop, this);
        }
    }
    // LEAP_DEVICE_INFO* info = leap.GetDeviceProperties();
    // std::cout << "leap connected with serial: " << info->serial << std::endl;
}

void LeapCPP::CloseConnection(void)
{
    if (!_isRunning)
    {
        return;
    }
    _isRunning = false;
    LeapCloseConnection(connectionHandle);
    pollingThread.join();
}

void LeapCPP::kill(void)
{
    if (!_isRunning)
    {
        return;
    }
    CloseConnection();
    LeapDestroyConnection(connectionHandle);
    std::cout << "Leap: Killed." << std::endl;
}

const char *LeapCPP::ResultString(eLeapRS r)
{
    switch (r)
    {
    case eLeapRS_Success:
        return "eLeapRS_Success";
    case eLeapRS_UnknownError:
        return "eLeapRS_UnknownError";
    case eLeapRS_InvalidArgument:
        return "eLeapRS_InvalidArgument";
    case eLeapRS_InsufficientResources:
        return "eLeapRS_InsufficientResources";
    case eLeapRS_InsufficientBuffer:
        return "eLeapRS_InsufficientBuffer";
    case eLeapRS_Timeout:
        return "eLeapRS_Timeout";
    case eLeapRS_NotConnected:
        return "eLeapRS_NotConnected";
    case eLeapRS_HandshakeIncomplete:
        return "eLeapRS_HandshakeIncomplete";
    case eLeapRS_BufferSizeOverflow:
        return "eLeapRS_BufferSizeOverflow";
    case eLeapRS_ProtocolError:
        return "eLeapRS_ProtocolError";
    case eLeapRS_InvalidClientID:
        return "eLeapRS_InvalidClientID";
    case eLeapRS_UnexpectedClosed:
        return "eLeapRS_UnexpectedClosed";
    case eLeapRS_UnknownImageFrameRequest:
        return "eLeapRS_UnknownImageFrameRequest";
    case eLeapRS_UnknownTrackingFrameID:
        return "eLeapRS_UnknownTrackingFrameID";
    case eLeapRS_RoutineIsNotSeer:
        return "eLeapRS_RoutineIsNotSeer";
    case eLeapRS_TimestampTooEarly:
        return "eLeapRS_TimestampTooEarly";
    case eLeapRS_ConcurrentPoll:
        return "eLeapRS_ConcurrentPoll";
    case eLeapRS_NotAvailable:
        return "eLeapRS_NotAvailable";
    case eLeapRS_NotStreaming:
        return "eLeapRS_NotStreaming";
    case eLeapRS_CannotOpenDevice:
        return "eLeapRS_CannotOpenDevice";
    default:
        return "unknown result type.";
    }
}

void LeapCPP::setDevice(const LEAP_DEVICE_INFO *deviceProps)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (lastDevice)
    {
        free(lastDevice->serial);
    }
    else
    {
        lastDevice = (LEAP_DEVICE_INFO *)malloc(sizeof(*deviceProps));
    }
    *lastDevice = *deviceProps;
    lastDevice->serial = (char *)malloc(deviceProps->serial_length);
    memcpy(lastDevice->serial, deviceProps->serial, deviceProps->serial_length);
}

/** Returns a pointer to the cached device info. */
LEAP_DEVICE_INFO *LeapCPP::GetDeviceProperties()
{
    LEAP_DEVICE_INFO *currentDevice;
    std::lock_guard<std::mutex> guard(m_mutex);
    currentDevice = lastDevice;
    return currentDevice;
}

void LeapCPP::deepCopyTrackingEvent(LEAP_TRACKING_EVENT *dst, const LEAP_TRACKING_EVENT *src)
{
    memcpy(&dst->info, &src->info, sizeof(LEAP_FRAME_HEADER));
    dst->tracking_frame_id = src->tracking_frame_id;
    dst->nHands = src->nHands;
    dst->framerate = src->framerate;
    memcpy(dst->pHands, src->pHands, src->nHands * sizeof(LEAP_HAND));
}

void LeapCPP::setImage(const LEAP_IMAGE_EVENT *imageEvent)
{
    // std::cout << "Leap: Received image set for frame " << (long long int)imageEvent->info.frame_id << "with size " << (long long int)imageEvent->image[0].properties.width * (long long int)imageEvent->image[0].properties.height * 2 << std::endl;
    const LEAP_IMAGE_PROPERTIES *properties = &imageEvent->image[0].properties;
    if (properties->bpp != 1)
        return;
    std::lock_guard<std::mutex> guard(m_mutex);
    m_imageFrameID = imageEvent->info.frame_id;
    if (properties->width * properties->height != m_imageSize)
    {
        void *prev_image_buffer = m_imageBuffer;
        m_imageWidth = properties->width;
        m_imageHeight = properties->height;
        m_imageSize = m_imageWidth * m_imageHeight;
        m_imageBuffer = malloc(2 * m_imageSize);
        if (prev_image_buffer)
            free(prev_image_buffer);
        m_textureChanged = true;
    }

    memcpy(m_imageBuffer, (char *)imageEvent->image[0].data + imageEvent->image[0].offset, m_imageSize);
    memcpy((char *)m_imageBuffer + m_imageSize, (char *)imageEvent->image[1].data + imageEvent->image[1].offset, m_imageSize);
    // Save the distortion data if it's version id changes
    if (m_currentDistortionId != imageEvent->image[0].matrix_version)
    {
        size_t distortion_size = 64 * 64 * 2;
        distortion_buffer_left = malloc(sizeof(float) * distortion_size);
        distortion_buffer_right = malloc(sizeof(float) * distortion_size);
        memcpy(distortion_buffer_left, imageEvent->image[0].distortion_matrix, sizeof(float) * distortion_size);
        memcpy(distortion_buffer_right, imageEvent->image[1].distortion_matrix, sizeof(float) * distortion_size);
        m_currentDistortionId = imageEvent->image[0].matrix_version;
    }
    m_imageReady = true;
}

bool LeapCPP::getImage(std::vector<uint8_t> &image1, std::vector<uint8_t> &image2, uint32_t &width, uint32_t &height)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (m_imageReady)
    {
        std::vector<uint8_t> buffer1((char *)m_imageBuffer, (char *)m_imageBuffer + m_imageSize);
        std::vector<uint8_t> buffer2((char *)m_imageBuffer + m_imageSize, (char *)m_imageBuffer + 2 * m_imageSize);
        image1 = std::move(buffer1);
        image2 = std::move(buffer2);
        width = m_imageWidth;
        height = m_imageHeight;
    }
    else
    {
        width = 0;
        height = 0;
        return false;
    }
    return true;
}

bool LeapCPP::getDistortion(std::vector<float> &dist1, std::vector<float> &dist2, uint32_t &width, uint32_t &height)
{
    std::lock_guard<std::mutex> guard(m_mutex);
    size_t distortion_size = 64 * 64 * 2;
    if (m_imageReady)
    {
        std::vector<float> buffer1((float *)distortion_buffer_left, (float *)distortion_buffer_left + distortion_size);
        std::vector<float> buffer2((float *)distortion_buffer_right, (float *)distortion_buffer_right + distortion_size);
        dist1 = std::move(buffer1);
        dist2 = std::move(buffer2);
        width = 64;
        height = 64;
    }
    else
    {
        width = 0;
        height = 0;
        return false;
    }
    return true;
}
/**
 * Caches the newest frame by copying the tracking event struct returned by
 * LeapC.
 */
void LeapCPP::setFrame(const LEAP_TRACKING_EVENT *frame)
{
    // LockMutex(&dataLock);
    // if (!lastFrame)
    //     lastFrame = (LEAP_TRACKING_EVENT *)malloc(sizeof(*frame));
    // *lastFrame = *frame;
    // UnlockMutex(&dataLock);
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!lastFrame)
    {
        lastFrame = (LEAP_TRACKING_EVENT *)malloc(sizeof(LEAP_TRACKING_EVENT));
        lastFrame->pHands = (LEAP_HAND *)malloc(2 * sizeof(LEAP_HAND));
    }

    if (frame != NULL)
    {
        deepCopyTrackingEvent(lastFrame, frame);
    }
}

LEAP_TRACKING_EVENT *LeapCPP::getFrame()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    if (lastFrame == NULL)
    {
        return NULL;
    }
    LEAP_TRACKING_EVENT *currentFrame = NULL;
    currentFrame = (LEAP_TRACKING_EVENT *)malloc(sizeof(LEAP_TRACKING_EVENT));
    currentFrame->pHands = (LEAP_HAND *)malloc(lastFrame->nHands * sizeof(LEAP_HAND));
    deepCopyTrackingEvent(currentFrame, lastFrame);
    return currentFrame;
    // LEAP_TRACKING_EVENT *currentFrame = NULL;
    // LockMutex(&dataLock);
    // currentFrame = lastFrame;
    // UnlockMutex(&dataLock);
    // return currentFrame;
}

std::vector<float> LeapCPP::getIndexTip()
{
    // todo: deep copy frame instead of assigning pointers
    LEAP_TRACKING_EVENT *currentFrame;
    std::lock_guard<std::mutex> guard(m_mutex);
    currentFrame = lastFrame;
    std::vector<float> tip;
    if (NULL == currentFrame)
        return tip;
    for (uint32_t h = 0; h < currentFrame->nHands; h++)
    {
        LEAP_HAND *hand = &currentFrame->pHands[h];
        if (hand->type == eLeapHandType_Left)
            continue;
        LEAP_DIGIT *finger = &hand->digits[1];
        LEAP_BONE *bone = &finger->bones[3];
        tip.push_back(bone->next_joint.x);
        tip.push_back(bone->next_joint.y);
        tip.push_back(bone->next_joint.z);
    }
    return tip;
}

// glm::vec3 LeapCPP::triangulate(const glm::vec2 &leap1,
//                                    const glm::vec2 &leap2,
//                                    const int leap_width, const int leap_height)
// {
//     // leap image plane is x right, and y up like opengl...
//     glm::vec2 l1_vert = Helpers::NDCtoScreen(leap1, leap_width, leap_height, false);
//     LEAP_VECTOR l1_vert_leap = {l1_vert.x, l1_vert.y, 1.0f};
//     glm::vec2 l2_vert = Helpers::NDCtoScreen(leap2, leap_width, leap_height, false);
//     LEAP_VECTOR l2_vert_leap = {l2_vert.x, l2_vert.y, 1.0f};
//     LEAP_VECTOR leap1_rays_2d = LeapPixelToRectilinear(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, l1_vert_leap);
//     LEAP_VECTOR leap2_rays_2d = LeapPixelToRectilinear(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, l2_vert_leap);
//     // glm::mat4 leap1_extrinsic = glm::mat4(1.0f);
//     // glm::mat4 leap2_extrinsic = glm::mat4(1.0f);
//     // LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, glm::value_ptr(leap1_extrinsic));
//     // LeapExtrinsicCameraMatrix(*leap.getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, glm::value_ptr(leap2_extrinsic));
//     float baseline = 40.0f;
//     // see https://forums.leapmotion.com/t/sdk-2-1-raw-data-get-pixel-position-xyz/1604/12
//     float z = baseline / (leap2_rays_2d.x - leap1_rays_2d.x);
//     float alty1 = z * leap2_rays_2d.y;
//     float alty2 = z * leap1_rays_2d.y;
//     float x = z * leap2_rays_2d.x - baseline / 2.0f;
//     // float altx = z * leap1_rays_2d.x + baseline / 2.0f;
//     float y = (alty1 + alty2) / 2.0f;
//     glm::vec3 point_3d = glm::vec3(x, -z, y);
//     return point_3d;
// }
// std::vector<glm::vec3> LeapCPP::triangulate(const std::vector<glm::vec2> &leap1,
//                                                 const std::vector<glm::vec2> &leap2,
//                                                 const int leap_width, const int leap_height)
// {
//     // extract 3d points from leap1_verts and leap2_verts
//     // first get rays from the leap camera corrected for distortion, in 2D camera space
//     if (leap1.size() != leap2.size())
//     {
//         std::cout << "leap1 and leap2 must be the same size" << std::endl;
//         exit(1);
//     }
//     std::vector<LEAP_VECTOR> leap1_rays_2d, leap2_rays_2d;
//     for (int i = 0; i < leap1.size(); i++)
//     {
//         glm::vec2 l1_vert = Helpers::NDCtoScreen(leap1[i], leap_width, leap_height, false);
//         LEAP_VECTOR l1_vert_leap = {l1_vert.x, l1_vert.y, 1.0f};
//         glm::vec2 l2_vert = Helpers::NDCtoScreen(leap2[i], leap_width, leap_height, false);
//         LEAP_VECTOR l2_vert_leap = {l2_vert.x, l2_vert.y, 1.0f};
//         // LEAP_VECTOR l1_verts = {leap_width * (leap1_verts[i].x + 1) / 2, leap_height * (leap1_verts[i].y + 1) / 2, 1.0f};
//         // LEAP_VECTOR l2_verts = {leap_width * (leap2_verts[i].x + 1) / 2, leap_height * (leap2_verts[i].y + 1) / 2, 1.0f};
//         LEAP_VECTOR l1_ray = LeapPixelToRectilinear(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, l1_vert_leap);
//         leap1_rays_2d.push_back(l1_ray);
//         LEAP_VECTOR l2_ray = LeapPixelToRectilinear(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, l2_vert_leap);
//         leap2_rays_2d.push_back(l2_ray);
//     }
//     // second convert rays to 3D leap space using the extrinsics matrix
//     glm::mat4 leap1_extrinsic = glm::mat4(1.0f);
//     glm::mat4 leap2_extrinsic = glm::mat4(1.0f);
//     LeapExtrinsicCameraMatrix(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_left, glm::value_ptr(leap1_extrinsic));
//     LeapExtrinsicCameraMatrix(*getConnectionHandle(), eLeapPerspectiveType::eLeapPerspectiveType_stereo_right, glm::value_ptr(leap2_extrinsic));
//     /*
//     std::vector<cv::Point3f> leap1_ray_dirs_3d, leap2_ray_dirs_3d;
//     cv::Point3f leap1_origin = {leap1_extrinsic[3][0], leap1_extrinsic[3][1], leap1_extrinsic[3][2]};
//     cv::Point3f leap2_origin = {leap2_extrinsic[3][0], leap2_extrinsic[3][1], leap2_extrinsic[3][2]};
//     for (int i = 0; i < leap1_verts.size(); i++)
//     {
//         glm::vec4 leap1_ray_3d = glm::inverse(leap1_extrinsic) * glm::vec4(leap1_rays_2d[i].x, leap1_rays_2d[i].y, leap1_rays_2d[i].z, 1.0f);
//         leap1_ray_3d.x /= leap1_ray_3d.z;
//         leap1_ray_3d.y /= leap1_ray_3d.z;
//         leap1_ray_dirs_3d.push_back(cv::Point3f(leap1_ray_3d.x, leap1_ray_3d.y, 1.0f));
//         glm::vec4 leap2_ray_3d = glm::inverse(leap2_extrinsic) * glm::vec4(leap2_rays_2d[i].x, leap2_rays_2d[i].y, leap2_rays_2d[i].z, 1.0f);
//         leap2_ray_3d.x /= leap2_ray_3d.z;
//         leap2_ray_3d.y /= leap2_ray_3d.z;
//         leap2_ray_dirs_3d.push_back(cv::Point3f(leap2_ray_3d.x, leap2_ray_3d.y, 1.0f));
//     }
//     // triangulate the 3D rays to get the 3D points
//     for (int i = 0; i < leap2_ray_dirs_3d.size(); i++)
//     {
//         cv::Point3f point = approximate_ray_intersection(leap1_ray_dirs_3d[i], leap1_origin, leap2_ray_dirs_3d[i], leap2_origin, NULL, NULL, NULL);
//         object_points.push_back(point);
//     }
//     */
//     std::vector<glm::vec3> points;
//     float baseline = leap2_extrinsic[3][0] - leap1_extrinsic[3][0];
//     for (int i = 0; i < leap1_rays_2d.size(); i++)
//     {
//         // see https://forums.leapmotion.com/t/sdk-2-1-raw-data-get-pixel-position-xyz/1604/12
//         float z = baseline / (leap2_rays_2d[i].x - leap1_rays_2d[i].x);
//         float alty1 = z * -leap2_rays_2d[i].y; // reason for negative is that the y direction is flipped in openGL
//         float alty2 = z * -leap1_rays_2d[i].y; // reason for negative is that the y direction is flipped in openGL
//         float x = z * leap2_rays_2d[i].x - baseline / 2.0f;
//         float altx = z * leap1_rays_2d[i].x + baseline / 2.0f;
//         float y = (alty1 + alty2) / 2.0f;
//         points.push_back(glm::vec3(x, -z, y));
//         // obj_to_save.push_back(x);
//         // obj_to_save.push_back(-z);
//         // obj_to_save.push_back(y);
//     }
//     return points;
// }

// std::vector<float> LeapCPP::getFrame()
// {
//     LEAP_TRACKING_EVENT *currentFrame;

//     LockMutex(&dataLock);
//     currentFrame = lastFrame;
//     UnlockMutex(&dataLock);
//     std::vector<float> frame;
//     for (uint32_t h = 0; h < currentFrame->nHands; h++)
//     {
//         LEAP_HAND *hand = &currentFrame->pHands[h];
//         if (hand->type == eLeapHandType_Right)
//             continue;
//         for (uint32_t f = 0; f < 5; f++)
//         {
//             LEAP_DIGIT *finger = &hand->digits[f];
//             for (uint32_t b = 0; b < 4; b++)
//             {
//                 LEAP_BONE *bone = &finger->bones[b];
//                 frame.push_back(bone->prev_joint.x);
//                 frame.push_back(bone->prev_joint.y);
//                 frame.push_back(bone->prev_joint.z);
//                 // frame->push_back(bone->next_joint.x);
//                 // frame->push_back(bone->next_joint.y);
//                 // frame->push_back(bone->next_joint.z);
//             }
//         }
//     }
//     return frame;
// }
