#include "leap.h"

/** Called by serviceMessageLoop() when a connection event is returned by LeapPollConnection(). */
void LeapConnect::handleConnectionEvent(const LEAP_CONNECTION_EVENT *connection_event)
{
  IsConnected = true;
  std::cout << "leap connected." << std::endl;
}

/** Called by serviceMessageLoop() when a connection lost event is returned by LeapPollConnection(). */
void LeapConnect::handleConnectionLostEvent(const LEAP_CONNECTION_LOST_EVENT *connection_lost_event)
{
  IsConnected = false;
}

void LeapConnect::handlePolicyEvent(const LEAP_POLICY_EVENT *policy_event)
{
  std::cout << "Policy: " << policy_event->current_policy << std::endl;
}

void LeapConnect::handleConfigChangeEvent(const LEAP_CONFIG_CHANGE_EVENT *config_change_event)
{
  if (config_change_event->status)
  {
    std::cout << "Setting config request id: " << config_change_event->requestID << " was successfull." << std::endl;
  }
  else
  {
    std::cout << "Setting config request id: " << config_change_event->requestID << " failed." << std::endl;
  }
}
void LeapConnect::handleConfigResponseEvent(const LEAP_CONFIG_RESPONSE_EVENT *config_response_event)
{
  std::cout << "The config for request id: " << config_response_event->requestID << " is: " << config_response_event->value.strValue << std::endl;
}

void LeapConnect::handleImageEvent(const LEAP_IMAGE_EVENT *imageEvent)
{
  std::cout << "Received image set for frame " << (long long int)imageEvent->info.frame_id << "with size " << (long long int)imageEvent->image[0].properties.width * (long long int)imageEvent->image[0].properties.height * 2 << std::endl;
}

void LeapConnect::handleDeviceEvent(const LEAP_DEVICE_EVENT *device_event)
{
  LEAP_DEVICE deviceHandle;
  // Open device using LEAP_DEVICE_REF from event struct.
  eLeapRS result = LeapOpenDevice(device_event->device, &deviceHandle);
  if (result != eLeapRS_Success)
  {
    printf("Could not open device %s.\n", ResultString(result));
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
      printf("Failed to get device info %s.\n", ResultString(result));
      free(deviceProperties.serial);
      return;
    }
  }
  setDevice(&deviceProperties);
  free(deviceProperties.serial);
  LeapCloseDevice(deviceHandle);
}

void LeapConnect::handleTrackingEvent(const LEAP_TRACKING_EVENT *tracking_event)
{
  if (m_poll)
    setFrame(tracking_event); // support polling tracking data from different thread
}

void LeapConnect::serviceMessageLoop()
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
      // handlePointMappingChangeEvent(msg.point_mapping_change_event);
      break;
    case eLeapEventType_TrackingMode:
      // handleTrackingModeEvent(msg.tracking_mode_event);
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
      std::cout << "Unhandled message type: " << msg.type << std::endl;
    } // switch on msg.type
  }
  // std::cout << "leap service loop finished." << std::endl;
}
void LeapConnect::OpenConnection(void)
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
      InitializeCriticalSection(&dataLock);
      pollingThread = std::thread(&LeapConnect::serviceMessageLoop, this);
    }
  }
  // LEAP_DEVICE_INFO* info = leap.GetDeviceProperties();
  // std::cout << "leap connected with serial: " << info->serial << std::endl;
}

void LeapConnect::CloseConnection(void)
{
  if (!_isRunning)
  {
    return;
  }
  _isRunning = false;
  LeapCloseConnection(connectionHandle);
  pollingThread.join();
}

void LeapConnect::kill(void)
{
  if (!_isRunning)
  {
    return;
  }
  CloseConnection();
  LeapDestroyConnection(connectionHandle);
  std::cout << "leap killed." << std::endl;
}

const char *LeapConnect::ResultString(eLeapRS r)
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

void LeapConnect::setDevice(const LEAP_DEVICE_INFO *deviceProps)
{
  LockMutex(&dataLock);
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
  UnlockMutex(&dataLock);
}

/** Returns a pointer to the cached device info. */
LEAP_DEVICE_INFO *LeapConnect::GetDeviceProperties()
{
  LEAP_DEVICE_INFO *currentDevice;
  LockMutex(&dataLock);
  currentDevice = lastDevice;
  UnlockMutex(&dataLock);
  return currentDevice;
}

/**
 * Caches the newest frame by copying the tracking event struct returned by
 * LeapC.
 */
void LeapConnect::setFrame(const LEAP_TRACKING_EVENT *frame)
{
  LockMutex(&dataLock);
  if (!lastFrame)
    lastFrame = (LEAP_TRACKING_EVENT *)malloc(sizeof(*frame));
  *lastFrame = *frame;
  UnlockMutex(&dataLock);
}

/** Returns a pointer to the cached tracking frame. */
std::vector<float> *LeapConnect::getFrame()
{
  LEAP_TRACKING_EVENT *currentFrame;

  LockMutex(&dataLock);
  currentFrame = lastFrame;
  UnlockMutex(&dataLock);
  std::vector<float> *frame = new std::vector<float>();
  for (uint32_t h = 0; h < currentFrame->nHands; h++)
  {
    LEAP_HAND *hand = &currentFrame->pHands[h];
    for (uint32_t f = 0; f < 5; f++)
    {
      LEAP_DIGIT *finger = &hand->digits[f];
      for (uint32_t b = 0; b < 4; b++)
      {
        LEAP_BONE *bone = &finger->bones[b];
        frame->push_back(bone->prev_joint.x);
        frame->push_back(bone->prev_joint.y);
        frame->push_back(bone->prev_joint.z);
        // frame->push_back(bone->next_joint.x);
        // frame->push_back(bone->next_joint.y);
        // frame->push_back(bone->next_joint.z);
      }
    }
  }
  return frame;
}