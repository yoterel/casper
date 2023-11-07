#include "camera.h"

// Contains a Configuration Event Handler that prints a message for each event method call.

class CConfigurationEventPrinter : public CConfigurationEventHandler
{
public:
    void OnAttach(CInstantCamera & /*camera*/)
    {
        std::cout << "Baser API: OnAttach event" << std::endl;
    }

    void OnAttached(CInstantCamera &camera)
    {
        std::cout << "Baser API: OnAttached event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnOpen(CInstantCamera &camera)
    {
        // std::cout << "OnOpen event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnOpened(CInstantCamera &camera)
    {
        // std::cout << "OnOpened event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnGrabStart(CInstantCamera &camera)
    {
        // std::cout << "OnGrabStart event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnGrabStarted(CInstantCamera &camera)
    {
        // std::cout << "OnGrabStarted event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnGrabStop(CInstantCamera &camera)
    {
        // std::cout << "OnGrabStop event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnGrabStopped(CInstantCamera &camera)
    {
        // std::cout << "OnGrabStopped event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnClose(CInstantCamera &camera)
    {
        // std::cout << "OnClose event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnClosed(CInstantCamera &camera)
    {
        // std::cout << "OnClosed event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnDestroy(CInstantCamera &camera)
    {
        // std::cout << "OnDestroy event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnDestroyed(CInstantCamera & /*camera*/)
    {
        // std::cout << "OnDestroyed event" << std::endl;
    }

    void OnDetach(CInstantCamera &camera)
    {
        std::cout << "Baser API: OnDetach event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnDetached(CInstantCamera &camera)
    {
        std::cout << "Baser API: OnDetached event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }

    void OnGrabError(CInstantCamera &camera, const char *errorMessage)
    {
        std::cout << "Baser API: OnGrabError event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
        std::cout << "Baser API: Error Message: " << errorMessage << std::endl;
    }

    void OnCameraDeviceRemoved(CInstantCamera &camera)
    {
        std::cout << "Baser API: OnCameraDeviceRemoved event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
    }
};

class SingleFrameConfiguration : public CConfigurationEventHandler
{
public:
    SingleFrameConfiguration(float exposure_time)
    {
        if (exposure_time < 1850.0)
        {
            std::cout << "Baser API: exposure time cannot be less than 1850.0, setting to 1850.0" << std::endl;
            exposure_time = 1850.0;
        }
        if (exposure_time > 15000.0)
        {
            std::cout << "Baser API: exposure time cannot be greater than 15000.0, setting to 15000.0" << std::endl;
            exposure_time = 15000.0;
        }
        m_exposure_time = exposure_time;
    }
    /// Apply acquire single frame configuration.
    static void ApplyConfiguration(GenApi::INodeMap &nodemap, float exposure_time)
    {
        using namespace GenApi;

        // Disable all trigger types.
        CConfigurationHelper::DisableAllTriggers(nodemap);

        // Disable compression mode.
        CConfigurationHelper::DisableCompression(nodemap);

        // Disable GenDC streaming.
        CConfigurationHelper::DisableGenDC(nodemap);

        // Set image component.
        CConfigurationHelper::SelectRangeComponent(nodemap);

        // Set acquisition mode.
        CEnumParameter(nodemap, "AcquisitionMode").SetValue("SingleFrame");
        CFloatParameter(nodemap, "ExposureTime").SetValue(exposure_time); // 1850.0=for max fps, 1904.0f = for 500 fps
    }

    // Set basic camera settings
    virtual void OnOpened(CInstantCamera &camera)
    {
        try
        {
            ApplyConfiguration(camera.GetNodeMap(), m_exposure_time);
            // Probe max packet size
            CConfigurationHelper::ProbePacketSize(camera.GetStreamGrabberNodeMap());
        }
        catch (const GenericException &e)
        {
            throw RUNTIME_EXCEPTION("Baser API: Could not apply configuration. Pylon::GenericException caught in OnOpened method msg=%hs", e.what());
        }
        catch (const std::exception &e)
        {
            throw RUNTIME_EXCEPTION("Baser API: Could not apply configuration. std::exception caught in OnOpened method msg=%hs", e.what());
        }
        catch (...)
        {
            throw RUNTIME_EXCEPTION("Baser API: Could not apply configuration. Unknown exception caught in OnOpened method.");
        }
    }

private:
    float m_exposure_time;
};

// hardware trigger configuration for basler camera

class HardwareTriggerConfiguration : public CConfigurationEventHandler
{
public:
    HardwareTriggerConfiguration(float exposureTime, bool hardwareTrigger)
    {
        if (exposureTime < 1850.0)
        {
            std::cout << "Baser API: exposure time cannot be less than 1850.0, setting to 1850.0" << std::endl;
            exposureTime = 1850.0;
        }
        if (exposureTime > 15000.0)
        {
            std::cout << "Baser API: exposure time cannot be greater than 15000.0, setting to 15000.0" << std::endl;
            exposureTime = 15000.0;
        }
        m_exposure_time = exposureTime;
        m_hardwareTrigger = hardwareTrigger;
    }
    /// Apply trigger configuration.
    static void ApplyConfiguration(GenApi::INodeMap &nodemap, float exposure_time, bool hardwareTrigger)
    {
        using namespace GenApi;
        // bool hardwareTrigger = false;

        // Disable compression mode.
        CConfigurationHelper::DisableCompression(nodemap);

        // Disable GenDC streaming.
        CConfigurationHelper::DisableGenDC(nodemap);

        // Select image component.
        CConfigurationHelper::SelectRangeComponent(nodemap);

        // Disable all trigger types except the trigger type used for triggering the acquisition of
        // frames.
        {
            // Get required enumerations.
            CEnumParameter triggerSelector(nodemap, "TriggerSelector");
            CEnumParameter triggerMode(nodemap, "TriggerMode");

            // Check the available camera trigger mode(s) to select the appropriate one: acquisition start trigger mode
            // (used by older cameras, i.e. for cameras supporting only the legacy image acquisition control mode;
            // do not confuse with acquisition start command) or frame start trigger mode
            // (used by newer cameras, i.e. for cameras using the standard image acquisition control mode;
            // equivalent to the acquisition start trigger mode in the legacy image acquisition control mode).
            String_t triggerName("FrameStart");
            if (!triggerSelector.CanSetValue(triggerName))
            {
                triggerName = "AcquisitionStart";
                if (!triggerSelector.CanSetValue(triggerName))
                {
                    throw RUNTIME_EXCEPTION("Could not select trigger. Neither FrameStart nor AcquisitionStart is available.");
                }
            }

            // Get all enumeration entries of trigger selector.
            StringList_t triggerSelectorEntries;
            triggerSelector.GetSettableValues(triggerSelectorEntries);

            // Turn trigger mode off for all trigger selector entries except for the frame trigger given by triggerName.
            for (StringList_t::const_iterator it = triggerSelectorEntries.begin(); it != triggerSelectorEntries.end(); ++it)
            {
                // Set trigger mode to off.
                triggerSelector.SetValue(*it);

                if (hardwareTrigger)
                {
                    if (triggerName == *it)
                    {
                        // Activate trigger.
                        triggerMode.SetValue("On");

                        // The trigger source must be set to 'Software'.
                        // CEnumParameter( nodemap, "TriggerSource" ).SetValue( "Software" );

                        //// Alternative hardware trigger configuration:
                        //// This configuration can be copied and modified to create a hardware trigger configuration.
                        //// Remove setting the 'TriggerSource' to 'Software' (see above) and
                        //// use the commented lines as a starting point.
                        //// The camera user's manual contains more information about available configurations.
                        //// The Basler pylon Viewer tool can be used to test the selected settings first.

                        //// The trigger source must be set to the trigger input, e.g. 'Line1'.
                        CEnumParameter(nodemap, "TriggerSource").SetValue("Line4");

                        ////The trigger activation must be set to e.g. 'RisingEdge'.
                        CEnumParameter(nodemap, "TriggerActivation").SetValue("RisingEdge");
                    }
                    else
                    {
                        triggerMode.TrySetValue("Off");
                    }
                }
                else
                {
                    triggerMode.TrySetValue("Off");
                }
            }
            // Finally select the frame trigger type (resp. acquisition start type
            // for older cameras). Issuing a software trigger will now trigger
            // the acquisition of a frame.
            triggerSelector.SetValue(triggerName);
        }

        // Set acquisition mode to "continuous"
        CEnumParameter(nodemap, "AcquisitionMode").SetValue("Continuous");
        CFloatParameter(nodemap, "ExposureTime").SetValue(exposure_time); // 1850.0=for max fps, 1904.0f = for 500 fps
    }

    // Set basic camera settings.
    virtual void OnOpened(CInstantCamera &camera)
    {
        try
        {
            ApplyConfiguration(camera.GetNodeMap(), m_exposure_time, m_hardwareTrigger);
            // Probe max packet size
            CConfigurationHelper::ProbePacketSize(camera.GetStreamGrabberNodeMap());
        }
        catch (const GenericException &e)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. Pylon::GenericException caught in OnOpened method msg=%hs", e.what());
        }
        catch (const std::exception &e)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. std::exception caught in OnOpened method msg=%hs", e.what());
        }
        catch (...)
        {
            throw RUNTIME_EXCEPTION("Could not apply configuration. Unknown exception caught in OnOpened method.");
        }
    }

private:
    float m_exposure_time;
    bool m_hardwareTrigger;
};

// image event handler
class MyImageEventHandler : public CImageEventHandler
{
public:
    MyImageEventHandler(blocking_queue<CPylonImage> &camera_queue, bool &close_signal, uint32_t &height, uint32_t &width) : myqueue(camera_queue), close_signal(close_signal), height(height), width(width) {}
    blocking_queue<CPylonImage> &myqueue;
    bool &close_signal;
    uint32_t &height;
    uint32_t &width;
    virtual void OnImagesSkipped(CInstantCamera &camera, size_t countOfSkippedImages)
    {
        // std::cout << "OnImagesSkipped event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
        // std::cout << countOfSkippedImages << " images have been skipped." << std::endl;
        // std::cout << std::endl;
    }

    virtual void OnImageGrabbed(CInstantCamera &camera, const CGrabResultPtr &ptrGrabResult)
    {
        if (close_signal)
        {
            camera.Close();
            return;
        }
        // std::cout << "OnImageGrabbed event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
        // Image grabbed successfully?
        if (ptrGrabResult->GrabSucceeded())
        {
            // std::cout << "Image grabbed !!! " << std::endl;
            /********************** to Mat conversion *************************/
            // // std::cout << "SizeX: " << ptrGrabResult->GetWidth() << std::endl;
            // // std::cout << "SizeY: " << ptrGrabResult->GetHeight() << std::endl;
            // // const uint8_t* pImageBuffer = (uint8_t*) ptrGrabResult->GetBuffer();
            // // std::cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << std::endl;
            // // std::cout << std::endl;
            // // Pylon::DisplayImage( 1, ptrGrabResult );
            // CImageFormatConverter formatConverter;
            // formatConverter.OutputPixelFormat= PixelType_BGR8packed;
            // // CPylonImage* pylonImage = new CPylonImage();
            // CPylonImage pylonImage;
            // formatConverter.Convert(pylonImage, ptrGrabResult);
            // cv::Mat myimage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*) pylonImage.GetBuffer());
            // // cv::namedWindow("image", cv::WINDOW_AUTOSIZE );
            // // cv::imshow("image", myimage);
            // // cv::waitKey(1);
            // // cv::resize(myimage, myimage, cv::Size(1024, 768));
            // myqueue.push(myimage.clone());
            /********************** end to Mat conversion **********************/
            // auto start = std::chrono::system_clock::now();
            if (height == 0 || width == 0)
            {
                // std::cout << "setting camera height width " << std::endl;
                height = ptrGrabResult->GetHeight();
                width = ptrGrabResult->GetWidth();
            }
            CImageFormatConverter formatConverter;
            formatConverter.OutputPixelFormat = PixelType_BGRA8packed;
            CPylonImage pylonImage;
            formatConverter.Convert(pylonImage, ptrGrabResult);
            // uint8_t* buffer = (uint8_t*) pylonImage.GetBuffer();
            // std::cout << "Gray value of first pixel: " << (uint32_t) buffer[0] << std::endl;
            // std::cout << "SizeX: " << ptrGrabResult->GetWidth() << std::endl;
            // std::cout << "SizeY: " << ptrGrabResult->GetHeight() << std::endl;
            // cv::Mat myimage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*) pylonImage.GetBuffer());
            // cv::imwrite("test1.png", myimage);
            myqueue.push(pylonImage);
            // auto runtime = std::chrono::system_clock::now() - start;
            //     std::cout << "ms: "
            //     << (std::chrono::duration_cast<std::chrono::microseconds>(runtime)).count()*1.0/1000
            //     << "\n";
        }
        else
        {
            std::cout << "Baser API: error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << std::endl;
        }
    }
};

bool BaslerCamera::init(blocking_queue<CPylonImage> &camera_queue, bool &close_signal,
                        uint32_t height, uint32_t width, float exposureTime, bool hardwareTrigger)
{
    PylonInitialize();
    try
    {
        camera.Attach(CTlFactory::GetInstance().CreateFirstDevice());
        camera.RegisterConfiguration(new HardwareTriggerConfiguration(exposureTime, hardwareTrigger), RegistrationMode_ReplaceAll, Cleanup_Delete);
        camera.RegisterConfiguration(new CConfigurationEventPrinter, RegistrationMode_Append, Cleanup_Delete);
        camera.RegisterImageEventHandler(new MyImageEventHandler(camera_queue, close_signal, height, width), RegistrationMode_Append, Cleanup_Delete);
        camera.Open();
        is_open = true;
        std::cout << "Basler API: camera initialized." << std::endl;
        return true;
    }
    catch (const GenericException &e)
    {
        std::cerr << "Baser API: an exception occurred: " << e.GetDescription() << std::endl;
        return false;
    }
}

void BaslerCamera::acquire()
{
    if (!is_open)
        return;
    try
    {
        if (camera.CanWaitForFrameTriggerReady())
        {
            // Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
            // to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
            // The GrabStrategy_OneByOne default grab strategy is used.
            camera.StartGrabbing(GrabStrategy_LatestImageOnly, GrabLoop_ProvidedByInstantCamera); // GrabStrategy_LatestImageOnly
            // Wait for user input to trigger the camera or exit the program.
            // The grabbing is stopped, the device is closed and destroyed automatically when the camera object goes out of scope.

            // bool runLoop = true;
            // while (runLoop)
            // {
            //     // std::cout << std::endl << "Enter \"t\" to trigger the camera or \"e\" to exit and press enter? (t/e) "; std::cout.flush();

            //     // std::string userInput;
            //     // std::getline(std::cin, userInput);

            //     // for (size_t i = 0; i < userInput.size(); ++i)
            //     // {
            //         // char key = userInput[i];
            //         if (true)
            //         {
            //             // Execute the software trigger. Wait up to 1000 ms for the camera to be ready for trigger.
            //             if (camera.WaitForFrameTriggerReady( 1000, TimeoutHandling_ThrowException ))
            //             {
            //                 camera.ExecuteSoftwareTrigger();
            //             }
            //         }
            //         // else if ((key == 'e') || (key == 'E'))
            //         // {
            //         //     runLoop = false;
            //         //     break;
            //         // }
            //     // }

            //     // Wait some time to allow the OnImageGrabbed handler print its output,
            //     // so the printed text on the console is in the expected order.
            //     WaitObject::Sleep( 1 );
            // }
        }
        else
        {
            // See the documentation of CInstantCamera::CanWaitForFrameTriggerReady() for more information.
            std::cout << std::endl
                      << "This sample can only be used with cameras that can be queried whether they are ready to accept the next frame trigger." << std::endl;
        }
    }
    catch (const GenericException &e)
    {
        std::cerr << "Baser API: An exception occurred: " << e.GetDescription() << std::endl;
    }
}

void BaslerCamera::balance_white()
{
    camera.LightSourcePreset.SetValue(LightSourcePreset_Off);
    camera.BalanceWhiteAuto.SetValue(BalanceWhiteAuto_Once);
}

void BaslerCamera::kill()
{
    if (!is_open)
        return;
    camera.Close();
    // PylonTerminate();
    is_open = false;
    std::cout << "Baser API: basler camera killed." << std::endl;
}

#ifdef PYTHON_BINDINGS_BUILD

void BaslerCamera::init_single(float exposure_time)
{
    PylonInitialize();
    try
    {
        camera.Attach(CTlFactory::GetInstance().CreateFirstDevice());
        camera.RegisterConfiguration(new SingleFrameConfiguration(exposure_time), RegistrationMode_ReplaceAll, Cleanup_Delete);
        camera.Open();
        is_open = true;
    }
    catch (const GenericException &e)
    {
        std::cerr << "Baser API: An exception occurred: " << e.GetDescription() << std::endl;
    }
}

nb::ndarray<nb::numpy, const uint8_t> BaslerCamera::capture_single()
{
    CGrabResultPtr ptrGrabResult;
    if (camera.GrabOne(5000, ptrGrabResult))
    {
        CImageFormatConverter formatConverter;
        formatConverter.OutputPixelFormat = PixelType_RGB8packed;
        CPylonImage pylonImage;
        formatConverter.Convert(pylonImage, ptrGrabResult);
        uint8_t *buffer = new uint8_t[pylonImage.GetImageSize()];
        memcpy(buffer, (uint8_t *)pylonImage.GetBuffer(), pylonImage.GetImageSize());
        size_t shape[3] = {pylonImage.GetHeight(), pylonImage.GetWidth(), 3};
        return nb::ndarray<nb::numpy, const uint8_t>(buffer, 3, shape);
    }
    else
    {
        std::cout << "Baser API: Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << std::endl;
        return nb::ndarray<nb::numpy, const uint8_t>();
    }
}
#endif