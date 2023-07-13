#include "camera.h"

//configuration for hardware trigger
class HardwareTriggerConfiguration : public CConfigurationEventHandler
    {
    public:
        /// Apply software trigger configuration.
        static void ApplyConfiguration( GenApi::INodeMap& nodemap )
        {
            using namespace GenApi;

            //Disable compression mode.
            CConfigurationHelper::DisableCompression( nodemap );

            //Disable GenDC streaming.
            CConfigurationHelper::DisableGenDC( nodemap );

            //Select image component.
            CConfigurationHelper::SelectRangeComponent( nodemap );

            // Disable all trigger types except the trigger type used for triggering the acquisition of
            // frames.
            {
                // Get required enumerations.
                CEnumParameter triggerSelector( nodemap, "TriggerSelector" );
                CEnumParameter triggerMode( nodemap, "TriggerMode" );

                // Check the available camera trigger mode(s) to select the appropriate one: acquisition start trigger mode
                // (used by older cameras, i.e. for cameras supporting only the legacy image acquisition control mode;
                // do not confuse with acquisition start command) or frame start trigger mode
                // (used by newer cameras, i.e. for cameras using the standard image acquisition control mode;
                // equivalent to the acquisition start trigger mode in the legacy image acquisition control mode).
                String_t triggerName( "FrameStart" );
                if (!triggerSelector.CanSetValue( triggerName ))
                {
                    triggerName = "AcquisitionStart";
                    if (!triggerSelector.CanSetValue( triggerName ))
                    {
                        throw RUNTIME_EXCEPTION( "Could not select trigger. Neither FrameStart nor AcquisitionStart is available." );
                    }
                }

                // Get all enumeration entries of trigger selector.
                StringList_t triggerSelectorEntries;
                triggerSelector.GetSettableValues( triggerSelectorEntries );

                // Turn trigger mode off for all trigger selector entries except for the frame trigger given by triggerName.
                for (StringList_t::const_iterator it = triggerSelectorEntries.begin(); it != triggerSelectorEntries.end(); ++it)
                {
                    // Set trigger mode to off.
                    triggerSelector.SetValue( *it );
                    bool trigger = false;
                    if (trigger)
                    {
                        if (triggerName == *it)
                        {
                            // Activate trigger.
                            triggerMode.SetValue( "On" );

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
                            triggerMode.TrySetValue( "Off" );
                        }
                    }
                    else
                    {
                        triggerMode.TrySetValue( "Off" );
                    }
                }
                // Finally select the frame trigger type (resp. acquisition start type
                // for older cameras). Issuing a software trigger will now trigger
                // the acquisition of a frame.
                triggerSelector.SetValue( triggerName );
            }

            
            //Set acquisition mode to "continuous"
            CEnumParameter( nodemap, "AcquisitionMode" ).SetValue( "Continuous" );
            CFloatParameter( nodemap, "ExposureTime" ).SetValue(1850.0f);  // 1850.0=for max fps, 1904.0f = for 500 fps
        }

        //Set basic camera settings.
        virtual void OnOpened( CInstantCamera& camera )
        {
            try
            {
                ApplyConfiguration( camera.GetNodeMap() );
                // Probe max packet size
                CConfigurationHelper::ProbePacketSize( camera.GetStreamGrabberNodeMap() );
            }
            catch (const GenericException& e)
            {
                throw RUNTIME_EXCEPTION( "Could not apply configuration. Pylon::GenericException caught in OnOpened method msg=%hs", e.what() );
            }
            catch (const std::exception& e)
            {
                throw RUNTIME_EXCEPTION( "Could not apply configuration. std::exception caught in OnOpened method msg=%hs", e.what() );
            }
            catch (...)
            {
                throw RUNTIME_EXCEPTION( "Could not apply configuration. Unknown exception caught in OnOpened method." );
            }
        }
    };

//image event handler
class MyImageEventHandler : public CImageEventHandler
{
public:
    MyImageEventHandler(blocking_queue<CPylonImage>& camera_queue, bool& close_signal, uint32_t& height, uint32_t& width) :
                        myqueue(camera_queue), close_signal(close_signal), height(height), width(width) {}
    // blocking_queue<cv::Mat>& myqueue;
    blocking_queue<CPylonImage>& myqueue;
    bool& close_signal;
    uint32_t& height;
    uint32_t& width;
    virtual void OnImagesSkipped( CInstantCamera& camera, size_t countOfSkippedImages )
    {
        // std::cout << "OnImagesSkipped event for device " << camera.GetDeviceInfo().GetModelName() << std::endl;
        // std::cout << countOfSkippedImages << " images have been skipped." << std::endl;
        // std::cout << std::endl;
    }

    virtual void OnImageGrabbed( CInstantCamera& camera, const CGrabResultPtr& ptrGrabResult )
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
	        formatConverter.OutputPixelFormat= PixelType_RGB8packed;
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
            std::cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << std::endl;
        }
    }
};

BaslerCamera::BaslerCamera()
{
}

BaslerCamera::~BaslerCamera()
{
    // std::cerr << std::endl << "Terminating pylon camera stack...press any key to continue" << std::endl;
    // while (std::cin.get() != '\n')
    // PylonTerminate();
}

void BaslerCamera::init(blocking_queue<CPylonImage>& camera_queue, bool& close_signal, uint32_t& height, uint32_t& width)
{
    PylonInitialize();
    try
    {
        camera.Attach(CTlFactory::GetInstance().CreateFirstDevice());
        camera.RegisterConfiguration( new HardwareTriggerConfiguration, RegistrationMode_ReplaceAll, Cleanup_Delete );
        camera.RegisterConfiguration( new CConfigurationEventPrinter, RegistrationMode_Append, Cleanup_Delete );
        camera.RegisterImageEventHandler( new MyImageEventHandler(camera_queue, close_signal, height, width), RegistrationMode_Append, Cleanup_Delete );
        camera.Open();
        is_open = true;
    }
    catch (const GenericException& e)
    {
        std::cerr << "An exception occurred." << std::endl << e.GetDescription() << std::endl;
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
            camera.StartGrabbing( GrabStrategy_LatestImageOnly, GrabLoop_ProvidedByInstantCamera );  // GrabStrategy_LatestImageOnly
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
            std::cout << std::endl << "This sample can only be used with cameras that can be queried whether they are ready to accept the next frame trigger." << std::endl;
        }
    }
    catch (const GenericException& e)
    {
        std::cerr << "An exception occurred." << std::endl << e.GetDescription() << std::endl;
    }
}

