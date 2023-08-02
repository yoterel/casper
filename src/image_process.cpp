
#include "image_process.h"
// #include <helper_cuda.h>
// #include <Exceptions.h>
// #include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <npp.h>
#include "utils.h"

bool NPP_wrapper::printfNPPinfo()
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
            libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
            (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
            (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    // bool bVal = checkCudaCapabilities(1, 0);
    return true;
}

bool NPP_wrapper::process(uint8_t* device_buffer_io, unsigned int width, unsigned int height)
{
        try {
        NppiSize oSizeROI = {(int)width, (int)height};
        int stepSize = 4 * width * sizeof(uint8_t);
        Npp32f aCoeffs[3] = {0.333f, 0.333f, 0.333f};
        npp::ImageNPP_8u_C1 oDeviceDst1C(width, height);
        npp::ImageNPP_8u_C4 oDeviceDst4C(width, height);
        nppiColorToGray_8u_AC4C1R((Npp8u*)device_buffer_io, stepSize,
                                oDeviceDst1C.data(), oDeviceDst1C.pitch(),
                                oSizeROI, aCoeffs);
        nppiThreshold_Val_8u_C1IR(oDeviceDst1C.data(), oDeviceDst1C.pitch(), oSizeROI, 10, 255, NPP_CMP_GREATER);
        nppiDup_8u_C1AC4R(oDeviceDst1C.data(), oDeviceDst1C.pitch(), oDeviceDst4C.data(), oDeviceDst4C.pitch(), oSizeROI);
        // inplace gamma
        // npp::ImageNPP_8u_C4 oDeviceDst(width, height);
        // checkErrorNPP(nppiGammaFwd_8u_AC4IR((Npp8u*)device_buffer_io, stepSize, oSizeROI));
        // out of place gamma
        // npp::ImageNPP_8u_C4 oDeviceDst(width, height);
        // checkErrorNPP(nppiGammaFwd_8u_AC4R((Npp8u*)device_buffer_io, stepSize, oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI));
        cudaError_t eResult;
        eResult = cudaMemcpy2D((Npp8u*)device_buffer_io, stepSize, oDeviceDst4C.data(), oDeviceDst4C.pitch(), width * 4 * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
        NPP_ASSERT(cudaSuccess == eResult);
        // device_buffer_io = oDeviceDst.data();
        nppiFree(oDeviceDst1C.data());
        nppiFree(oDeviceDst4C.data());
        return true;


        } catch (npp::Exception &rException) {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        return false;
        // exit(1);
        } catch (...) {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        // exit(1);
        return false;
    }
}

bool NPP_wrapper::process(uint8_t* host_buffer_in, uint8_t* host_buffer_out, unsigned int width, unsigned int height)
{
    try {
    npp::ImageCPU_8u_C4 oHostSrc(width, height);
    memcpy(oHostSrc.data(), host_buffer_in, width * height * 4 * sizeof(uint8_t));
    npp::ImageNPP_8u_C4 oDeviceSrc(oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    Npp32f aCoeffs[3] = {0.333f, 0.333f, 0.333f};
    nppiColorToGray_8u_AC4C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                              oDeviceDst.data(), oDeviceDst.pitch(),
                              oSizeROI, aCoeffs);
    nppiThreshold_Val_8u_C1IR(oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, 10, 255, NPP_CMP_GREATER);
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // checkErrorNPP(nppiColorToGray_8u_C4C1R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceDst.data(), oDeviceSrc.pitch(),
    //     oSizeROI, aCoeffs));
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    memcpy(host_buffer_out, oHostDst.data(), width * height * sizeof(uint8_t));
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
    return true;


    } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;
    return false;
    // exit(1);
    } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;
    // exit(1);
    return false;
    }
}

bool distanceTransform(uint8_t* host_buffer_in, 
                       uint8_t* host_buffer_out,
                       unsigned int width, unsigned int height)
{
try {
    npp::ImageCPU_8u_C4 oHostSrc(width, height);
    memcpy(oHostSrc.data(), host_buffer_in, width * height * 4 * sizeof(uint8_t));
    npp::ImageNPP_8u_C4 oDeviceSrc(oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    Npp32f aCoeffs[3] = {0.333f, 0.333f, 0.333f};
    // color to gray
    nppiColorToGray_8u_AC4C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                              oDeviceDst.data(), oDeviceDst.pitch(),
                              oSizeROI, aCoeffs);
    // threshold
    nppiThreshold_Val_8u_C1IR(oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, 10, 255, NPP_CMP_GREATER);
    npp::ImageNPP_16s_C1 oDeviceDstVoronoiIndices(width * 2, height);
    npp::ImageNPP_16s_C1 oDeviceDstVoronoiIndices(width * 2, height);
    /* closest point transform */
    // get site indices from distance transform
    nppiDistanceTransformPBA_8u16u_C1R_Ctx(oDeviceDst.data(), oDeviceDst.pitch(), 255, 255, 0, 0, oDeviceDstVoronoiIndices.data(), oDeviceDstVoronoiIndices.pitch(), 0, 0, 0, 0, oSizeROI, oDeviceBuffer.data(), 0);
    // split result into x and y maps (is there an easier way?)
    //prepare a mask that is checkerboard pattern of 0 and 1 (ymap)
    //nppiCopy_16s_C1MR
    //invert the mask and copy again to get (xmap)
    //nppiCopy_16s_C1MR
    //increase bit depth and convert to float
    //nppiConvert_16s32f_C1R
    //divide by width and height
    //nppiDivC_32f_C1IR
    // finally, remap the input using the maps...
    // nppiRemap_8u_C4R
    // save result back to buffer
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // checkErrorNPP(nppiColorToGray_8u_C4C1R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceDst.data(), oDeviceSrc.pitch(),
    //     oSizeROI, aCoeffs));
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    memcpy(host_buffer_out, oHostDst.data(), width * height * sizeof(uint8_t));
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
    return true;


    } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;
    return false;
    // exit(1);
    } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;
    // exit(1);
    return false;
    }
}