
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

bool NPP_wrapper::distanceTransform(uint8_t* host_buffer_in, 
                       uint8_t* host_buffer_out,
                    //    uint8_t* image_to_sample_from,
                       unsigned int width, unsigned int height)
{
try {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = NULL; // default stream
    npp::ImageCPU_8u_C4 oHostSrc(width, height);
    memcpy(oHostSrc.data(), host_buffer_in, width * height * 4 * sizeof(uint8_t));
    npp::ImageNPP_8u_C4 oDeviceSrc(oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceDst1C(oDeviceSrc.size());
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiSize oSizeDoubleROI = {(int)oDeviceSrc.width()*2, (int)oDeviceSrc.height()};
    Npp32f aCoeffs[3] = {0.333f, 0.333f, 0.333f};
    // color to gray
    nppiColorToGray_8u_AC4C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                              oDeviceDst1C.data(), oDeviceDst1C.pitch(),
                              oSizeROI, aCoeffs);
    // threshold
    nppiThreshold_Val_8u_C1IR(oDeviceDst1C.data(), oDeviceDst1C.pitch(), oSizeROI, 10, 255, NPP_CMP_GREATER);
    npp::ImageNPP_16s_C1 oDeviceDstVoronoiIndices(width * 2, height);
    // npp::ImageNPP_16u_C1 oDeviceDstTransform(width, height);
    /* closest point transform */
    // get site indices from distance transform
    size_t nScratchBufferSize;
    Npp8u* pScratchDeviceBuffer = NULL;
    nppiDistanceTransformPBAGetBufferSize(oSizeROI, &nScratchBufferSize);
    cudaMalloc((void **) &pScratchDeviceBuffer, nScratchBufferSize);
    nppiDistanceTransformPBA_8u16u_C1R_Ctx(oDeviceDst1C.data(), oDeviceDst1C.pitch(),
                                           255, 255, 0, 0,
                                           oDeviceDstVoronoiIndices.data(),
                                           oDeviceDstVoronoiIndices.pitch(),
                                           0, 0, 0, 0,
                                           oSizeROI, pScratchDeviceBuffer, nppStreamCtx);
    //sanity
    // npp::ImageCPU_16u_C1 oHostDstTransform(oDeviceDst.size());
    // oDeviceDstTransform.copyTo(oHostDstTransform.data(), oHostDstTransform.pitch());
    // memcpy(distance_transform_buffer_out, oHostDstTransform.data(), width * height * sizeof(uint16_t));


    // split result into x and y maps (is there an easier way?)
    //prepare a mask that is checkerboard pattern of 0 and 1 (ymap)
    uint8_t* flattened_mask = new uint8_t[width * height];
    for (unsigned int i = 0; i < width * height; i++)
    {
        flattened_mask[i] = (i % 2 == 0) ? 0 : 255;
    }
    npp::ImageCPU_8u_C1 oHostMask(width, height);
    memcpy(oHostMask.data(), flattened_mask, width * height * sizeof(uint8_t));
    npp::ImageNPP_8u_C1 oDeviceMask(oHostMask);
    npp::ImageNPP_16s_C1 oDeviceYMap(width, height);
    nppiCopy_16s_C1MR(oDeviceDstVoronoiIndices.data(), oDeviceDstVoronoiIndices.pitch(),
                      oDeviceYMap.data(), oDeviceYMap.pitch(),
                      oSizeDoubleROI, oDeviceMask.data(), oDeviceMask.pitch());
    //invert the mask and copy again to get (xmap)
    npp::ImageNPP_8u_C1 oDeviceMaskInverted(oDeviceMask.size());
    nppiNot_8u_C1R(oDeviceMask.data(), oDeviceMask.pitch(),
                   oDeviceMaskInverted.data(), oDeviceMaskInverted.pitch(),
                   oSizeROI);
    npp::ImageNPP_16s_C1 oDeviceXMap(width, height);
    nppiCopy_16s_C1MR(oDeviceDstVoronoiIndices.data(), oDeviceDstVoronoiIndices.pitch(),
                      oDeviceXMap.data(), oDeviceXMap.pitch(),
                      oSizeDoubleROI, oDeviceMaskInverted.data(), oDeviceMaskInverted.pitch());
    //increase bit depth and convert to float
    npp::ImageNPP_32f_C1 oDeviceXMapFloat(width, height);
    nppiConvert_16s32f_C1R(oDeviceXMap.data(), oDeviceXMap.pitch(),
                           oDeviceXMapFloat.data(), oDeviceXMapFloat.pitch(),
                           oSizeROI);
    npp::ImageNPP_32f_C1 oDeviceYMapFloat(width, height);
    nppiConvert_16s32f_C1R(oDeviceYMap.data(), oDeviceYMap.pitch(),
                           oDeviceYMapFloat.data(), oDeviceYMapFloat.pitch(),
                           oSizeROI);
    //divide by width and height
    nppiDivC_32f_C1IR((float)width, oDeviceXMapFloat.data(), oDeviceXMapFloat.pitch(), oSizeROI);
    nppiDivC_32f_C1IR((float)height, oDeviceYMapFloat.data(), oDeviceYMapFloat.pitch(), oSizeROI);
    // finally, remap the input using the maps...
    // npp::ImageCPU_8u_C4 oHostSampleFrom(width, height);
    // memcpy(oHostSampleFrom.data(), host_buffer_in, width * height * 4 * sizeof(uint8_t));
    // npp::ImageNPP_8u_C4 oDeviceSampleFrom(oHostSampleFrom);
    npp::ImageNPP_8u_C4 oDeviceDst4C(width, height);
    NppiRect oRectROI = {0, 0, (int)width, (int)height};
    nppiRemap_8u_C4R(oDeviceSrc.data(), oSizeROI, oDeviceSrc.pitch(),
                     oRectROI, 
                     oDeviceXMapFloat.data(), oDeviceXMapFloat.pitch(),
                     oDeviceYMapFloat.data(), oDeviceYMapFloat.pitch(),
                     oDeviceDst4C.data(), oDeviceDst4C.pitch(), oSizeROI, 
                     NPPI_INTER_LINEAR);
    // save result back to host
    npp::ImageCPU_8u_C4 oHostDst(oDeviceDst4C.size());
    oDeviceDst4C.copyTo(oHostDst.data(), oHostDst.pitch());
    // NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // checkErrorNPP(nppiColorToGray_8u_C4C1R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceDst.data(), oDeviceSrc.pitch(),
    //     oSizeROI, aCoeffs));
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    memcpy(host_buffer_out, oHostDst.data(), width * height * sizeof(uint8_t) * 4);
    cudaFree(pScratchDeviceBuffer);
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst1C.data());
    nppiFree(oDeviceDst4C.data());
    nppiFree(oDeviceDstVoronoiIndices.data());
    // nppiFree(oDeviceDstTransform.data());
    nppiFree(oDeviceYMap.data());
    nppiFree(oDeviceXMap.data());
    nppiFree(oDeviceMask.data());
    nppiFree(oDeviceMaskInverted.data());
    nppiFree(oDeviceXMapFloat.data());
    nppiFree(oDeviceYMapFloat.data());
    // nppiFree(oDeviceSampleFrom.data());
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

bool NPP_wrapper::distanceTransform(uint8_t* device_buffer_io, unsigned int width, unsigned int height)
{
try {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = NULL; // default stream
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
    npp::ImageNPP_16s_C1 oDeviceDstVoronoiIndices(width * 2, height);
    npp::ImageNPP_16u_C1 oDeviceDstTransform(width, height);
    /* closest point transform */
    // get site indices from distance transform
    size_t nScratchBufferSize;
    Npp8u* pScratchDeviceBuffer = NULL;
    nppiDistanceTransformPBAGetBufferSize(oSizeROI, &nScratchBufferSize);
    cudaMalloc((void **) &pScratchDeviceBuffer, nScratchBufferSize);
    nppiDistanceTransformPBA_8u16u_C1R_Ctx(oDeviceDst1C.data(), oDeviceDst1C.pitch(),
                                           255, 255, 0, 0,
                                           oDeviceDstVoronoiIndices.data(),
                                           oDeviceDstVoronoiIndices.pitch(),
                                           0, 0, oDeviceDstTransform.data(), oDeviceDstTransform.pitch(),
                                           oSizeROI, pScratchDeviceBuffer, nppStreamCtx);
    //sanity
    // npp::ImageCPU_16u_C1 oHostDstTransform(oDeviceDst.size());
    // oDeviceDstTransform.copyTo(oHostDstTransform.data(), oHostDstTransform.pitch());
    // memcpy(distance_transform_buffer_out, oHostDstTransform.data(), width * height * sizeof(uint16_t));

    cudaFree(pScratchDeviceBuffer);
    cudaError_t eResult;
    eResult = cudaMemcpy2D((Npp8u*)device_buffer_io, stepSize, oDeviceDst4C.data(), oDeviceDst4C.pitch(), width * 4 * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
    NPP_ASSERT(cudaSuccess == eResult);
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
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // checkErrorNPP(nppiColorToGray_8u_C4C1R(
    //     oDeviceSrc.data(), oDeviceSrc.pitch(),
    //     oDeviceDst.data(), oDeviceSrc.pitch(),
    //     oSizeROI, aCoeffs));
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // memcpy(host_buffer_out, oHostDst.data(), width * height * sizeof(uint8_t));
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