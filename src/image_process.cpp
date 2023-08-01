
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
bool NPP_wrapper::cuda_process(uint8_t* buffer_in, uint8_t* buffer_out, unsigned int width, unsigned int height)
{
    try {
    npp::ImageCPU_8u_C4 oHostSrc(width, height);
    memcpy(oHostSrc.data(), buffer_in, width * height * 4 * sizeof(uint8_t));
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
    memcpy(buffer_out, oHostDst.data(), width * height * sizeof(uint8_t));
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