// #include <cooperative_groups.h>
#include <helper_cuda.h>

cudaTextureObject_t inTexObject;

// get pixel from 2D image, with clamping to border
__device__ float4 getPixel(int x, int y, cudaTextureObject_t inTex) {
  float4 res = tex2D<float4>(inTex, x, y);
  return res;
}


__global__ void cudaProcess(float *g_odata, int imgw, int imgh,
                            int tilew, int r, float threshold, float highlight,
                            cudaTextureObject_t inTex) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    g_odata[y * imgw + x] = getPixel(x, y, inTex);
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   cudaArray *g_data_array,
                                   float *g_odata, int imgw, int imgh,
                                   int tilew, int radius, float threshold,
                                   float highlight) {
  struct cudaChannelFormatDesc desc;
  checkCudaErrors(cudaGetChannelDesc(&desc, g_data_array));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = g_data_array;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(
      cudaCreateTextureObject(&inTexObject, &texRes, &texDescr, NULL));

    cudaProcess<<<grid, block, sbytes>>>(g_odata, imgw, imgh,
                                         block.x + (2 * radius), radius, 0.8f,
                                         4.0f, inTexObject);

}
