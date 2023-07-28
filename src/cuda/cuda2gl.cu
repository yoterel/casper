// Utilities and system includes

// #include <helper_cuda.h>

// clamp x to range [a, b]
// __device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

// __device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

// // convert floating point rgb color to 8-bit integer
// __device__ int rgbToInt(float r, float g, float b) {
//   r = clamp(r, 0.0f, 255.0f);
//   g = clamp(g, 0.0f, 255.0f);
//   b = clamp(b, 0.0f, 255.0f);
//   return (int(b) << 16) | (int(g) << 8) | int(r);
// }

// __global__ void cudaProcess(unsigned int *g_odata, int imgw) {
//   extern __shared__ uchar4 sdata[];

//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int bw = blockDim.x;
//   int bh = blockDim.y;
//   int x = blockIdx.x * bw + tx;
//   int y = blockIdx.y * bh + ty;

//   uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
//   g_odata[y * imgw + x] = rgbToInt(c4.z, c4.y, c4.x);
// }
// copy kernel
__global__ void copy_kernel(cudaSurfaceObject_t surface, int nx, int ny){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < nx && y < ny){
        uchar4 data = make_uchar4(x % 255, 
                                  y % 255, 
                                  0, 255);
        surf2Dwrite(data, surface, x * sizeof(uchar4), y);
    }
}
// __global__ void cudaProcess(cudaSurfaceObject_t surface, unsigned int width, unsigned int height, float cx, float cy, float r) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x >= width || y >= height) {
//         return;
//     }

//     float dx = x - cx;
//     float dy = y - cy;
//     float distance = sqrt(dx*dx + dy*dy);

//     if (distance <= r) {
//         uchar4 color = make_uchar4(255, 0, 0, 255);
//         surf2Dwrite(color, surface, x * sizeof(uchar4), y);
//     }
// }

extern "C" void launch_cudaProcess(dim3 grid, dim3 block,
                                   cudaSurfaceObject_t surface,
                                   unsigned int width, unsigned int height)
{
    // cudaProcess<<<grid, block>>>(surface, width, height, width / 2.0f, height / 2.0f, width / 4.0f);
    copy_kernel<<<grid, block>>>(surface, width, height);
}
