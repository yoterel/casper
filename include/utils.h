#ifndef AHANDS_UTILS_H
#define AHANDS_UTILS_H

//GL includes
#include <glad/glad.h>
// CUDA includes
#include <cuda_runtime.h>
// NPP includes
#include "nppdefs.h"

GLenum glCheckError_(const char *file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__)
const char *_cudaGetErrorEnum(cudaError_t error);
const char *_cudaGetErrorEnum_NPP(NppStatus error);

// template <typename T>
void check(cudaError_t result, char const *const func, const char *const file, int const line);
void check2(NppStatus result);
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define checkErrorNPP(S) check2((S))

#endif // AHANDS_UTILS