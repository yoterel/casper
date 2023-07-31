#ifndef AHANDS_UTILS_H
#define AHANDS_UTILS_H

#include <iostream>
//GL includes
#include <glad/glad.h>
// CUDA includes
#include <cuda_runtime.h>

GLenum glCheckError_(const char *file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__)
const char *_cudaGetErrorEnum(cudaError_t error);

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#endif // AHANDS_UTILS