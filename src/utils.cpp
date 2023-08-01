#include "utils.h"
// #include <Exceptions.h>
#include <iostream>

// template <typename T>
void check(cudaError_t result, char const *const func, const char *const file, int const line)
{
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

void check2(NppStatus eStatusNPP)
{
    if (eStatusNPP != NPP_SUCCESS) 
    {
        std::cout << "NPP_CHECK_NPP - eStatusNPP = " << _cudaGetErrorEnum_NPP(eStatusNPP) << "("<< eStatusNPP << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
const char *_cudaGetErrorEnum_NPP(NppStatus error)
{ 
    return "npp error"; 
}