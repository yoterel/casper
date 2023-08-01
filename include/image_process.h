#ifndef THRESHOLD_H
#define THRESHOLD_H
#include <stdio.h>
#include <stdint.h>

class NPP_wrapper
{
public:
    static bool printfNPPinfo();
    static bool cuda_process(uint8_t* buffer_in, uint8_t* buffer_out, unsigned int width, unsigned int height);
};
#endif // THRESHOLD_H