#ifndef THRESHOLD_H
#define THRESHOLD_H
#include <stdio.h>
#include <stdint.h>

class NPP_wrapper
{
public:
    static bool printfNPPinfo();
    static bool process(uint8_t* host_buffer_in, uint8_t* host_buffer_out, unsigned int width, unsigned int height);
    static bool process(uint8_t* device_buffer_io, unsigned int width, unsigned int height);
    static bool distanceTransform(uint8_t* host_buffer_in, 
                                  uint8_t* host_buffer_out,
                                  unsigned int width, unsigned int height);
};
#endif // THRESHOLD_H