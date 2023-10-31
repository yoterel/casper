#ifndef DIFFUSE_H
#define DIFFUSE_H


#include <stdio.h>
#include <iostream>
#include <stdlib.h>

// #include "HTTPRequest.h"
// #include "numpy/arrayobject.h"

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

class Diffuse
{
public:
    Diffuse();
    void print_backend_config();
    void txt2img(const std::string prompt);
    void img2img() {};
private:
};

#endif /* DIFFUSE_H */
