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
    // std::string base64_decode(const std::string& input);
    // bool is_base64(unsigned char c) {return (isalnum(c) || (c == '+') || (c == '/'));};
    // std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
};

#endif /* DIFFUSE_H */
