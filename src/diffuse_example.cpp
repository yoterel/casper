#include <chrono>
#include <stdio.h>
#include "diffuse.h"
using namespace std::chrono;

int main(int argc, char* argv[])
{
    Diffuse diffuseObject = Diffuse();
    auto start = high_resolution_clock::now();
    diffuseObject.txt2img("a rainbow rabbit");
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    return 0;
}