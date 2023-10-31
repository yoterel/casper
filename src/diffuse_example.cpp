#include "diffuse.h"

int main(int argc, char* argv[])
{
    Diffuse diffuseObject = Diffuse();
    diffuseObject.genImage("a rabbit");
    return 0;
}