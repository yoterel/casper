#pragma once
#include <iostream>
#include "display.h"
#include "opencv2/opencv.hpp"

#ifdef NATIVECPPLIBRARY_EXPORTS
#define NATIVECPPLIBRARY_API __declspec(dllexport)
#else
#define NATIVECPPLIBRARY_API __declspec(dllimport)
# endif

struct Color32
{
	uchar red;
	uchar green;
	uchar blue;
	uchar alpha;
};
class UnityPlugin {
    public:
        UnityPlugin(int projector_width, int projector_height);
        ~UnityPlugin(){
            projector.gracefully_close();
        };
        bool initialize_projector();
        void projector_show_white(int iterations);
        void buffer_to_image(Color32 **rawImage, int width, int height);
        int debug(int debug_value);
    private:
        DynaFlashProjector projector;
};

extern "C" {
    NATIVECPPLIBRARY_API UnityPlugin* createUnityPlugin(int width, int height);
    NATIVECPPLIBRARY_API void freeUnityPlugin(UnityPlugin* instance);
    NATIVECPPLIBRARY_API int debug(UnityPlugin* instance, int debug_value);
    NATIVECPPLIBRARY_API bool initialize_projector(UnityPlugin* instance);
    NATIVECPPLIBRARY_API void projector_show_white(UnityPlugin* instance, int iterations);
    NATIVECPPLIBRARY_API void buffer_to_image(UnityPlugin* instance, Color32 **rawImage, int width, int height);
}