#include "UnityPlugin.h"
UnityPlugin::UnityPlugin(int projector_width, int projector_height) : projector(projector_width, projector_height)
{
}

bool UnityPlugin::initialize_projector()
{
    return projector.init();
}

UnityPlugin::~UnityPlugin()
{
    if (projector.is_initialized())
    {
        projector.gracefully_close();
    }
}

int UnityPlugin::debug(int debug_value)
{
    return debug_value+1;
}

void UnityPlugin::buffer_to_image(char* buffer, int width, int height)
{
    cv::Mat image(height, width, CV_8UC3, buffer);
    cv::Mat myimage = cv::Mat(height, width, CV_8UC3, buffer);
    cv::imwrite("test.png", myimage);
}

void UnityPlugin::projector_show_white(int iterations)
{
    if (projector.is_initialized())
    {
        cv::Mat white_image(projector.width, projector.height, CV_8UC3, cv::Scalar(255, 255, 255));
        for (int i = 0; i < iterations; i++)
        {
            std::cout << i << "\n";
            projector.show(white_image);
        }
    }
}

UnityPlugin* createUnityPlugin()
{
    return new UnityPlugin(1024, 768);
};
void freeUnityPlugin(UnityPlugin* instance)
{
    delete instance;
};
int debug(UnityPlugin* instance, int debug_value)
{
    return instance->debug(debug_value);
};
bool initialize_projector(UnityPlugin* instance)
{
    return instance->initialize_projector();
};
void projector_show_white(UnityPlugin* instance, int iterations)
{
    instance->projector_show_white(iterations);
};
void buffer_to_image(UnityPlugin* instance, char* buffer, int width, int height)
{
    instance->buffer_to_image(buffer, width, height);
};