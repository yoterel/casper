# Root CMakeLists.txt for the overall Pylon project
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0074 NEW)    # respect <PACKAGE>_ROOT variables in "find_package"
include(CMakePrintHelpers)

project(augmented_hands VERSION 0.0)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third-party/opencv" ${CMAKE_PREFIX_PATH})

find_package(pylon 7.3.0 REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED)

add_executable(ahand 
    src/main.cpp
    src/ConfigurationEventPrinter.h
    src/display.cpp
    src/display.h
    src/camera.cpp
    src/camera.h
    src/SerialPort.cpp
    src/SerialPort.h
    third-party/glad/src/glad.c
)
target_link_libraries(ahand PRIVATE pylon::pylon )
target_link_libraries(ahand PRIVATE ${OpenCV_LIBS})
# dynaflash
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad/include")

if (MSVC)
    file(GLOB OPENCV_DLLS "${PROJECT_SOURCE_DIR}/third-party/opencv/x64/vc15/bin/*.dll")
    add_custom_command(TARGET ahand 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:ahand>)

    file(GLOB OPENCV_DLLS "${PROJECT_SOURCE_DIR}/third-party/opencv/bin/*.dll")
    add_custom_command(TARGET ahand 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:ahand>)

    file(GLOB DYNAFLASH_DLLS "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/*.dll")
    add_custom_command(TARGET ahand 
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${DYNAFLASH_DLLS}
                    $<TARGET_FILE_DIR:ahand>)
endif (MSVC)

add_library(AHUnityPlugin SHARED 
src/UnityPlugin.cpp
src/UnityPlugin.h
)