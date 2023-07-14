# Root CMakeLists.txt for the overall Pylon project
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(GLAD_VER 4.3)
set(Python_EXECUTABLE "C:/Users/sens/.conda/envs/sandbox/python.exe")
cmake_policy(SET CMP0074 NEW)    # respect <PACKAGE>_ROOT variables in "find_package"
include(CMakePrintHelpers)

project(augmented_hands VERSION 0.0)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third-party/opencv" ${CMAKE_PREFIX_PATH})

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pylon 7.3.0 REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(dynaflash 
    src/display.cpp
    src/display.h
)
target_compile_definitions(dynaflash PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
target_link_libraries(dynaflash PRIVATE ${OpenCV_LIBS})

add_executable(ahand 
    src/main.cpp
    src/ConfigurationEventPrinter.h
    src/display.cpp
    src/display.h
    src/camera.cpp
    src/camera.h
    src/SerialPort.cpp
    src/SerialPort.h
    src/shader.h
    src/timer.cpp
    src/timer.h
    src/stb_image.h
    src/stb_image_write.h
    third-party/glad${GLAD_VER}/src/glad.c
)
target_compile_definitions(ahand PRIVATE STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
target_link_libraries(ahand PRIVATE pylon::pylon)
target_link_libraries(ahand PRIVATE ${OpenCV_LIBS})
# dynaflash
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
####################################### unity plugin
add_library(AHUnityPlugin SHARED 
    src/UnityPlugin.cpp
    src/UnityPlugin.h
    src/display.cpp
    src/display.h
)
target_compile_definitions(AHUnityPlugin PRIVATE NATIVECPPLIBRARY_EXPORTS DLL_BUILD)
target_include_directories(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
target_link_libraries(AHUnityPlugin PRIVATE ${OpenCV_LIBS})
#########################################
######################################### sample program
# add_executable(sample_program 
#     src/sample_program.cpp
# )
# target_link_libraries(sample_program PRIVATE ${OpenCV_LIBS})
# target_include_directories(sample_program PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
# target_link_libraries(sample_program PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
#########################################
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