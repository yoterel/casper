cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(GLAD_VER 4.3)
set(Python_EXECUTABLE "C:/Users/sens/.conda/envs/sandbox/python.exe")
set(ULTRALEAP_PATH_ROOT "$ENV{ProgramFiles}/Ultraleap")
cmake_policy(SET CMP0074 NEW)    # respect <PACKAGE>_ROOT variables in "find_package"
include(CMakePrintHelpers)

project(augmented_hands VERSION 0.1 LANGUAGES C CXX CUDA)
#project(augmented_hands LANGUAGES CUDA CXX)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third-party/opencv" ${CMAKE_PREFIX_PATH})

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pylon 7.3.0 REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED)
find_package(LeapSDK 5 REQUIRED PATHS "${ULTRALEAP_PATH_ROOT}")
################ nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

################ dyna flash python bindings
nanobind_add_module(dynaflash 
    src/display.cpp
)
target_compile_definitions(dynaflash PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
target_link_libraries(dynaflash PRIVATE ${OpenCV_LIBS})

################ basler camera python bindings
nanobind_add_module(basler 
    src/camera.cpp
)
target_compile_definitions(basler PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(basler PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(basler PRIVATE pylon::pylon)
target_link_libraries(basler PRIVATE ${OpenCV_LIBS})

################ leap motion python bindings
nanobind_add_module(leap 
    src/leap.cpp
)
target_compile_definitions(leap PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(leap PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(leap PRIVATE LeapSDK::LeapC)

################ main application
add_executable(ahand 
    src/main.cpp
    src/display.cpp
    src/camera.cpp
    src/gl_camera.cpp
    src/SerialPort.cpp
    src/text.cpp
    src/timer.cpp
    src/leap.cpp
    src/texture.cpp
    src/skinned_model.cpp
    src/skinned_shader.cpp
    src/canvas.cpp
    src/utils.cpp
    src/cuda/cuda2gl.cu
    src/image_process.cpp
    src/cnpy.cpp
    third-party/glad${GLAD_VER}/src/glad.c
)
# target_compile_definitions(ahand PRIVATE STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
set_target_properties(ahand PROPERTIES CUDA_ARCHITECTURES "72")
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common")
# cuda
target_include_directories(ahand PRIVATE "$ENV{CUDA_PATH}/include")
target_link_libraries(ahand PRIVATE "$ENV{CUDA_PATH}/lib/x64/cudart.lib")
# npp
set_target_properties(ahand PROPERTIES CUDA_ARCHITECTURES "72")
# target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common")
target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common/UtilNPP")
# target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common/FreeImage/Dist/x64")
target_link_libraries(ahand PRIVATE 
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppc.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppial.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppicc.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppidei.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppif.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppig.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppim.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppist.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppisu.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppitc.lib"
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/npps.lib")
# pylon
target_link_libraries(ahand PRIVATE pylon::pylon)
# opencv
target_link_libraries(ahand PRIVATE ${OpenCV_LIBS})
# dynaflash
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# assimp
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# glm
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# leap
target_link_libraries(ahand PRIVATE LeapSDK::LeapC)
# stb
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# freetype
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
target_link_libraries(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
# zlib
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/include")
target_link_libraries(ahand PRIVATE debug "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib_debug.lib" optimized "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib.lib")
####################################### unity plugin
add_library(AHUnityPlugin SHARED 
    src/UnityPlugin.cpp
    src/display.cpp
)
target_compile_definitions(AHUnityPlugin PRIVATE NATIVECPPLIBRARY_EXPORTS DLL_BUILD)
target_include_directories(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
target_link_libraries(AHUnityPlugin PRIVATE ${OpenCV_LIBS})

######################################### cuda sample program
# add_executable(cuda_example 
#     # src/add.cu
#     src/cuda/boxFilter.cu
#     # src/cuda/cuda2gl.cpp
#     # src/cuda/cuda2gl.cu
# )
# set_target_properties(cuda_example PROPERTIES CUDA_ARCHITECTURES "72")
# target_include_directories(cuda_example PRIVATE "C:/src/cuda-samples/Common")
# target_include_directories(cuda_example PRIVATE "C:/src/cuda-samples/Common/UtilNPP")
# target_include_directories(cuda_example PRIVATE "C:/src/cuda-samples/Common/FreeImage/Dist/x64")
# target_link_libraries(cuda_example PRIVATE "C:/src/cuda-samples/Common/FreeImage/Dist/x64/freeimage.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppc.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppial.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppicc.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppidei.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppif.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppig.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppim.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppist.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppisu.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/nppitc.lib")
# target_link_libraries(cuda_example PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64/npps.lib")
######################################### dynaflash example program
add_executable(dyna_example 
    src/dyna_example.cpp
)
target_link_libraries(dyna_example PRIVATE ${OpenCV_LIBS})
target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
######################################### post build copy dlls
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

get_target_property(LEAPC_IMPORTED_CONFIG LeapSDK::LeapC IMPORTED_CONFIGURATIONS)
get_target_property(LEAPC_SHARED_LIB_PATH LeapSDK::LeapC IMPORTED_LOCATION_${LEAPC_IMPORTED_CONFIG})
                
add_custom_command(TARGET ahand
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${LEAPC_SHARED_LIB_PATH}
                $<TARGET_FILE_DIR:ahand>)

file(GLOB ASSIMP_DLLS "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/*.dll") 
add_custom_command(TARGET ahand
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ASSIMP_DLLS}
                $<TARGET_FILE_DIR:ahand>)

file(GLOB FREETYPE_DLLS "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/*.dll") 
add_custom_command(TARGET ahand
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FREETYPE_DLLS}
                $<TARGET_FILE_DIR:ahand>)
# file(GLOB ZLIB_DLLS "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.3-bin/bin/*.dll") 
# add_custom_command(TARGET ahand
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${ZLIB_DLLS}
#                     $<TARGET_FILE_DIR:ahand>)