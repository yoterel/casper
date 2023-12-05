cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(GLAD_VER 4.3)
set(Python_EXECUTABLE "C:/Users/sens/.conda/envs/sandbox/python.exe")  # 558: C:/Users/sens/miniconda3/envs/sd/python.exe  559: C:/Users/sens/.conda/envs/sandbox/python.exe
set(Python_ROOT_DIR "C:/Users/sens/miniconda3/envs/sd/")
set(ULTRALEAP_PATH_ROOT "$ENV{ProgramFiles}/Ultraleap")
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third-party/opencv")
set(OPENCV_COMPILER "vc15")
# set(OpenCV_DIR "C:/OpenCV_Build/install_debug")
# set(OPENCV_COMPILER "vc16")
set(CUDA_EXAMPLES "C:/src/cuda-samples")
set(CUDA_NPP_LIBS_LOC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")  # 11.8, 12.2
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/")  # 11.8, 12.2
# set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third-party/opencv" ${CMAKE_PREFIX_PATH})
cmake_policy(SET CMP0074 NEW)    # respect <PACKAGE>_ROOT variables in "find_package"
include(CMakePrintHelpers)
project(augmented_hands VERSION 0.1 LANGUAGES C CXX CUDA)
#project(augmented_hands LANGUAGES CUDA CXX)
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
find_package(pylon 7.3.0 REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED)
find_package(LeapSDK 5 REQUIRED PATHS "${ULTRALEAP_PATH_ROOT}")
# message(PYTHON_FOUND="${Python_FOUND}")
# message(Python_Interpreter_FOUND="${Python_Interpreter_FOUND}")
# message(Python_EXECUTABLE="${Python_EXECUTABLE}")
# message(Python_STDLIB="${Python_STDLIB}")
# message(Python_Development_FOUND="${Python_Development_FOUND}")
# message(PYTHON_VERSION="${Python_VERSION}")
# message(PYTHON_INCLUDE_DIRS="${PYTHON_INCLUDE_DIRS}")
# message(PYTHON_LIBRARIES="${PYTHON_LIBRARIES}")
# message(Python_LIBRARY_DIRS="${Python_LIBRARY_DIRS}")
# message(Python_RUNTIME_LIBRARY_DIRS="${Python_RUNTIME_LIBRARY_DIRS}")
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
    src/post_process.cpp
    src/utils.cpp
    src/cuda/cuda2gl.cu
    src/image_process.cpp
    src/cnpy.cpp
    src/fbo.cpp
    src/quad.cpp
    src/helpers.cpp
    third-party/imgui/imgui.cpp
    third-party/imgui/imgui_draw.cpp
    third-party/imgui/imgui_tables.cpp
    third-party/imgui/imgui_widgets.cpp
    third-party/imgui/imgui_demo.cpp
    third-party/imgui/backends/imgui_impl_glfw.cpp
    third-party/imgui/backends/imgui_impl_opengl3.cpp
    third-party/imgui/misc/cpp/imgui_stdlib.cpp
    third-party/glad${GLAD_VER}/src/glad.c
)
# target_compile_definitions(ahand PRIVATE STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
# target_compile_definitions(ahand PRIVATE CV_IGNORE_DEBUG_BUILD_GUARD)
set_target_properties(ahand PROPERTIES CUDA_ARCHITECTURES "72")
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ahand PRIVATE "${CUDA_EXAMPLES}/Common")
# cuda
target_include_directories(ahand PRIVATE "$ENV{CUDA_PATH}/include")
target_link_libraries(ahand PRIVATE "$ENV{CUDA_PATH}/lib/x64/cudart.lib")
# npp
set_target_properties(ahand PROPERTIES CUDA_ARCHITECTURES "72")
# target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common")
target_include_directories(ahand PRIVATE "${CUDA_EXAMPLES}/Common/UtilNPP")
# target_include_directories(ahand PRIVATE "C:/src/cuda-samples/Common/FreeImage/Dist/x64")
target_link_libraries(ahand PRIVATE 
                        "${CUDA_NPP_LIBS_LOC}/nppc.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppial.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppicc.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppidei.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppif.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppig.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppim.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppist.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppisu.lib"
                        "${CUDA_NPP_LIBS_LOC}/nppitc.lib"
                        "${CUDA_NPP_LIBS_LOC}/npps.lib")
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
# imgui
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui")
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/backends")
target_include_directories(ahand PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/misc/cpp")
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

####################################### coaxial calibration
add_executable(coaxial_calib 
    src/coaxial_calibration.cpp
    src/display
    src/camera.cpp
    src/timer.cpp
    src/texture.cpp
    src/quad.cpp
    src/text.cpp
    src/point_cloud.cpp
    third-party/glad${GLAD_VER}/src/glad.c
)
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/include")
# pylon
target_link_libraries(coaxial_calib PRIVATE pylon::pylon)
# opencv
target_link_libraries(coaxial_calib PRIVATE ${OpenCV_LIBS})
# dynaflash
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# glm
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# assimp
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# stb
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# freetype
target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
####################################### leap->proejctor calibration
add_executable(leap_calib 
    src/leap_calibration.cpp
    src/display
    src/camera.cpp
    src/timer.cpp
    src/texture.cpp
    src/quad.cpp
    src/text.cpp
    src/fbo.cpp
    src/leap.cpp
    src/cnpy.cpp
    src/point_cloud.cpp
    src/helpers.cpp
    third-party/glad${GLAD_VER}/src/glad.c
)
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/include")
# pylon
target_link_libraries(leap_calib PRIVATE pylon::pylon)
# opencv
target_link_libraries(leap_calib PRIVATE ${OpenCV_LIBS})
# dynaflash
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# glm
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# assimp
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# stb
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# freetype
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
# leap
target_link_libraries(leap_calib PRIVATE LeapSDK::LeapC)
# zlib
target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/include")
target_link_libraries(leap_calib PRIVATE debug "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib_debug.lib" optimized "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib.lib")
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
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppc.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppial.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppicc.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppidei.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppif.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppig.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppim.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppist.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppisu.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/nppitc.lib")
# target_link_libraries(cuda_example PRIVATE "${CUDA_NPP_LIBS_LOC}/npps.lib")
######################################### dynaflash example program
add_executable(dyna_example 
    src/dyna_example.cpp
)
target_link_libraries(dyna_example PRIVATE ${OpenCV_LIBS})
target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
######################################### diffuse example program
add_executable(diffuse_example 
    src/diffuse_example.cpp
    src/diffuse.cpp
    src/base64.cpp
)
# target_include_directories(diffuse_example PRIVATE ${PYTHON_INCLUDE_DIRS})
# target_link_libraries(diffuse_example PRIVATE )
target_include_directories(diffuse_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(diffuse_example PRIVATE wsock32 ws2_32 ${OpenCV_LIBS})
# target_link_libraries(diffuse_example PRIVATE ${PYTHON_LIBRARIES})
# target_link_libraries(diffuse_example PRIVATE Python::Python)  # Python::NumPy
# target_link_libraries(diffuse_example PRIVATE ${Python_LIBRARY_DIRS}/python310.lib)
######################################### opencv benchmark example program
# add_executable(benchmark_example
#     src/benchmark.cpp
#     src/timer.cpp
# )
# target_include_directories(benchmark_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(benchmark_example PRIVATE wsock32 ws2_32 ${OpenCV_LIBS})
######################################### post build copy dlls
file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/${OPENCV_COMPILER}/bin/*.dll")
# message(OPENCV_DLLS="${OPENCV_DLLS}")
list(LENGTH OPENCV_DLLS RES_LEN)
if(NOT RES_LEN EQUAL 0)
# DIR is empty, do something
    add_custom_command(TARGET ahand 
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${OPENCV_DLLS}
                        $<TARGET_FILE_DIR:ahand>)
endif()

# file(GLOB OPENCV_DLLS "${OpenCV_DIR}/bin/*.dll")
# # message(OPENCV_DLLS="${OPENCV_DLLS}")
# list(LENGTH OPENCV_DLLS RES_LEN)
# if(NOT RES_LEN EQUAL 0)
#     add_custom_command(TARGET ahand 
#                         POST_BUILD
#                         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                         ${OPENCV_DLLS}
#                         $<TARGET_FILE_DIR:ahand>)
# endif()

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