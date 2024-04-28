cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(GLAD_VER 4.3)
set(Python_EXECUTABLE "C:/Users/sens/AppData/Local/Programs/Python/Python311/python.exe")  # replace with a python environment with numpy, mediapipe, and dev tools installed
set(ULTRALEAP_PATH_ROOT "$ENV{ProgramFiles}/Ultraleap")
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third-party/opencv")  # replace with opencv install path (e.g. "${CMAKE_SOURCE_DIR}/third-party/opencv" or "C:/OpenCV/build/install")
file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDA_TOOLKIT_ROOT_DIR)  # replaces # set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH}) for normalizing the path
file(TO_CMAKE_PATH "$ENV{PYLON_DEV_DIR}" PYLON_DEV_DIR)  # replaces # set(PYLON_DEV_DIR $ENV{PYLON_DEV_DIR}) for normalizing the path
set(PYLON_RT_DIR "${PYLON_DEV_DIR}/../Runtime/x64")
cmake_policy(SET CMP0074 NEW)  # respect <PACKAGE>_ROOT variables in "find_package"
# add_compile_definitions(CV_IGNORE_DEBUG_BUILD_GUARD)
include(CMakePrintHelpers)
project(casper VERSION 0.1 LANGUAGES C CXX)
find_package(Python 3.9 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pylon 7.3.0 REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED)
find_package(LeapSDK 5 REQUIRED PATHS "${ULTRALEAP_PATH_ROOT}")
if(DEFINED OpenCV_CUDA_VERSION)
    add_compile_definitions(OPENCV_WITH_CUDA)
    message("OpenCV was compiled with CUDA")
else()
    message("OpenCV was compiled without CUDA")
endif()
# message(OpenCV_CUDA_VERSION="${OpenCV_CUDA_VERSION}")
# message(OpenCV_COMPUTE_CAPABILITIES="${OpenCV_COMPUTE_CAPABILITIES}")
# message(OpenCV_RUNTIME="${OpenCV_RUNTIME}")
# message(Python_Interpreter_FOUND="${Python_Interpreter_FOUND}")
# message(Python_EXECUTABLE="${Python_EXECUTABLE}")
# message(Python_STDLIB="${Python_STDLIB}")
# message(Python_Development_FOUND="${Python_Development_FOUND}")
# message(PYTHON_VERSION="${Python_VERSION}")
# message(PYTHON_INCLUDE_DIRS="${PYTHON_INCLUDE_DIRS}")
# message(PYTHON_LIBRARIES="${PYTHON_LIBRARIES}")
# message(Python_LIBRARY_DIRS="${Python_LIBRARY_DIRS}")
# message(Python_RUNTIME_LIBRARY_DIRS="${Python_RUNTIME_LIBRARY_DIRS}")
################ nanobind boilerplate
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
target_include_directories(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
target_link_libraries(dynaflash PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
target_link_libraries(dynaflash PRIVATE ${OpenCV_LIBS})

################ basler camera python bindings
nanobind_add_module(basler 
    src/camera.cpp
    src/timer.cpp
)
target_compile_definitions(basler PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(basler PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(basler PRIVATE pylon::pylon)
target_link_libraries(basler PRIVATE ${OpenCV_LIBS})

################ leap motion python bindings
nanobind_add_module(leap 
    src/leapCPP.cpp
)
target_compile_definitions(leap PRIVATE PYTHON_BINDINGS_BUILD)
target_include_directories(leap PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(leap PRIVATE LeapSDK::LeapC)

################ main application
add_executable(casper 
    src/main.cpp
    src/display.cpp
    src/camera.cpp
    src/gl_camera.cpp
    src/SerialPort.cpp
    src/text.cpp
    src/timer.cpp
    src/leapCPP.cpp
    src/texture.cpp
    src/diffuse.cpp
    src/base64.cpp
    src/skinned_model.cpp
    src/skinned_shader.cpp
    src/post_process.cpp
    src/utils.cpp
    src/cnpy.cpp
    src/point_cloud.cpp
    src/fbo.cpp
    src/quad.cpp
    src/helpers.cpp
    src/kalman.cpp
    src/grid.cpp
    src/user_study.cpp
    src/MidiControllerAPI.cpp
    src/guess_pose_game.cpp
    src/guess_char_game.cpp
    src/guess_animal_game.cpp
    src/dear_widgets.cpp
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
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/include")
# pylon
target_link_libraries(casper PRIVATE pylon::pylon)
# opencv
target_link_libraries(casper PRIVATE ${OpenCV_LIBS})
# winsock
target_link_libraries(casper PRIVATE wsock32 ws2_32)
# dynaflash
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
target_link_libraries(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# GLFW
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
target_link_libraries(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# GLAD
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# assimp
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
target_link_libraries(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# glm
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# leap
target_link_libraries(casper PRIVATE LeapSDK::LeapC)
# stb
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# freetype
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
target_link_libraries(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
# zlib
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/include")
target_link_libraries(casper PRIVATE debug "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib_debug.lib" optimized "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib.lib")
# imgui
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui")
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/backends")
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/misc/cpp")
# libicp
target_include_directories(casper PRIVATE "C:/src/boost_1_84_0")
target_include_directories(casper PRIVATE "${PROJECT_SOURCE_DIR}/third-party/libicp")
# python
target_link_libraries(casper PRIVATE Python::Python Python::NumPy)
####################################### registration example
# add_executable(registration_example 
#     src/registration_example.cpp
#     src/timer.cpp
#     ${LIBICP_SOURCES}
# )
# target_include_directories(registration_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_include_directories(registration_example PRIVATE "C:/src/boost_1_84_0")
# target_include_directories(registration_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/libicp")
# target_link_libraries(registration_example PRIVATE ${OpenCV_LIBS})
####################################### deformation example
# add_executable(deformation_example
#     src/deformation_example.cpp
#     src/timer.cpp
#     src/cnpy.cpp
#     src/point_cloud.cpp
#     src/helpers.cpp
#     src/fbo.cpp
#     src/quad.cpp
#     src/texture.cpp
#     src/grid.cpp
#     third-party/glad${GLAD_VER}/src/glad.c
# )
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # GLFW
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
# target_link_libraries(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# # GLAD
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# # opencv
# target_link_libraries(deformation_example PRIVATE ${OpenCV_LIBS})
# # stb
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# # glm
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# # python
# target_link_libraries(deformation_example PRIVATE Python::Python Python::NumPy)
# # zlib
# target_include_directories(deformation_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/include")
# target_link_libraries(deformation_example PRIVATE debug "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib_debug.lib" optimized "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib.lib")

####################################### kalman filter example
# add_executable(kalman_example
#     src/kalman_example.cpp
#     src/kalman.cpp
#     src/timer.cpp
# )
# target_include_directories(kalman_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # opencv
# target_link_libraries(kalman_example PRIVATE ${OpenCV_LIBS})
####################################### ONYX example
# add_executable(onyx_example
#     src/onyx_example.cpp
#     src/timer.cpp
# )
# target_include_directories(onyx_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # onyx
# target_link_libraries(onyx_example PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
# target_include_directories(onyx_example PRIVATE "${ONNXRUNTIME_ROOTDIR}/include")
# target_include_directories(onyx_example PRIVATE "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
# target_compile_definitions(onyx_example PRIVATE USE_CUDA)
# # opencv
# target_link_libraries(onyx_example PRIVATE ${OpenCV_LIBS})
####################################### unity plugin
# add_library(AHUnityPlugin SHARED 
#     src/UnityPlugin.cpp
#     src/display.cpp
# )
# target_compile_definitions(AHUnityPlugin PRIVATE NATIVECPPLIBRARY_EXPORTS DLL_BUILD)
# target_include_directories(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_include_directories(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
# target_link_libraries(AHUnityPlugin PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# target_link_libraries(AHUnityPlugin PRIVATE ${OpenCV_LIBS})

####################################### coaxial calibration
# add_executable(coaxial_calib 
#     src/coaxial_calibration.cpp
#     src/display
#     src/camera.cpp
#     src/timer.cpp
#     src/texture.cpp
#     src/quad.cpp
#     src/text.cpp
#     src/point_cloud.cpp
#     third-party/glad${GLAD_VER}/src/glad.c
# )
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # pylon
# target_link_libraries(coaxial_calib PRIVATE pylon::pylon)
# # opencv
# target_link_libraries(coaxial_calib PRIVATE ${OpenCV_LIBS})
# # dynaflash
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
# target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# # GLFW
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
# target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# # GLAD
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# # glm
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# # assimp
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
# target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# # stb
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# # freetype
# target_include_directories(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
# target_link_libraries(coaxial_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
####################################### leap->projector calibration
# add_executable(leap_calib 
#     src/leap_calibration.cpp
#     src/display
#     src/camera.cpp
#     src/timer.cpp
#     src/texture.cpp
#     src/quad.cpp
#     src/text.cpp
#     src/fbo.cpp
#     src/leapCPP.cpp
#     src/cnpy.cpp
#     src/point_cloud.cpp
#     src/helpers.cpp
#     third-party/glad${GLAD_VER}/src/glad.c
# )
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # pylon
# target_link_libraries(leap_calib PRIVATE pylon::pylon)
# # opencv
# target_link_libraries(leap_calib PRIVATE ${OpenCV_LIBS})
# # dynaflash
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
# target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
# # GLFW
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
# target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# # GLAD
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# # glm
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
# # assimp
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/include")
# target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/assimp-vc143-mt.lib")
# # stb
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# # freetype
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/include")
# target_link_libraries(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/freetype.lib")
# # leap
# target_link_libraries(leap_calib PRIVATE LeapSDK::LeapC)
# # zlib
# target_include_directories(leap_calib PRIVATE "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/include")
# target_link_libraries(leap_calib PRIVATE debug "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib_debug.lib" optimized "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.11/lib/zlib.lib")
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
# add_executable(dyna_example 
#     src/dyna_example.cpp
# )
# target_link_libraries(dyna_example PRIVATE ${OpenCV_LIBS})
# target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/include")
# target_include_directories(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(dyna_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/DynaFlash200.lib")
######################################### diffuse example program
# add_executable(diffuse_example 
#     src/diffuse_example.cpp
#     src/diffuse.cpp
#     src/base64.cpp
#     src/timer.cpp
# )
# target_include_directories(diffuse_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(diffuse_example PRIVATE wsock32 ws2_32 ${OpenCV_LIBS})
# target_include_directories(diffuse_example PRIVATE "${PROJECT_SOURCE_DIR}/third-party/stb")
# target_link_libraries(diffuse_example PRIVATE Python::Python Python::NumPy)
######################################### download benchmark example program
# add_executable(benchmark_download
#     src/benchmark_download.cpp
#     src/timer.cpp
#     third-party/imgui/imgui.cpp
#     third-party/imgui/imgui_draw.cpp
#     third-party/imgui/imgui_tables.cpp
#     third-party/imgui/imgui_widgets.cpp
#     third-party/imgui/imgui_demo.cpp
#     third-party/imgui/backends/imgui_impl_glfw.cpp
#     third-party/imgui/backends/imgui_impl_opengl3.cpp
#     third-party/imgui/misc/cpp/imgui_stdlib.cpp
#     third-party/glad${GLAD_VER}/src/glad.c
# )
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/include")
# # pylon
# target_link_libraries(benchmark_download PRIVATE pylon::pylon)
# # opencv
# target_link_libraries(benchmark_download PRIVATE ${OpenCV_LIBS})
# # GLFW
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/include")
# target_link_libraries(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glfw-3.3.8/lib/glfw3.lib")
# # GLAD
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glad${GLAD_VER}/include")
# # imgui
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui")
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/backends")
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/imgui/misc/cpp")
# # glm
# target_include_directories(benchmark_download PRIVATE "${PROJECT_SOURCE_DIR}/third-party/glm")
######################################### optical flow benchmark example program
# add_executable(benchmark_example
#     src/benchmark_flow.cpp
#     src/timer.cpp
# )
# target_include_directories(benchmark_example PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(benchmark_example PRIVATE wsock32 ws2_32 ${OpenCV_LIBS})
######################################### post build copy dlls
file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/${OpenCV_RUNTIME}/bin/*.dll")
# message(OPENCV_DLLS="${OPENCV_DLLS}")
list(LENGTH OPENCV_DLLS RES_LEN)
if(NOT RES_LEN EQUAL 0)
    add_custom_command(TARGET casper 
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${OPENCV_DLLS}
                        $<TARGET_FILE_DIR:casper>)
endif()

file(GLOB DYNAFLASH_DLLS "${PROJECT_SOURCE_DIR}/third-party/dynaflash/lib/*.dll")
add_custom_command(TARGET casper 
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DYNAFLASH_DLLS}
                $<TARGET_FILE_DIR:casper>)

get_target_property(LEAPC_IMPORTED_CONFIG LeapSDK::LeapC IMPORTED_CONFIGURATIONS)
get_target_property(LEAPC_SHARED_LIB_PATH LeapSDK::LeapC IMPORTED_LOCATION_${LEAPC_IMPORTED_CONFIG})
                
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${LEAPC_SHARED_LIB_PATH}
                $<TARGET_FILE_DIR:casper>)

file(GLOB ASSIMP_DLLS "${PROJECT_SOURCE_DIR}/third-party/assimp/lib/*.dll") 
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ASSIMP_DLLS}
                $<TARGET_FILE_DIR:casper>)

file(GLOB FREETYPE_DLLS "${PROJECT_SOURCE_DIR}/third-party/freetype/lib/*.dll") 
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FREETYPE_DLLS}
                $<TARGET_FILE_DIR:casper>)

file(GLOB PYTHON_DLL "${Python_RUNTIME_LIBRARY_DIRS}/python*.dll")
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PYTHON_DLL}
                $<TARGET_FILE_DIR:casper>)

file(GLOB BASLER_DLLS "${PYLON_RT_DIR}/*.dll")
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${BASLER_DLLS}
                $<TARGET_FILE_DIR:casper>)

file(GLOB PYTHON_EMBEDDING "${PROJECT_SOURCE_DIR}/src/predict.py")
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PYTHON_EMBEDDING}
                $<TARGET_FILE_DIR:casper>)

file(GLOB PYTHON_EMBEDDING "${PROJECT_SOURCE_DIR}/src/chatgpt.py")
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PYTHON_EMBEDDING}
                $<TARGET_FILE_DIR:casper>)

file(GLOB PYTHON_EMBEDDING "${PROJECT_SOURCE_DIR}/src/calibration.py")
add_custom_command(TARGET casper
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PYTHON_EMBEDDING}
                $<TARGET_FILE_DIR:casper>)
# file(GLOB ZLIB_DLLS "${PROJECT_SOURCE_DIR}/third-party/zlib-1.2.3-bin/bin/*.dll") 
# add_custom_command(TARGET casper
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${ZLIB_DLLS}
#                     $<TARGET_FILE_DIR:casper>)