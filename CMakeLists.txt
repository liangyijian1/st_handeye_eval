cmake_minimum_required(VERSION 3.20)

project(st_handeye_eval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    if (MSVC)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /Od /Ob0 /DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O0 -g -DNDEBUG")
    endif()
endif()

if(MSVC)
    add_compile_options("/utf-8")
    add_definitions(-DNOMINMAX)
endif()



message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# 禁用 CMake 3.24+ 的 SYSTEM 包含目录 /external: 标志（会导致 MSVC 找不到标准库）
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

set(OpenCV_DIR "D:/3rdParty/opencv-4.13.0/build/install/x64/vc18/lib" CACHE PATH "OpenCV CMake package directory" FORCE)
if(NOT DEFINED Eigen3_DIR OR NOT EXISTS "${Eigen3_DIR}/Eigen3Config.cmake")
    set(Eigen3_DIR "C:/Program Files (x86)/PCL/3rdParty/Eigen/share/eigen3/cmake" CACHE PATH "Eigen3 CMake package directory" FORCE)
endif()
if(NOT DEFINED glog_DIR OR NOT EXISTS "${glog_DIR}/glog-config.cmake")
    set(glog_DIR "D:/3rdParty/glog-master/install/lib/cmake/glog" CACHE PATH "glog CMake package directory" FORCE)
endif()
if(NOT DEFINED gflags_DIR OR NOT EXISTS "${gflags_DIR}/gflags-config.cmake")
    set(gflags_DIR "D:/3rdParty/gflags-master/install/lib/cmake/gflags" CACHE PATH "gflags CMake package directory" FORCE)
endif()
if(NOT DEFINED Ceres_DIR OR NOT EXISTS "${Ceres_DIR}/CeresConfig.cmake")
    set(Ceres_DIR "D:/3rdParty/ceres-solver-2.2.0/install/lib/cmake/Ceres" CACHE PATH "Ceres CMake package directory" FORCE)
endif()
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA Toolkit used by the local Ceres build" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" CACHE PATH "CUDA Toolkit used by the local Ceres build" FORCE)
foreach(cuda_cache_var IN ITEMS
    CUDAToolkit_BIN_DIR
    CUDAToolkit_NVCC_EXECUTABLE
    CUDA_CUDART
    CUDA_cudart_LIBRARY
    CUDA_cudart_static_LIBRARY
    CUDA_cublas_LIBRARY
    CUDA_cublasLt_LIBRARY
    CUDA_cusolver_LIBRARY
    CUDA_cusparse_LIBRARY
    CUDA_nvJitLink_LIBRARY
)
    unset(${cuda_cache_var} CACHE)
endforeach()

find_package(Ceres REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d features2d ximgproc)

add_executable(st_handeye_eval
    main.cpp
    handeye_eval.cpp
    handeye_eval.h
    packalgo_circle_grid_detector.cpp
    packalgo_circle_grid_detector.h
    super_edge_detector.cpp
    super_edge_detector.h
    stability_analysis.cpp
    stability_analysis.h
)


if(TARGET opencv_world)
    set(ST_HANDEYE_OPENCV_TARGET opencv_world)
else()
    set(ST_HANDEYE_OPENCV_TARGET ${OpenCV_LIBS})
endif()

target_link_libraries(st_handeye_eval PRIVATE
    ${ST_HANDEYE_OPENCV_TARGET}
    Ceres::ceres
)


target_compile_definitions(st_handeye_eval PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(st_handeye_eval PRIVATE /utf-8 /W3)
    target_link_options(st_handeye_eval PRIVATE $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>)
endif()

set(OPENCV_RUNTIME_DIR "D:/3rdParty/opencv-4.13.0/build/install/x64/vc18/bin")
if(EXISTS "${OPENCV_RUNTIME_DIR}")
    file(GLOB OPENCV_RUNTIME_DLLS "${OPENCV_RUNTIME_DIR}/opencv_*.dll")
    if(OPENCV_RUNTIME_DLLS)
        add_custom_command(TARGET st_handeye_eval POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${OPENCV_RUNTIME_DLLS}
                    $<TARGET_FILE_DIR:st_handeye_eval>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

set(CUDA_RUNTIME_DIR "${CUDAToolkit_ROOT}/bin")
if(EXISTS "${CUDA_RUNTIME_DIR}")
    set(CUDA_RUNTIME_DLL_NAMES
        cudart64_12.dll
        cublas64_12.dll
        cublasLt64_12.dll
        cusolver64_11.dll
        cusparse64_12.dll
        nvJitLink_120_0.dll
    )
    set(CUDA_RUNTIME_DLLS)
    foreach(cuda_runtime_dll IN LISTS CUDA_RUNTIME_DLL_NAMES)
        if(EXISTS "${CUDA_RUNTIME_DIR}/${cuda_runtime_dll}")
            list(APPEND CUDA_RUNTIME_DLLS "${CUDA_RUNTIME_DIR}/${cuda_runtime_dll}")
        endif()
    endforeach()
    if(CUDA_RUNTIME_DLLS)
        add_custom_command(TARGET st_handeye_eval POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CUDA_RUNTIME_DLLS}
                    $<TARGET_FILE_DIR:st_handeye_eval>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()
