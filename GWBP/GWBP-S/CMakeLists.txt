# @file CMakeLists.txt
# @author Fande Yu 
# @version 0.1 
# @copyright 
#
# ChangeLog
# AUTHOR      | TIME       | VERSION       | DESCRIPTION
# ------      | ----       | -------       | -----------
# Fande Yu    | 2020/10/14 | 0.1.20201014  | Create this file 
# Fande Yu	  | 2020/12/26 | 0.1.20201226  | Build Seperatedly

#Notice
#1. export CC=gcc and CXX=g++ ahead of running this CMake file.
#2. If this program will be run on multi-nodes where NCCL may be necessary, this file need to be modified.

# CMAKE REQUIREMENT
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(TomoProject LANGUAGES CXX CUDA)

set(TomoProject_VERSION_MAJOR 0)
set(TomoProject_VERSION_MINOR 1)
set(TomoProject_VERSION_ADDIT 20201014)

#Get git commit
execute_process(
    COMMAND git rev-parse HEAD 
    OUTPUT_VARIABLE COMMIT_VERSION 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Welcome to TomoProject ${TomoProject_VERSION_MAJOR}.${TomoProject_VERSION_MINOR}.${TomoProject_VERSION_ADDIT}!")


# 配置CUDA使用gcc编译，同时保持C++使用icpc
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)  # Tesla V100对应的架构

# 强制CUDA使用gcc作为主机编译器（关键配置）
set(CMAKE_CUDA_HOST_COMPILER gcc)
# 向CUDA传递gcc兼容的编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -Xcompiler -fopenmp,-Wall")

#For TomoConfig.h.in
set(COMMIT_VERSION_QUOTE "\"${COMMIT_VERSION}\"")

# C++代码使用Intel编译器icpc
set(CMAKE_C_COMPILER icpc)
set(CMAKE_CXX_COMPILER icpc)
set(COMMON_FLAGS "-std=c++11 ${COMMON_FLAGS}")

#For FFTW
option(SINGLE_PRECISION "Whether to use single float precision in calculation?" ON)

#Only option is useful for #cmakedefine in TomoConfig.h.in
option(CPU_VERSION "Whether to use CPU to calculate?" ON)
# 新增CUDA版本开关
option(CUDA_VERSION "Whether to use CUDA to calculate?" ON)


#Add compilation flag part1: build type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "release")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")
if("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    set(COMMON_FLAGS "${COMMON_FLAGS} -O3 -fp-model fast=2 -ipo -qopenmp -fno-alias -xCORE-AVX512 -qopt-zmm-usage=high")
    message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile ${CMAKE_PROJECT_NAME} with ${COMMON_FLAGS} flag.")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    set(COMMON_FLAGS "${COMMON_FLAGS} -O3 -fp-model fast=2 -ipo -qopenmp -fno-alias -xCORE-AVX512 -qopt-zmm-usage=high")
    set(COMMON_FLAGS "${COMMON_FLAGS} -g")
    message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}, compile ${CMAKE_PROJECT_NAME} with ${COMMON_FLAGS} flag.")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "release")

if("${CPU_VERSION}")
    message(STATUS "Try to build Tomo-Project in CPU version (using icpc).")
endif("${CPU_VERSION}")

if("${CUDA_VERSION}")
    message(STATUS "Try to build Tomo-Project in CUDA version (using nvcc with gcc).")
endif("${CUDA_VERSION}")


#Add compilation flags part2: library flags
#set(COMMON_FLAGS "${COMMON_FLAGS} -lutil -lcore -lz -lpthread -lm -lfftw3f -lfftw3f_threads")
set(COMMON_FLAGS "${COMMON_FLAGS} -lz -lpthread -mkl ")

set(CMAKE_C_FLAGS "${COMMON_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -g")

#output flags
message(STATUS "CMAKE_C_FLAGS : ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")  # 验证CUDA主机编译器


# 载入FFTW的CMakeLists文件
include(${CMAKE_SOURCE_DIR}/cmake/BuildFFTW.cmake)
message(STATUS "FFTW Path: ${FFTW_PATH}")
message(STATUS "FFTW Libraries: ${FFTW_LIBRARIES}")
include_directories(${FFTW_PATH}/include)

include_directories(${CMAKE_SOURCE_DIR}/external/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/src/include)


# CPU版本配置
if("${CPU_VERSION}")
    file(GLOB_RECURSE CORE_SOURCES_CPU ${CMAKE_SOURCE_DIR}/src/core/*.cpp)
    add_library(core_CPU STATIC ${CORE_SOURCES_CPU})
    add_dependencies(core_CPU FFTW)
    target_link_libraries(core_CPU PUBLIC ${FFTW_LIBRARIES})

    file(GLOB_RECURSE UTIL_SOURCES ${CMAKE_SOURCE_DIR}/src/util/*.cpp)
    add_library(util STATIC ${UTIL_SOURCES})
    
    file(GLOB_RECURSE MAIN_SOURCES ${CMAKE_SOURCE_DIR}/src/main/*.cpp)
    foreach(MAINSRC ${MAIN_SOURCES})
        get_filename_component(MAINNAME ${MAINSRC} NAME_WE)
        set(MAINNAME "${MAINNAME}_cpu")
        add_executable(${MAINNAME} ${MAINSRC})
        target_link_libraries(${MAINNAME} PUBLIC core_CPU util ${FFTW_LIBRARIES})
    endforeach()
endif("${CPU_VERSION}")


if("${CUDA_VERSION}")
    # 1. 收集所有核心源文件：
    #    - 包含修改后的 .cu 文件
    #    - 其他未修改的 .cpp 文件（通过 GLOB 自动收集，避免手动列举）
    file(GLOB_RECURSE CORE_SOURCES_CUDA 
        ${CMAKE_SOURCE_DIR}/src/core/*.cu  # 包含修改后的 ReconstructionAlgo_WBP_RAM.cu
        ${CMAKE_SOURCE_DIR}/src/core/*.cpp  # 自动包含其他所有 .cpp（如 MRC.cpp、CTF.cpp 等）
    )

    if(CORE_SOURCES_CUDA)
        add_library(core_CUDA STATIC ${CORE_SOURCES_CUDA})
        
        # 2. 依赖和编译选项：
        #    - 保留 FFTW 依赖（若 CUDA 版本仍需使用 FFTW，否则可移除）
        add_dependencies(core_CUDA FFTW)
        
        # 编译选项：
        target_compile_options(core_CUDA PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                # 调试模式：仅在开发时启用（-g -G 会降低性能）
                # -g -G  
                -arch=sm_70  # V100 计算能力（根据实际显卡调整，如 sm_80 对应 A100）
                -Xcompiler "-mavx512f,-mavx512dq"  # 传递 AVX-512 给主机编译器
                --expt-relaxed-constexpr  # 允许 CUDA 中的 constexpr 扩展（可选）
            >
            $<$<COMPILE_LANGUAGE:CXX>:  # 对 .cpp 文件单独设置 C++ 选项
                -mavx512f -mavx512dq
            >
        )
        
        # 3. 链接库：
        #    - 必选：CUDA 运行时（cudart）和 FFT 库（cufft）
        #    - 保留 FFTW（若仍需）、zlib、util 工具库
        target_link_libraries(core_CUDA PUBLIC 
            util
            ${FFTW_LIBRARIES}  # 若 CUDA 版本完全改用 cufft，可移除此行
            ${ZLIB_LIBRARIES}
            cufft  # CUDA FFT 库（替代 FFTW 或与 FFTW 共存）
            cudart # CUDA 运行时库
        )
        
        # 4. 头文件路径：确保包含所有依赖的头文件（Eigen、外部库等）
        target_include_directories(core_CUDA PUBLIC
            ${FFTW_PATH}/include
            ${CMAKE_SOURCE_DIR}/external/Eigen
            ${CMAKE_SOURCE_DIR}/src/include
            ${CUDA_INCLUDE_DIRS}  # 显式添加 CUDA 头文件路径（可选，现代 CMake 通常自动处理）
        )

        # 5. 生成 CUDA 版本可执行文件
        #    - MAIN_SOURCES 需提前定义（与 CPU 版本共用，或在此处重新收集）
        file(GLOB_RECURSE MAIN_SOURCES ${CMAKE_SOURCE_DIR}/src/main/*.cpp)  # 若未提前定义
        foreach(MAINSRC ${MAIN_SOURCES})
            get_filename_component(MAINNAME ${MAINSRC} NAME_WE)
            set(MAINNAME "${MAINNAME}_cuda")
            add_executable(${MAINNAME} ${MAINSRC})
            # 可执行文件编译选项（与库保持一致）
            target_compile_options(${MAINNAME} PRIVATE -mavx512f -mavx512dq)
            # 链接核心库和依赖
            target_link_libraries(${MAINNAME} PUBLIC core_CUDA util ${ZLIB_LIBRARIES})
        endforeach()
    else()
        message(WARNING "No CUDA source files (.cu) or core .cpp files found in src/core/")
    endif()
endif("${CUDA_VERSION}")