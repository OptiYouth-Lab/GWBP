/*******************************************************************
 *       Filename:  cuda_utils.h
 *                                                                 
 *    Description:  CUDA工具类 - 提供错误检查、内存管理等基础功能
 *                                                                 
 *        Version:  1.0 (GPU Migration)                                           
 *        Created:  12/18/2024
 *       Compiler:  nvcc                                          
 *                                                                 
 *         Author:  Based on CPU version by Ruan Huabin
 *                  GPU Migration Implementation                             
 *                                                                 
 *******************************************************************/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 错误检查宏
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error at %s:%d - Error code %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Launch Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// GPU设备信息
// ============================================================================

void printGPUInfo();
int getGPUDeviceCount();
void setGPUDevice(int device);

// ============================================================================
// 内存管理工具
// ============================================================================

template<typename T>
class GPUBuffer {
private:
    T* d_ptr;
    size_t size;
    bool allocated;

public:
    // 构造函数
    GPUBuffer() : d_ptr(nullptr), size(0), allocated(false) {}
    
    explicit GPUBuffer(size_t count) : d_ptr(nullptr), size(count), allocated(false) {
        allocate(count);
    }
    
    // 析构函数
    ~GPUBuffer() {
        free();
    }
    
    // 禁止拷贝
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    
    // 分配内存
    void allocate(size_t count) {
        if (allocated) {
            free();
        }
        size = count;
        CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
        allocated = true;
    }
    
    // 释放内存
    void free() {
        if (allocated && d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
            allocated = false;
        }
    }
    
    // 从CPU拷贝到GPU
    void copyFromHost(const T* h_ptr, size_t count = 0) {
        if (count == 0) count = size;
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), 
                              cudaMemcpyHostToDevice));
    }
    
    // 从GPU拷贝到CPU
    void copyToHost(T* h_ptr, size_t count = 0) const {
        if (count == 0) count = size;
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), 
                              cudaMemcpyDeviceToHost));
    }
    
    // 异步拷贝
    void copyFromHostAsync(const T* h_ptr, cudaStream_t stream, size_t count = 0) {
        if (count == 0) count = size;
        CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, count * sizeof(T), 
                                   cudaMemcpyHostToDevice, stream));
    }
    
    void copyToHostAsync(T* h_ptr, cudaStream_t stream, size_t count = 0) const {
        if (count == 0) count = size;
        CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, count * sizeof(T), 
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    // 内存清零
    void memset(int value = 0) {
        CUDA_CHECK(cudaMemset(d_ptr, value, size * sizeof(T)));
    }
    
    // 获取指针
    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    
    // 获取大小
    size_t getSize() const { return size; }
    bool isAllocated() const { return allocated; }
};

// ============================================================================
// cuFFT计划管理
// ============================================================================

class CuFFTPlanManager {
private:
    cufftHandle plan;
    bool created;

public:
    CuFFTPlanManager() : created(false) {}
    
    ~CuFFTPlanManager() {
        destroy();
    }
    
    // 创建1D R2C计划
    void create1D_R2C(int nx) {
        if (created) destroy();
        CUFFT_CHECK(cufftPlan1d(&plan, nx, CUFFT_R2C, 1));
        created = true;
    }
    
    // 创建1D C2R计划
    void create1D_C2R(int nx) {
        if (created) destroy();
        CUFFT_CHECK(cufftPlan1d(&plan, nx, CUFFT_C2R, 1));
        created = true;
    }
    
    // 创建2D R2C计划
    void create2D_R2C(int nx, int ny) {
        if (created) destroy();
        CUFFT_CHECK(cufftPlan2d(&plan, ny, nx, CUFFT_R2C));
        created = true;
    }
    
    // 创建2D C2R计划
    void create2D_C2R(int nx, int ny) {
        if (created) destroy();
        CUFFT_CHECK(cufftPlan2d(&plan, ny, nx, CUFFT_C2R));
        created = true;
    }
    
    // 创建1D Many R2C计划 (batch FFT)
    void create1D_Many_R2C(int nx, int batch) {
        if (created) destroy();
        int n[1] = {nx};
        CUFFT_CHECK(cufftPlanMany(&plan, 1, n, 
                                  nullptr, 1, nx,      // input
                                  nullptr, 1, nx/2+1,  // output
                                  CUFFT_R2C, batch));
        created = true;
    }
    
    // 创建1D Many C2R计划
    void create1D_Many_C2R(int nx, int batch) {
        if (created) destroy();
        int n[1] = {nx};
        CUFFT_CHECK(cufftPlanMany(&plan, 1, n, 
                                  nullptr, 1, nx/2+1,  // input
                                  nullptr, 1, nx,      // output
                                  CUFFT_C2R, batch));
        created = true;
    }
    
    // 设置CUDA流
    void setStream(cudaStream_t stream) {
        if (created) {
            CUFFT_CHECK(cufftSetStream(plan, stream));
        }
    }
    
    // 执行R2C变换
    void executeR2C(cufftReal* input, cufftComplex* output) {
        CUFFT_CHECK(cufftExecR2C(plan, input, output));
    }
    
    // 执行C2R变换
    void executeC2R(cufftComplex* input, cufftReal* output) {
        CUFFT_CHECK(cufftExecC2R(plan, input, output));
    }
    
    // 销毁计划
    void destroy() {
        if (created) {
            CUFFT_CHECK(cufftDestroy(plan));
            created = false;
        }
    }
    
    cufftHandle getPlan() { return plan; }
};

// ============================================================================
// 性能计时工具
// ============================================================================

class GPUTimer {
private:
    cudaEvent_t start, stop;
    
public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void startTimer() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    
    float stopTimer() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================================
// 常用GPU工具函数
// ============================================================================

// 计算grid和block维度
inline dim3 computeGridSize(int total_threads, int block_size) {
    return dim3((total_threads + block_size - 1) / block_size);
}

inline dim3 computeGridSize2D(int width, int height, int block_x, int block_y) {
    return dim3((width + block_x - 1) / block_x, 
                (height + block_y - 1) / block_y);
}

// 检查GPU是否可用
bool isGPUAvailable();

// 同步设备
inline void syncDevice() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // CUDA_UTILS_H
