/*******************************************************************
 *       Filename:  cuda_utils.cu
 *                                                                 
 *    Description:  CUDA工具类实现
 *                                                                 
 *        Version:  1.0                                           
 *        Created:  12/18/2024
 *       Compiler:  nvcc                                          
 *                                                                 
 *******************************************************************/

#include "cuda_utils.h"
#include <iostream>

// ============================================================================
// GPU设备信息函数实现
// ============================================================================

void printGPUInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("\n========== GPU Information ==========\n");
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  CUDA Cores: ~%d\n", 
               prop.multiProcessorCount * 
               (prop.major == 7 ? 64 : (prop.major == 8 ? 128 : 64)));
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.2f GHz\n", 
               prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  L2 Cache Size: %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
        printf("  Shared Memory per Block: %.2f KB\n",
               prop.sharedMemPerBlock / 1024.0);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("\n");
    }
    printf("=====================================\n\n");
}

int getGPUDeviceCount() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

void setGPUDevice(int device) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (device < 0 || device >= deviceCount) {
        fprintf(stderr, "Invalid device ID: %d. Available devices: 0-%d\n",
                device, deviceCount - 1);
        exit(EXIT_FAILURE);
    }
    
    CUDA_CHECK(cudaSetDevice(device));
    printf("Using GPU device %d\n", device);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
}

bool isGPUAvailable() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    // 尝试在设备上分配一小块内存来确认GPU可用
    void* test_ptr;
    err = cudaMalloc(&test_ptr, 1024);
    if (err != cudaSuccess) {
        return false;
    }
    cudaFree(test_ptr);
    
    return true;
}

// ============================================================================
// 测试函数 - 验证GPU功能
// ============================================================================

__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void testGPUFunctionality() {
    printf("Testing GPU functionality...\n");
    
    const int n = 1024;
    float* h_data = new float[n];
    float* h_result = new float[n];
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    // 分配GPU内存
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    
    // 拷贝到GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    // 执行kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    test_kernel<<<grid_size, block_size>>>(d_data, n);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 拷贝回CPU
    CUDA_CHECK(cudaMemcpy(h_result, d_data, n * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float expected = h_data[i] * 2.0f + 1.0f;
        if (fabs(h_result[i] - expected) > 1e-5) {
            passed = false;
            printf("Error at index %d: expected %f, got %f\n",
                   i, expected, h_result[i]);
            break;
        }
    }
    
    if (passed) {
        printf("GPU functionality test PASSED!\n\n");
    } else {
        printf("GPU functionality test FAILED!\n\n");
    }
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    delete[] h_result;
}
