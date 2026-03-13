/*******************************************************************
 *       Filename:  wbp_backproject_gpu.cu
 *
 *    Description:  WBP backprojection GPU implementation
 *                  - Recon volume stays on GPU across tilts
 *                  - Per-tilt xz mapping is precomputed once and reused for all y
 *
 *******************************************************************/

#include "cuda_utils.h"
#include "wbp_backproject_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <cstdlib>
#include <vector>

// =============================
// Device global caches
// =============================
static float* g_d_recon = nullptr;          // Nx*Ny*h
static size_t g_recon_elems = 0;

static float* g_d_corrected = nullptr;      // n_zz*Nx*Ny (host-input API only)
static size_t g_corrected_elems_capacity = 0;

// Per-tilt xz map: size Nx*h
static int* g_d_map_t1t = nullptr;
static int* g_d_map_nz = nullptr;
static float* g_d_map_coeff = nullptr;
static size_t g_map_elems_capacity = 0;

// Precomputed xz maps for all tilts: [n_maps][Nx*h]
static int* g_d_map_t1t_all = nullptr;
static int* g_d_map_nz_all = nullptr;
static float* g_d_map_coeff_all = nullptr;
static size_t g_map_all_elems_capacity = 0;
static int g_map_all_count = 0;
static size_t g_map_all_map_elems = 0;

// Cached tilt parameters for batched map build.
static float* g_d_tmp_h_all = nullptr;
static float* g_d_sin_all = nullptr;
static float* g_d_cos_all = nullptr;
static int g_tilt_param_capacity = 0;

// Partial reductions for recon stats
static float* g_d_partial_min = nullptr;
static float* g_d_partial_max = nullptr;
static double* g_d_partial_sum = nullptr;
static size_t g_partial_capacity = 0;

static int g_Nx = 0;
static int g_Ny = 0;
static int g_h = 0;

// =============================
// Build xz map once per tilt
// =============================
__global__ void wbp_build_xz_map_kernel(
    int* __restrict__ d_t1t,      // [h*Nx]
    int* __restrict__ d_nz,       // [h*Nx], -1 means invalid
    float* __restrict__ d_coeff,  // [h*Nx]
    int Nx, int h,
    int defocus_step,
    float tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1
) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)Nx * (size_t)h;
    if (idx >= total) return;

    int k = (int)(idx / (size_t)Nx);
    int i = (int)(idx - (size_t)k * (size_t)Nx);

    float xi = (float)i - x_orig_offset;
    float zk = (float)k - z_orig_offset;

    float x_orig = xi * cos1 - zk * sin1 + x_orig_offset;
    float z_orig = xi * sin1 + zk * cos1;

    int t1t = (int)floorf(x_orig);
    float coeff = x_orig - (float)t1t;
    int n_z = (int)floorf((z_orig + tmp_h) / (float)defocus_step);

    if (t1t < 0 || (t1t + 1) >= Nx) {
        d_t1t[idx] = 0;
        d_coeff[idx] = 0.0f;
        d_nz[idx] = -1;
        return;
    }

    d_t1t[idx] = t1t;
    d_coeff[idx] = coeff;
    d_nz[idx] = n_z;
}

__global__ void wbp_build_xz_map_batch_kernel(
    int* __restrict__ d_t1t_all,      // [n_maps][h*Nx]
    int* __restrict__ d_nz_all,       // [n_maps][h*Nx]
    float* __restrict__ d_coeff_all,  // [n_maps][h*Nx]
    int Nx, int h,
    int defocus_step,
    float x_orig_offset,
    float z_orig_offset,
    const float* __restrict__ d_tmp_h_all,  // [n_maps]
    const float* __restrict__ d_sin_all,    // [n_maps]
    const float* __restrict__ d_cos_all,    // [n_maps]
    int n_maps
) {
    int map_idx = (int)blockIdx.y;
    if (map_idx < 0 || map_idx >= n_maps) return;

    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t map_elems = (size_t)Nx * (size_t)h;
    if (idx >= map_elems) return;

    const float tmp_h = d_tmp_h_all[map_idx];
    const float sin1 = d_sin_all[map_idx];
    const float cos1 = d_cos_all[map_idx];

    int k = (int)(idx / (size_t)Nx);
    int i = (int)(idx - (size_t)k * (size_t)Nx);

    float xi = (float)i - x_orig_offset;
    float zk = (float)k - z_orig_offset;

    float x_orig = xi * cos1 - zk * sin1 + x_orig_offset;
    float z_orig = xi * sin1 + zk * cos1;

    int t1t = (int)floorf(x_orig);
    float coeff = x_orig - (float)t1t;
    int n_z = (int)floorf((z_orig + tmp_h) / (float)defocus_step);

    size_t out = (size_t)map_idx * map_elems + idx;
    if (t1t < 0 || (t1t + 1) >= Nx) {
        d_t1t_all[out] = 0;
        d_coeff_all[out] = 0.0f;
        d_nz_all[out] = -1;
        return;
    }

    d_t1t_all[out] = t1t;
    d_coeff_all[out] = coeff;
    d_nz_all[out] = n_z;
}

// =============================
// Accumulate using prebuilt map
// =============================
__global__ void wbp_backproject_accumulate_mapped_kernel(
    const float* __restrict__ d_stack_corrected, // [n_zz][Ny][Nx]
    float* __restrict__ d_recon,                 // [Ny][h][Nx], flattened
    const int* __restrict__ d_t1t,               // [h*Nx]
    const int* __restrict__ d_nz,                // [h*Nx]
    const float* __restrict__ d_coeff,           // [h*Nx]
    int Nx, int Ny, int h,
    int n_zz,
    float corrected_scale
) {
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)Nx * (size_t)h * (size_t)Ny;
    if (tid >= total) return;

    int x_h = Nx * h;
    int j = (int)(tid / (size_t)x_h);
    int xz = (int)(tid - (size_t)j * (size_t)x_h); // [h*Nx]

    int t1t = d_t1t[xz];
    int n_z = d_nz[xz];
    float coeff = d_coeff[xz];

    if (n_z < 0 || n_z >= n_zz) return;

    size_t plane = (size_t)Nx * (size_t)Ny;
    size_t base = (size_t)n_z * plane + (size_t)j * (size_t)Nx + (size_t)t1t;

    float v0 = d_stack_corrected[base];
    float v1 = d_stack_corrected[base + 1];
    d_recon[tid] += ((1.0f - coeff) * v0 + coeff * v1) * corrected_scale;
}

// One thread handles one (x,z) and loops over all y to reuse map data.
__global__ void wbp_backproject_accumulate_mapped_yloop_kernel(
    const float* __restrict__ d_stack_corrected, // [n_zz][Ny][Nx]
    float* __restrict__ d_recon,                 // [Ny][h][Nx], flattened
    const int* __restrict__ d_t1t,               // [h*Nx]
    const int* __restrict__ d_nz,                // [h*Nx]
    const float* __restrict__ d_coeff,           // [h*Nx]
    int Nx, int Ny, int h,
    int n_zz,
    float corrected_scale
) {
    size_t xz = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t map_elems = (size_t)Nx * (size_t)h;
    if (xz >= map_elems) return;

    int t1t = d_t1t[xz];
    int n_z = d_nz[xz];
    float coeff = d_coeff[xz];
    if (n_z < 0 || n_z >= n_zz) return;

    const size_t plane = (size_t)Nx * (size_t)Ny;
    const size_t x_h = (size_t)Nx * (size_t)h;
    const float* corr_ptr = d_stack_corrected + (size_t)n_z * plane + (size_t)t1t;
    float* recon_ptr = d_recon + xz;

    for (int j = 0; j < Ny; ++j) {
        const size_t off_img = (size_t)j * (size_t)Nx;
        const size_t off_rec = (size_t)j * x_h;
        float v0 = corr_ptr[off_img];
        float v1 = corr_ptr[off_img + 1];
        recon_ptr[off_rec] += ((1.0f - coeff) * v0 + coeff * v1) * corrected_scale;
    }
}

__global__ void reduce_recon_partial_kernel(
    const float* __restrict__ d_data,
    size_t n,
    float* __restrict__ d_partial_min,
    float* __restrict__ d_partial_max,
    double* __restrict__ d_partial_sum
) {
    extern __shared__ unsigned char smem[];
    float* s_min = reinterpret_cast<float*>(smem);
    float* s_max = s_min + blockDim.x;
    double* s_sum = reinterpret_cast<double*>(s_max + blockDim.x);

    const unsigned int tid = threadIdx.x;
    const size_t global_stride = (size_t)blockDim.x * (size_t)gridDim.x;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)tid;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    double local_sum = 0.0;

    for (; idx < n; idx += global_stride) {
        float v = d_data[idx];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
        local_sum += (double)v;
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial_min[blockIdx.x] = s_min[0];
        d_partial_max[blockIdx.x] = s_max[0];
        d_partial_sum[blockIdx.x] = s_sum[0];
    }
}

static void ensure_map_capacity(size_t elems) {
    if (elems <= g_map_elems_capacity && g_d_map_t1t && g_d_map_nz && g_d_map_coeff) return;

    if (g_d_map_t1t) { CUDA_CHECK(cudaFree(g_d_map_t1t)); g_d_map_t1t = nullptr; }
    if (g_d_map_nz) { CUDA_CHECK(cudaFree(g_d_map_nz)); g_d_map_nz = nullptr; }
    if (g_d_map_coeff) { CUDA_CHECK(cudaFree(g_d_map_coeff)); g_d_map_coeff = nullptr; }

    CUDA_CHECK(cudaMalloc(&g_d_map_t1t, elems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_map_nz, elems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_map_coeff, elems * sizeof(float)));
    g_map_elems_capacity = elems;
}

static bool ensure_map_all_capacity(size_t total_elems) {
    if (total_elems <= g_map_all_elems_capacity &&
        g_d_map_t1t_all && g_d_map_nz_all && g_d_map_coeff_all) {
        return true;
    }

    if (g_d_map_t1t_all) { CUDA_CHECK(cudaFree(g_d_map_t1t_all)); g_d_map_t1t_all = nullptr; }
    if (g_d_map_nz_all) { CUDA_CHECK(cudaFree(g_d_map_nz_all)); g_d_map_nz_all = nullptr; }
    if (g_d_map_coeff_all) { CUDA_CHECK(cudaFree(g_d_map_coeff_all)); g_d_map_coeff_all = nullptr; }

    cudaError_t e1 = cudaMalloc(&g_d_map_t1t_all, total_elems * sizeof(int));
    if (e1 != cudaSuccess) return false;
    cudaError_t e2 = cudaMalloc(&g_d_map_nz_all, total_elems * sizeof(int));
    if (e2 != cudaSuccess) {
        cudaFree(g_d_map_t1t_all);
        g_d_map_t1t_all = nullptr;
        return false;
    }
    cudaError_t e3 = cudaMalloc(&g_d_map_coeff_all, total_elems * sizeof(float));
    if (e3 != cudaSuccess) {
        cudaFree(g_d_map_t1t_all);
        cudaFree(g_d_map_nz_all);
        g_d_map_t1t_all = nullptr;
        g_d_map_nz_all = nullptr;
        return false;
    }

    g_map_all_elems_capacity = total_elems;
    return true;
}

static bool ensure_tilt_param_capacity(int n_maps) {
    if (n_maps <= g_tilt_param_capacity &&
        g_d_tmp_h_all && g_d_sin_all && g_d_cos_all) {
        return true;
    }

    if (g_d_tmp_h_all) { CUDA_CHECK(cudaFree(g_d_tmp_h_all)); g_d_tmp_h_all = nullptr; }
    if (g_d_sin_all) { CUDA_CHECK(cudaFree(g_d_sin_all)); g_d_sin_all = nullptr; }
    if (g_d_cos_all) { CUDA_CHECK(cudaFree(g_d_cos_all)); g_d_cos_all = nullptr; }

    cudaError_t e1 = cudaMalloc(&g_d_tmp_h_all, (size_t)n_maps * sizeof(float));
    if (e1 != cudaSuccess) return false;
    cudaError_t e2 = cudaMalloc(&g_d_sin_all, (size_t)n_maps * sizeof(float));
    if (e2 != cudaSuccess) {
        cudaFree(g_d_tmp_h_all);
        g_d_tmp_h_all = nullptr;
        return false;
    }
    cudaError_t e3 = cudaMalloc(&g_d_cos_all, (size_t)n_maps * sizeof(float));
    if (e3 != cudaSuccess) {
        cudaFree(g_d_tmp_h_all);
        cudaFree(g_d_sin_all);
        g_d_tmp_h_all = nullptr;
        g_d_sin_all = nullptr;
        return false;
    }

    g_tilt_param_capacity = n_maps;
    return true;
}

static void ensure_partial_capacity(size_t blocks) {
    if (blocks <= g_partial_capacity && g_d_partial_min && g_d_partial_max && g_d_partial_sum) return;

    if (g_d_partial_min) { CUDA_CHECK(cudaFree(g_d_partial_min)); g_d_partial_min = nullptr; }
    if (g_d_partial_max) { CUDA_CHECK(cudaFree(g_d_partial_max)); g_d_partial_max = nullptr; }
    if (g_d_partial_sum) { CUDA_CHECK(cudaFree(g_d_partial_sum)); g_d_partial_sum = nullptr; }

    CUDA_CHECK(cudaMalloc(&g_d_partial_min, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_partial_max, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_partial_sum, blocks * sizeof(double)));
    g_partial_capacity = blocks;
}

static bool use_bp_yloop_kernel() {
    static int mode = -1;
    if (mode < 0) {
        mode = 1; // default enabled
        const char* env = std::getenv("WBP_BP_YLOOP");
        if (env) mode = (std::atoi(env) != 0) ? 1 : 0;
    }
    return mode != 0;
}

static void launch_backproject_mapped(
    const float* d_stack_corrected,
    float* d_recon,
    const int* d_map_t1t,
    const int* d_map_nz,
    const float* d_map_coeff,
    int Nx, int Ny, int h,
    int n_zz,
    float corrected_scale,
    cudaStream_t stream
) {
    const int block_size = 256;
    if (use_bp_yloop_kernel()) {
        const size_t map_elems = (size_t)Nx * (size_t)h;
        const int grid_size = (int)((map_elems + (size_t)block_size - 1) / (size_t)block_size);
        wbp_backproject_accumulate_mapped_yloop_kernel<<<grid_size, block_size, 0, stream>>>(
            d_stack_corrected,
            d_recon,
            d_map_t1t,
            d_map_nz,
            d_map_coeff,
            Nx, Ny, h,
            n_zz,
            corrected_scale
        );
        CUDA_CHECK_LAST_ERROR();
        return;
    }

    const int n_voxels = Nx * Ny * h;
    const int grid_size = (n_voxels + block_size - 1) / block_size;
    wbp_backproject_accumulate_mapped_kernel<<<grid_size, block_size, 0, stream>>>(
        d_stack_corrected,
        d_recon,
        d_map_t1t,
        d_map_nz,
        d_map_coeff,
        Nx, Ny, h,
        n_zz,
        corrected_scale
    );
    CUDA_CHECK_LAST_ERROR();
}

// =============================
// Host API
// =============================
extern "C" void wbp_recon_init(int Nx, int Ny, int h) {
    g_Nx = Nx;
    g_Ny = Ny;
    g_h = h;

    size_t elems = (size_t)Nx * (size_t)Ny * (size_t)h;
    g_recon_elems = elems;

    if (g_d_recon) {
        CUDA_CHECK(cudaFree(g_d_recon));
        g_d_recon = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&g_d_recon, elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_d_recon, 0, elems * sizeof(float)));

    // Recon geometry may change across runs; invalidate precomputed map metadata.
    g_map_all_count = 0;
    g_map_all_map_elems = 0;
}

extern "C" void wbp_backproject_accumulate_gpu(
    const float* h_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1
) {
    if (!g_d_recon || Nx != g_Nx || Ny != g_Ny || h != g_h) {
        wbp_recon_init(Nx, Ny, h);
    }

    size_t plane = (size_t)Nx * (size_t)Ny;
    size_t needed = (size_t)n_zz * plane;
    if (!g_d_corrected || needed > g_corrected_elems_capacity) {
        if (g_d_corrected) {
            CUDA_CHECK(cudaFree(g_d_corrected));
            g_d_corrected = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&g_d_corrected, needed * sizeof(float)));
        g_corrected_elems_capacity = needed;
    }

    CUDA_CHECK(cudaMemcpy(g_d_corrected, h_stack_corrected, needed * sizeof(float), cudaMemcpyHostToDevice));

    size_t map_elems = (size_t)Nx * (size_t)h;
    ensure_map_capacity(map_elems);
    {
        int block = 256;
        int grid = (int)((map_elems + (size_t)block - 1) / (size_t)block);
        wbp_build_xz_map_kernel<<<grid, block>>>(
            g_d_map_t1t, g_d_map_nz, g_d_map_coeff,
            Nx, h,
            defocus_step,
            (float)tmp_h,
            x_orig_offset,
            z_orig_offset,
            sin1,
            cos1
        );
        CUDA_CHECK_LAST_ERROR();
    }

    launch_backproject_mapped(
        g_d_corrected,
        g_d_recon,
        g_d_map_t1t,
        g_d_map_nz,
        g_d_map_coeff,
        Nx, Ny, h,
        n_zz,
        1.0f,
        (cudaStream_t)0
    );
}

extern "C" void wbp_backproject_accumulate_gpu_device_stream(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1,
    float corrected_scale,
    void* stream_handle
);

extern "C" void wbp_backproject_accumulate_gpu_device(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1,
    float corrected_scale
) {
    wbp_backproject_accumulate_gpu_device_stream(
        d_stack_corrected,
        Nx, Ny, h,
        n_zz,
        defocus_step,
        tmp_h,
        x_orig_offset,
        z_orig_offset,
        sin1,
        cos1,
        corrected_scale,
        nullptr
    );
}

extern "C" void wbp_backproject_accumulate_gpu_device_stream(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1,
    float corrected_scale,
    void* stream_handle
) {
    if (!d_stack_corrected) return;
    if (!g_d_recon || g_Nx != Nx || g_Ny != Ny || g_h != h) {
        fprintf(stderr, "[GPU][WBP] recon not initialized or size mismatch in accumulate_gpu_device.\n");
        return;
    }
    cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : (cudaStream_t)0;

    size_t map_elems = (size_t)Nx * (size_t)h;
    ensure_map_capacity(map_elems);
    {
        int block = 256;
        int grid = (int)((map_elems + (size_t)block - 1) / (size_t)block);
        wbp_build_xz_map_kernel<<<grid, block, 0, stream>>>(
            g_d_map_t1t, g_d_map_nz, g_d_map_coeff,
            Nx, h,
            defocus_step,
            (float)tmp_h,
            x_orig_offset,
            z_orig_offset,
            sin1,
            cos1
        );
        CUDA_CHECK_LAST_ERROR();
    }

    launch_backproject_mapped(
        d_stack_corrected,
        g_d_recon,
        g_d_map_t1t,
        g_d_map_nz,
        g_d_map_coeff,
        Nx, Ny, h,
        n_zz,
        corrected_scale,
        stream
    );
}

extern "C" int wbp_backproject_prepare_xz_maps(
    int Nx, int h,
    int n_maps,
    int defocus_step,
    float x_orig_offset,
    float z_orig_offset,
    const float* h_tmp_h_all,
    const float* h_sin_all,
    const float* h_cos_all,
    void* stream_handle
) {
    if (Nx <= 0 || h <= 0 || n_maps <= 0) return 0;
    if (!h_tmp_h_all || !h_sin_all || !h_cos_all) return 0;
    if (g_Nx != Nx || g_h != h) return 0;

    const size_t map_elems = (size_t)Nx * (size_t)h;
    const size_t total_elems = map_elems * (size_t)n_maps;
    if (!ensure_map_all_capacity(total_elems)) return 0;

    cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : (cudaStream_t)0;

    bool ok = ensure_tilt_param_capacity(n_maps);
    if (ok && cudaMemcpyAsync(g_d_tmp_h_all, h_tmp_h_all, (size_t)n_maps * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) ok = false;
    if (ok && cudaMemcpyAsync(g_d_sin_all, h_sin_all, (size_t)n_maps * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) ok = false;
    if (ok && cudaMemcpyAsync(g_d_cos_all, h_cos_all, (size_t)n_maps * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) ok = false;

    if (ok) {
        int block = 256;
        int grid_x = (int)((map_elems + (size_t)block - 1) / (size_t)block);
        dim3 grid((unsigned int)grid_x, (unsigned int)n_maps, 1u);
        wbp_build_xz_map_batch_kernel<<<grid, block, 0, stream>>>(
            g_d_map_t1t_all,
            g_d_map_nz_all,
            g_d_map_coeff_all,
            Nx, h,
            defocus_step,
            x_orig_offset,
            z_orig_offset,
            g_d_tmp_h_all,
            g_d_sin_all,
            g_d_cos_all,
            n_maps
        );
        if (cudaGetLastError() != cudaSuccess) ok = false;
    }

    if (!ok) {
        g_map_all_count = 0;
        g_map_all_map_elems = 0;
        return 0;
    }

    g_map_all_count = n_maps;
    g_map_all_map_elems = map_elems;
    return 1;
}

extern "C" void wbp_backproject_accumulate_gpu_device_precomputed_map_stream(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int map_index,
    float corrected_scale,
    void* stream_handle
) {
    if (!d_stack_corrected) return;
    if (!g_d_recon || g_Nx != Nx || g_Ny != Ny || g_h != h) return;
    if (!g_d_map_t1t_all || !g_d_map_nz_all || !g_d_map_coeff_all) return;
    if (map_index < 0 || map_index >= g_map_all_count) return;

    const size_t map_elems = (size_t)Nx * (size_t)h;
    if (g_map_all_map_elems != map_elems) return;

    cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : (cudaStream_t)0;

    const int* map_t1t = g_d_map_t1t_all + (size_t)map_index * map_elems;
    const int* map_nz = g_d_map_nz_all + (size_t)map_index * map_elems;
    const float* map_coeff = g_d_map_coeff_all + (size_t)map_index * map_elems;

    launch_backproject_mapped(
        d_stack_corrected,
        g_d_recon,
        map_t1t,
        map_nz,
        map_coeff,
        Nx, Ny, h,
        n_zz,
        corrected_scale,
        stream
    );
}

extern "C" void wbp_recon_finalize(float* h_recon_out) {
    if (!g_d_recon || g_recon_elems == 0) return;
    CUDA_CHECK(cudaMemcpy(h_recon_out, g_d_recon, g_recon_elems * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" void wbp_recon_copy_range(float* h_out, size_t offset_elems, size_t count_elems) {
    if (!h_out || !g_d_recon || g_recon_elems == 0 || count_elems == 0) return;
    if (offset_elems >= g_recon_elems) return;

    size_t valid_count = count_elems;
    size_t remain = g_recon_elems - offset_elems;
    if (valid_count > remain) valid_count = remain;

    CUDA_CHECK(cudaMemcpy(
        h_out,
        g_d_recon + offset_elems,
        valid_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

extern "C" void wbp_recon_get_stats(float* h_min_out, float* h_max_out, double* h_mean_out) {
    if (h_min_out) *h_min_out = 0.0f;
    if (h_max_out) *h_max_out = 0.0f;
    if (h_mean_out) *h_mean_out = 0.0;

    if (!g_d_recon || g_recon_elems == 0) return;

    const int threads = 256;
    int blocks = (int)((g_recon_elems + (size_t)threads - 1) / (size_t)threads);
    if (blocks > 4096) blocks = 4096; // enough parallelism, smaller partial copy overhead
    if (blocks < 1) blocks = 1;

    ensure_partial_capacity((size_t)blocks);

    size_t shared_bytes = (size_t)threads * (sizeof(float) + sizeof(float) + sizeof(double));
    reduce_recon_partial_kernel<<<blocks, threads, shared_bytes>>>(
        g_d_recon,
        g_recon_elems,
        g_d_partial_min,
        g_d_partial_max,
        g_d_partial_sum
    );
    CUDA_CHECK_LAST_ERROR();

    std::vector<float> h_partial_min((size_t)blocks);
    std::vector<float> h_partial_max((size_t)blocks);
    std::vector<double> h_partial_sum((size_t)blocks);

    CUDA_CHECK(cudaMemcpy(h_partial_min.data(), g_d_partial_min, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_max.data(), g_d_partial_max, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_sum.data(), g_d_partial_sum, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));

    float min_v = h_partial_min[0];
    float max_v = h_partial_max[0];
    double sum_v = 0.0;
    for (int i = 0; i < blocks; ++i) {
        if (h_partial_min[(size_t)i] < min_v) min_v = h_partial_min[(size_t)i];
        if (h_partial_max[(size_t)i] > max_v) max_v = h_partial_max[(size_t)i];
        sum_v += h_partial_sum[(size_t)i];
    }

    if (h_min_out) *h_min_out = min_v;
    if (h_max_out) *h_max_out = max_v;
    if (h_mean_out) *h_mean_out = sum_v / (double)g_recon_elems;
}

extern "C" void wbp_recon_free() {
    if (g_d_recon) {
        CUDA_CHECK(cudaFree(g_d_recon));
        g_d_recon = nullptr;
    }
    if (g_d_corrected) {
        CUDA_CHECK(cudaFree(g_d_corrected));
        g_d_corrected = nullptr;
    }
    if (g_d_map_t1t) {
        CUDA_CHECK(cudaFree(g_d_map_t1t));
        g_d_map_t1t = nullptr;
    }
    if (g_d_map_nz) {
        CUDA_CHECK(cudaFree(g_d_map_nz));
        g_d_map_nz = nullptr;
    }
    if (g_d_map_coeff) {
        CUDA_CHECK(cudaFree(g_d_map_coeff));
        g_d_map_coeff = nullptr;
    }
    if (g_d_map_t1t_all) {
        CUDA_CHECK(cudaFree(g_d_map_t1t_all));
        g_d_map_t1t_all = nullptr;
    }
    if (g_d_map_nz_all) {
        CUDA_CHECK(cudaFree(g_d_map_nz_all));
        g_d_map_nz_all = nullptr;
    }
    if (g_d_map_coeff_all) {
        CUDA_CHECK(cudaFree(g_d_map_coeff_all));
        g_d_map_coeff_all = nullptr;
    }
    if (g_d_tmp_h_all) {
        CUDA_CHECK(cudaFree(g_d_tmp_h_all));
        g_d_tmp_h_all = nullptr;
    }
    if (g_d_sin_all) {
        CUDA_CHECK(cudaFree(g_d_sin_all));
        g_d_sin_all = nullptr;
    }
    if (g_d_cos_all) {
        CUDA_CHECK(cudaFree(g_d_cos_all));
        g_d_cos_all = nullptr;
    }
    if (g_d_partial_min) {
        CUDA_CHECK(cudaFree(g_d_partial_min));
        g_d_partial_min = nullptr;
    }
    if (g_d_partial_max) {
        CUDA_CHECK(cudaFree(g_d_partial_max));
        g_d_partial_max = nullptr;
    }
    if (g_d_partial_sum) {
        CUDA_CHECK(cudaFree(g_d_partial_sum));
        g_d_partial_sum = nullptr;
    }

    g_recon_elems = 0;
    g_corrected_elems_capacity = 0;
    g_map_elems_capacity = 0;
    g_map_all_elems_capacity = 0;
    g_map_all_count = 0;
    g_map_all_map_elems = 0;
    g_tilt_param_capacity = 0;
    g_partial_capacity = 0;
    g_Nx = g_Ny = g_h = 0;
}

