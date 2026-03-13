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
    int n_zz
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
    d_recon[tid] += (1.0f - coeff) * v0 + coeff * v1;
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

    size_t total = (size_t)Nx * (size_t)Ny * (size_t)h;
    int block = 256;
    int grid = (int)((total + (size_t)block - 1) / (size_t)block);

    wbp_backproject_accumulate_mapped_kernel<<<grid, block>>>(
        g_d_corrected,
        g_d_recon,
        g_d_map_t1t,
        g_d_map_nz,
        g_d_map_coeff,
        Nx, Ny, h,
        n_zz
    );
    CUDA_CHECK_LAST_ERROR();
}

extern "C" void wbp_backproject_accumulate_gpu_device(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1
) {
    if (!d_stack_corrected) return;
    if (!g_d_recon || g_Nx != Nx || g_Ny != Ny || g_h != h) {
        fprintf(stderr, "[GPU][WBP] recon not initialized or size mismatch in accumulate_gpu_device.\n");
        return;
    }

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

    const int block_size = 256;
    const int n_voxels = Nx * Ny * h;
    const int grid_size = (n_voxels + block_size - 1) / block_size;

    wbp_backproject_accumulate_mapped_kernel<<<grid_size, block_size>>>(
        d_stack_corrected,
        g_d_recon,
        g_d_map_t1t,
        g_d_map_nz,
        g_d_map_coeff,
        Nx, Ny, h,
        n_zz
    );
    CUDA_CHECK_LAST_ERROR();
}

extern "C" void wbp_recon_finalize(float* h_recon_out) {
    if (!g_d_recon || g_recon_elems == 0) return;
    CUDA_CHECK(cudaMemcpy(h_recon_out, g_d_recon, g_recon_elems * sizeof(float), cudaMemcpyDeviceToHost));
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

    g_recon_elems = 0;
    g_corrected_elems_capacity = 0;
    g_map_elems_capacity = 0;
    g_Nx = g_Ny = g_h = 0;
}

