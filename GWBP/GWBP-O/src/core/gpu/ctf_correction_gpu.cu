/*******************************************************************
 *       Filename:  ctf_correction_gpu.cu
 *
 *    Description:  CTF校正的GPU实现 - WBP算法的核心计算热点
 *                  + 增加 Device 输入/输出版本接口（方案A）
 *
 *        Version:  1.1
 *        Created:  12/18/2024
 *       Modified:  01/18/2026
 *       Compiler:  nvcc
 *
 *******************************************************************/

#include "cuda_utils.h"
#include "CTF.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

// ============================================================================
// CTF计算Kernel
// ============================================================================

__global__ void ctf_correction_kernel(
    cufftComplex* d_freq,
    int Nx, int Ny,
    float defocus1, float defocus2, float astig,
    float phase_shift, float w_cos, float w_sin,
    float pix_1, float lambda2, float lambda3,
    bool flip_contrast, float z_offset, float pix
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int freq_width = Nx / 2 + 1;
    if (x >= freq_width || y >= Ny) return;

    //int Nx_ce = (Nx + 2) / 2;
    int Ny_ce = (Ny + 2) / 2;

    float x_norm = (float)x;
    float y_norm = (y >= Ny_ce) ? (float)(y - Ny) : (float)y;

    float x_real = x_norm * pix_1 / (float)Nx;
    float y_real = y_norm * pix_1 / (float)Ny;

    float alpha;
    if (fabsf(x_norm) < 1e-10f) {
        alpha = (y_norm > 0) ? (M_PI / 2.0f) :
                (y_norm < 0) ? (-M_PI / 2.0f) : 0.0f;
    } else {
        alpha = atan2f(y_real, x_real);
    }

    float freq2 = x_real * x_real + y_real * y_real;

    float df = 0.5f * (defocus1 + defocus2 - 2.0f * z_offset * pix +
                       (defocus1 - defocus2) * __cosf(2.0f * (alpha - astig)));

    float chi = (lambda2 * df - lambda3 * freq2) * freq2 + phase_shift;

    float sin_chi, cos_chi;
    __sincosf(chi, &sin_chi, &cos_chi);
    float ctf_val = w_sin * sin_chi + w_cos * cos_chi;

    if (flip_contrast) {
        ctf_val = -ctf_val;
    }

    int idx = y * freq_width + x;
    if (ctf_val < 0.0f) {
        d_freq[idx].x = -d_freq[idx].x;
        d_freq[idx].y = -d_freq[idx].y;
    }
}

// ============================================================================
// 归一化Kernel
// ============================================================================

__global__ void normalize_kernel(float* d_data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] *= scale;
    }
}

// ============================================================================
// 主CTF校正函数 - CPU端接口（保持原样）
// ============================================================================

__global__ void pad_rows_linear_kernel(
    const float* __restrict__ d_in,
    float* __restrict__ d_pad,
    int Nx_orig, int Nx_final, int nxb, int Ny
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (y >= Ny || x >= nxb) return;

    int in_row = y * Nx_orig;
    int out_row = y * nxb;

    if (x < Nx_orig) {
        d_pad[out_row + x] = d_in[in_row + x];
    } else if (x < Nx_final) {
        int nxp = Nx_final - Nx_orig + 1;
        float nxp_1 = 1.0f / (float)nxp;
        float tmp1 = d_in[in_row] * nxp_1;
        float tmp2 = d_in[in_row + Nx_orig - 1] * nxp_1;
        d_pad[out_row + x] = (float)(x - Nx_orig + 1) * tmp1 + (float)(Nx_final - x) * tmp2;
    } else {
        d_pad[out_row + x] = 0.0f;
    }
}

__global__ void build_weight_lut_kernel(
    float* __restrict__ d_w,
    int freq_width,
    int radial_Nx,
    float sigma_inv
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= freq_width) return;

    if (k == 0) {
        d_w[k] = 0.2f;
    } else if (k <= radial_Nx) {
        d_w[k] = (float)k;
    } else {
        int dk = k - radial_Nx;
        d_w[k] = (float)radial_Nx * expf(-(float)(dk * dk) * sigma_inv);
    }
}

__global__ void apply_weight_kernel(
    cufftComplex* __restrict__ d_freq,
    const float* __restrict__ d_w,
    int freq_width, int Ny
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (k >= freq_width || y >= Ny) return;

    int idx = y * freq_width + k;
    float w = d_w[k];
    d_freq[idx].x *= w;
    d_freq[idx].y *= w;
}

__global__ void unpack_rows_norm_kernel(
    float* __restrict__ d_out,
    const float* __restrict__ d_pad,
    int Nx_orig, int nxb, int Ny,
    float norm
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= Nx_orig || y >= Ny) return;
    d_out[y * Nx_orig + x] = d_pad[y * nxb + x] * norm;
}

namespace {
struct Weighting1DCache {
    int Nx = 0, Ny = 0, Nx_final = 0, nxb = 0, freq_width = 0;
    cufftHandle plan_fwd = 0, plan_inv = 0;
    float* d_padded = nullptr;
    cufftComplex* d_freq = nullptr;
    float* d_weights = nullptr;

    void destroy() {
        if (plan_fwd) { cufftDestroy(plan_fwd); plan_fwd = 0; }
        if (plan_inv) { cufftDestroy(plan_inv); plan_inv = 0; }
        if (d_padded) { cudaFree(d_padded); d_padded = nullptr; }
        if (d_freq) { cudaFree(d_freq); d_freq = nullptr; }
        if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
        Nx = Ny = Nx_final = nxb = freq_width = 0;
    }

    void ensure(int nx, int ny) {
        int nx_padding = nx / 10;
        int nx_final = nx + nx_padding;
        int nxb_local = nx_final + 2 - (nx_final & 1);
        int fw = nx_final / 2 + 1;

        if (Nx == nx && Ny == ny && Nx_final == nx_final &&
            nxb == nxb_local && freq_width == fw &&
            plan_fwd && plan_inv && d_padded && d_freq && d_weights) {
            return;
        }

        destroy();
        Nx = nx;
        Ny = ny;
        Nx_final = nx_final;
        nxb = nxb_local;
        freq_width = fw;

        CUDA_CHECK(cudaMalloc(&d_padded, (size_t)Ny * (size_t)nxb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_freq, (size_t)Ny * (size_t)freq_width * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_weights, (size_t)freq_width * sizeof(float)));

        int n[1] = {Nx_final};

        int inembed_r2c[1] = {nxb};
        int onembed_r2c[1] = {freq_width};
        CUFFT_CHECK(cufftPlanMany(
            &plan_fwd, 1, n,
            inembed_r2c, 1, nxb,
            onembed_r2c, 1, freq_width,
            CUFFT_R2C, Ny
        ));

        int inembed_c2r[1] = {freq_width};
        int onembed_c2r[1] = {nxb};
        CUFFT_CHECK(cufftPlanMany(
            &plan_inv, 1, n,
            inembed_c2r, 1, freq_width,
            onembed_c2r, 1, nxb,
            CUFFT_C2R, Ny
        ));
    }
};

static Weighting1DCache g_weight_cache;
} // namespace

extern "C" void filter_weighting_1d_many_gpu_inplace_stream(
    float* d_image,
    int Nx, int Ny,
    float radial,
    float sigma,
    void* stream_handle
);

extern "C" void filter_weighting_1d_many_gpu_inplace(
    float* d_image,
    int Nx, int Ny,
    float radial,
    float sigma
) {
    filter_weighting_1d_many_gpu_inplace_stream(
        d_image, Nx, Ny, radial, sigma, nullptr
    );
}

extern "C" void filter_weighting_1d_many_gpu_inplace_stream(
    float* d_image,
    int Nx, int Ny,
    float radial,
    float sigma,
    void* stream_handle
) {
    if (!d_image || Nx <= 0 || Ny <= 0) return;
    cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : (cudaStream_t)0;

    g_weight_cache.ensure(Nx, Ny);
    CUFFT_CHECK(cufftSetStream(g_weight_cache.plan_fwd, stream));
    CUFFT_CHECK(cufftSetStream(g_weight_cache.plan_inv, stream));

    int radial_Nx = (int)floorf((float)g_weight_cache.Nx_final * radial);
    if (radial_Nx < 0) radial_Nx = 0;

    float sigma_Nx = (float)g_weight_cache.Nx_final * sigma;
    float sigma_inv = (sigma_Nx > 1e-12f) ? (1.0f / (sigma_Nx * sigma_Nx)) : 1e30f;

    {
        int tpb = 256;
        dim3 block(tpb);
        dim3 grid((g_weight_cache.nxb + tpb - 1) / tpb, Ny);
        pad_rows_linear_kernel<<<grid, block, 0, stream>>>(
            d_image, g_weight_cache.d_padded,
            Nx, g_weight_cache.Nx_final, g_weight_cache.nxb, Ny
        );
        CUDA_CHECK_LAST_ERROR();
    }

    CUFFT_CHECK(cufftExecR2C(
        g_weight_cache.plan_fwd,
        (cufftReal*)g_weight_cache.d_padded,
        (cufftComplex*)g_weight_cache.d_freq
    ));

    {
        int tpb = 256;
        int bpg = (g_weight_cache.freq_width + tpb - 1) / tpb;
        build_weight_lut_kernel<<<bpg, tpb, 0, stream>>>(
            g_weight_cache.d_weights,
            g_weight_cache.freq_width,
            radial_Nx,
            sigma_inv
        );
        CUDA_CHECK_LAST_ERROR();

        dim3 block(tpb);
        dim3 grid((g_weight_cache.freq_width + tpb - 1) / tpb, Ny);
        apply_weight_kernel<<<grid, block, 0, stream>>>(
            g_weight_cache.d_freq,
            g_weight_cache.d_weights,
            g_weight_cache.freq_width,
            Ny
        );
        CUDA_CHECK_LAST_ERROR();
    }

    CUFFT_CHECK(cufftExecC2R(
        g_weight_cache.plan_inv,
        (cufftComplex*)g_weight_cache.d_freq,
        (cufftReal*)g_weight_cache.d_padded
    ));

    {
        int tpb = 256;
        dim3 block(tpb);
        dim3 grid((Nx + tpb - 1) / tpb, Ny);
        float norm = 1.0f / (float)Nx; // keep consistent with CPU path
        unpack_rows_norm_kernel<<<grid, block, 0, stream>>>(
            d_image, g_weight_cache.d_padded, Nx, g_weight_cache.nxb, Ny, norm
        );
        CUDA_CHECK_LAST_ERROR();
    }
}

extern "C" void ctf_correction_gpu(
    float* h_image,
    float* h_output,
    int Nx, int Ny,
    CTF ctf,
    bool flip_contrast,
    float z_offset
) {
    int img_size = Nx * Ny;
    int freq_size = (Nx / 2 + 1) * Ny;

    float* d_image = nullptr;
    cufftComplex* d_freq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_image, (size_t)img_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, (size_t)freq_size * sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(d_image, h_image, (size_t)img_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, Ny, Nx, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&plan_inv, Ny, Nx, CUFFT_C2R));

    CUFFT_CHECK(cufftExecR2C(plan_fwd, (cufftReal*)d_image, d_freq));

    float defocus1, defocus2, astig, phase_shift, w_cos, w_sin;
    float pix, pix_1, lambda, lambda2, lambda3;
    ctf.computeCTF2D_P(defocus1, defocus2, astig, phase_shift,
                       w_cos, w_sin, pix, pix_1, lambda, lambda2, lambda3);

    dim3 block(16, 16);
    dim3 grid((Nx / 2 + 1 + 15) / 16, (Ny + 15) / 16);

    ctf_correction_kernel<<<grid, block>>>(
        d_freq, Nx, Ny,
        defocus1, defocus2, astig, phase_shift, w_cos, w_sin,
        pix_1, lambda2, lambda3, flip_contrast, z_offset, pix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CTF kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    CUFFT_CHECK(cufftExecC2R(plan_inv, d_freq, (cufftReal*)d_image));

    float scale = 1.0f / (float)(Nx * Ny);
    int block_size = 256;
    int grid_size = (img_size + block_size - 1) / block_size;
    normalize_kernel<<<grid_size, block_size>>>(d_image, img_size, scale);

    CUDA_CHECK(cudaMemcpy(h_output, d_image, (size_t)img_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan_fwd));
    CUFFT_CHECK(cufftDestroy(plan_inv));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_freq));
}

// ============================================================================
// 3D-CTF校正 - CPU端接口（保持原样）
// ============================================================================

extern "C" void ctf_correction_3d_gpu(
    float* h_image,
    float* h_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets
) {
    int img_size = Nx * Ny;
    int freq_size = (Nx / 2 + 1) * Ny;

    float* d_image = nullptr;
    float* d_corrected = nullptr;
    cufftComplex* d_freq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_image, (size_t)img_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corrected, (size_t)img_size * (size_t)num_slices * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, (size_t)freq_size * sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(d_image, h_image, (size_t)img_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, Ny, Nx, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&plan_inv, Ny, Nx, CUFFT_C2R));

    float defocus1, defocus2, astig, phase_shift, w_cos, w_sin;
    float pix, pix_1, lambda, lambda2, lambda3;
    ctf.computeCTF2D_P(defocus1, defocus2, astig, phase_shift,
                       w_cos, w_sin, pix, pix_1, lambda, lambda2, lambda3);

    dim3 block(16, 16);
    dim3 grid((Nx / 2 + 1 + 15) / 16, (Ny + 15) / 16);

    const float scale = 1.0f / (float)(Nx * Ny);
    const int block_size = 256;
    const int grid_size = (img_size + block_size - 1) / block_size;

    for (int z = 0; z < num_slices; z++) {
        CUFFT_CHECK(cufftExecR2C(plan_fwd, (cufftReal*)d_image, d_freq));

        ctf_correction_kernel<<<grid, block>>>(
            d_freq, Nx, Ny,
            defocus1, defocus2, astig, phase_shift, w_cos, w_sin,
            pix_1, lambda2, lambda3, flip_contrast,
            z_offsets ? z_offsets[z] : 0.0f, pix);
        CUDA_CHECK_LAST_ERROR();

        float* d_output_slice = d_corrected + (size_t)z * (size_t)img_size;
        CUFFT_CHECK(cufftExecC2R(plan_inv, d_freq, (cufftReal*)d_output_slice));

        normalize_kernel<<<grid_size, block_size>>>(d_output_slice, img_size, scale);
        CUDA_CHECK_LAST_ERROR();
    }

    CUDA_CHECK(cudaMemcpy(h_corrected, d_corrected,
                          (size_t)img_size * (size_t)num_slices * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan_fwd));
    CUFFT_CHECK(cufftDestroy(plan_inv));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_corrected));
    CUDA_CHECK(cudaFree(d_freq));
}

// ============================================================================
// 3D-CTF校正（Device输入/输出版本） - 方案A
//  - d_image:        device端输入micrograph (Nx*Ny)
//  - d_corrected:    device端输出stack_corrected (num_slices * Nx*Ny)
//  - z_offsets_host: host端z_offset数组（几十~几百float，允许）
//
// 目的：
//  1) 避免大块 HtoD/DtoH 往返（stack_corrected 不再拷回 CPU）
//  2) 与背投影GPU（device版）直连，形成GPU pipeline
//
// 说明：
//  - CTF常量仍在 host 侧计算（与原实现一致）
//  - 仅临时分配 d_freq（频域缓冲）
// ============================================================================



// ============================================================================
// [OPT] Batched 3D-CTF correction helpers (reduce per-slice FFT calls)
//  - Compute R2C once per micrograph
//  - Expand into a batched frequency buffer for all slices
//  - One batched C2R (cufftPlanMany) for all slices
//  - Optional cache of plans & buffers to avoid per-image alloc/plan overhead
// ============================================================================

__global__ void ctf_phaseflip_expand_kernel(
    const cufftComplex* __restrict__ d_freq_base,
    cufftComplex* __restrict__ d_freq_batch,
    const float* __restrict__ d_z_offsets,
    const float* __restrict__ d_freq2_base,
    const float* __restrict__ d_cos2alpha,
    const float* __restrict__ d_sin2alpha,
    int Nx, int Ny,
    float defocus_sum, float defocus_diff,
    float phase_shift, float w_cos, float w_sin,
    float pix_1_sq, float lambda2, float lambda3,
    float cos2astig, float sin2astig,
    bool flip_contrast, float pix,
    int batch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z;

    int freq_width = Nx / 2 + 1;
    if (x >= freq_width || y >= Ny || s >= batch) return;

    float z_off = d_z_offsets ? d_z_offsets[s] : 0.0f;
    int idx = y * freq_width + x;
    float freq2 = d_freq2_base[idx] * pix_1_sq;
    float cos2a_minus_astig = d_cos2alpha[idx] * cos2astig + d_sin2alpha[idx] * sin2astig;

    float df = 0.5f * (defocus_sum - 2.0f * z_off * pix + defocus_diff * cos2a_minus_astig);

    float chi = (lambda2 * df - lambda3 * freq2) * freq2 + phase_shift;

    float sin_chi, cos_chi;
    __sincosf(chi, &sin_chi, &cos_chi);
    float ctf_val = w_sin * sin_chi + w_cos * cos_chi;
    if (flip_contrast) ctf_val = -ctf_val;

    cufftComplex v = d_freq_base[idx];
    if (ctf_val < 0.0f) {
        v.x = -v.x;
        v.y = -v.y;
    }

    d_freq_batch[(size_t)s * (size_t)freq_width * (size_t)Ny + (size_t)idx] = v;
}

__global__ void precompute_freq_geom_kernel(
    float* __restrict__ d_freq2_base,
    float* __restrict__ d_cos2alpha,
    float* __restrict__ d_sin2alpha,
    int Nx, int Ny
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int freq_width = Nx / 2 + 1;
    if (x >= freq_width || y >= Ny) return;

    const int Ny_ce = (Ny + 2) / 2;
    float x_norm = (float)x;
    float y_norm = (y >= Ny_ce) ? (float)(y - Ny) : (float)y;

    float xr = x_norm / (float)Nx;
    float yr = y_norm / (float)Ny;
    float r2 = xr * xr + yr * yr;
    int idx = y * freq_width + x;

    d_freq2_base[idx] = r2;
    if (r2 > 0.0f) {
        float inv_r2 = 1.0f / r2;
        d_cos2alpha[idx] = (xr * xr - yr * yr) * inv_r2;
        d_sin2alpha[idx] = (2.0f * xr * yr) * inv_r2;
    } else {
        d_cos2alpha[idx] = 1.0f;
        d_sin2alpha[idx] = 0.0f;
    }
}

namespace {
constexpr int kFreqBlockX = 32;
constexpr int kFreqBlockY = 8;
static long long g_ctf3d_inv_plan_create_count = 0;
static double g_ctf3d_inv_plan_create_ms = 0.0;

struct Ctf3dGpuProfile {
    bool inited = false;
    bool enabled = false;
    bool per_call = false;

    int calls = 0;
    double ensure_ms = 0.0;
    double zcopy_ms = 0.0;
    double const_ms = 0.0;
    double fft_fwd_ms = 0.0;
    double expand_ms = 0.0;
    double fft_inv_ms = 0.0;
    double norm_ms = 0.0;
    double total_ms = 0.0;

    cudaEvent_t evt_begin = nullptr;
    cudaEvent_t evt_end = nullptr;

    void init_if_needed() {
        if (inited) return;
        inited = true;

        const char* env = getenv("WBP_CTF3D_PROFILE");
        enabled = (env && atoi(env) != 0);
        const char* env_per_call = getenv("WBP_CTF3D_PROFILE_PER_CALL");
        per_call = (env_per_call && atoi(env_per_call) != 0);

        if (enabled) {
            CUDA_CHECK(cudaEventCreate(&evt_begin));
            CUDA_CHECK(cudaEventCreate(&evt_end));
        }
    }

    void report_and_reset() {
        if (!enabled || calls <= 0) return;
        const double inv_n = 1.0 / (double)calls;
        printf("\n[PROFILE][GPU][CTF3D] calls=%d\n", calls);
        printf("[PROFILE][GPU][CTF3D] ensure/cache total=%.3f ms, avg=%.3f ms/call\n", ensure_ms, ensure_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] zcopy       total=%.3f ms, avg=%.3f ms/call\n", zcopy_ms, zcopy_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] ctf const    total=%.3f ms, avg=%.3f ms/call\n", const_ms, const_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] fft r2c      total=%.3f ms, avg=%.3f ms/call\n", fft_fwd_ms, fft_fwd_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] phaseflip    total=%.3f ms, avg=%.3f ms/call\n", expand_ms, expand_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] fft c2r      total=%.3f ms, avg=%.3f ms/call\n", fft_inv_ms, fft_inv_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] normalize    total=%.3f ms, avg=%.3f ms/call\n", norm_ms, norm_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] total staged total=%.3f ms, avg=%.3f ms/call\n\n", total_ms, total_ms * inv_n);
        printf("[PROFILE][GPU][CTF3D] inv plan create count=%lld, total=%.3f ms\n\n",
               g_ctf3d_inv_plan_create_count, g_ctf3d_inv_plan_create_ms);

        calls = 0;
        ensure_ms = zcopy_ms = const_ms = 0.0;
        fft_fwd_ms = expand_ms = fft_inv_ms = norm_ms = total_ms = 0.0;
        g_ctf3d_inv_plan_create_count = 0;
        g_ctf3d_inv_plan_create_ms = 0.0;
    }

    void shutdown() {
        if (evt_begin) { cudaEventDestroy(evt_begin); evt_begin = nullptr; }
        if (evt_end) { cudaEventDestroy(evt_end); evt_end = nullptr; }
    }
};

static Ctf3dGpuProfile g_ctf3d_prof;

struct CtfBatchCache {
    int Nx = 0;
    int Ny = 0;
    int batch_capacity = 0;
    size_t freq_width = 0;
    size_t freq_size = 0; // freq_width * Ny

    cufftHandle plan_fwd = 0;
    cufftHandle plan_inv_many = 0; // active plan for current call batch
    int plan_inv_many_batch = 0;

    struct InvPlanEntry {
        int batch = 0;
        cufftHandle plan = 0;
    };
    std::vector<InvPlanEntry> inv_plan_cache;

    cufftComplex* d_freq_base = nullptr;
    cufftComplex* d_freq_batch = nullptr;
    float* d_z_offsets = nullptr;
    float* d_freq2_base = nullptr;
    float* d_cos2alpha = nullptr;
    float* d_sin2alpha = nullptr;

    void destroy_inv_plans() {
        for (size_t i = 0; i < inv_plan_cache.size(); ++i) {
            if (inv_plan_cache[i].plan) {
                cufftDestroy(inv_plan_cache[i].plan);
                inv_plan_cache[i].plan = 0;
            }
        }
        inv_plan_cache.clear();
        plan_inv_many = 0;
        plan_inv_many_batch = 0;
    }

    void destroy() {
        if (plan_fwd) { cufftDestroy(plan_fwd); plan_fwd = 0; }
        destroy_inv_plans();
        if (d_freq_base) { cudaFree(d_freq_base); d_freq_base = nullptr; }
        if (d_freq_batch) { cudaFree(d_freq_batch); d_freq_batch = nullptr; }
        if (d_z_offsets) { cudaFree(d_z_offsets); d_z_offsets = nullptr; }
        if (d_freq2_base) { cudaFree(d_freq2_base); d_freq2_base = nullptr; }
        if (d_cos2alpha) { cudaFree(d_cos2alpha); d_cos2alpha = nullptr; }
        if (d_sin2alpha) { cudaFree(d_sin2alpha); d_sin2alpha = nullptr; }
        Nx = Ny = batch_capacity = 0;
        freq_width = freq_size = 0;
    }

    cufftHandle find_inv_plan(int b) const {
        for (size_t i = 0; i < inv_plan_cache.size(); ++i) {
            if (inv_plan_cache[i].batch == b) return inv_plan_cache[i].plan;
        }
        return 0;
    }

    void ensure_batch_buffers(int b) {
        if (b <= batch_capacity && d_freq_batch && d_z_offsets) return;

        if (d_freq_batch) { CUDA_CHECK(cudaFree(d_freq_batch)); d_freq_batch = nullptr; }
        if (d_z_offsets) { CUDA_CHECK(cudaFree(d_z_offsets)); d_z_offsets = nullptr; }

        CUDA_CHECK(cudaMalloc(&d_freq_batch, freq_size * (size_t)b * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_z_offsets, (size_t)b * sizeof(float)));
        batch_capacity = b;
    }

    cufftHandle create_inv_plan(int b) {
        auto t_begin = std::chrono::steady_clock::now();
        cufftHandle plan = 0;
        int n[2] = {Ny, Nx};
        int inembed[2] = {Ny, (int)freq_width};
        int onembed[2] = {Ny, Nx};
        int istride = 1, ostride = 1;
        int idist = (int)freq_size;
        int odist = Nx * Ny;
        CUFFT_CHECK(cufftPlanMany(&plan,
                                 2, n,
                                 inembed, istride, idist,
                                 onembed, ostride, odist,
                                 CUFFT_C2R, b));
        InvPlanEntry entry;
        entry.batch = b;
        entry.plan = plan;
        inv_plan_cache.push_back(entry);
        auto t_end = std::chrono::steady_clock::now();
        g_ctf3d_inv_plan_create_count++;
        g_ctf3d_inv_plan_create_ms += std::chrono::duration<double, std::milli>(t_end - t_begin).count();
        return plan;
    }

    void ensure(int nx, int ny, int b) {
        // Rebuild geometry-dependent resources only when image size changes.
        if (nx != Nx || ny != Ny ||
            !plan_fwd || !d_freq_base || !d_freq2_base || !d_cos2alpha || !d_sin2alpha) {
            destroy();
            Nx = nx;
            Ny = ny;
            freq_width = (size_t)(Nx / 2 + 1);
            freq_size  = freq_width * (size_t)Ny;

            CUDA_CHECK(cudaMalloc(&d_freq_base,  freq_size * sizeof(cufftComplex)));
            CUDA_CHECK(cudaMalloc(&d_freq2_base, freq_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_cos2alpha, freq_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sin2alpha, freq_size * sizeof(float)));

            CUFFT_CHECK(cufftPlan2d(&plan_fwd, Ny, Nx, CUFFT_R2C));

            dim3 block(kFreqBlockX, kFreqBlockY);
            dim3 grid((int)((freq_width + block.x - 1) / block.x),
                      (Ny + block.y - 1) / block.y);
            precompute_freq_geom_kernel<<<grid, block>>>(
                d_freq2_base,
                d_cos2alpha,
                d_sin2alpha,
                Nx, Ny
            );
            CUDA_CHECK_LAST_ERROR();
        }

        // Batch-dependent resources are grown lazily and never shrunk in-run.
        ensure_batch_buffers(b);

        cufftHandle cached = find_inv_plan(b);
        if (!cached) cached = create_inv_plan(b);
        plan_inv_many = cached;
        plan_inv_many_batch = b;
    }
};

static CtfBatchCache g_ctf_cache;
} // namespace

#define CTF3D_PROFILE_STAGE(stage_acc, ...)                               \
    do {                                                                   \
        if (g_ctf3d_prof.enabled) {                                        \
            CUDA_CHECK(cudaEventRecord(g_ctf3d_prof.evt_begin, _ctf3d_stream));        \
            do { __VA_ARGS__; } while (0);                                 \
            CUDA_CHECK(cudaEventRecord(g_ctf3d_prof.evt_end, _ctf3d_stream));          \
            CUDA_CHECK(cudaEventSynchronize(g_ctf3d_prof.evt_end));        \
            float _ctf3d_ms = 0.0f;                                        \
            CUDA_CHECK(cudaEventElapsedTime(&_ctf3d_ms,                    \
                                            g_ctf3d_prof.evt_begin,         \
                                            g_ctf3d_prof.evt_end));         \
            (stage_acc) += (double)_ctf3d_ms;                              \
            _ctf3d_call_total += (double)_ctf3d_ms;                        \
        } else {                                                            \
            do { __VA_ARGS__; } while (0);                                 \
        }                                                                   \
    } while (0)

static void ctf_correction_3d_gpu_device_impl(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host,
    bool do_normalize,
    cudaStream_t stream
) {
    if (!d_image || !d_corrected || num_slices <= 0) return;
    cudaStream_t _ctf3d_stream = stream;
    g_ctf3d_prof.init_if_needed();
    double _ctf3d_call_total = 0.0;

    // Cache plans/buffers for this (Nx, Ny, num_slices)
    auto t_ensure_begin = std::chrono::steady_clock::now();
    g_ctf_cache.ensure(Nx, Ny, num_slices);
    auto t_ensure_end = std::chrono::steady_clock::now();
    double ensure_ms = std::chrono::duration<double, std::milli>(t_ensure_end - t_ensure_begin).count();
    g_ctf3d_prof.ensure_ms += ensure_ms;
    _ctf3d_call_total += ensure_ms;
    CUFFT_CHECK(cufftSetStream(g_ctf_cache.plan_fwd, _ctf3d_stream));
    CUFFT_CHECK(cufftSetStream(g_ctf_cache.plan_inv_many, _ctf3d_stream));

    // Copy z_offsets to device (small)
    if (z_offsets_host) {
        CTF3D_PROFILE_STAGE(g_ctf3d_prof.zcopy_ms,
            CUDA_CHECK(cudaMemcpyAsync(g_ctf_cache.d_z_offsets, z_offsets_host,
                                  (size_t)num_slices * sizeof(float),
                                  cudaMemcpyHostToDevice, _ctf3d_stream));
        );
    } else {
        CTF3D_PROFILE_STAGE(g_ctf3d_prof.zcopy_ms,
            CUDA_CHECK(cudaMemsetAsync(g_ctf_cache.d_z_offsets, 0, (size_t)num_slices * sizeof(float), _ctf3d_stream));
        );
    }

    // Compute CTF constants (host)
    float defocus1, defocus2, astig, phase_shift, w_cos, w_sin;
    float pix, pix_1, lambda, lambda2, lambda3;
    auto t_const_begin = std::chrono::steady_clock::now();
    ctf.computeCTF2D_P(defocus1, defocus2, astig, phase_shift,
                       w_cos, w_sin, pix, pix_1, lambda, lambda2, lambda3);
    auto t_const_end = std::chrono::steady_clock::now();
    double const_ms = std::chrono::duration<double, std::milli>(t_const_end - t_const_begin).count();
    g_ctf3d_prof.const_ms += const_ms;
    _ctf3d_call_total += const_ms;

    // 1) Forward FFT once: image -> freq_base
    CTF3D_PROFILE_STAGE(g_ctf3d_prof.fft_fwd_ms,
        CUFFT_CHECK(cufftExecR2C(g_ctf_cache.plan_fwd, (cufftReal*)d_image, (cufftComplex*)g_ctf_cache.d_freq_base));
    );

    // 2) Expand into batched frequency buffer for all slices
    dim3 block(kFreqBlockX, kFreqBlockY);
    dim3 grid((Nx / 2 + 1 + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              num_slices);

    const float defocus_sum = defocus1 + defocus2;
    const float defocus_diff = defocus1 - defocus2;
    const float pix_1_sq = pix_1 * pix_1;
    const float cos2astig = cosf(2.0f * astig);
    const float sin2astig = sinf(2.0f * astig);

    CTF3D_PROFILE_STAGE(g_ctf3d_prof.expand_ms,
        ctf_phaseflip_expand_kernel<<<grid, block, 0, _ctf3d_stream>>>(
            g_ctf_cache.d_freq_base,
            g_ctf_cache.d_freq_batch,
            g_ctf_cache.d_z_offsets,
            g_ctf_cache.d_freq2_base,
            g_ctf_cache.d_cos2alpha,
            g_ctf_cache.d_sin2alpha,
            Nx, Ny,
            defocus_sum, defocus_diff,
            phase_shift, w_cos, w_sin,
            pix_1_sq, lambda2, lambda3,
            cos2astig, sin2astig,
            flip_contrast, pix,
            num_slices
        );
        CUDA_CHECK_LAST_ERROR();
    );

    // 3) Batched inverse FFT: freq_batch -> corrected stack
    CTF3D_PROFILE_STAGE(g_ctf3d_prof.fft_inv_ms,
        CUFFT_CHECK(cufftExecC2R(g_ctf_cache.plan_inv_many,
                                 (cufftComplex*)g_ctf_cache.d_freq_batch,
                                 (cufftReal*)d_corrected));
    );

    if (do_normalize) {
        // 4) Normalize all slices in one kernel
        const int img_size = Nx * Ny;
        const int total = img_size * num_slices;
        const float scale = 1.0f / (float)(Nx * Ny);
        int tpb = 256;
        int bpg = (total + tpb - 1) / tpb;
        CTF3D_PROFILE_STAGE(g_ctf3d_prof.norm_ms,
            normalize_kernel<<<bpg, tpb, 0, _ctf3d_stream>>>(d_corrected, total, scale);
            CUDA_CHECK_LAST_ERROR();
        );
    }

    g_ctf3d_prof.calls++;
    g_ctf3d_prof.total_ms += _ctf3d_call_total;
    if (g_ctf3d_prof.enabled && g_ctf3d_prof.per_call) {
        printf("[PROFILE][GPU][CTF3D] call=%d total=%.3f ms\n",
               g_ctf3d_prof.calls, _ctf3d_call_total);
    }
}

extern "C" void ctf_correction_3d_gpu_device(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host
) {
    ctf_correction_3d_gpu_device_impl(
        d_image,
        d_corrected,
        Nx, Ny,
        num_slices,
        ctf,
        flip_contrast,
        z_offsets_host,
        true,
        (cudaStream_t)0
    );
}

extern "C" void ctf_correction_3d_gpu_device_no_norm(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host
) {
    ctf_correction_3d_gpu_device_impl(
        d_image,
        d_corrected,
        Nx, Ny,
        num_slices,
        ctf,
        flip_contrast,
        z_offsets_host,
        false,
        (cudaStream_t)0
    );
}

extern "C" void ctf_correction_3d_gpu_device_no_norm_stream(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host,
    void* stream_handle
) {
    cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : (cudaStream_t)0;
    ctf_correction_3d_gpu_device_impl(
        d_image,
        d_corrected,
        Nx, Ny,
        num_slices,
        ctf,
        flip_contrast,
        z_offsets_host,
        false,
        stream
    );
}

extern "C" void ctf_correction_3d_gpu_device_profile_report() {
    g_ctf3d_prof.report_and_reset();
}

extern "C" void ctf_correction_3d_gpu_device_profile_shutdown() {
    g_ctf3d_prof.shutdown();
}

#undef CTF3D_PROFILE_STAGE
