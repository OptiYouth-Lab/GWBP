/*******************************************************************
 *       Filename:  wbp_backproject_gpu.h
 *
 *    Description:  WBP backprojection GPU interface
 *******************************************************************/

#ifndef WBP_BACKPROJECT_GPU_H
#define WBP_BACKPROJECT_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void wbp_recon_init(int Nx, int Ny, int h);

void wbp_backproject_accumulate_gpu(
    const float* h_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1
);

void wbp_backproject_accumulate_gpu_device(
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
);

void wbp_backproject_accumulate_gpu_device_stream(
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

// Precompute xz mapping tables for all tilts and reuse during backprojection.
// Return 1 on success, 0 on failure.
int wbp_backproject_prepare_xz_maps(
    int Nx, int h,
    int n_maps,
    int defocus_step,
    float x_orig_offset,
    float z_orig_offset,
    const float* h_tmp_h_all,
    const float* h_sin_all,
    const float* h_cos_all,
    void* stream_handle
);

void wbp_backproject_accumulate_gpu_device_precomputed_map_stream(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int map_index,
    float corrected_scale,
    void* stream_handle
);

void wbp_recon_finalize(float* h_recon_out);
void wbp_recon_copy_range(float* h_out, size_t offset_elems, size_t count_elems);

// Compute min/max/mean of the GPU recon volume and return to host.
void wbp_recon_get_stats(float* h_min_out, float* h_max_out, double* h_mean_out);

void wbp_recon_free();

#ifdef __cplusplus
}
#endif

#endif // WBP_BACKPROJECT_GPU_H

