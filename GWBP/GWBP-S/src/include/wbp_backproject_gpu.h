/*******************************************************************
 *       Filename:  wbp_backproject_gpu.h
 *
 *    Description:  WBP backprojection GPU interface
 *******************************************************************/

#ifndef WBP_BACKPROJECT_GPU_H
#define WBP_BACKPROJECT_GPU_H

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

/*
 * Legacy/rollback API: device-stack backprojection accumulate.
 * corrected_scale was not part of the original interface.
 * If scaling is needed, it should be handled internally in the .cu/.cpp implementation.
 */
void wbp_backproject_accumulate_gpu_device(
    const float* d_stack_corrected,
    int Nx, int Ny, int h,
    int n_zz,
    int defocus_step,
    int tmp_h,
    float x_orig_offset,
    float z_orig_offset,
    float sin1,
    float cos1
);

void wbp_recon_finalize(float* h_recon_out);

// Compute min/max/mean of the GPU recon volume and return to host.
void wbp_recon_get_stats(float* h_min_out, float* h_max_out, double* h_mean_out);

void wbp_recon_free();

#ifdef __cplusplus
}
#endif

#endif // WBP_BACKPROJECT_GPU_H