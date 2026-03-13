/*******************************************************************
 *       Filename:  ReconstructionAlgo.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/07/2020 05:40:48 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/
#include "../include/ReconstructionAlgo_WBP_RAM.h"
#include "../include/mrc.h"
#include "../include/CTF.h"
#include "math.h"
#include "fftw3.h"
#include "omp.h"
#include <immintrin.h>
#include <sys/time.h> 
#include "util.h"
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <sys/stat.h>
#include <errno.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <vector>

#ifdef USE_GPU
// GPU path: CTF correction + WBP backprojection
#include <cuda_runtime.h>
#include "cuda_utils.h"          // CUDA_CHECK / CUDA_CHECK_LAST_ERROR
#include "wbp_backproject_gpu.h" // WBP backprojection (GPU)

// CTF校正GPU版本（device-output版本：避免每张图把 corrected stack 拷回 host）
extern "C" void ctf_correction_3d_gpu_device(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host
);
extern "C" void ctf_correction_3d_gpu_device_no_norm(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host
);
extern "C" void ctf_correction_3d_gpu_device_no_norm_stream(
    const float* d_image,
    float* d_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets_host,
    void* stream_handle
);
extern "C" void ctf_correction_3d_gpu_device_profile_shutdown();

// CTF校正GPU版本
extern "C" void ctf_correction_gpu(
    float* h_image,
    float* h_output,
    int Nx, int Ny,
    CTF ctf,
    bool flip_contrast,
    float z_offset
);

// 3D-CTF校正GPU版本
extern "C" void ctf_correction_3d_gpu(
    float* h_image,
    float* h_corrected,
    int Nx, int Ny,
    int num_slices,
    CTF ctf,
    bool flip_contrast,
    float* z_offsets
);

// 1D weighting on GPU (in-place on device image [Ny][Nx])
extern "C" void filter_weighting_1d_many_gpu_inplace(
    float* d_image,
    int Nx, int Ny,
    float radial,
    float sigma
);
extern "C" void filter_weighting_1d_many_gpu_inplace_stream(
    float* d_image,
    int Nx, int Ny,
    float radial,
    float sigma,
    void* stream_handle
);

// Recon volume stats on GPU
extern "C" void wbp_recon_get_stats(
    float* h_min_out,
    float* h_max_out,
    double* h_mean_out
);
#endif

static void buf2fft(float *buf, float *fft, int nx, int ny)
{
    int nxb=nx+2-nx%2;
    int i;
	#pragma omp parallel for schedule(static)
    for(i=0;i<(nx+2-nx%2)*ny;i++)
    {
        fft[i]=0.0;
    }
	#pragma omp parallel for schedule(static)
    for(i=0;i<ny;i++)
    {
        memcpy(fft+i*nxb,buf+i*nx,sizeof(float)*nx);
    }
}

static void fft2buf(float *buf, float *fft, int nx, int ny)
{
    int nxb=nx+2-nx%2;
    int i;
    for(i=0;i<nx*ny;i++)
    {
        buf[i]=0.0;
    }
    for(i=0;i<ny;i++)
    {
        memcpy(buf+i*nx,fft+i*nxb,sizeof(float)*nx);
    }
}

static void buf2fft_padding_1D(float *buf, float *fft, int nx_orig, int nx_final, int ny_orig, int ny_final)
{
    int nxb = nx_final + 2 - (nx_final & 1);
    int nxp = nx_final - nx_orig + 1;
    int i,j;

	float nxp_1 = 1.0 / float(nxp);

	#pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for(i=0;i<ny_orig;i++)
        {
            memcpy(fft+i*nxb,buf+i*nx_orig,sizeof(float)*nx_orig);
            float tmp1 = buf[i*nx_orig] * nxp_1;
            float tmp2 = buf[(i+1)*nx_orig-1] * nxp_1;
            
            for(j=nx_orig;j<nx_final;j++)    // padding for continuity (FFT本质是周期延拓后做DFT，将首尾连接以保证连续性)
            {
                fft[i*nxb+j]=(float(j-nx_orig+1))*tmp1+(float(nx_final-j))*tmp2;
            }
        }
    }
}

static void fft2buf_padding_1D(float *buf, float *fft, int nx_orig, int nx_final, int ny_orig, int ny_final)
{
    int nxb = nx_final + 2 - (nx_final & 1);

    int i;

	int loop_num = nx_orig * ny_orig;

	#pragma omp parallel for schedule(static)
    for(i = 0; i < ny_orig; i++)
    {
        memcpy(buf + i * nx_orig, fft + i * nxb, sizeof(float) * nx_orig);

        int cnt = (i + 1) * nx_orig;

        for(int j = (i + 1) * nxb; j < cnt; j++)
        {
            buf[j] = 0.0;
        }
    }
}

static void filter_weighting(float *data,int Nx,float radial,float sigma)
{
    int Nx_padding=int(Nx/10);
    int Nx_final=Nx+Nx_padding;
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[Nx_final+2-Nx_final%2];
    plan_fft=fftwf_plan_dft_r2c_1d(Nx_final,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft=fftwf_plan_dft_c2r_1d(Nx_final,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
    buf2fft_padding_1D(data,bufc,Nx,Nx_final,1,1);
    fftwf_execute(plan_fft);

    int radial_Nx=int(floor((Nx_final)*radial));
    float sigma_Nx=float(Nx_final)*sigma;

    // loop: Nx_final+2-Nx_final%2 (all Fourier components)
    for(int i=0;i<Nx_final+2-Nx_final%2;i+=2)   // radial filtering
    {
        if(i==0)    // DC
        {
            bufc[i]*=0.2;
            bufc[i+1]*=0.2;
        }
        else if(i/2<=radial_Nx)  // radial
        {
            bufc[i]*=(i/2);
            bufc[i+1]*=(i/2);
        }
        else    // Gaussian falloff
        {
            bufc[i]=bufc[i]*float(radial_Nx)*exp(-float((i/2-radial_Nx)*(i/2-radial_Nx))/(sigma_Nx*sigma_Nx));
            bufc[i+1]=bufc[i+1]*float(radial_Nx)*exp(-float((i/2-radial_Nx)*(i/2-radial_Nx))/(sigma_Nx*sigma_Nx));
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf_padding_1D(data,bufc,Nx,Nx_final,1,1);
    for(int i=0;i<Nx;i++)   // normalization
    {
        data[i]/=Nx;
    }
    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
}


static void filter_weighting_1D_many1(float *data, float * bufc, int Nx,int Ny,float radial,float sigma)
{
    int Nx_padding = int(Nx / 10);
    int Nx_final   = Nx + Nx_padding;

    fftwf_plan plan_fft,plan_ifft;
    
    // FFTW threads: avoid oversubscription with OpenMP. Default to 1 thread.
    // You can override via: export WBP_FFTW_THREADS=<n>
    static bool fftw_threads_inited = false;
    if (!fftw_threads_inited) {
        fftwf_init_threads();
        fftw_threads_inited = true;
    }
    int fftw_nthreads = 1;
    if (const char* env = std::getenv("WBP_FFTW_THREADS")) {
        int v = std::atoi(env);
        if (v > 0) fftw_nthreads = v;
    }
    fftwf_plan_with_nthreads(fftw_nthreads);

    plan_fft  = fftwf_plan_many_dft_r2c(1,&Nx_final,Ny,(float*)bufc,NULL,1,(Nx_final+2-Nx_final%2),
									    reinterpret_cast<fftwf_complex*>(bufc),NULL,1,(Nx_final+2-Nx_final%2)/2,FFTW_ESTIMATE);

    plan_ifft = fftwf_plan_many_dft_c2r(1,&Nx_final,Ny,reinterpret_cast<fftwf_complex*>(bufc),NULL,1,(Nx_final+2-Nx_final%2)/2,
									    (float*)bufc,NULL,1,(Nx_final+2-Nx_final%2),FFTW_ESTIMATE);
    
    buf2fft_padding_1D(data,bufc,Nx,Nx_final,Ny,Ny);

    fftwf_execute(plan_fft);

    int  radial_Nx = int(floor((Nx_final) * radial));

    float sigma_Nx = float(Nx_final) * sigma;
	sigma_Nx *= sigma_Nx;

	register int loop_i = Nx_final + 2 - (Nx_final & 1);

    float sigma_Nx_1 = 1.0 / sigma_Nx;

    // loop: Ny (all Fourier components for y-axis)
	#pragma omp parallel for schedule(static)
    for(int j = 0; j < Ny; j++)
    {
		int j_index = j * loop_i;
		int i;

		int num = min(0, loop_i);
		for(i = 0; i <=num; i +=2)
		{
			bufc[i  +j_index] *= 0.2;
            bufc[i+1+j_index] *= 0.2;
		}

		num = min(radial_Nx<<1, loop_i-1);
		for( ; i <=num; i +=2)
        {
            bufc[i  +j_index] *= (i>>1);
            bufc[i+1+j_index] *= (i>>1);
        }

		for( ; i <loop_i; i +=2)
		{
			int i_tmp = (i>>1) - radial_Nx;
            i_tmp *= i_tmp;
            float tmp1 = float(radial_Nx)*exp(-float(i_tmp) * sigma_Nx_1);

            bufc[i  +j_index] = bufc[i  +j_index] * tmp1;
            bufc[i+1+j_index] = bufc[i+1+j_index] * tmp1;
		}
    }

    fftwf_execute(plan_ifft);

    fft2buf_padding_1D(data,bufc,Nx,Nx_final,Ny,Ny);

	double Nx_1 = 1.0 / double(Nx);
	int  Nx_Ny  = Nx  * Ny;

    #pragma omp parallel for simd schedule(static)
    for(int i = 0; i < Nx_Ny; i++)   // normalization
    {
        data[i] *= Nx_1;
    }

    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
}


static void filter_weighting_1D_many(float *data,int Nx,int Ny,float radial,float sigma)
{
    int Nx_padding = int(Nx / 10);
    int Nx_final   = Nx + Nx_padding;

    fftwf_plan plan_fft,plan_ifft;

	float * bufc = (float *)aligned_alloc(4096, sizeof(float) * (Nx_final+2-Nx_final%2) * Ny);
    
    

    plan_fft  = fftwf_plan_many_dft_r2c(1,&Nx_final,Ny,(float*)bufc,NULL,1,(Nx_final+2-Nx_final%2),
									    reinterpret_cast<fftwf_complex*>(bufc),NULL,1,(Nx_final+2-Nx_final%2)/2,FFTW_ESTIMATE);

    plan_ifft = fftwf_plan_many_dft_c2r(1,&Nx_final,Ny,reinterpret_cast<fftwf_complex*>(bufc),NULL,1,(Nx_final+2-Nx_final%2)/2,
									    (float*)bufc,NULL,1,(Nx_final+2-Nx_final%2),FFTW_ESTIMATE);
    
    buf2fft_padding_1D(data,bufc,Nx,Nx_final,Ny,Ny);

    fftwf_execute(plan_fft);

    int  radial_Nx = int(floor((Nx_final) * radial));

    float sigma_Nx = float(Nx_final) * sigma;
	sigma_Nx *= sigma_Nx;

	register int loop_i = Nx_final + 2 - (Nx_final & 1);

    // loop: Ny (all Fourier components for y-axis)
	#pragma omp parallel for schedule(static)
    for(int j = 0; j < Ny; j++)
    {
		int j_index = j * loop_i;
		int i;

		int num = min(0, loop_i);
		for(i = 0; i <=num; i +=2)
		{
			bufc[i  +j_index] *= 0.2;
            bufc[i+1+j_index] *= 0.2;
		}

		num = min(radial_Nx<<1, loop_i-1);
		for( ; i <=num; i +=2)
        {
            bufc[i  +j_index] *= (i>>1);
            bufc[i+1+j_index] *= (i>>1);
        }

		for( ; i <loop_i; i +=2)
		{
			int i_tmp = (i>>1) - radial_Nx;
            i_tmp *= i_tmp;
            float tmp1 = float(radial_Nx)*exp(-float(i_tmp)/sigma_Nx);

            bufc[i  +j_index] = bufc[i  +j_index] * tmp1;
            bufc[i+1+j_index] = bufc[i+1+j_index] * tmp1;
		}
    }

    fftwf_execute(plan_ifft);

    fft2buf_padding_1D(data,bufc,Nx,Nx_final,Ny,Ny);

	double Nx_1 = 1.0 / double(Nx);
	int Nx_Ny  = Nx  * Ny;

	#pragma omp parallel for schedule(static)
    for(int i = 0; i < Nx_Ny; i++)   // normalization
    {
        data[i] *= Nx_1;
    }

    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);

	free(bufc);
}

static void filter_weighting_2D(float *image,int Nx,int Ny,float radial,float sigma,float psi_rad)
{
    int Nx_padding=0;
    int Nx_final=Nx_padding+Nx;
    fftwf_plan plan_fft,plan_ifft;
    float *bufc=new float[(Nx_final+2-Nx_final%2)*Ny];
    plan_fft=fftwf_plan_dft_r2c_2d(Ny,Nx_final,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft=fftwf_plan_dft_c2r_2d(Ny,Nx_final,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
    buf2fft(image,bufc,Nx,Ny);
    fftwf_execute(plan_fft);

    float radial_pi=M_PI*radial,sigma_pi=M_PI*sigma;
    float fft_x[Nx_final],fft_y[Ny];
    // loop: Nx_final (all Fourier components for x-axis)
    for(int i=0;i<Nx_final;i++)
    {
        fft_x[i]=2*M_PI/(Nx_final)*i;
        if(fft_x[i]>M_PI)
        {
            fft_x[i]=fft_x[i]-2*M_PI;
        }
    }
    // loop: Ny (all Fourier components for y-axis)
    for(int j=0;j<Ny;j++)
    {
        fft_y[j]=2*M_PI/Ny*j;
        if(fft_y[j]>M_PI)
        {
            fft_y[j]=fft_y[j]-2*M_PI;
        }
    }


	#pragma omp parallel for schedule(static)
    // loop: Ny (all Fourier components for y-axis)
    for(int j=0;j<Ny;j++)
    {
        // loop: Nx_final+2-Nx_final%2 (all Fourier components for x-axis)
        for(int i=0;i<Nx_final+2-Nx_final%2;i+=2)   // 2D radial filtering
        {
            float x_rot=fft_x[i/2]*cos(-psi_rad)-fft_y[j]*sin(-psi_rad),y_rot=fft_x[i/2]*sin(-psi_rad)+fft_y[j]*cos(-psi_rad);
            if(fabs(x_rot)<1e-6)    // DC
            {
                bufc[i+j*(Nx_final+2-Nx_final%2)]*=(0.2*fft_x[1]*cos(-psi_rad));
                bufc[i+1+j*(Nx_final+2-Nx_final%2)]*=(0.2*fft_x[1]*cos(-psi_rad));
            }
            else if(fabs(x_rot)<=radial_pi)   // radial
            {
                bufc[i+j*(Nx_final+2-Nx_final%2)]*=x_rot;
                bufc[i+1+j*(Nx_final+2-Nx_final%2)]*=x_rot;
            }
            else    // Gaussian falloff
            {
                bufc[i+j*(Nx_final+2-Nx_final%2)]=bufc[i+j*(Nx_final+2-Nx_final%2)]*radial_pi*exp(-((x_rot-radial_pi)*(x_rot-radial_pi))/(sigma_pi*sigma_pi));
                bufc[i+1+j*(Nx_final+2-Nx_final%2)]=bufc[i+1+j*(Nx_final+2-Nx_final%2)]*radial_pi*exp(-((x_rot-radial_pi)*(x_rot-radial_pi))/(sigma_pi*sigma_pi));
            }
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf(image,bufc,Nx,Ny);
    for(int i=0;i<Nx*Ny;i++)   // normalization
    {
        image[i]=image[i]/(Nx*Ny);
    }
    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
    delete [] bufc;
}






static void ctf_correction1(float *image, float *image1, float * bufc, int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset)   // z_offset in pixels
{
    float defocus1    ;
    float defocus2    ;
    float astig       ;
    float phase_shift ;
    float w_cos       ;
    float w_sin       ;
    float pix_1       ;
    float pix         ;
    float lambda      ;
    float lambda2     ;
    float lambda3     ;

    ctf.computeCTF2D_P(defocus1,defocus2,astig,phase_shift,w_cos,w_sin,pix, pix_1, lambda, lambda2, lambda3);

    fftwf_plan plan_fft;
    fftwf_plan plan_ifft;

	int loop_num = Nx + 2 - (Nx & 1);

    plan_fft  = fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
	plan_ifft = fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);

    int Nx_Ny     = Nx       * Ny;
	int loop_num1 = loop_num * Ny;

	float * p1 = bufc;
	float * p2 = image;
    int Nx_num = Nx - 16;

    for(int i=0; i<Ny; i++)
    {
		int j = 0;

        #pragma unroll(8)
		for(j=0; j<=Nx_num; j+=16)
		{
            _mm512_store_ps(&p1[j],_mm512_load_ps(&p2[j]));
		}

        for(; j<Nx; j++)
		{
			p1[j] = p2[j];
		}

        for(; j<loop_num; j++)
		{
			p1[j] = 0.0;
		}
		
		p1 += loop_num;
		p2 += Nx;
    }

	fftwf_execute(plan_fft);

    float Nx_Ny_1  = 1.0 / (float)Nx_Ny;
	int   i        = 0;
	int   cnt      = loop_num1 - 8;
    float Nx_pix_1 = pix_1 / float(Nx);
    float Ny_pix_1 = pix_1 / float(Ny);
    float tmp_num1 = defocus1 + defocus2 - 2 * z_offset * pix;
    float tmp_num2 = defocus1 - defocus2;

    int Nx_ce = int(ceil(float(Nx+1)/2));
    int Ny_ce = int(ceil(float(Ny+1)/2));

    if(flip_contrast)
    {
        for(i=0; i<Ny; i++)
        {
    #pragma omp parallel for simd schedule(static)
            for(int j=0; j<loop_num; j+=2)
            {
                int x = j>>1;
                int y = i;
                
                float x_norm = (x >= Nx_ce)?(x-Nx):(x);
                float y_norm = (y >= Ny_ce)?(y-Ny):(y);
                float x_real = float(x_norm) * (Nx_pix_1);
                float y_real = float(y_norm) * (Ny_pix_1);

                float alpha;
                if(x_norm==0)
                {
                    alpha = M_PI_2 * ((y_norm>0) - (y_norm<0));
                }
                else
                {
                    alpha = atan(y_real/x_real);
                }

                float freq2   = x_real * x_real + y_real * y_real;
                float df_now  = (tmp_num1 + tmp_num2 * cos(2 * (alpha - astig))) * 0.5;
                float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
                float ctf_now = -(w_sin * sin(chi) + w_cos * cos(chi));
                
                if(ctf_now<0)
                {
                    bufc[i*loop_num+j]   =-bufc[i*loop_num+j];
                    bufc[i*loop_num+j+1] =-bufc[i*loop_num+j+1];
                }
            }
        }
    }
    else
    {
        for(i=0; i<Ny; i++)
        {
    #pragma omp parallel for simd schedule(static)
            for(int j=0; j<loop_num; j+=2)
            {
                int x = j>>1;
                int y = i;
                
                float x_norm = (x>=Nx_ce)?(x-Nx):(x);
                float y_norm = (y>=Ny_ce)?(y-Ny):(y);

                float x_real = float(x_norm) * (Nx_pix_1);
                float y_real = float(y_norm) * (Ny_pix_1);

                float alpha;
                if(x_norm==0)
                {
                    alpha = M_PI_2 * ((y_norm>0) - (y_norm<0));
                }
                else
                {
                    alpha = atan(y_real/x_real);
                }

                float freq2   = x_real * x_real + y_real * y_real;
                float df_now  = (tmp_num1 + tmp_num2 * cos(2 * (alpha - astig))) * 0.5;
                float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
                float ctf_now = w_sin * sin(chi) + w_cos * cos(chi);
                
                if(ctf_now<0)
                {
                    bufc[i*loop_num+j]   =-bufc[i*loop_num+j];
                    bufc[i*loop_num+j+1] =-bufc[i*loop_num+j+1];
                }
            }
        }
    }
    
	
    fftwf_execute(plan_ifft);

    int loop_num4 = Ny * Nx;

    p1 = bufc;
	p2 = image1;

    __m512 Nx_Ny_1_tmp = _mm512_set1_ps(Nx_Ny_1);

    for(int i=0; i<Ny; i++)
    {
        int j=0;
        #pragma unroll(8)
        for(; j<Nx_num; j+=16)
        {
            _mm512_store_ps(&p2[j],_mm512_mul_ps(_mm512_load_ps(&p1[j]), Nx_Ny_1_tmp));
        }

    #pragma omp parallel for simd schedule(static)
        for(; j<Nx; j++)
        {
            p2[j] = p1[j] * Nx_Ny_1;
        }

        p1 += loop_num;
		p2 += Nx;
    }
    
    fftwf_destroy_plan(plan_fft);
    fftwf_destroy_plan(plan_ifft);
}



static void ctf_correction3(float *image, float *image1, float * bufc, int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset)   // z_offset in pixels
{
    fftwf_plan plan_fft;
	fftwf_plan plan_ifft;

    register float defocus1    ;
    register float defocus2    ;
    register float astig       ;
    register float phase_shift ;
    register float w_cos       ;
    register float w_sin       ;
    register float pix_1       ;
    register float pix         ;
    register float lambda      ;
    register float lambda2     ;
    register float lambda3     ;

    ctf.computeCTF2D_P(defocus1, defocus2, astig, phase_shift, w_cos, w_sin, pix, pix_1, lambda, lambda2, lambda3);

	register int loop_num = Nx + 2 - (Nx & 1);
    plan_fft              = fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft             = fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);

    register int Nx_Ny     = Nx * Ny;
	register int loop_num1 = loop_num * Ny;

    #pragma omp parallel for schedule(static)
    for(int z=0; z<Nx_Ny; z++)
    {
        int i = z / Nx;
        int j = z - i * Nx;
        bufc[i*loop_num+j] = image[z];
    }

    #pragma omp parallel for schedule(static)
    for(int i=0;i<Ny;i++)
    {
		for(int j=i*loop_num+Nx;j<(i+1)*loop_num;j++)
		{
			bufc[j] = 0.0;
		}
    }

    fftwf_execute(plan_fft);
	
    register float Nx_Ny_1  = 1.0   / (float)Nx_Ny;
    register float Nx_pix_1 = pix_1 / float(Nx);
    register float Ny_pix_1 = pix_1 / float(Ny);
    register float tmp_num1 = defocus1 + defocus2 - 2 * z_offset * pix;
    register float tmp_num2 = defocus1 - defocus2;

    if(flip_contrast)
    {
    #pragma omp parallel for simd schedule(static)
        for(int i=0; i<loop_num1; i+=2)
        {
            register int tmp1      = i / loop_num;
            register int tmp2      = i - tmp1 * loop_num;
            register int  x        = tmp2>>1;
            register int  y        = tmp1;
            register float x_norm  = (x>=int(ceil(float(Nx+1)/2)))?(x-Nx):(x);
            register float y_norm  = (y>=int(ceil(float(Ny+1)/2)))?(y-Ny):(y);
            register float x_real  = float(x_norm) * (Nx_pix_1);
            register float y_real  = float(y_norm) * (Ny_pix_1);
            register float alpha;

            if(x_norm==0)
                alpha = M_PI_2 * ((y_norm>0) - (y_norm<0));
            else
                alpha = atan(y_real / x_real);

            register float freq2   = x_real * x_real + y_real * y_real;
            register float df_now  = ( tmp_num1 + tmp_num2 * cos(2*(alpha-astig)) ) * 0.5;
            register float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
            register float ctf_now = -(w_sin * sin(chi) + w_cos * cos(chi));
        
            if(ctf_now<0)
            {
                bufc[i]   = -bufc[i];
                bufc[i+1] = -bufc[i+1];
            }
        }
    }
    else
    {
    #pragma omp parallel for simd schedule(static)
        for(int i=0; i<loop_num1; i+=2)
        {
            register int tmp1      = i / loop_num;
            register int tmp2      = i - tmp1 * loop_num;
            register int  x        = tmp2>>1;
            register int  y        = tmp1;
            register float x_norm  = (x>=int(ceil(float(Nx+1)/2)))?(x-Nx):(x);
            register float y_norm  = (y>=int(ceil(float(Ny+1)/2)))?(y-Ny):(y);
            register float x_real  = float(x_norm) * (Nx_pix_1);
            register float y_real  = float(y_norm) * (Ny_pix_1);
            register float alpha;

            if(x_norm==0)
            {
                alpha = M_PI_2 * ((y_norm>0) - (y_norm<0));
            }
            else
            {
                alpha = atan(y_real/x_real);
            }

            register float freq2   = x_real * x_real + y_real * y_real;
            register float df_now  = (tmp_num1 + tmp_num2 * cos(2*(alpha-astig))) * 0.5;
            register float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
            register float ctf_now = w_sin * sin(chi) + w_cos * cos(chi);
        
            if(ctf_now<0)
            {
                bufc[i]   = -bufc[i];
                bufc[i+1] = -bufc[i+1];
            }
        }
    }

    fftwf_execute(plan_ifft);

    int loop_num4 = Ny * Nx;

    #pragma unroll(16)
    #pragma omp parallel for simd schedule(static)
    for(int z=0; z<loop_num4; z++)
    {
        int i = z / Nx;
        int j = z - i * Nx;
        image1[i*Nx + j] = bufc[i*loop_num+j] * Nx_Ny_1;
    }

    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);
}

static void ctf_correction(float *image,int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset)   // z_offset in pixels
{

    fftwf_plan plan_fft;
	fftwf_plan plan_ifft;

	int loop_num = Nx + 2 - (Nx & 1);

	float * bufc = (float *)aligned_alloc(4096, sizeof(float) * loop_num * Ny);

    plan_fft  = fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
    plan_ifft = fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);

    int Nx_Ny = Nx*Ny;

	int loop_num1 = loop_num * Ny;

    #pragma omp parallel for schedule(static)
    for(int z=0; z<Nx_Ny; z++)
    {
        int i = z / Nx;
        int j = z - i * Nx;
        bufc[i*loop_num+j] = image[z];
    }

    #pragma omp parallel for schedule(static)
    for(int i=0;i<Ny;i++)
    {
		for(int j=i*loop_num+Nx;j<(i+1)*loop_num;j++)
		{
			bufc[j] = 0.0;
		}
    }

    fftwf_execute(plan_fft);
	
	
     double Nx_Ny_1 = 1.0 / (double)Nx_Ny;

	#pragma omp parallel for schedule(static)
    for(int i=0; i<loop_num1; i+=2)
    {
		int tmp1 = i / loop_num;
		int tmp2 = i - tmp1 * loop_num;

        float ctf_now = ctf.computeCTF2D(tmp2/2,tmp1,Nx,Ny,true,flip_contrast,z_offset);

        bufc[i]   *=ctf_now;
        bufc[i+1] *=ctf_now;
    }

    fftwf_execute(plan_ifft);

    int loop_num4 = Ny * Nx;

    #pragma omp parallel for schedule(static)
    for(int z=0; z<loop_num4; z++)
    {
        int i = z / Nx;
        int j = z - i * Nx;
        image[i*Nx + j] = bufc[i*loop_num+j] * Nx_Ny_1;
    }

    fftwf_destroy_plan(plan_fft);
	fftwf_destroy_plan(plan_ifft);

	free(bufc);
}





ReconstructionAlgo_WBP_RAM::~ReconstructionAlgo_WBP_RAM()
{

}

void ReconstructionAlgo_WBP_RAM::doReconstruction(map<string, string> &inputPara, map<string, string> &outputPara)
{
    // Input
    map<string,string>::iterator it=inputPara.find("path");
    string path;
    if(it!=inputPara.end())
    {
        path=it->second;
    }
    else
    {
        path="./";
    }

    it=inputPara.find("input_mrc");
    string input_mrc;
    if(it!=inputPara.end())
    {
        input_mrc=path+"/"+it->second;
    }
    else
    {
        cerr << "No input file name!" << endl;
        abort();
    }
    MRC stack_orig(input_mrc.c_str(),"rb");
    if(!stack_orig.hasFile())
    {
        cerr << "Cannot open input mrc stack!" << endl;
        abort();
    }

    it=inputPara.find("output_mrc");
    string output_mrc;
    if(it!=inputPara.end())
    {
        output_mrc=path+"/"+it->second;
    }
    else
    {
        output_mrc="tomo.rec";
    }

it=inputPara.find("h");
    int h;
    if(it!=inputPara.end())
    {
        h=atoi(it->second.c_str());
    }
    else
    {
        h=int(stack_orig.getNx()/4);
    }

    bool skip_ctfcorrection,skip_weighting,skip_3dctf;
    it=inputPara.find("skip_ctfcorrection");
    if(it!=inputPara.end())
    {
        skip_ctfcorrection=atoi(it->second.c_str());
    }
    else
    {
        skip_ctfcorrection=0;
    }
    it=inputPara.find("skip_3dctf");
    if(it!=inputPara.end())
    {
        skip_3dctf=atoi(it->second.c_str());
    }
    else
    {
        skip_3dctf=0;
    }
    it=inputPara.find("skip_weighting");
    if(it!=inputPara.end())
    {
        skip_weighting=atoi(it->second.c_str());
    }
    else
    {
        skip_weighting=0;
    }
    
    float weighting_radial=0.05,weighting_sigma=0.5;
    if(!skip_weighting)
    {
        it=inputPara.find("weighting_radial");
        if(it!=inputPara.end())
        {
            weighting_radial=atof(it->second.c_str());
        }
        else
        {
            weighting_radial=0.05;
        }
        it=inputPara.find("weighting_sigma");
        if(it!=inputPara.end())
        {
            weighting_sigma=atof(it->second.c_str());
        }
        else
        {
            weighting_sigma=0.5;
        }
    }

    it=inputPara.find("input_tlt");
    string input_tlt;
    if(it!=inputPara.end())
    {
        input_tlt=path+"/"+it->second;
    }
    else
    {
        cerr << "No input tlt file name!" << endl;
        abort();
    }
    FILE *ftlt=fopen(input_tlt.c_str(),"r");
    if(ftlt==NULL)
    {
        cerr << "Cannot open tlt file!" << endl;
        abort();
    }
    float theta[stack_orig.getNz()];
    float theta_max=0.0;
    for(int n=0;n<stack_orig.getNz();n++)
    {
        fscanf(ftlt,"%f",&theta[n]);
        if(fabs(theta[n])>theta_max)
        {
            theta_max=fabs(theta[n]);
        }
    }
    fflush(ftlt);
    fclose(ftlt);

    bool unrotated_stack=false;
    it=inputPara.find("unrotated_stack");
    if(it!=inputPara.end())
    {
        unrotated_stack=atoi(it->second.c_str());
    }
    else
    {
        unrotated_stack=0;
    }

    int h_tilt_max=int(ceil(float(stack_orig.getNx())*sin(theta_max/180*M_PI)+float(h)*cos(theta_max/180*M_PI)))+1;    // the maximum height after tilt

    string path_psi=path+"/"+"psi.txt";
    FILE *fpsi=fopen(path_psi.c_str(),"r");
    if(!fpsi)
    {
        cerr << "No psi found!" << endl;
        abort();
    }
    float psi_deg,psi_rad;
    fscanf(fpsi,"%f",&psi_deg);
    psi_rad=psi_deg*M_PI/180;
    fflush(fpsi);
    fclose(fpsi);

    CTF ctf_para[stack_orig.getNz()];
    float Cs,pix,volt,w_cos;
    bool flip_contrast=false;
    int defocus_step=1;
    it=inputPara.find("pixel_size");
    if(it!=inputPara.end())
    {
        pix=atof(it->second.c_str());
    }
    else
    {
        cerr << "No pixel size!" << endl;
        abort();
    }
    if(!skip_ctfcorrection) // read in defocus file for CTF correction
    {
        it=inputPara.find("Cs");
        if(it!=inputPara.end())
        {
            Cs=atof(it->second.c_str());
        }
        else
        {
            cerr << "No Cs!" << endl;
            abort();
        }
        it=inputPara.find("voltage");
        if(it!=inputPara.end())
        {
            volt=atof(it->second.c_str());
        }
        else
        {
            cerr << "No accelerating voltage!" << endl;
            abort();
        }
        it=inputPara.find("w");
        if(it!=inputPara.end())
        {
            w_cos=atof(it->second.c_str());
        }
        else
        {
            cerr << "No amplitude contrast!" << endl;
            abort();
        }

        it=inputPara.find("flip_contrast");
        if(it!=inputPara.end())
        {
            flip_contrast=atoi(it->second.c_str());
        }
        else
        {
            flip_contrast=false;
        }

        it=inputPara.find("defocus_file");
        string defocus_file;
        if(it!=inputPara.end())
        {
            defocus_file=path+"/"+it->second;
        }
        else
        {
            defocus_file="defocus_file.txt";
        }
        FILE *fdefocus=fopen(defocus_file.c_str(),"r");
        if(!fdefocus)
        {
            cerr << "Cannot open defocus file!" << endl;
            abort();
        }

        for(int n=0;n<stack_orig.getNz();n++)
        {
            ctf_para[n].setN(n);
            ctf_para[n].setAllImagePara(pix,volt,Cs);
            float defocus_tmp[7];
            for(int i=0;i<7;i++)    // CTFFIND4 style
            {
                fscanf(fdefocus,"%f",&defocus_tmp[i]);
            }
            if(unrotated_stack)
            {
                ctf_para[n].setAllCTFPara(defocus_tmp[1],defocus_tmp[2],defocus_tmp[3],defocus_tmp[4],w_cos);
            }
            else
            {
                ctf_para[n].setAllCTFPara(defocus_tmp[1],defocus_tmp[2],defocus_tmp[3]-psi_deg,defocus_tmp[4],w_cos);   // 特别注意：目前的CTF估计结果取自CTFFIND4，是用原图（即未经旋转的图）估计的，因此对于重构旋转后的图，像散角（astig）也要旋转对应的角度！！！
            }
        }
        fflush(fdefocus);
        fclose(fdefocus);

        if(!skip_3dctf)
        {
            it=inputPara.find("defocus_step");
            if(it!=inputPara.end())
            {
                defocus_step=atoi(it->second.c_str());
            }
            else
            {
                defocus_step=1;
            }
        }
    }
    // Reconstruction
    //cout << endl << "Reconstruction with (W)BP in RAM:" << endl << endl;

    printf("\nReconstruction with (W)BP in RAM:\n");

    if(!unrotated_stack)    // input rotated stack (y-axis as tilt axis)
    {
        printf("Using rotated stack\n");

        int stack_orig_getNx = stack_orig.getNx();
		int stack_orig_getNy = stack_orig.getNy();
		int stack_orig_getNz = stack_orig.getNz();

        float * stack_recon1 = nullptr;
        const size_t recon_bytes = (size_t)stack_orig_getNx * (size_t)stack_orig_getNy * (size_t)h * sizeof(float);
        bool stack_recon1_pinned = false;
        bool volume_written_in_finalize = false;
#ifdef USE_GPU
        const bool defer_stack_recon1_alloc = (!skip_ctfcorrection && !skip_3dctf);
#else
        const bool defer_stack_recon1_alloc = false;
#endif

        auto alloc_stack_recon1 = [&]() {
            if (stack_recon1) return;
#ifdef USE_GPU
            if (cudaMallocHost((void**)&stack_recon1, recon_bytes) == cudaSuccess) {
                stack_recon1_pinned = true;
            } else {
                stack_recon1 = (float *)aligned_alloc(4096, recon_bytes);
            }
#else
            stack_recon1 = (float *)aligned_alloc(4096, recon_bytes);
#endif
            if (!stack_recon1) {
                cerr << "Failed to allocate stack_recon1" << endl;
                abort();
            }
        };

        if (!defer_stack_recon1_alloc)
        {
            alloc_stack_recon1();
            #pragma omp parallel for schedule(static)
            for(int j=0;j<stack_orig_getNy;j++)
            {
                memset(&stack_recon1[j*stack_orig_getNx*h], 0.0, sizeof(float) * stack_orig_getNx * h);
            }
        }

        printf("Start reconstruction:\n");
        float x_orig_offset = float(stack_orig_getNx) * 0.5;
        float z_orig_offset = float(h) * 0.5;

        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(stack_orig_getNx, h, stack_orig_getNy, 2); // (x,z,y)
        // High-throughput output path: large stdio buffer + file preallocation.
        char* final_stdio_buf = nullptr;
        size_t final_stdio_buf_size = 64ull * 1024ull * 1024ull;
        if (const char* env_stdio_mb = std::getenv("WBP_STDIO_BUF_MB")) {
            long long mb = std::atoll(env_stdio_mb);
            if (mb > 0) final_stdio_buf_size = (size_t)mb * 1024ull * 1024ull;
        }
        if (final_stdio_buf_size >= 4096ull) {
            size_t alloc_bytes = ((final_stdio_buf_size + 4095ull) / 4096ull) * 4096ull;
            final_stdio_buf = (char*)aligned_alloc(4096, alloc_bytes);
            if (final_stdio_buf) {
                final_stdio_buf_size = alloc_bytes;
                setvbuf(stack_final.m_fp, final_stdio_buf, _IOFBF, final_stdio_buf_size);
            } else {
                final_stdio_buf_size = 0;
                setvbuf(stack_final.m_fp, NULL, _IOFBF, 0);
            }
        } else {
            final_stdio_buf_size = 0;
            setvbuf(stack_final.m_fp, NULL, _IOFBF, 0);
        }

        bool output_prealloc_enable = true;
        if (const char* env_prealloc = std::getenv("WBP_OUTPUT_PREALLOC")) {
            output_prealloc_enable = (std::atoi(env_prealloc) != 0);
        }
        if (output_prealloc_enable) {
            const size_t im_size = (size_t)stack_final.getImSize();
            const size_t total_data_bytes = im_size * (size_t)stack_orig_getNy;
            const size_t full_file_bytes = (size_t)1024 + (size_t)stack_final.getSymdatasize() + total_data_bytes;
            int fd = fileno(stack_final.m_fp);
            if (fd >= 0) {
                int rc = posix_fallocate(fd, (off_t)0, (off_t)full_file_bytes);
                if (rc != 0) {
                    fprintf(stderr, "[PROFILE][HOST][IO] prealloc failed rc=%d (%s)\n", rc, strerror(rc));
                }
            }
            const long data_offset = (long)((size_t)1024 + (size_t)stack_final.getSymdatasize());
            fseek(stack_final.m_fp, data_offset, SEEK_SET);
        }
#ifdef USE_GPU
        bool have_gpu_volume_stats = false;
        float gpu_min_all = 0.0f;
        float gpu_max_all = 0.0f;
        double gpu_mean_all = 0.0;
#endif

        if(skip_ctfcorrection)  // no correction, simple (W)BP
        {
			float *stack_recon[stack_orig_getNy]; // (x,z,y)

        	#pragma omp parallel for schedule(static)
        	for(int j=0;j<stack_orig_getNy;j++)
        	{
            	stack_recon[j] = (float *)aligned_alloc(4096, stack_orig_getNx * h * sizeof(float));
            	memset(stack_recon[j], 0.0, sizeof(float) * stack_orig_getNx * h);
        	}

			float * image_now        = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * sizeof(float));
            float * image_now_backup = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * sizeof(float));
            for(int n=0;n<stack_orig.getNz();n++)   // loop for every micrograph
            {
                float theta_rad=theta[n]/180*M_PI;

                stack_orig.read2DIm_32bit(image_now,n);
                memcpy(image_now_backup,image_now,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());

                if(!skip_weighting)
                {
                    filter_weighting_1D_many(image_now,stack_orig.getNx(),stack_orig.getNy(),weighting_radial,weighting_sigma);
                }

                // reconstruction
                float *strip_now;
                float *recon_now;
                
                strip_now=new float[stack_orig.getNx()];
                recon_now=new float[stack_orig.getNx()*h];  // 第一维x，第二维z
            
                // loop: Ny (number of xz-slices)
                for(int j=0;j<stack_orig.getNy();j++)
                {
                    memcpy(strip_now,image_now+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());

                    // BP
                    // loop: Nx*h (whole xz-slice)
                    for(int i=0;i<stack_orig.getNx()*h;i++)
                    {
                        recon_now[i]=0.0;
                    }
                    // loop: Nx*h (whole xz-slice)
                    for(int k=0;k<h;k++)
                    {
                        for(int i=0;i<stack_orig.getNx();i++)   // loop for the xz-plane to perform BP
                        {
                            float x_orig=(float(i)-x_orig_offset)*cos(theta_rad)-(float(k)-z_orig_offset)*sin(theta_rad)+x_orig_offset;
                            float z_orig=(float(i)-x_orig_offset)*sin(theta_rad)+(float(k)-z_orig_offset)*cos(theta_rad)+z_orig_offset;
                            float coeff=x_orig-floor(x_orig);
                            if(floor(x_orig)>=0 && ceil(x_orig)<stack_orig.getNx())
                            {
                                recon_now[i+k*stack_orig.getNx()]=(1-coeff)*strip_now[int(floor(x_orig))]+(coeff)*strip_now[int(ceil(x_orig))];
                            }
                            else
                            {
                                recon_now[i+k*stack_orig.getNx()]=0.0;
                            }
                        }
                    }
                    // loop: Nx*h (whole xz-slice)
                    for(int i=0;i<stack_orig.getNx()*h;i++)
                    {
                        stack_recon[j][i]+=recon_now[i];
                    }
                }
            
                delete [] strip_now;
                delete [] recon_now;
            }
			free(image_now);
        	free(image_now_backup);
        }
        else    // perform CTF correction
        {
            if(skip_3dctf)  // perform simple CTF correction (no consideration of height)
            {
				float *stack_recon[stack_orig_getNy]; // (x,z,y)

        		#pragma omp parallel for schedule(static)
        		for(int j=0;j<stack_orig_getNy;j++)
        		{
            		stack_recon[j] = (float *)aligned_alloc(4096, stack_orig_getNx * h * sizeof(float));
        		    memset(stack_recon[j], 0.0, sizeof(float) * stack_orig_getNx * h);
		        }

				float * image_now        = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * sizeof(float));
                float * image_now_backup = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * sizeof(float));
                for(int n=0;n<stack_orig.getNz();n++)   // loop for every micrograph
                {
                    float theta_rad=theta[n]/180*M_PI;

                    stack_orig.read2DIm_32bit(image_now,n);

                    memcpy(image_now_backup,image_now,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());

                    // correction
                    ctf_correction(image_now,stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],flip_contrast,0.0);

                    // weighting
                    if(!skip_weighting)
                    {
                        filter_weighting_1D_many(image_now,stack_orig.getNx(),stack_orig.getNy(),weighting_radial,weighting_sigma);
                    }

                    // recontruction
                    float *strip_now;
                    float *recon_now;

                    strip_now=new float[stack_orig.getNx()];
                    recon_now=new float[stack_orig.getNx()*h];  // 第一维x，第二维z
                    
                    #pragma omp parallel for
                    // loop: Ny (number of xz-slices)
                    for(int j=0;j<stack_orig.getNy();j++)
                    {
                        memcpy(strip_now,image_now+j*stack_orig.getNx(),sizeof(float)*stack_orig.getNx());

                        // BP
                        // loop: Nx*h (whole xz-slice)
                        for(int i=0;i<stack_orig.getNx()*h;i++)
                        {
                            recon_now[i]=0.0;
                        }
                        // loop: Nx*h (whole xz-slice)
                        for(int k=0;k<h;k++)
                        {
                            for(int i=0;i<stack_orig.getNx();i++)   // loop for the xz-plane to perform BP
                            {
                                float x_orig=(float(i)-x_orig_offset)*cos(theta_rad)-(float(k)-z_orig_offset)*sin(theta_rad)+x_orig_offset;
                                float z_orig=(float(i)-x_orig_offset)*sin(theta_rad)+(float(k)-z_orig_offset)*cos(theta_rad)+z_orig_offset;
                                float coeff=x_orig-floor(x_orig);
                                if(floor(x_orig)>=0 && ceil(x_orig)<stack_orig.getNx())
                                {
                                    recon_now[i+k*stack_orig.getNx()]=(1-coeff)*strip_now[int(floor(x_orig))]+(coeff)*strip_now[int(ceil(x_orig))];
                                }
                                else
                                {
                                    recon_now[i+k*stack_orig.getNx()]=0.0;
                                }
                            }
                        }
                        // loop: Nx*h (whole xz-slice)
                        for(int i=0;i<stack_orig.getNx()*h;i++)
                        {
                            stack_recon[j][i]+=recon_now[i];
                        }
                    }
                
                    delete [] strip_now;
                    delete [] recon_now;
                }
				free(image_now);
        		free(image_now_backup);
            }
            else    // perform 3D-CTF correction
            {
                // Use pinned host memory to accelerate host/device transfers.
                const size_t image_bytes = (size_t)stack_orig_getNx * (size_t)stack_orig_getNy * sizeof(float);

                float* stack_corrected = nullptr;
#ifdef USE_GPU
                float* image_now[2] = {nullptr, nullptr};
                bool image_now_pinned[2] = {false, false};
                for (int s = 0; s < 2; ++s) {
                    if (cudaMallocHost((void**)&image_now[s], image_bytes) == cudaSuccess) {
                        image_now_pinned[s] = true;
                    } else {
                        image_now[s] = (float*)aligned_alloc(4096, image_bytes);
                    }
                    if (!image_now[s]) {
                        cerr << "Failed to allocate image_now host buffer" << endl;
                        abort();
                    }
                }
#else
                float* image_now = (float*)aligned_alloc(4096, image_bytes);
                bool image_now_pinned = false;
                if (!image_now) {
                    cerr << "Failed to allocate image_now host buffer" << endl;
                    abort();
                }
#endif

                // [GPU] stack_corrected host buffer is NOT needed in device pipeline; skip large cudaMallocHost to reduce overhead
                stack_corrected = NULL;
				int ttt          = stack_orig_getNx * stack_orig_getNy;
                const float corrected_scale = 1.0f / float(ttt);
                
                const size_t ttt_bytes = (size_t)ttt * sizeof(float);
#ifndef USE_GPU
                int thread_nums = omp_get_max_threads();
                int Nx_padding11 = int(stack_orig_getNx / 10);
                int Nx_final11   = stack_orig_getNx + Nx_padding11;
                float * bufc12   = (float *)aligned_alloc(4096, sizeof(float) * (Nx_final11 + 2 - (Nx_final11 & 1)) * stack_orig_getNy);
				int chunk        = (stack_orig_getNx + 2 - (stack_orig_getNx & 1)) * stack_orig_getNy;
				float * bufc     = (float *)aligned_alloc(4096, sizeof(float) * (stack_orig_getNx + 2 - (stack_orig_getNx & 1)) * stack_orig_getNy * thread_nums);
	
                fftwf_init_threads();
#endif

                size_t ImSize = (size_t)stack_final.getImSize();
                // stdio buffering is configured at file creation stage.

			// ------------------------------
			// GPU pipeline resources (allocated once)
			// ------------------------------
			#ifdef USE_GPU
            struct GpuProfileStats {
                double read_ms = 0.0;
                double h2d_ms = 0.0;
                double weighting_ms = 0.0;
                double ctf3d_ms = 0.0;
                double backproj_ms = 0.0;
                double loop_wall_ms = 0.0;
                double finalize_d2h_ms = 0.0;
                double recon_stats_ms = 0.0;
                double zrange_host_ms = 0.0;
                double h2d_wall_ms = 0.0;
                double weighting_wall_ms = 0.0;
                double ctf3d_wall_ms = 0.0;
                double backproj_wall_ms = 0.0;
                double loop_known_ms = 0.0;
                double loop_other_ms = 0.0;
                double finalize_alloc_ms = 0.0;
                double finalize_chunk_loop_ms = 0.0;
                double finalize_chunk_d2h_ms = 0.0;
                double finalize_chunk_write_ms = 0.0;
                double finalize_stage_free_ms = 0.0;
                double finalize_total_ms = 0.0;
                double finalize_cleanup_recon_free_ms = 0.0;
                double finalize_cleanup_dev_free_ms = 0.0;
                double finalize_cleanup_zoffset_free_ms = 0.0;
                double finalize_cleanup_profile_shutdown_ms = 0.0;
                double finalize_cleanup_event_destroy_ms = 0.0;
                double finalize_chunk_d2h_min_ms = 1e100;
                double finalize_chunk_d2h_max_ms = 0.0;
                double finalize_chunk_write_min_ms = 1e100;
                double finalize_chunk_write_max_ms = 0.0;
                int images = 0;
                long long ctf_slices = 0;
                long long finalize_chunks = 0;
                long long finalize_chunk_bytes = 0;
                long long finalize_total_bytes = 0;
            } prof;

            // Stage-level GPU profiling inserts stream sync and hurts overlap.
            // Default: disabled for performance. Enable with WBP_GPU_STAGE_PROFILE=1.
            bool gpu_stage_profile = false;
            if (const char* env_stage_prof = std::getenv("WBP_GPU_STAGE_PROFILE")) {
                gpu_stage_profile = (std::atoi(env_stage_prof) != 0);
            }
            bool host_fine_profile = false;
            if (const char* env_host_fine = std::getenv("WBP_HOST_FINE_PROFILE")) {
                host_fine_profile = (std::atoi(env_host_fine) != 0);
            }

            cudaEvent_t evt_begin, evt_end;
            CUDA_CHECK(cudaEventCreate(&evt_begin));
            CUDA_CHECK(cudaEventCreate(&evt_end));
#define GPU_PROFILE_TIME_ADD(acc, code_block)            \
            do {                                         \
                if (gpu_stage_profile) {                 \
                    CUDA_CHECK(cudaEventRecord(evt_begin, 0)); \
                    do { code_block } while (0);         \
                    CUDA_CHECK(cudaEventRecord(evt_end, 0)); \
                    CUDA_CHECK(cudaEventSynchronize(evt_end)); \
                    float _gpu_prof_ms = 0.0f;           \
                    CUDA_CHECK(cudaEventElapsedTime(&_gpu_prof_ms, evt_begin, evt_end)); \
                    (acc) += (double)_gpu_prof_ms;       \
                } else {                                 \
                    do { code_block } while (0);         \
                }                                        \
            } while (0)

#define GPU_PROFILE_TIME_ADD_ON(stream_, acc, code_block) \
            do {                                           \
                if (gpu_stage_profile) {                   \
                    CUDA_CHECK(cudaEventRecord(evt_begin, (stream_))); \
                    do { code_block } while (0);           \
                    CUDA_CHECK(cudaEventRecord(evt_end, (stream_))); \
                    CUDA_CHECK(cudaEventSynchronize(evt_end)); \
                    float _gpu_prof_ms = 0.0f;             \
                    CUDA_CHECK(cudaEventElapsedTime(&_gpu_prof_ms, evt_begin, evt_end)); \
                    (acc) += (double)_gpu_prof_ms;         \
                } else {                                   \
                    do { code_block } while (0);           \
                }                                          \
            } while (0)

            float* d_image_now[2] = {nullptr, nullptr};
			float* d_stack_corrected_dev[2] = {nullptr, nullptr};
			float* z_offsets_host = nullptr;
            cudaStream_t stream_preprocess = nullptr;
            cudaStream_t stream_bp = nullptr;
            cudaEvent_t evt_ctf_done[2] = {nullptr, nullptr};
            cudaEvent_t evt_slot_free[2] = {nullptr, nullptr};
            cudaEvent_t evt_h2d_done[2] = {nullptr, nullptr};
			const int max_slices = (int)(h_tilt_max / defocus_step) + 1;
            const int tmp_h = int(h_tilt_max >> 1);
            const int cnt11 = (2 * tmp_h) / defocus_step;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_preprocess, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_bp, cudaStreamNonBlocking));
            for (int s = 0; s < 2; ++s) {
                CUDA_CHECK(cudaMalloc((void**)&d_image_now[s], ttt_bytes));
			    CUDA_CHECK(cudaMalloc((void**)&d_stack_corrected_dev[s], (size_t)max_slices * (size_t)ttt * sizeof(float)));
                CUDA_CHECK(cudaEventCreateWithFlags(&evt_ctf_done[s], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&evt_slot_free[s], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&evt_h2d_done[s], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(evt_slot_free[s], stream_bp));
                CUDA_CHECK(cudaEventRecord(evt_h2d_done[s], stream_preprocess));
            }
			z_offsets_host = new float[max_slices];
            for (int zz = 0; zz < cnt11; zz++) {
                z_offsets_host[zz] = float(zz * defocus_step - tmp_h) + float(defocus_step - 1) / 2.0f;
            }
            bool dynamic_ctf_slices = true;
            if (const char* env_dynamic = std::getenv("WBP_CTF3D_DYNAMIC_SLICES")) {
                dynamic_ctf_slices = (std::atoi(env_dynamic) != 0);
            }
            int ctf_slice_bucket = 4; // default bucket=4; 1 means no bucket/alignment.
            if (const char* env_bucket = std::getenv("WBP_CTF3D_SLICE_BUCKET")) {
                int v = std::atoi(env_bucket);
                if (v > 1) ctf_slice_bucket = v;
            }
            std::vector<float> tilt_sin((size_t)stack_orig_getNz, 0.0f);
            std::vector<float> tilt_cos((size_t)stack_orig_getNz, 1.0f);
            std::vector<int> tilt_z_slice_begin((size_t)stack_orig_getNz, 0);
            std::vector<int> tilt_num_slices((size_t)stack_orig_getNz, cnt11);
            std::vector<int> tilt_tmp_h_this_tilt((size_t)stack_orig_getNz, tmp_h);
            std::vector<float> tilt_tmp_h_float((size_t)stack_orig_getNz, (float)tmp_h);

            for (int n = 0; n < stack_orig_getNz; ++n) {
                float theta_rad = theta[n] / 180.0f * (float)M_PI;
                float sin1 = sinf(theta_rad);
                float cos1 = cosf(theta_rad);
                tilt_sin[(size_t)n] = sin1;
                tilt_cos[(size_t)n] = cos1;

                // Dynamic z-slice range per tilt to reduce unnecessary 3D-CTF work.
                const float xi_min = -x_orig_offset;
                const float xi_max = (float)stack_orig_getNx - 1.0f - x_orig_offset;
                const float zk_min = -z_orig_offset;
                const float zk_max = (float)h - 1.0f - z_orig_offset;

                const float x_term_min = (sin1 >= 0.0f) ? (xi_min * sin1) : (xi_max * sin1);
                const float x_term_max = (sin1 >= 0.0f) ? (xi_max * sin1) : (xi_min * sin1);
                const float z_term_min = (cos1 >= 0.0f) ? (zk_min * cos1) : (zk_max * cos1);
                const float z_term_max = (cos1 >= 0.0f) ? (zk_max * cos1) : (zk_min * cos1);

                const float z_orig_min = x_term_min + z_term_min;
                const float z_orig_max = x_term_max + z_term_max;

                int z_slice_begin = (int)floor((z_orig_min + (float)tmp_h) / (float)defocus_step);
                int z_slice_end = (int)floor((z_orig_max + (float)tmp_h) / (float)defocus_step);

                if (!dynamic_ctf_slices) {
                    z_slice_begin = 0;
                    z_slice_end = cnt11 - 1;
                } else {
                    int z_begin_raw = z_slice_begin;
                    int z_end_raw = z_slice_end;

                    if (z_begin_raw < 0) z_begin_raw = 0;
                    if (z_end_raw >= cnt11) z_end_raw = cnt11 - 1;
                    if (z_end_raw < z_begin_raw) {
                        z_begin_raw = 0;
                        z_end_raw = 0;
                    }

                    if (ctf_slice_bucket > 1) {
                        int n_raw = z_end_raw - z_begin_raw + 1;
                        int n_aligned = ((n_raw + ctf_slice_bucket - 1) / ctf_slice_bucket) * ctf_slice_bucket;
                        if (n_aligned > cnt11) n_aligned = cnt11;

                        int z_begin_aligned = z_begin_raw;
                        if (z_begin_aligned + n_aligned > cnt11) {
                            z_begin_aligned = cnt11 - n_aligned;
                        }
                        if (z_begin_aligned < 0) z_begin_aligned = 0;

                        z_slice_begin = z_begin_aligned;
                        z_slice_end = z_begin_aligned + n_aligned - 1;
                    } else {
                        z_slice_begin = z_begin_raw;
                        z_slice_end = z_end_raw;
                    }
                }

                int num_slices_this_tilt = z_slice_end - z_slice_begin + 1;
                int tmp_h_this_tilt = tmp_h - z_slice_begin * defocus_step;
                tilt_z_slice_begin[(size_t)n] = z_slice_begin;
                tilt_num_slices[(size_t)n] = num_slices_this_tilt;
                tilt_tmp_h_this_tilt[(size_t)n] = tmp_h_this_tilt;
                tilt_tmp_h_float[(size_t)n] = (float)tmp_h_this_tilt;
            }
            if (host_fine_profile) {
                // keep API compatibility for profile fields; per-tilt z-range is precomputed once now
                prof.zrange_host_ms += 0.0;
            }

			wbp_recon_init(stack_orig_getNx, stack_orig_getNy, h);
            bool precompute_xz_map_enable = true;
            if (const char* env_pre_map = std::getenv("WBP_PRECOMPUTE_XZ_MAP")) {
                precompute_xz_map_enable = (std::atoi(env_pre_map) != 0);
            }
            bool precomputed_xz_map = false;
            if (precompute_xz_map_enable) {
                int ok = wbp_backproject_prepare_xz_maps(
                    stack_orig_getNx,
                    h,
                    stack_orig_getNz,
                    defocus_step,
                    x_orig_offset,
                    z_orig_offset,
                    tilt_tmp_h_float.data(),
                    tilt_sin.data(),
                    tilt_cos.data(),
                    (void*)stream_bp
                );
                precomputed_xz_map = (ok != 0);
                if (!precomputed_xz_map) {
                    fprintf(stderr, "[GPU][WBP] precompute xz-map failed, fallback to per-tilt build.\n");
                }
            }
			#endif
            #ifndef USE_GPU
            const int tmp_h = int(h_tilt_max >> 1);
            const int cnt11 = (2 * tmp_h) / defocus_step;
            const int cnt22 = cnt11 - (cnt11 & 63);
            const bool host_fine_profile = false;
            #endif

                std::chrono::steady_clock::time_point t_pipeline_begin;
                if (host_fine_profile) t_pipeline_begin = std::chrono::steady_clock::now();
                int progress_interval = 10;
                if (const char* env_prog = std::getenv("WBP_PROGRESS_INTERVAL")) {
                    int v = std::atoi(env_prog);
                    if (v >= 0) progress_interval = v;
                }
                for(int n=0;n<stack_orig_getNz;n++)   // loop for every micrograph
                {
#ifdef USE_GPU
                    const int slot = (n & 1);
                    if (n >= 2) {
                        CUDA_CHECK(cudaEventSynchronize(evt_h2d_done[slot]));
                    }
                    if (n >= 2) {
                        CUDA_CHECK(cudaStreamWaitEvent(stream_preprocess, evt_slot_free[slot], 0));
                    }
#endif
                    if (progress_interval > 0 &&
                        ((n % progress_interval) == 0 || n == stack_orig_getNz - 1)) {
                        printf("Image %d/%d\n", n + 1, stack_orig_getNz);
                    }
                    float sin1 = 0.0f;
                    float cos1 = 1.0f;
                    int z_slice_begin = 0;
                    int num_slices_this_tilt = 1;
                    int tmp_h_this_tilt = tmp_h;
#ifdef USE_GPU
                    sin1 = tilt_sin[(size_t)n];
                    cos1 = tilt_cos[(size_t)n];
                    z_slice_begin = tilt_z_slice_begin[(size_t)n];
                    num_slices_this_tilt = tilt_num_slices[(size_t)n];
                    tmp_h_this_tilt = tilt_tmp_h_this_tilt[(size_t)n];
                    float* z_offsets_this_tilt = z_offsets_host + z_slice_begin;
#else
                    float theta_rad = theta[n] / 180 * M_PI;
                    sin1 = sin(theta_rad);
                    cos1 = cos(theta_rad);
#endif

                    std::chrono::steady_clock::time_point t_read_begin;
                    if (host_fine_profile) t_read_begin = std::chrono::steady_clock::now();
#ifdef USE_GPU
                    stack_orig.read2DIm_32bit(image_now[slot],n);
#else
                    stack_orig.read2DIm_32bit(image_now,n);
#endif
                    if (host_fine_profile) {
                        auto t_read_end = std::chrono::steady_clock::now();
                        prof.read_ms += std::chrono::duration<double, std::milli>(t_read_end - t_read_begin).count();
                    }

                    int n_zz=0;

#ifndef USE_GPU
                    if(!skip_weighting)
                    {
                          fftwf_plan_with_nthreads(omp_get_max_threads());
                          filter_weighting_1D_many1(image_now, bufc12, stack_orig_getNx, stack_orig_getNy, weighting_radial, weighting_sigma);
                    }
#endif

				#ifdef USE_GPU
				    // ========== GPU device pipeline ==========
				    // 1) copy micrograph to device
				    // 2) optional 1D weighting on device
				    // 2) 3D-CTF correction: device->device (no big DtoH)
				    // 3) WBP backprojection: device corrected stack -> device recon volume
                    std::chrono::steady_clock::time_point t_h2d_begin;
                    if (host_fine_profile) t_h2d_begin = std::chrono::steady_clock::now();
                    GPU_PROFILE_TIME_ADD_ON(stream_preprocess, prof.h2d_ms,
				        CUDA_CHECK(cudaMemcpyAsync(d_image_now[slot], image_now[slot], ttt_bytes, cudaMemcpyHostToDevice, stream_preprocess));
                    );
                    CUDA_CHECK(cudaEventRecord(evt_h2d_done[slot], stream_preprocess));
                    if (host_fine_profile) {
                        auto t_h2d_end = std::chrono::steady_clock::now();
                        prof.h2d_wall_ms += std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_begin).count();
                    }
                    if (!skip_weighting) {
                        std::chrono::steady_clock::time_point t_weighting_begin;
                        if (host_fine_profile) t_weighting_begin = std::chrono::steady_clock::now();
                        GPU_PROFILE_TIME_ADD_ON(stream_preprocess, prof.weighting_ms,
                            filter_weighting_1d_many_gpu_inplace_stream(
                                d_image_now[slot],
                                stack_orig_getNx,
                                stack_orig_getNy,
                                weighting_radial,
                                weighting_sigma,
                                (void*)stream_preprocess
                            );
                        );
                        if (host_fine_profile) {
                            auto t_weighting_end = std::chrono::steady_clock::now();
                            prof.weighting_wall_ms += std::chrono::duration<double, std::milli>(t_weighting_end - t_weighting_begin).count();
                        }
                    }
                    std::chrono::steady_clock::time_point t_ctf3d_begin;
                    if (host_fine_profile) t_ctf3d_begin = std::chrono::steady_clock::now();
                    GPU_PROFILE_TIME_ADD_ON(stream_preprocess, prof.ctf3d_ms,
				        ctf_correction_3d_gpu_device_no_norm_stream(
				            d_image_now[slot],
				            d_stack_corrected_dev[slot],
				            stack_orig_getNx,
				            stack_orig_getNy,
				            num_slices_this_tilt,
				            ctf_para[n],
				            flip_contrast,
				            z_offsets_this_tilt,
                            (void*)stream_preprocess
				        );
                    );
                    if (host_fine_profile) {
                        auto t_ctf3d_end = std::chrono::steady_clock::now();
                        prof.ctf3d_wall_ms += std::chrono::duration<double, std::milli>(t_ctf3d_end - t_ctf3d_begin).count();
                    }
                    n_zz = num_slices_this_tilt;
                    prof.ctf_slices += (long long)n_zz;
                    CUDA_CHECK(cudaEventRecord(evt_ctf_done[slot], stream_preprocess));
				#else
                    // ========== 原始CPU版本 ==========
                    #pragma omp parallel
                    {
                        #pragma omp for schedule(static) reduction(+:n_zz) nowait
                        for(int zz=0; zz<cnt22; zz++)
                        {
                            ctf_correction1(image_now, &stack_corrected[zz * stack_orig_getNx * stack_orig_getNy], 
                                        &bufc[chunk * omp_get_thread_num()],
                                        stack_orig_getNx, stack_orig_getNy, ctf_para[n], 
                                        flip_contrast, float(zz * defocus_step - tmp_h) + float(defocus_step-1)/2);
                            n_zz++;
                        }
                    }
				#endif

				    // NOTE:
				    // - Old path used: 3D GPU correction (cnt22 slices) + per-slice GPU/CPU tail.
				    // - That created many device->host copies.
				    // - Now we correct all cnt11 slices in one device->device pass.

					#ifdef USE_GPU
					    // GPU WBP backprojection accumulates directly into the device recon volume.
                        CUDA_CHECK(cudaStreamWaitEvent(stream_bp, evt_ctf_done[slot], 0));
                        std::chrono::steady_clock::time_point t_backproj_begin;
                        if (host_fine_profile) t_backproj_begin = std::chrono::steady_clock::now();
                        if (precomputed_xz_map) {
                            GPU_PROFILE_TIME_ADD_ON(stream_bp, prof.backproj_ms,
                                wbp_backproject_accumulate_gpu_device_precomputed_map_stream(
                                    d_stack_corrected_dev[slot],
                                    stack_orig_getNx,
                                    stack_orig_getNy,
                                    h,
                                    n_zz,
                                    n,
                                    corrected_scale,
                                    (void*)stream_bp
                                );
                            );
                        } else {
                            GPU_PROFILE_TIME_ADD_ON(stream_bp, prof.backproj_ms,
                                wbp_backproject_accumulate_gpu_device_stream(
                                    d_stack_corrected_dev[slot],
                                    stack_orig_getNx,
                                    stack_orig_getNy,
                                    h,
                                    n_zz,
                                    defocus_step,
                                    tmp_h_this_tilt,
                                    x_orig_offset,
                                    z_orig_offset,
                                    sin1,
                                    cos1,
                                    corrected_scale,
                                    (void*)stream_bp
                                );
                            );
                        }
                        if (host_fine_profile) {
                            auto t_backproj_end = std::chrono::steady_clock::now();
                            prof.backproj_wall_ms += std::chrono::duration<double, std::milli>(t_backproj_end - t_backproj_begin).count();
                        }
                        CUDA_CHECK(cudaEventRecord(evt_slot_free[slot], stream_bp));
					#else
					    int cnt = ttt * h;
					    int x_h = stack_orig_getNx * h;
					
					    #pragma omp parallel
					    {
							#pragma unroll(16)
    #pragma omp parallel for simd schedule(static)
					        #pragma omp for schedule(static) nowait
					        for(int z11=0; z11<cnt; z11++)
					        {
					            register int j = z11 / x_h;
					            register int k = (z11 - j * x_h) / stack_orig_getNx;
					            register int i = z11 - z11 / stack_orig_getNx * stack_orig_getNx;
				
					            register float tmp_tmp1 = (float(i)-x_orig_offset);
					            register float tmp_tmp2 = (float(k)-z_orig_offset);
					            register float x_orig   = tmp_tmp1 * cos1 - tmp_tmp2 * sin1 + x_orig_offset;
					            register float z_orig   = tmp_tmp1 * sin1 + tmp_tmp2 * cos1;
					            register int t1t        = int(floor(x_orig));
					            register float coeff    = x_orig - t1t;
					            register int n_z        = floor((z_orig + tmp_h) / defocus_step);
							register int idx1       = i + k * stack_orig_getNx;
					            register int idx2       = j * stack_orig_getNx + t1t;
				
					            if(((n_z>=0 && n_z<n_zz) && t1t >=0 && ceil(x_orig)<stack_orig_getNx))
							    stack_recon1[z11] += ((1-coeff) * stack_corrected[n_z*ttt+idx2] + (coeff) * stack_corrected[n_z*ttt + idx2 + 1]);
					        }
					    }
					#endif
                    prof.images++;
                }
#ifdef USE_GPU
                CUDA_CHECK(cudaStreamSynchronize(stream_bp));
                CUDA_CHECK(cudaStreamSynchronize(stream_preprocess));
#endif
                if (host_fine_profile) {
                    auto t_pipeline_end = std::chrono::steady_clock::now();
                    prof.loop_wall_ms += std::chrono::duration<double, std::milli>(t_pipeline_end - t_pipeline_begin).count();
                }
					#ifdef USE_GPU
                    auto t_finalize_total_begin = std::chrono::steady_clock::now();
                    GPU_PROFILE_TIME_ADD(prof.recon_stats_ms,
                        wbp_recon_get_stats(&gpu_min_all, &gpu_max_all, &gpu_mean_all);
                    );
                    have_gpu_volume_stats = true;

                    bool chunked_finalize_write = true;
                    if (const char* env_chunked = std::getenv("WBP_FINALIZE_CHUNKED_WRITE")) {
                        chunked_finalize_write = (std::atoi(env_chunked) != 0);
                    }
                    const size_t total_bytes = ImSize * (size_t)stack_orig_getNy;
                    const size_t total_elems = total_bytes / sizeof(float);
                    prof.finalize_total_bytes = (long long)total_bytes;
                    size_t chunk_bytes = 64ull * 1024ull * 1024ull;
                    if (const char* env_chunk_mb = std::getenv("WBP_FINALIZE_CHUNK_MB")) {
                        long long mb = std::atoll(env_chunk_mb);
                        if (mb > 0) chunk_bytes = (size_t)mb * 1024ull * 1024ull;
                    }
                    if (chunk_bytes > total_bytes) chunk_bytes = total_bytes;
                    if (chunk_bytes < sizeof(float)) chunk_bytes = sizeof(float);
                    chunk_bytes -= (chunk_bytes % sizeof(float));
                    prof.finalize_chunk_bytes = (long long)chunk_bytes;

                    auto t_finalize_alloc_begin = std::chrono::steady_clock::now();
                    float* finalize_stage = nullptr;
                    bool finalize_stage_pinned = false;
                    if (chunked_finalize_write) {
                        if (cudaMallocHost((void**)&finalize_stage, chunk_bytes) == cudaSuccess) {
                            finalize_stage_pinned = true;
                        } else {
                            size_t alloc_bytes = ((chunk_bytes + 4095ull) / 4096ull) * 4096ull;
                            finalize_stage = (float*)aligned_alloc(4096, alloc_bytes);
                        }
                    }
                    auto t_finalize_alloc_end = std::chrono::steady_clock::now();
                    prof.finalize_alloc_ms += std::chrono::duration<double, std::milli>(
                        t_finalize_alloc_end - t_finalize_alloc_begin).count();

                    if (finalize_stage) {
                        size_t copied = 0;
                        const size_t chunk_elems = chunk_bytes / sizeof(float);
                        auto t_finalize_chunk_loop_begin = std::chrono::steady_clock::now();
                        while (copied < total_elems) {
                            size_t todo = total_elems - copied;
                            if (todo > chunk_elems) todo = chunk_elems;
                            auto t_chunk_d2h_begin = std::chrono::steady_clock::now();
                            GPU_PROFILE_TIME_ADD(prof.finalize_d2h_ms,
					            wbp_recon_copy_range(finalize_stage, copied, todo);
                            );
                            auto t_chunk_d2h_end = std::chrono::steady_clock::now();
                            const double this_chunk_d2h_ms = std::chrono::duration<double, std::milli>(
                                t_chunk_d2h_end - t_chunk_d2h_begin).count();
                            prof.finalize_chunk_d2h_ms += this_chunk_d2h_ms;
                            if (this_chunk_d2h_ms < prof.finalize_chunk_d2h_min_ms) prof.finalize_chunk_d2h_min_ms = this_chunk_d2h_ms;
                            if (this_chunk_d2h_ms > prof.finalize_chunk_d2h_max_ms) prof.finalize_chunk_d2h_max_ms = this_chunk_d2h_ms;
                            auto t_chunk_write_begin = std::chrono::steady_clock::now();
                            if (fwrite(finalize_stage, sizeof(float), todo, stack_final.m_fp) != todo) {
                                cerr << "Failed to write reconstruction chunk to output file" << endl;
                                abort();
                            }
                            auto t_chunk_write_end = std::chrono::steady_clock::now();
                            const double this_chunk_write_ms = std::chrono::duration<double, std::milli>(
                                t_chunk_write_end - t_chunk_write_begin).count();
                            prof.finalize_chunk_write_ms += this_chunk_write_ms;
                            if (this_chunk_write_ms < prof.finalize_chunk_write_min_ms) prof.finalize_chunk_write_min_ms = this_chunk_write_ms;
                            if (this_chunk_write_ms > prof.finalize_chunk_write_max_ms) prof.finalize_chunk_write_max_ms = this_chunk_write_ms;
                            prof.finalize_chunks += 1;
                            copied += todo;
                        }
                        auto t_finalize_chunk_loop_end = std::chrono::steady_clock::now();
                        prof.finalize_chunk_loop_ms += std::chrono::duration<double, std::milli>(
                            t_finalize_chunk_loop_end - t_finalize_chunk_loop_begin).count();
                        volume_written_in_finalize = true;
                    } else {
                        alloc_stack_recon1();
                        auto t_full_d2h_begin = std::chrono::steady_clock::now();
                        GPU_PROFILE_TIME_ADD(prof.finalize_d2h_ms,
					        wbp_recon_finalize(stack_recon1);
                        );
                        auto t_full_d2h_end = std::chrono::steady_clock::now();
                        prof.finalize_chunk_d2h_ms += std::chrono::duration<double, std::milli>(
                            t_full_d2h_end - t_full_d2h_begin).count();
                    }
                    auto t_finalize_free_begin = std::chrono::steady_clock::now();
                    if (finalize_stage_pinned) cudaFreeHost(finalize_stage);
                    else free(finalize_stage);
                    auto t_finalize_free_end = std::chrono::steady_clock::now();
                    prof.finalize_stage_free_ms += std::chrono::duration<double, std::milli>(
                        t_finalize_free_end - t_finalize_free_begin).count();

                    auto t_cleanup_recon_free_begin = std::chrono::steady_clock::now();
					wbp_recon_free();
                    auto t_cleanup_recon_free_end = std::chrono::steady_clock::now();
                    prof.finalize_cleanup_recon_free_ms += std::chrono::duration<double, std::milli>(
                        t_cleanup_recon_free_end - t_cleanup_recon_free_begin).count();

                    auto t_cleanup_dev_free_begin = std::chrono::steady_clock::now();
                    for (int s = 0; s < 2; ++s) {
                        CUDA_CHECK(cudaFree(d_image_now[s]));
                        CUDA_CHECK(cudaFree(d_stack_corrected_dev[s]));
                    }
                    auto t_cleanup_dev_free_end = std::chrono::steady_clock::now();
                    prof.finalize_cleanup_dev_free_ms += std::chrono::duration<double, std::milli>(
                        t_cleanup_dev_free_end - t_cleanup_dev_free_begin).count();

                    auto t_cleanup_zoffset_begin = std::chrono::steady_clock::now();
					delete [] z_offsets_host;
                    auto t_cleanup_zoffset_end = std::chrono::steady_clock::now();
                    prof.finalize_cleanup_zoffset_free_ms += std::chrono::duration<double, std::milli>(
                        t_cleanup_zoffset_end - t_cleanup_zoffset_begin).count();

                    auto t_cleanup_prof_shutdown_begin = std::chrono::steady_clock::now();
                    ctf_correction_3d_gpu_device_profile_shutdown();
                    auto t_cleanup_prof_shutdown_end = std::chrono::steady_clock::now();
                    prof.finalize_cleanup_profile_shutdown_ms += std::chrono::duration<double, std::milli>(
                        t_cleanup_prof_shutdown_end - t_cleanup_prof_shutdown_begin).count();

                    auto t_cleanup_event_destroy_begin = std::chrono::steady_clock::now();
                    for (int s = 0; s < 2; ++s) {
                        CUDA_CHECK(cudaEventDestroy(evt_ctf_done[s]));
                        CUDA_CHECK(cudaEventDestroy(evt_slot_free[s]));
                        CUDA_CHECK(cudaEventDestroy(evt_h2d_done[s]));
                    }
                    CUDA_CHECK(cudaStreamDestroy(stream_preprocess));
                    CUDA_CHECK(cudaStreamDestroy(stream_bp));
                    CUDA_CHECK(cudaEventDestroy(evt_begin));
                    CUDA_CHECK(cudaEventDestroy(evt_end));
                    auto t_cleanup_event_destroy_end = std::chrono::steady_clock::now();
                    prof.finalize_cleanup_event_destroy_ms += std::chrono::duration<double, std::milli>(
                        t_cleanup_event_destroy_end - t_cleanup_event_destroy_begin).count();

                    auto t_finalize_total_end = std::chrono::steady_clock::now();
                    prof.finalize_total_ms += std::chrono::duration<double, std::milli>(
                        t_finalize_total_end - t_finalize_total_begin).count();

                    if (prof.images > 0) {
                        prof.loop_known_ms = prof.read_ms + prof.zrange_host_ms + prof.h2d_wall_ms +
                                             prof.weighting_wall_ms + prof.ctf3d_wall_ms + prof.backproj_wall_ms;
                        prof.loop_other_ms = prof.loop_wall_ms - prof.loop_known_ms;
                        if (prof.loop_other_ms < 0.0) prof.loop_other_ms = 0.0;
                    }
#undef GPU_PROFILE_TIME_ADD
#undef GPU_PROFILE_TIME_ADD_ON
					#endif

                    if (!volume_written_in_finalize) {
				        fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
                    }
                //fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
#ifndef USE_GPU
				free(bufc);
				free(bufc12);
#endif

				// Free pinned/non-pinned host buffers.
#ifdef USE_GPU
                for (int s = 0; s < 2; ++s) {
				    if (image_now_pinned[s]) cudaFreeHost(image_now[s]);
				    else free(image_now[s]);
                }
#else
                free(image_now);
#endif
                // [GPU] stack_corrected was not allocated (device pipeline)
            }
        }

        // update MRC header
        float min_all = 0.0f;
        float max_all = 0.0f;
        double mean_all = 0.0;
        bool used_gpu_stats = false;
#ifdef USE_GPU
        if (have_gpu_volume_stats) {
            min_all = gpu_min_all;
            max_all = gpu_max_all;
            mean_all = gpu_mean_all;
            used_gpu_stats = true;
        }
#endif
        if (!used_gpu_stats) {
            if (!stack_recon1) {
                cerr << "Host reconstruction volume is not available for header stats." << endl;
                abort();
            }
            min_all = stack_recon1[0];
            max_all = stack_recon1[0];
            const size_t voxel_num = (size_t)stack_orig_getNx * (size_t)h * (size_t)stack_orig_getNy;
            double sum_all = 0.0;

            #pragma omp parallel for schedule(static) reduction(+:sum_all) reduction(min:min_all) reduction(max:max_all)
            for(long long idx = 0; idx < (long long)voxel_num; ++idx)
            {
                float v = stack_recon1[idx];
                sum_all += (double)v;
                if(v < min_all) min_all = v;
                if(v > max_all) max_all = v;
            }
            mean_all = sum_all / (double)voxel_num;
        }
        stack_final.computeHeader(pix,false,min_all,max_all,float(mean_all));
        stack_final.close();
        if (final_stdio_buf) {
            free(final_stdio_buf);
            final_stdio_buf = nullptr;
        }

        printf("Done\n");
#ifdef USE_GPU
        if (stack_recon1_pinned) cudaFreeHost(stack_recon1);
        else free(stack_recon1);
#else
        free(stack_recon1);
#endif
    }
    stack_orig.close();
	printf("\nFinish reconstruction successfully!\n");
	printf("All results save in: ");
	cout<<path;
	printf("\n\n");
}

