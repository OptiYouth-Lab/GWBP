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
#include <limits>
#include <fcntl.h>

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
extern "C" void ctf_correction_3d_gpu_device_profile_report();
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
    cout<<"Run doReconstruction() in ReconstructionAlgo_WBP_RAM"<<endl;

    // Input
    map<string,string>::iterator it=inputPara.find("path");
    string path;
    if(it!=inputPara.end())
    {
        path=it->second;
        cout << "File path: " << path << endl;
    }
    else
    {
        cout << "No specifit file path, set default: ./" << endl;
        path="./";
    }

    auto resolveWithPath = [&](const string& file_or_path) -> string {
        if (!file_or_path.empty() && file_or_path[0] == '/') return file_or_path;
        return path + "/" + file_or_path;
    };

    it=inputPara.find("input_mrc");
    string input_mrc;
    if(it!=inputPara.end())
    {
        input_mrc=resolveWithPath(it->second);
        cout << "Input file name: " << input_mrc << endl;
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
        output_mrc=resolveWithPath(it->second);
        cout << "Output file name: " << output_mrc << endl;
    }
    else
    {
        cout << "No output file name, set default: tomo.rec" << endl;
        output_mrc=resolveWithPath("tomo.rec");
    }

    it=inputPara.find("prfx");
    string prfx;
    if(it!=inputPara.end())
    {
        prfx=resolveWithPath(it->second);
        cout << "Prefix: " << prfx << endl;
    }
    else
    {
        cout << "No prfx, set default: tomo" << endl;
        prfx=resolveWithPath("tomo");
    }

    it=inputPara.find("h");
    int h;
    if(it!=inputPara.end())
    {
        h=atoi(it->second.c_str());
        cout << "Reconstruction height: " << h << endl;
    }
    else
    {
        h=int(stack_orig.getNx()/4);
        cout << "No height for reconstruction, set default (Nx/4): " << h << endl;
    }

    bool skip_ctfcorrection,skip_weighting,skip_3dctf;
    it=inputPara.find("skip_ctfcorrection");
    if(it!=inputPara.end())
    {
        skip_ctfcorrection=atoi(it->second.c_str());
        cout << "Skip CTF correction: " << skip_ctfcorrection << endl;
    }
    else
    {
        cout << "No skip_ctfcorrection, set default: 0" << endl;
        skip_ctfcorrection=0;
    }
    it=inputPara.find("skip_3dctf");
    if(it!=inputPara.end())
    {
        skip_3dctf=atoi(it->second.c_str());
        cout << "Skip 3D-CTF: " << skip_3dctf << endl;
    }
    else
    {
        cout << "No skip_3dctf, set default: 0 (Perform 3D-CTF correction)" << endl;
        skip_3dctf=0;
    }
    it=inputPara.find("skip_weighting");
    if(it!=inputPara.end())
    {
        skip_weighting=atoi(it->second.c_str());
        cout << "Skip weighting: " << skip_weighting << endl;
    }
    else
    {
        cout << "No skip_weighting, set default: 0 (Perform WBP)" << endl;
        skip_weighting=0;
    }

    bool verbose = false;
    it=inputPara.find("verbose");
    if(it!=inputPara.end())
    {
        verbose=atoi(it->second.c_str());
        cout << "Verbose logging: " << verbose << endl;
    }
    else
    {
        cout << "No verbose flag, set default: 0 (sparse progress output)" << endl;
        verbose=false;
    }
    
    float weighting_radial=0.05,weighting_sigma=0.5;
    if(!skip_weighting)
    {
        cout << "Weighting parameters:" << endl;
        it=inputPara.find("weighting_radial");
        if(it!=inputPara.end())
        {
            weighting_radial=atof(it->second.c_str());
            cout << "Radial: " << weighting_radial << endl;
        }
        else
        {
            cout << "No weighting_radial, set default: 0.05" << endl;
            weighting_radial=0.05;
        }
        it=inputPara.find("weighting_sigma");
        if(it!=inputPara.end())
        {
            weighting_sigma=atof(it->second.c_str());
            cout << "Sigma: " << weighting_sigma << endl;
        }
        else
        {
            cout << "No weighting_sigma, set default: 0.5" << endl;
            weighting_sigma=0.5;
        }
    }

    it=inputPara.find("input_tlt");
    string input_tlt;
    if(it!=inputPara.end())
    {
        input_tlt=resolveWithPath(it->second);
        cout << "Input tlt file name: " << input_tlt << endl;
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
        cout << "Input unrotated stack: " << unrotated_stack << endl;
    }
    else
    {
        cout << "No unrotated stack, input rotated stack" << endl;
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
    cout << "psi: " << psi_deg << endl;

    CTF ctf_para[stack_orig.getNz()];
    float Cs,pix,volt,w_cos;
    bool flip_contrast=false;
    int defocus_step=1;
    it=inputPara.find("pixel_size");
    if(it!=inputPara.end())
    {
        pix=atof(it->second.c_str());
        cout << "Pixel size (A): " << pix << endl;
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
            printf("Cs (mm): %f\n", Cs);
            //cout << "Cs (mm): " << Cs << endl;
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
            printf("Accelerating voltage (kV): %d\n", volt);
            cout << "Accelerating voltage (kV): " << volt << endl;
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
            cout << "Amplitude contrast: " << w_cos << endl;
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
            cout << "Flip contrast: " << flip_contrast << endl;
        }
        else
        {
            cout << "No flip contrast, set default: 0" << endl;
            flip_contrast=false;
        }

        it=inputPara.find("defocus_file");
        string defocus_file;
        if(it!=inputPara.end())
        {
            defocus_file=resolveWithPath(it->second);
            cout << "Defocus file name: " << defocus_file << endl;
        }
        else
        {
            cout << "No defocus file name, set default: defocus_file.txt" << endl;
            defocus_file=resolveWithPath("defocus_file.txt");
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
                cout << "Defocus step (pixels): " << defocus_step << endl;
            }
            else
            {
                cout << "No defocus step, set default: 1" << endl;
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
                cout << "[GPU] stack_recon1 uses pinned host memory" << endl;
            } else {
                stack_recon1 = (float *)aligned_alloc(4096, recon_bytes);
                cout << "[GPU] stack_recon1 pinned alloc failed, fallback to aligned_alloc" << endl;
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
#ifdef USE_GPU
        bool have_gpu_volume_stats = false;
        float gpu_min_all = 0.0f;
        float gpu_max_all = 0.0f;
        double gpu_mean_all = 0.0;
#endif
        char* stdio_buf = nullptr;

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
                printf("Image %d:\n",n);
                float theta_rad=theta[n]/180*M_PI;

                stack_orig.read2DIm_32bit(image_now,n);
                memcpy(image_now_backup,image_now,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());

                cout << "\tSkip CTF correction, use raw micrograph!" << endl;

                if(!skip_weighting)
                {
                    cout << "\tStart weighting: " << endl;
                    filter_weighting_1D_many(image_now,stack_orig.getNx(),stack_orig.getNy(),weighting_radial,weighting_sigma);
                    cout << "\tDone!" << endl;
                }

                // reconstruction
                cout << "\tStart reconstruction, loop over xz-plane:" << endl;
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

                cout << "\tDone" << endl;
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
                    printf("Image %d:\n",n);

                    float theta_rad=theta[n]/180*M_PI;

                    stack_orig.read2DIm_32bit(image_now,n);

                    memcpy(image_now_backup,image_now,sizeof(float)*stack_orig.getNx()*stack_orig.getNy());

                    cout << "\tPerform uniform CTF correction!" << endl;

                    // correction
                    cout << "\tStart correction: " << endl;
                    ctf_correction(image_now,stack_orig.getNx(),stack_orig.getNy(),ctf_para[n],flip_contrast,0.0);
                    cout << "\tDone!" << endl;

                    // weighting
                    if(!skip_weighting)
                    {
                        cout << "\tStart weighting: " << endl;
                        filter_weighting_1D_many(image_now,stack_orig.getNx(),stack_orig.getNy(),weighting_radial,weighting_sigma);
                        cout << "\tDone!" << endl;
                    }

                    // recontruction
                    cout << "\tStart reconstuction, loop over xz-plane:" << endl;

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
                    
                    cout << "\tDone" << endl;
                }
				free(image_now);
        		free(image_now_backup);
            }
            else    // perform 3D-CTF correction
            {
                // Use pinned host memory to accelerate host/device transfers.
                const size_t image_bytes = (size_t)stack_orig_getNx * (size_t)stack_orig_getNy * sizeof(float);

                float* image_now = nullptr;
                float* stack_corrected = nullptr;
                bool image_now_pinned = false;

                if (cudaMallocHost((void**)&image_now, image_bytes) == cudaSuccess) image_now_pinned = true;
                else image_now = (float*)aligned_alloc(4096, image_bytes);

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
                // Host I/O fine profile is enabled by default; set WBP_HOST_IO_FINE_PROFILE=0 to disable.
                bool host_io_fine_profile = true;
                if (const char* env_host_io_prof = std::getenv("WBP_HOST_IO_FINE_PROFILE")) {
                    host_io_fine_profile = (std::atoi(env_host_io_prof) != 0);
                }

                struct HostIoFineStats {
                    int setvbuf_rc = 0;
                    int setvbuf_mode = _IOFBF;
                    size_t setvbuf_size = 0;
                    bool prealloc_enabled = false;
                    bool prealloc_ok = false;
                    int prealloc_rc = 0;
                    size_t prealloc_bytes = 0;
                    double prealloc_ms = 0.0;
                    bool chunked_enabled = false;
                    size_t chunk_bytes = 0;
                    size_t total_bytes = 0;
                    long long chunk_count = 0;
                    double chunk_copy_ms = 0.0;
                    double chunk_copy_min_ms = std::numeric_limits<double>::infinity();
                    double chunk_copy_max_ms = 0.0;
                    double chunk_write_ms = 0.0;
                    double chunk_write_min_ms = std::numeric_limits<double>::infinity();
                    double chunk_write_max_ms = 0.0;
                    double finalize_full_d2h_ms = 0.0;
                    double finalize_full_write_ms = 0.0;
                } host_io;

                // Set explicit large stdio buffer for output file (default 64MB).
                size_t stdio_buf_bytes = 64ull * 1024ull * 1024ull;
                if (const char* env_stdio_mb = std::getenv("WBP_STDIO_BUF_MB")) {
                    long long mb = std::atoll(env_stdio_mb);
                    if (mb > 0) stdio_buf_bytes = (size_t)mb * 1024ull * 1024ull;
                }
                if (stdio_buf_bytes >= 4096ull) {
                    size_t alloc_bytes = ((stdio_buf_bytes + 4095ull) / 4096ull) * 4096ull;
                    stdio_buf = (char*)aligned_alloc(4096, alloc_bytes);
                    if (stdio_buf) {
                        host_io.setvbuf_size = alloc_bytes;
                    } else {
                        host_io.setvbuf_size = 0;
                    }
                } else {
                    host_io.setvbuf_size = 0;
                }
                host_io.setvbuf_mode = _IOFBF;
                host_io.setvbuf_rc = setvbuf(stack_final.m_fp, stdio_buf, _IOFBF, host_io.setvbuf_size);

                // Preallocate full output file to reduce metadata/extent work during close.
                bool prealloc_enable = true;
                if (const char* env_prealloc = std::getenv("WBP_OUTPUT_PREALLOC")) {
                    prealloc_enable = (std::atoi(env_prealloc) != 0);
                }
                host_io.prealloc_enabled = prealloc_enable;
                const size_t data_bytes_total = ImSize * (size_t)stack_orig_getNy;
                const size_t full_file_bytes = (size_t)1024 + (size_t)stack_final.getSymdatasize() + data_bytes_total;
                host_io.prealloc_bytes = full_file_bytes;
                host_io.total_bytes = data_bytes_total;
                if (prealloc_enable) {
                    auto t_prealloc_begin = std::chrono::steady_clock::now();
                    int fd = fileno(stack_final.m_fp);
                    if (fd >= 0) {
                        int rc = posix_fallocate(fd, (off_t)0, (off_t)full_file_bytes);
                        host_io.prealloc_rc = rc;
                        host_io.prealloc_ok = (rc == 0);
                    } else {
                        host_io.prealloc_rc = errno ? errno : -1;
                        host_io.prealloc_ok = false;
                    }
                    auto t_prealloc_end = std::chrono::steady_clock::now();
                    host_io.prealloc_ms = std::chrono::duration<double, std::milli>(t_prealloc_end - t_prealloc_begin).count();
                    if (!host_io.prealloc_ok && host_io_fine_profile) {
                        fprintf(stderr, "[PROFILE][HOST][IO] prealloc failed rc=%d (%s)\n",
                                host_io.prealloc_rc,
                                strerror(host_io.prealloc_rc));
                    }
                }
                // Ensure subsequent writes start from MRC data region.
                const long data_offset = (long)((size_t)1024 + (size_t)stack_final.getSymdatasize());
                if (fseek(stack_final.m_fp, data_offset, SEEK_SET) != 0 && host_io_fine_profile) {
                    fprintf(stderr, "[PROFILE][HOST][IO] warning: fseek to data offset failed errno=%d\n", errno);
                }

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
                int images = 0;
                long long ctf_slices = 0;
            } prof;

            // Stage-level GPU profiling inserts stream sync and hurts overlap.
            // Default: disabled for performance. Enable with WBP_GPU_STAGE_PROFILE=1.
            bool gpu_stage_profile = false;
            if (const char* env_stage_prof = std::getenv("WBP_GPU_STAGE_PROFILE")) {
                gpu_stage_profile = (std::atoi(env_stage_prof) != 0);
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

            float* d_image_now = nullptr;
			float* d_stack_corrected_dev = nullptr;
			float* z_offsets_host = nullptr;
			const int max_slices = (int)(h_tilt_max / defocus_step) + 1;
            const int tmp_h = int(h_tilt_max >> 1);
            const int cnt11 = (2 * tmp_h) / defocus_step;
			CUDA_CHECK(cudaMalloc((void**)&d_image_now, ttt_bytes));
			CUDA_CHECK(cudaMalloc((void**)&d_stack_corrected_dev, (size_t)max_slices * (size_t)ttt * sizeof(float)));
			z_offsets_host = new float[max_slices];
            for (int zz = 0; zz < cnt11; zz++) {
                z_offsets_host[zz] = float(zz * defocus_step - tmp_h) + float(defocus_step - 1) / 2.0f;
            }
            bool dynamic_ctf_slices = true;
            if (const char* env_dynamic = std::getenv("WBP_CTF3D_DYNAMIC_SLICES")) {
                dynamic_ctf_slices = (std::atoi(env_dynamic) != 0);
            }
            int ctf_slice_bucket = 4; // 1 means no bucket/alignment.
            if (const char* env_bucket = std::getenv("WBP_CTF3D_SLICE_BUCKET")) {
                int v = std::atoi(env_bucket);
                if (v > 1) ctf_slice_bucket = v;
            }
			wbp_recon_init(stack_orig_getNx, stack_orig_getNy, h);
			#endif
            #ifndef USE_GPU
            const int tmp_h = int(h_tilt_max >> 1);
            const int cnt11 = (2 * tmp_h) / defocus_step;
            const int cnt22 = cnt11 - (cnt11 & 63);
            #endif

                for(int n=0;n<stack_orig_getNz;n++)   // loop for every micrograph
                {
                    if (verbose) {
                        printf("Image %d:\n", n);
                    } else if ((n % 10) == 0 || n == stack_orig_getNz - 1) {
                        printf("Image %d/%d\n", n + 1, stack_orig_getNz);
                    }
                    float theta_rad = theta[n] / 180 * M_PI;
                    float sin1 = sin(theta_rad);
                    float cos1 = cos(theta_rad);

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

                            // Keep aligned window length whenever possible by shifting start,
                            // instead of truncating near upper boundary.
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

                    const int num_slices_this_tilt = z_slice_end - z_slice_begin + 1;
                    const int tmp_h_this_tilt = tmp_h - z_slice_begin * defocus_step;
#ifdef USE_GPU
                    float* z_offsets_this_tilt = z_offsets_host + z_slice_begin;
#endif

                    auto t_loop_begin = std::chrono::steady_clock::now();
                    auto t_read_begin = std::chrono::steady_clock::now();
                    stack_orig.read2DIm_32bit(image_now,n);
                    auto t_read_end = std::chrono::steady_clock::now();
                    prof.read_ms += std::chrono::duration<double, std::milli>(t_read_end - t_read_begin).count();

                    if (verbose) {
                        printf("\tPerform 3D-CTF correction!\n\tPerform 3D correction & save corrected stack:\n");
                    }
                    int n_zz=0;

#ifndef USE_GPU
                    if(!skip_weighting)
                    {
                          printf("\tStart weighting...\n");
                          fftwf_plan_with_nthreads(omp_get_max_threads());
                          filter_weighting_1D_many1(image_now, bufc12, stack_orig_getNx, stack_orig_getNy, weighting_radial, weighting_sigma);
                          printf("\tDone\n");
                    }
#endif

                    if (verbose) {
                        printf("\tStart 3D-CTF correction...\n");
                    }

				#ifdef USE_GPU
				    // ========== GPU device pipeline ==========
				    // 1) copy micrograph to device
				    // 2) optional 1D weighting on device
				    // 2) 3D-CTF correction: device->device (no big DtoH)
				    // 3) WBP backprojection: device corrected stack -> device recon volume
				    if (verbose) {
                        printf("\t[GPU] Using GPU for 3D-CTF correction (device output)...\n");
                    }
				
                    GPU_PROFILE_TIME_ADD(prof.h2d_ms,
				        CUDA_CHECK(cudaMemcpy(d_image_now, image_now, ttt_bytes, cudaMemcpyHostToDevice));
                    );
                    if (!skip_weighting) {
                        if (verbose) {
                            printf("\t[GPU] Start weighting...\n");
                        }
                        GPU_PROFILE_TIME_ADD(prof.weighting_ms,
                            filter_weighting_1d_many_gpu_inplace(
                                d_image_now,
                                stack_orig_getNx,
                                stack_orig_getNy,
                                weighting_radial,
                                weighting_sigma
                            );
                        );
                        if (verbose) {
                            printf("\t[GPU] Weighting done.\n");
                        }
                    }
                    GPU_PROFILE_TIME_ADD(prof.ctf3d_ms,
				        ctf_correction_3d_gpu_device_no_norm(
				            d_image_now,
				            d_stack_corrected_dev,
				            stack_orig_getNx,
				            stack_orig_getNy,
				            num_slices_this_tilt,
				            ctf_para[n],
				            flip_contrast,
				            z_offsets_this_tilt
				        );
                    );
				    n_zz = num_slices_this_tilt;
                    prof.ctf_slices += (long long)n_zz;
				
				    if (verbose) {
                        printf("\t[GPU] 3D-CTF correction completed (device).\n");
                    }
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

					if (verbose) {
                        printf("\tDone!\n\tPerform reconstruction:\n");
                    }
				
					#ifdef USE_GPU
					    // GPU WBP backprojection accumulates directly into the device recon volume.
                        GPU_PROFILE_TIME_ADD(prof.backproj_ms,
					        wbp_backproject_accumulate_gpu_device(
					            d_stack_corrected_dev,
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
                                corrected_scale
					        );
                        );
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
				
					if (verbose) {
                        printf("\tDone\n");
                    }
                    auto t_loop_end = std::chrono::steady_clock::now();
                    prof.loop_wall_ms += std::chrono::duration<double, std::milli>(t_loop_end - t_loop_begin).count();
                    prof.images++;
                }
                    double write_volume_ms = 0.0;
					#ifdef USE_GPU
                    ctf_correction_3d_gpu_device_profile_report();
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
                    size_t chunk_bytes = 64ull * 1024ull * 1024ull;
                    if (const char* env_chunk_mb = std::getenv("WBP_FINALIZE_CHUNK_MB")) {
                        long long mb = std::atoll(env_chunk_mb);
                        if (mb > 0) chunk_bytes = (size_t)mb * 1024ull * 1024ull;
                    }
                    if (chunk_bytes > total_bytes) chunk_bytes = total_bytes;
                    if (chunk_bytes < sizeof(float)) chunk_bytes = sizeof(float);
                    chunk_bytes -= (chunk_bytes % sizeof(float));
                    host_io.chunked_enabled = chunked_finalize_write;
                    host_io.chunk_bytes = chunk_bytes;
                    host_io.total_bytes = total_bytes;

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

                    if (finalize_stage) {
                        size_t copied = 0;
                        const size_t chunk_elems = chunk_bytes / sizeof(float);
                        while (copied < total_elems) {
                            size_t todo = total_elems - copied;
                            if (todo > chunk_elems) todo = chunk_elems;
                            auto t_chunk_copy_begin = std::chrono::steady_clock::now();
                            GPU_PROFILE_TIME_ADD(prof.finalize_d2h_ms,
					            wbp_recon_copy_range(finalize_stage, copied, todo);
                            );
                            auto t_chunk_copy_end = std::chrono::steady_clock::now();
                            double chunk_copy_ms = std::chrono::duration<double, std::milli>(
                                t_chunk_copy_end - t_chunk_copy_begin).count();
                            host_io.chunk_copy_ms += chunk_copy_ms;
                            if (chunk_copy_ms < host_io.chunk_copy_min_ms) host_io.chunk_copy_min_ms = chunk_copy_ms;
                            if (chunk_copy_ms > host_io.chunk_copy_max_ms) host_io.chunk_copy_max_ms = chunk_copy_ms;
                            auto t_chunk_write_begin = std::chrono::steady_clock::now();
                            if (fwrite(finalize_stage, sizeof(float), todo, stack_final.m_fp) != todo) {
                                cerr << "Failed to write reconstruction chunk to output file" << endl;
                                abort();
                            }
                            auto t_chunk_write_end = std::chrono::steady_clock::now();
                            double chunk_write_ms = std::chrono::duration<double, std::milli>(
                                t_chunk_write_end - t_chunk_write_begin).count();
                            write_volume_ms += chunk_write_ms;
                            host_io.chunk_write_ms += chunk_write_ms;
                            if (chunk_write_ms < host_io.chunk_write_min_ms) host_io.chunk_write_min_ms = chunk_write_ms;
                            if (chunk_write_ms > host_io.chunk_write_max_ms) host_io.chunk_write_max_ms = chunk_write_ms;
                            host_io.chunk_count++;
                            copied += todo;
                        }
                        volume_written_in_finalize = true;
                    } else {
                        alloc_stack_recon1();
                        auto t_finalize_d2h_begin = std::chrono::steady_clock::now();
                        GPU_PROFILE_TIME_ADD(prof.finalize_d2h_ms,
					        wbp_recon_finalize(stack_recon1);
                        );
                        auto t_finalize_d2h_end = std::chrono::steady_clock::now();
                        host_io.finalize_full_d2h_ms = std::chrono::duration<double, std::milli>(
                            t_finalize_d2h_end - t_finalize_d2h_begin).count();
                    }
                    if (finalize_stage_pinned) cudaFreeHost(finalize_stage);
                    else free(finalize_stage);

					wbp_recon_free();
                    CUDA_CHECK(cudaFree(d_image_now));
                    CUDA_CHECK(cudaFree(d_stack_corrected_dev));
					delete [] z_offsets_host;
                    ctf_correction_3d_gpu_device_profile_shutdown();
                    CUDA_CHECK(cudaEventDestroy(evt_begin));
                    CUDA_CHECK(cudaEventDestroy(evt_end));

                    if (prof.images > 0) {
                        const double inv_n = 1.0 / (double)prof.images;
                        printf("\n[PROFILE][GPU] images=%d\n", prof.images);
                        printf("[PROFILE][GPU] stage profile=%d\n", gpu_stage_profile ? 1 : 0);
                        printf("[PROFILE][GPU] dynamic ctf slices=%d\n", dynamic_ctf_slices ? 1 : 0);
                        printf("[PROFILE][GPU] ctf slice bucket=%d\n", ctf_slice_bucket);
                        printf("[PROFILE][GPU] ctf slices total=%lld, avg=%.2f/image\n",
                               prof.ctf_slices, (double)prof.ctf_slices * inv_n);
                        printf("[PROFILE][GPU] read2D     total=%.3f ms, avg=%.3f ms/image\n", prof.read_ms, prof.read_ms * inv_n);
                        if (gpu_stage_profile) {
                            printf("[PROFILE][GPU] H2D copy    total=%.3f ms, avg=%.3f ms/image\n", prof.h2d_ms, prof.h2d_ms * inv_n);
                            printf("[PROFILE][GPU] weighting   total=%.3f ms, avg=%.3f ms/image\n", prof.weighting_ms, prof.weighting_ms * inv_n);
                            printf("[PROFILE][GPU] 3D-CTF      total=%.3f ms, avg=%.3f ms/image\n", prof.ctf3d_ms, prof.ctf3d_ms * inv_n);
                            printf("[PROFILE][GPU] backproject total=%.3f ms, avg=%.3f ms/image\n", prof.backproj_ms, prof.backproj_ms * inv_n);
                        } else {
                            printf("[PROFILE][GPU] stage breakdown disabled (set WBP_GPU_STAGE_PROFILE=1 to enable)\n");
                        }
                        printf("[PROFILE][GPU] loop wall   total=%.3f ms, avg=%.3f ms/image\n", prof.loop_wall_ms, prof.loop_wall_ms * inv_n);
                        printf("[PROFILE][GPU] recon stats total=%.3f ms\n", prof.recon_stats_ms);
                        printf("[PROFILE][GPU] finalize D2H total=%.3f ms\n\n", prof.finalize_d2h_ms);
                    }
#undef GPU_PROFILE_TIME_ADD
					#endif

                if (!volume_written_in_finalize) {
                    auto t_write_begin = std::chrono::steady_clock::now();
			        fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
                    auto t_write_end = std::chrono::steady_clock::now();
                    write_volume_ms = std::chrono::duration<double, std::milli>(t_write_end - t_write_begin).count();
                    host_io.finalize_full_write_ms = write_volume_ms;
                }
                printf("[PROFILE][HOST] write_volume total=%.3f ms\n", write_volume_ms);
                if (host_io_fine_profile) {
                    const char* mode_str = (host_io.setvbuf_mode == _IONBF) ? "_IONBF"
                                           : (host_io.setvbuf_mode == _IOLBF) ? "_IOLBF"
                                           : "_IOFBF";
                    printf("[PROFILE][HOST][IO] setvbuf mode=%s size=%zu rc=%d\n",
                           mode_str, host_io.setvbuf_size, host_io.setvbuf_rc);
                    printf("[PROFILE][HOST][IO] prealloc enabled=%d ok=%d bytes=%zu rc=%d time=%.3f ms\n",
                           host_io.prealloc_enabled ? 1 : 0,
                           host_io.prealloc_ok ? 1 : 0,
                           host_io.prealloc_bytes,
                           host_io.prealloc_rc,
                           host_io.prealloc_ms);
                    if (host_io.chunked_enabled) {
                        const double inv_chunks = (host_io.chunk_count > 0) ? (1.0 / (double)host_io.chunk_count) : 0.0;
                        printf("[PROFILE][HOST][IO] chunked=1 total_bytes=%zu chunk_bytes=%zu chunks=%lld\n",
                               host_io.total_bytes, host_io.chunk_bytes, host_io.chunk_count);
                        printf("[PROFILE][HOST][IO] chunk D2H   total=%.3f ms, avg=%.3f ms, min=%.3f ms, max=%.3f ms\n",
                               host_io.chunk_copy_ms,
                               host_io.chunk_copy_ms * inv_chunks,
                               (host_io.chunk_count > 0 ? host_io.chunk_copy_min_ms : 0.0),
                               host_io.chunk_copy_max_ms);
                        printf("[PROFILE][HOST][IO] chunk fwrite total=%.3f ms, avg=%.3f ms, min=%.3f ms, max=%.3f ms\n",
                               host_io.chunk_write_ms,
                               host_io.chunk_write_ms * inv_chunks,
                               (host_io.chunk_count > 0 ? host_io.chunk_write_min_ms : 0.0),
                               host_io.chunk_write_max_ms);
                    } else {
                        printf("[PROFILE][HOST][IO] chunked=0 full D2H=%.3f ms, full fwrite=%.3f ms\n",
                               host_io.finalize_full_d2h_ms, host_io.finalize_full_write_ms);
                    }
                }
                //fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
#ifndef USE_GPU
				free(bufc);
				free(bufc12);
#endif

				// Free pinned/non-pinned host buffers.
				if (image_now_pinned) cudaFreeHost(image_now);
				else free(image_now);
                // [GPU] stack_corrected was not allocated (device pipeline)
            }
        }

        // write out final result
        printf("Wrtie out final reconstruction result:\n");

        //size_t ImSize = (size_t)stack_final.getImSize();
		//setvbuf(stack_final.m_fp, (char *)aligned_alloc(4096, ImSize * stack_orig_getNy), _IOFBF, ImSize * stack_orig_getNy);
		//fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);

        // update MRC header
        auto t_header_begin = std::chrono::steady_clock::now();
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
        auto t_header_end = std::chrono::steady_clock::now();
        printf("[PROFILE][HOST] header_write total=%.3f ms\n",
               std::chrono::duration<double, std::milli>(t_header_end - t_header_begin).count());

        auto t_fflush_begin = std::chrono::steady_clock::now();
        fflush(stack_final.m_fp);
        auto t_fflush_end = std::chrono::steady_clock::now();
        printf("[PROFILE][HOST] fflush total=%.3f ms\n",
               std::chrono::duration<double, std::milli>(t_fflush_end - t_fflush_begin).count());

        auto t_close_begin = std::chrono::steady_clock::now();
        stack_final.close();
        auto t_close_end = std::chrono::steady_clock::now();
        if (stdio_buf) {
            free(stdio_buf);
            stdio_buf = nullptr;
        }
        printf("[PROFILE][HOST] file_close total=%.3f ms\n",
               std::chrono::duration<double, std::milli>(t_close_end - t_close_begin).count());


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

