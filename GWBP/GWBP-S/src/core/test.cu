/*******************************************************************
 *       Filename:  ReconstructionAlgo.cpp -->cuda                                     
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
//#define __AMXTILEINTRIN_H__  // 放在第一行，屏蔽 amxtileintrin.h

#include "../include/ReconstructionAlgo_WBP_RAM.h"
#include "../include/mrc.h"
#include "../include/CTF.h"
#include "math.h"
#include "fftw3.h"
#include "omp.h"
#include <immintrin.h>
#include <sys/time.h> 
#include "util.h"
//#include "../include/util.h"
#include <iomanip>


#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// static void buf2fft(float *buf, float *fft, int nx, int ny)
// {
//     int nxb=nx+2-nx%2;
//     int i;
// 	#pragma omp parallel for schedule(static)
//     for(i=0;i<(nx+2-nx%2)*ny;i++)
//     {
//         fft[i]=0.0;
//     }
// 	#pragma omp parallel for schedule(static)
//     for(i=0;i<ny;i++)
//     {
//         memcpy(fft+i*nxb,buf+i*nx,sizeof(float)*nx);
//     }
// }

// 核函数：初始化 fft 数组为 0
__global__ void initFFTKernel(float* fft, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        fft[idx] = 0.0f;
    }
}

// 核函数：拷贝 buf 数据到 fft
__global__ void copyBufToFFTKernel(const float* buf, float* fft, 
                                   int nx, int nxb, int ny) {
    int row = blockIdx.x;                  // 行索引（每个块处理一行）
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // 列索引（多块处理一列）
    
    if (row < ny && col < nx) {            // 边界检查
        int fftOffset = row * nxb + col;
        int bufOffset = row * nx + col;
        fft[fftOffset] = buf[bufOffset];
    }
}

// 使用CUDA的buf2fft
void buf2fft(const float* d_buf, float* d_fft, 
             int nx, int ny, cudaStream_t stream = 0) {
    int nxb = nx + 2 - (nx % 2);  // 计算填充后的行宽
    int fftSize = nxb * ny;       // fft 总大小

    // 1. 初始化 fft 内存为 0
    dim3 blockInit(256);
    dim3 gridInit((fftSize + blockInit.x - 1) / blockInit.x);
    initFFTKernel<<<gridInit, blockInit, 0, stream>>>(d_fft, fftSize);

    // 2. 拷贝 buf 数据到 fft（支持 nx > 1024）
    int colsPerBlock = 256;                  // 块内线程数（≤1024）
    dim3 blockCopy(colsPerBlock);            // 线程块：(colsPerBlock, 1, 1)
    dim3 gridCopy(ny, (nx + colsPerBlock - 1) / colsPerBlock);  // 网格：(行数, 列块数)
    copyBufToFFTKernel<<<gridCopy, blockCopy, 0, stream>>>(d_buf, d_fft, nx, nxb, ny);
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

// static void filter_weighting(float *data,int Nx,float radial,float sigma)
// {
//     int Nx_padding=int(Nx/10);
//     int Nx_final=Nx+Nx_padding;
//     fftwf_plan plan_fft,plan_ifft;
//     float *bufc=new float[Nx_final+2-Nx_final%2];
//     plan_fft=fftwf_plan_dft_r2c_1d(Nx_final,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
//     plan_ifft=fftwf_plan_dft_c2r_1d(Nx_final,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);
//     buf2fft_padding_1D(data,bufc,Nx,Nx_final,1,1);
//     fftwf_execute(plan_fft);

//     int radial_Nx=int(floor((Nx_final)*radial));
//     float sigma_Nx=float(Nx_final)*sigma;

//     // loop: Nx_final+2-Nx_final%2 (all Fourier components)
//     for(int i=0;i<Nx_final+2-Nx_final%2;i+=2)   // radial filtering
//     {
//         if(i==0)    // DC
//         {
//             bufc[i]*=0.2;
//             bufc[i+1]*=0.2;
//         }
//         else if(i/2<=radial_Nx)  // radial
//         {
//             bufc[i]*=(i/2);
//             bufc[i+1]*=(i/2);
//         }
//         else    // Gaussian falloff
//         {
//             bufc[i]=bufc[i]*float(radial_Nx)*exp(-float((i/2-radial_Nx)*(i/2-radial_Nx))/(sigma_Nx*sigma_Nx));
//             bufc[i+1]=bufc[i+1]*float(radial_Nx)*exp(-float((i/2-radial_Nx)*(i/2-radial_Nx))/(sigma_Nx*sigma_Nx));
//         }
//     }

//     fftwf_execute(plan_ifft);
//     fft2buf_padding_1D(data,bufc,Nx,Nx_final,1,1);
//     for(int i=0;i<Nx;i++)   // normalization
//     {
//         data[i]/=Nx;
//     }
//     fftwf_destroy_plan(plan_fft);
// 	fftwf_destroy_plan(plan_ifft);
//     delete [] bufc;
// }


// 频域滤波核函数 - 必须定义在全局作用域
__global__ void paddingKernel(cufftReal* d_bufc, int Nx, int Nx_final) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx_final - Nx) {  // 仅处理填充区域
        int pos = Nx + idx;
        float nxp_1 = 1.0f / (Nx_final - Nx + 1);  // 与CPU版nxp_1一致
        float tmp1 = d_bufc[0] * nxp_1;
        float tmp2 = d_bufc[Nx - 1] * nxp_1;
        d_bufc[pos] = (idx + 1) * tmp1 + (Nx_final - Nx - idx) * tmp2;
    }
}

__global__ void filterKernel(cufftComplex* d_freq, int Nx_final, 
                             float radial, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx_final / 2 + 1) {  // R2C频域长度正确
        int radial_Nx = static_cast<int>(floor(Nx_final * radial));
        float sigma_Nx = Nx_final * sigma;
        sigma_Nx *= sigma_Nx;
        
        float weight;
        if (idx == 0) {
            weight = 0.2f;  // DC分量权重与CPU一致
        }
        else if (idx <= radial_Nx) {
            weight = static_cast<float>(idx);  // 径向权重
        }
        else {
            int diff = idx - radial_Nx;
            weight = static_cast<float>(radial_Nx) * exp(-static_cast<float>(diff * diff) / sigma_Nx);
        }

        d_freq[idx].x *= weight;
        d_freq[idx].y *= weight;
    }
}

__global__ void normalizeKernel(cufftReal* d_out, float* d_data, int Nx, int Nx_final) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx) {
        // 抵消CUFFT逆变换的内置缩放(Nx_final)，并保持与CPU一致的1/Nx归一化
        d_data[idx] = d_out[idx] / (static_cast<float>(Nx) * static_cast<float>(Nx_final));
    }
}

// 自定义CUFFT错误信息转换
const char* cufftGetErrorString(cufftResult result) {
    switch(result) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        default: return "Unknown CUFFT error";
    }
}

void filter_weighting(float* d_data, int Nx, float radial, float sigma,
                      cudaStream_t stream = 0) {
    // 输入参数验证（增强鲁棒性）
    if (d_data == nullptr || Nx <= 0 || radial < 0 || radial > 1.0f || sigma <= 0) {
        fprintf(stderr, "filter_weighting: 无效的输入参数\n");
        return;
    }

    int Nx_padding = Nx / 10;
    int Nx_final = Nx + Nx_padding;
    int fft_buffer_size = Nx_final;  // CUFFT无需额外+2

    // 1. 创建CUFFT计划
    cufftHandle plan_fft, plan_ifft;
    cufftResult fft_result = cufftPlan1d(&plan_fft, Nx_final, CUFFT_R2C, 1);
    if (fft_result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT FFT计划创建失败: %s\n", cufftGetErrorString(fft_result));
        return;
    }

    fft_result = cufftPlan1d(&plan_ifft, Nx_final, CUFFT_C2R, 1);
    if (fft_result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT IFFT计划创建失败: %s\n", cufftGetErrorString(fft_result));
        cufftDestroy(plan_fft);
        return;
    }

    // 2. 分配并初始化设备内存
    cufftReal* d_bufc;
    cudaError_t cuda_result = cudaMallocAsync(&d_bufc, sizeof(cufftReal) * fft_buffer_size, stream);
    if (cuda_result != cudaSuccess) {
        fprintf(stderr, "CUDA内存分配失败: %s\n", cudaGetErrorString(cuda_result));
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    cuda_result = cudaMemsetAsync(d_bufc, 0, sizeof(cufftReal) * fft_buffer_size, stream);
    if (cuda_result != cudaSuccess) {
        fprintf(stderr, "CUDA内存初始化失败: %s\n", cudaGetErrorString(cuda_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 3. 复制输入数据并执行填充（与CPU版buf2fft_padding_1D一致）
    cuda_result = cudaMemcpyAsync(d_bufc, d_data, sizeof(cufftReal) * Nx, 
                                 cudaMemcpyDeviceToDevice, stream);
    if (cuda_result != cudaSuccess) {
        fprintf(stderr, "CUDA数据复制失败: %s\n", cudaGetErrorString(cuda_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 执行填充操作
    dim3 padBlock(256);
    dim3 padGrid((Nx_final - Nx + padBlock.x - 1) / padBlock.x);
    paddingKernel<<<padGrid, padBlock, 0, stream>>>(d_bufc, Nx, Nx_final);

    // 4. 执行FFT
    fft_result = cufftExecR2C(plan_fft, d_bufc, (cufftComplex*)d_bufc);
    if (fft_result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT执行失败: %s\n", cufftGetErrorString(fft_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 5. 频域滤波
    dim3 block(256);
    dim3 grid((Nx_final / 2 + 1 + block.x - 1) / block.x);
    filterKernel<<<grid, block, 0, stream>>>((cufftComplex*)d_bufc, Nx_final, radial, sigma);

    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        fprintf(stderr, "滤波核函数启动失败: %s\n", cudaGetErrorString(cuda_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 6. 执行逆FFT
    fft_result = cufftExecC2R(plan_ifft, (cufftComplex*)d_bufc, d_bufc);
    if (fft_result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT逆变换执行失败: %s\n", cufftGetErrorString(fft_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 7. 归一化（修正CUFFT缩放）
    normalizeKernel<<<grid, block, 0, stream>>>(d_bufc, d_data, Nx, Nx_final);
    
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        fprintf(stderr, "归一化核函数启动失败: %s\n", cudaGetErrorString(cuda_result));
        cudaFreeAsync(d_bufc, stream);
        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        return;
    }

    // 8. 清理资源（确保流完成后释放）
    cudaStreamSynchronize(stream);  // 根据需求选择是否同步
    cudaFreeAsync(d_bufc, stream);
    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
}

    

static void filter_weighting_1D_many1(float *data, float * bufc, int Nx,int Ny,float radial,float sigma)
{
    int Nx_padding = int(Nx / 10);
    int Nx_final   = Nx + Nx_padding;

    fftwf_plan plan_fft,plan_ifft;
    
    fftwf_init_threads();
    fftwf_plan_with_nthreads(64);

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

    #pragma simd
    #pragma unroll(8)
	#pragma omp parallel for schedule(static)
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






static void ctf_correction1(float *image, float *image1, float * bufc, int Nx, int Ny, 
                           CTF ctf, bool flip_contrast, float z_offset) {  // z_offset in pixels
    // 1. 通过 CTF 类的 getter 函数获取参数（确保 CTF 对象已初始化）
    float defocus1    = ctf.getDefocus1();    // 获取离焦1
    float defocus2    = ctf.getDefocus2();    // 获取离焦2
    float astig       = ctf.getAstigmatism(); // 获取像散
    float phase_shift = ctf.getPhaseShift();  // 获取相移
    float w_cos       = ctf.getWcos();        // 获取 w_cos
    float w_sin       = ctf.getWsin();        // 获取 w_sin
    float pix         = ctf.getPixelSize();   // 获取像素大小（单位：m）
    float lambda      = ctf.getLambda();      // 获取波长（单位：m）

    // 2. 计算派生参数（确保所有变量均被初始化）
    float pix_1  = 1.0f / pix;                // 像素大小的倒数
    float lambda2 = lambda * lambda;          // 波长平方
    float lambda3 = lambda2 * lambda;         // 波长立方

    // 后续FFT和CTF校正逻辑保持不变...
    fftwf_plan plan_fft;
    fftwf_plan plan_ifft;

    int loop_num = Nx + 2 - (Nx & 1);  // 计算填充后的行宽

    plan_fft  = fftwf_plan_dft_r2c_2d(Ny, Nx, (float*)bufc, reinterpret_cast<fftwf_complex*>(bufc), FFTW_ESTIMATE);
    plan_ifft = fftwf_plan_dft_c2r_2d(Ny, Nx, reinterpret_cast<fftwf_complex*>(bufc), (float*)bufc, FFTW_ESTIMATE);

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
            #pragma simd
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
            #pragma simd
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

        #pragma simd
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

static void ctf_correction3(float *image, float *image1, float * bufc, int Nx, int Ny, 
                           CTF ctf, bool flip_contrast, float z_offset) {  // z_offset in pixels
    fftwf_plan plan_fft;
    fftwf_plan plan_ifft;

    // 1. 通过 CTF 类的 getter 函数获取参数（关键修正）
    register float defocus1    = ctf.getDefocus1();    // 从 CTF 对象获取离焦1
    register float defocus2    = ctf.getDefocus2();    // 获取离焦2
    register float astig       = ctf.getAstigmatism(); // 获取像散
    register float phase_shift = ctf.getPhaseShift();  // 获取相移
    register float w_cos       = ctf.getWcos();        // 获取 w_cos
    register float w_sin       = ctf.getWsin();        // 获取 w_sin
    register float pix         = ctf.getPixelSize();   // 获取像素大小
    register float lambda      = ctf.getLambda();      // 获取波长

    // 2. 计算派生参数（确保所有变量均初始化）
    register float pix_1       = 1.0f / pix;           // 像素大小的倒数
    register float lambda2     = lambda * lambda;      // 波长平方
    register float lambda3     = lambda2 * lambda;     // 波长立方

    // （移除原错误的 computeCTF2D_P 调用，无需通过函数传递参数）

    register int loop_num = Nx + 2 - (Nx & 1);
    plan_fft              = fftwf_plan_dft_r2c_2d(Ny, Nx, (float*)bufc, reinterpret_cast<fftwf_complex*>(bufc), FFTW_ESTIMATE);
    plan_ifft             = fftwf_plan_dft_c2r_2d(Ny, Nx, reinterpret_cast<fftwf_complex*>(bufc), (float*)bufc, FFTW_ESTIMATE);

    register int Nx_Ny     = Nx * Ny;
    register int loop_num1 = loop_num * Ny;

    // 复制图像数据到 buf 并填充零（保持原逻辑）
    #pragma omp parallel for schedule(static)
    for(int z = 0; z < Nx_Ny; z++) {
        int i = z / Nx;
        int j = z - i * Nx;
        bufc[i * loop_num + j] = image[z];
    }

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < Ny; i++) {
        for(int j = i * loop_num + Nx; j < (i + 1) * loop_num; j++) {
            bufc[j] = 0.0f;
        }
    }

    fftwf_execute(plan_fft);
    
    // 计算CTF校正相关参数（保持原逻辑）
    register float Nx_Ny_1  = 1.0f / (float)Nx_Ny;
    register float Nx_pix_1 = pix_1 / (float)Nx;
    register float Ny_pix_1 = pix_1 / (float)Ny;
    register float tmp_num1 = defocus1 + defocus2 - 2 * z_offset * pix;
    register float tmp_num2 = defocus1 - defocus2;

    // 根据 flip_contrast 执行CTF校正（保持原逻辑）
    if(flip_contrast) {
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < loop_num1; i += 2) {
            register int tmp1      = i / loop_num;
            register int tmp2      = i - tmp1 * loop_num;
            register int  x        = tmp2 >> 1;
            register int  y        = tmp1;
            register float x_norm  = (x >= (int)ceil((float)(Nx + 1)/2)) ? (x - Nx) : x;
            register float y_norm  = (y >= (int)ceil((float)(Ny + 1)/2)) ? (y - Ny) : y;
            register float x_real  = (float)x_norm * Nx_pix_1;
            register float y_real  = (float)y_norm * Ny_pix_1;
            register float alpha;

            if(x_norm == 0) {
                alpha = M_PI_2 * ((y_norm > 0) - (y_norm < 0));
            } else {
                alpha = atan(y_real / x_real);
            }

            register float freq2   = x_real * x_real + y_real * y_real;
            register float df_now  = (tmp_num1 + tmp_num2 * cos(2 * (alpha - astig))) * 0.5f;
            register float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
            register float ctf_now = -(w_sin * sin(chi) + w_cos * cos(chi));
        
            if(ctf_now < 0) {
                bufc[i]   = -bufc[i];
                bufc[i+1] = -bufc[i+1];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < loop_num1; i += 2) {
            register int tmp1      = i / loop_num;
            register int tmp2      = i - tmp1 * loop_num;
            register int  x        = tmp2 >> 1;
            register int  y        = tmp1;
            register float x_norm  = (x >= (int)ceil((float)(Nx + 1)/2)) ? (x - Nx) : x;
            register float y_norm  = (y >= (int)ceil((float)(Ny + 1)/2)) ? (y - Ny) : y;
            register float x_real  = (float)x_norm * Nx_pix_1;
            register float y_real  = (float)y_norm * Ny_pix_1;
            register float alpha;

            if(x_norm == 0) {
                alpha = M_PI_2 * ((y_norm > 0) - (y_norm < 0));
            } else {
                alpha = atan(y_real / x_real);
            }

            register float freq2   = x_real * x_real + y_real * y_real;
            register float df_now  = (tmp_num1 + tmp_num2 * cos(2 * (alpha - astig))) * 0.5f;
            register float chi     = (lambda2 * df_now - lambda3 * freq2) * freq2 + phase_shift;
            register float ctf_now = w_sin * sin(chi) + w_cos * cos(chi);
        
            if(ctf_now < 0) {
                bufc[i]   = -bufc[i];
                bufc[i+1] = -bufc[i+1];
            }
        }
    }

    fftwf_execute(plan_ifft);

    int loop_num4 = Ny * Nx;

    // 结果输出到 image1（保持原逻辑）
    #pragma omp parallel for schedule(static)
    for(int z = 0; z < loop_num4; z++) {
        int i = z / Nx;
        int j = z - i * Nx;
        image1[i * Nx + j] = bufc[i * loop_num + j] * Nx_Ny_1;
    }

    fftwf_destroy_plan(plan_fft);
    fftwf_destroy_plan(plan_ifft);
}


// static void ctf_correction(float *image,int Nx,int Ny,CTF ctf,bool flip_contrast,float z_offset)   // z_offset in pixels
// {

//     fftwf_plan plan_fft;
// 	fftwf_plan plan_ifft;

// 	int loop_num = Nx + 2 - (Nx & 1);

// 	float * bufc = (float *)aligned_alloc(4096, sizeof(float) * loop_num * Ny);

//     plan_fft  = fftwf_plan_dft_r2c_2d(Ny,Nx,(float*)bufc,reinterpret_cast<fftwf_complex*>(bufc),FFTW_ESTIMATE);
//     plan_ifft = fftwf_plan_dft_c2r_2d(Ny,Nx,reinterpret_cast<fftwf_complex*>(bufc),(float*)bufc,FFTW_ESTIMATE);

//     int Nx_Ny = Nx*Ny;

// 	int loop_num1 = loop_num * Ny;

//     #pragma omp parallel for schedule(static)
//     for(int z=0; z<Nx_Ny; z++)
//     {
//         int i = z / Nx;
//         int j = z - i * Nx;
//         bufc[i*loop_num+j] = image[z];
//     }

//     #pragma omp parallel for schedule(static)
//     for(int i=0;i<Ny;i++)
//     {
// 		for(int j=i*loop_num+Nx;j<(i+1)*loop_num;j++)
// 		{
// 			bufc[j] = 0.0;
// 		}
//     }

//     fftwf_execute(plan_fft);
	
	
//      double Nx_Ny_1 = 1.0 / (double)Nx_Ny;

// 	#pragma omp parallel for schedule(static)
//     for(int i=0; i<loop_num1; i+=2)
//     {
// 		int tmp1 = i / loop_num;
// 		int tmp2 = i - tmp1 * loop_num;

//         float ctf_now = ctf.computeCTF2D(tmp2/2,tmp1,Nx,Ny,true,flip_contrast,z_offset);

//         bufc[i]   *=ctf_now;
//         bufc[i+1] *=ctf_now;
//     }

//     fftwf_execute(plan_ifft);

//     int loop_num4 = Ny * Nx;

//     #pragma omp parallel for schedule(static)
//     for(int z=0; z<loop_num4; z++)
//     {
//         int i = z / Nx;
//         int j = z - i * Nx;
//         image[i*Nx + j] = bufc[i*loop_num+j] * Nx_Ny_1;
//     }

//     fftwf_destroy_plan(plan_fft);
// 	fftwf_destroy_plan(plan_ifft);

// 	free(bufc);
// }

// CTF 校正核函数（简化示例，需根据原公式完善）
__global__ void ctfCorrectionKernel(cufftComplex* d_freq, int Nx, int Ny, 
                                    CTF ctfParams, bool flip_contrast, 
                                    float z_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        // 计算频域坐标（归一化到[-0.5, 0.5]）
        float fx = (x < Nx/2) ? (float)x/Nx : (float)(x - Nx)/Nx;
        float fy = (y < Ny/2) ? (float)y/Ny : (float)(y - Ny)/Ny;
        float freq2 = fx*fx + fy*fy;  // 频率的平方

        // 从CTF参数计算散焦和相位
        float defocus = ctfParams.defocus1 + z_offset;  // 考虑z方向偏移的散焦
        float phase = 2 * M_PI * (0.5 * ctfParams.lambda * freq2 * defocus - ctfParams.phase_shift);

        // 计算CTF权重（振幅对比度+相位对比度）
        float ctf_now = ctfParams.w_cos * cos(phase) - ctfParams.w_sin * sin(phase);

        // 对比度翻转（如果需要）
        if (flip_contrast) {
            ctf_now = -ctf_now;
        }

        // 应用CTF权重到频域数据
        d_freq[y * (Nx / 2 + 1) + x].x *= ctf_now;
        d_freq[y * (Nx / 2 + 1) + x].y *= ctf_now;
    }
}

void ctf_correction(float* d_image, int Nx, int Ny, const CTF& ctf, 
                         bool flip_contrast, float z_offset,
                         cudaStream_t stream = 0) {
    // 1. 初始化 CUFFT 计划（2D R2C/C2R）
    cufftHandle plan_fft, plan_ifft;
    cufftPlan2d(&plan_fft, Ny, Nx, CUFFT_R2C);
    cufftPlan2d(&plan_ifft, Ny, Nx, CUFFT_C2R);

    // 2. 分配设备内存 + 执行 FFT（省略数据拷贝细节）
    cufftReal* d_bufc;
    cudaMalloc((void**)&d_bufc, sizeof(cufftReal) * (Nx + 2 - (Nx % 2)) * Ny);
    cufftExecR2C(plan_fft, d_bufc, (cufftComplex*)d_bufc);

    // 3. 启动 CTF 校正核函数
    dim3 block(16, 16); // 2D 线程块
    dim3 grid((Nx + block.x - 1) / block.x, (Ny + block.y - 1) / block.y);
    ctfCorrectionKernel<<<grid, block, 0, stream>>>((cufftComplex*)d_bufc, 
                                                     Nx, Ny, ctf, 
                                                     flip_contrast, z_offset);

    // 4. 逆 FFT + 结果处理
    cufftExecC2R(plan_ifft, (cufftComplex*)d_bufc, d_bufc);
    // 归一化等后处理（可再用核函数或直接 CPU 处理）

    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);
    cudaFree(d_bufc);
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

    it=inputPara.find("input_mrc");
    string input_mrc;
    if(it!=inputPara.end())
    {
        input_mrc=path+"/"+it->second;
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
        output_mrc=path+"/"+it->second;
        cout << "Output file name: " << output_mrc << endl;
    }
    else
    {
        cout << "No output file name, set default: tomo.rec" << endl;
        output_mrc="tomo.rec";
    }

    it=inputPara.find("prfx");
    string prfx;
    if(it!=inputPara.end())
    {
        prfx=path+"/"+it->second;
        cout << "Prefix: " << prfx << endl;
    }
    else
    {
        cout << "No prfx, set default: tomo" << endl;
        prfx=path+"/"+"tomo";
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
        input_tlt=path+"/"+it->second;
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
            defocus_file=path+"/"+it->second;
            cout << "Defocus file name: " << defocus_file << endl;
        }
        else
        {
            cout << "No defocus file name, set default: defocus_file.txt" << endl;
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

        if(!skip_3dctf)  {
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

        float * stack_recon1;
        stack_recon1 = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * h * sizeof(float));

        #pragma omp parallel for schedule(static)
        for(int j=0;j<stack_orig_getNy;j++)
        {
            memset(&stack_recon1[j*stack_orig_getNx*h], 0.0, sizeof(float) * stack_orig_getNx * h);
        }

        printf("Start reconstruction:\n");
        float x_orig_offset = float(stack_orig_getNx) * 0.5;
        float z_orig_offset = float(h) * 0.5;

        MRC stack_final(output_mrc.c_str(),"wb");
        stack_final.createMRC_empty(stack_orig_getNx, h, stack_orig_getNy, 2); // (x,z,y)

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
                int thread_nums = omp_get_max_threads();

				float * image_now       = (float *)aligned_alloc(4096, stack_orig_getNx * stack_orig_getNy * sizeof(float));
                int num_z_steps = (h_tilt_max + defocus_step - 1) / defocus_step; // 向上取整
                float * stack_corrected = (float *)aligned_alloc(4096, sizeof(float) * (num_z_steps + 1) * stack_orig_getNx * stack_orig_getNy);
                
				int ttt          = stack_orig_getNx * stack_orig_getNy;
                int Nx_padding11 = int(stack_orig_getNx / 10);
                int Nx_final11   = stack_orig_getNx + Nx_padding11;
                float * bufc12   = (float *)aligned_alloc(4096, sizeof(float) * (Nx_final11 + 2 - (Nx_final11 & 1)) * stack_orig_getNy);
				int chunk        = (stack_orig_getNx + 2 - (stack_orig_getNx & 1)) * stack_orig_getNy;
				float * bufc     = (float *)aligned_alloc(4096, sizeof(float) * (stack_orig_getNx + 2 - (stack_orig_getNx & 1)) * stack_orig_getNy * thread_nums);
	
                fftwf_init_threads();

                size_t ImSize = (size_t)stack_final.getImSize();
		        setvbuf(stack_final.m_fp, (char *)aligned_alloc(64, ImSize * stack_orig_getNy), _IOFBF, ImSize * stack_orig_getNy);

                for(int n=0;n<stack_orig_getNz;n++)   // loop for every micrograph
                {
                    printf("Image %d:\n",n);
                    float theta_rad = theta[n] / 180 * M_PI;
                    stack_orig.read2DIm_32bit(image_now,n);

                    printf("\tPerform 3D-CTF correction!\n\tPerform 3D correction & save corrected stack:\n");
                    int n_zz=0;

                    if(!skip_weighting)
                    {
                          printf("\tStart weighting...\n");
                          fftwf_plan_with_nthreads(omp_get_max_threads());
                          filter_weighting_1D_many1(image_now, bufc12, stack_orig_getNx, stack_orig_getNy, weighting_radial, weighting_sigma);
                          printf("\tDone\n");
                    }

                    printf("\tStart 3D-CTF correction...\n");
                    
                    int tmp_h = int(h_tilt_max >> 1);
		   			fftwf_plan_with_nthreads(1);
                    int cnt11 = (2 * tmp_h) / defocus_step;
					int cnt22 = cnt11 - (cnt11 & 63);

                    #pragma omp parallel
                    {
                        #pragma omp for schedule(static) reduction(+:n_zz) nowait
                        for(int zz=0; zz<cnt22; zz++)    // loop over every height (correct with different defocus)
                        {
                            ctf_correction1(image_now, &stack_corrected[zz * stack_orig_getNx * stack_orig_getNy], &bufc[chunk * omp_get_thread_num()],
                                stack_orig_getNx, stack_orig_getNy, ctf_para[n], flip_contrast, float(zz * defocus_step - tmp_h) + float(defocus_step-1)/2);
                            n_zz++;
                        }
                    }

                    fftwf_plan_with_nthreads(omp_get_max_threads());

					for(int zz = cnt22; zz < cnt11; zz++)    // loop over every height (correct with different defocus)
                    {
						int idx = zz * ttt;
                        ctf_correction3(image_now, &stack_corrected[zz * stack_orig_getNx * stack_orig_getNy], bufc,
                            stack_orig_getNx, stack_orig_getNy, ctf_para[n], flip_contrast, float(zz * defocus_step - tmp_h) + float(defocus_step-1)/2);
                        n_zz++;
                    }

					printf("\tDone!\n\tPerform reconstruction:\n");
                    
                    int cnt = ttt * h;
                    int x_h = stack_orig_getNx * h;

                    float sin1 = sin(theta_rad);
                    float cos1 = cos(theta_rad);

                    #pragma omp parallel
                    {
						#pragma unroll(16)
						#pragma simd
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
                    printf("\tDone\n");
                }
				setvbuf(stack_final.m_fp, (char *)aligned_alloc(4096, ImSize * stack_orig_getNy), _IOFBF, ImSize * stack_orig_getNy);
				fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
                //fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);
				free(bufc);
				free(bufc12);
				free(image_now);
            }
        }

        // write out final result
        printf("Wrtie out final reconstruction result:\n");

        //size_t ImSize = (size_t)stack_final.getImSize();
		//setvbuf(stack_final.m_fp, (char *)aligned_alloc(4096, ImSize * stack_orig_getNy), _IOFBF, ImSize * stack_orig_getNy);
		//fwrite(stack_recon1, 1, ImSize * stack_orig_getNy, stack_final.m_fp);

        // update MRC header
        int threads=3;
        float min_thread[threads],max_thread[threads];
        double mean_thread[threads];

        for(int th=0;th<threads;th++)
        {
            min_thread[th]  = stack_recon1[0];
            max_thread[th]  = stack_recon1[0];
            mean_thread[th] = 0.0;
        }

		float  min_1[64][threads];
        float  max_1[64][threads];
		double mean_1[64][threads];        

        #pragma omp parallel for
		for(int th=0;th<64;th++)
        {
            min_1[th][0]  = stack_recon1[0];
			min_1[th][1]  = stack_recon1[0];
			min_1[th][2]  = stack_recon1[0];
            max_1[th][0]  = stack_recon1[0];
			max_1[th][1]  = stack_recon1[0];
			max_1[th][2]  = stack_recon1[0];
            mean_1[th][0] = 0.0;
			mean_1[th][1] = 0.0;
			mean_1[th][2] = 0.0;
        }

        int tmp_loop2 = stack_orig_getNx * h;
		#pragma omp parallel for
		for(int j=0;j<stack_orig_getNy;j++)
        {
            double mean_now=0.0;
            for(int i=0; i<tmp_loop2; i++)
            {
                mean_now += stack_recon1[j * stack_orig_getNx * h + i];
                if(min_1[omp_get_thread_num()][j%threads]>stack_recon1[j*stack_orig.getNx()*h+i])
                {
                    min_1[omp_get_thread_num()][j%threads]=stack_recon1[j*stack_orig.getNx()*h+i];
                }
                if(max_1[omp_get_thread_num()][j%threads]<stack_recon1[j*stack_orig.getNx()*h+i])
                {
                    max_1[omp_get_thread_num()][j%threads]=stack_recon1[j*stack_orig.getNx()*h+i];
                }
            }
            mean_1[omp_get_thread_num()][j%threads]+=(mean_now/(stack_orig.getNx()*h));
        }

		for(int j=0;j<threads;j++)
        {
            for(int i=0;i<64;i++)
            {
                if(min_1[i][j%threads]<min_thread[j%threads])
                {
                    min_thread[j%threads]=min_1[i][j%threads];
                }
                if(max_1[i][j%threads]>max_thread[j%threads])
                {
                    max_thread[j%threads]=max_1[i][j%threads];
                }
				mean_thread[j%threads]+=mean_1[i][j%threads];
            }
        }

        float min_all=min_thread[0];
        float max_all=max_thread[0];
        double mean_all=0;

        for(int th=0;th<threads;th++)
        {
            mean_all+=mean_thread[th];
            if(min_all>min_thread[th])
            {
                min_all=min_thread[th];
            }
            if(max_all<max_thread[th])
            {
                max_all=max_thread[th];
            }
        }

        mean_all /= stack_orig_getNy;
        stack_final.computeHeader(pix,false,min_all,max_all,float(mean_all));
        stack_final.close();
        printf("Done\n");
    }
    stack_orig.close();
	printf("\nFinish reconstruction successfully!\n");
	printf("All results save in: ");
	cout<<path;
	printf("\n\n");
}

