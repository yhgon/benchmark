/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// build with command  nvcc -lcublas -lcudart -lcurand -arch=sm_70 mixed_precision.cu
// use max clock 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

// Must be multiples of 16 to fit TensorCore
#define SIZE 8192  //  4096 8192 10240 16384 24576
#define MATRIX_M SIZE 
#define MATRIX_N SIZE
#define MATRIX_K SIZE

#define num_clock 1530
#define num_SM 80
#define num_CUDA 64
#define num_FMA 2
#define num_DP_ratio 2
#define FP32_OP  num_clock*num_SM*num_CUDA*num_FMA
#define FP64_OP  FP32_OP / num_DP_ratio
#define TOTAL_OP  MATRIX_M * MATRIX_N * MATRIX_K * 2
#define TOTAL_OP2 (MATRIX_M*MATRIX_N) * (2*MATRIX_K+2) 



int main(int argc, char* argv[]) {

   printf("FP64 Matrix Memory Size : %f \n",  (float) (sizeof(double) * (float) (MATRIX_M*MATRIX_M)  / ( 1024 * 1024 ) )  );

   double *a_fp64;
   double *b_fp64;
   double *c_fp64;

	 float *a_fp32;
	 float *b_fp32;
	 float *c_fp32;

   double *c_cublas_fp64;
	 float *c_cublas_fp32;
   double *c_host_cublas_fp64; // for error tolorence 
	float *c_host_cublas_fp32; // for error tolorence 

   printf(" Step1. Initialize GPU API handles...\n");
   curandGenerator_t gen;

   cublasHandle_t cublasHandle;
   cublasCreate(&cublasHandle);
       
   cudaEvent_t startcublas_fp64;
	 cudaEvent_t startcublas_fp32;
	 cudaEvent_t stopcublas_fp64;   
   cudaEvent_t stopcublas_fp32;      

   cudaEventCreate(&startcublas_fp64);
 	 cudaEventCreate(&startcublas_fp32);
   cudaEventCreate(&stopcublas_fp64);
	 cudaEventCreate(&stopcublas_fp32);
   
   // Use tensor cores
//   cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

   printf(" Step2. Memory Mallocation ...\n");
   cudaMalloc((void**)&a_fp64, MATRIX_M * MATRIX_K * sizeof(double));
   cudaMalloc((void**)&b_fp64, MATRIX_K * MATRIX_N * sizeof(double));
	 cudaMalloc((void**)&c_fp64, MATRIX_M * MATRIX_N * sizeof(double));
	 
   cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float));
   cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float));
	 
   cudaMalloc((void**)&c_cublas_fp64, MATRIX_M * MATRIX_N * sizeof(double));
	 cudaMalloc((void**)&c_cublas_fp32, MATRIX_M * MATRIX_N * sizeof(float));

   c_host_cublas_fp64 = (double*)malloc(MATRIX_M * MATRIX_N * sizeof(double));
   c_host_cublas_fp32 = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   printf(" Step3. Data init with cuRAND ...\n");
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, 1337ULL);

	 curandGenerateUniformDouble(gen, a_fp64, MATRIX_M * MATRIX_K);
	 curandGenerateUniformDouble(gen, b_fp64, MATRIX_K * MATRIX_N);
	 curandGenerateUniformDouble(gen, c_fp64, MATRIX_M * MATRIX_N);

   curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K);
   curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N);
   curandGenerateUniform(gen, c_fp32, MATRIX_M * MATRIX_N);

	 cudaMemcpy(c_cublas_fp64, c_fp64, MATRIX_M * MATRIX_N * sizeof(double), cudaMemcpyDeviceToDevice);
   cudaMemcpy(c_cublas_fp32, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice);

	 
   curandDestroyGenerator(gen);

   double alpha_fp64 = 2.0f;
   double  beta_fp64 = 2.0f;

	 float alpha_fp32 = 2.0f;
	 float  beta_fp32 = 2.0f;

   printf(" Step5. Ready to Run...\n");
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha_fp64, beta_fp64);

   // Now using cuBLAS
   printf("warm up...");
	 cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
		           MATRIX_M, MATRIX_N, MATRIX_K, 
							  &alpha_fp32, 
								a_fp32, MATRIX_M, 
								b_fp32, MATRIX_K, 
								&beta_fp32, 
								c_fp32, MATRIX_M);
								
	 printf(" Step6.  Running with cuBLAS... sgemm\n");
	 cudaEventRecord(startcublas_fp32);
	 cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
		           MATRIX_M, MATRIX_N, MATRIX_K, 
							  &alpha_fp32, 
								a_fp32, MATRIX_M, 
								b_fp32, MATRIX_K, 
								&beta_fp32, 
								c_fp32, MATRIX_M);
	 cudaEventRecord(stopcublas_fp32);							
   cudaEventSynchronize(stopcublas_fp32);

	 
   printf(" Step6.  Running with cuBLAS...dgemm\n");
   cudaEventRecord(startcublas_fp64);
	 cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
		           MATRIX_M, MATRIX_N, MATRIX_K, 
							  &alpha_fp64, 
								a_fp64, MATRIX_M, 
								b_fp64, MATRIX_K, 
								&beta_fp64, 
								c_cublas_fp64, MATRIX_M);
   cudaEventRecord(stopcublas_fp64);
   cudaEventSynchronize(stopcublas_fp64);
	 
   printf(" Step7. Download results...\n");
   cudaMemcpy( c_host_cublas_fp64, c_cublas_fp64, MATRIX_M * MATRIX_N * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy( c_host_cublas_fp32, c_cublas_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
	 
	 float cublasTime_fp32, cublasTime_fp64;
	 cudaEventElapsedTime(&cublasTime_fp32, startcublas_fp32, stopcublas_fp32);
   cudaEventElapsedTime(&cublasTime_fp64, startcublas_fp64, stopcublas_fp64);

   printf("SGEMM cublas took %fms\n", cublasTime_fp32);
	 printf("DGEMM cublas took %fms  ", cublasTime_fp64);
   printf(" with   %f Operation and clock %d Mhz \n", (double) TOTAL_OP,  num_clock );

   printf(" FP64 RPeak: %f TFLOPS ",      (double) FP64_OP /(1000000)  );
	 printf("  DGEMM  :  %.2f TFLOPS \n",      (double) ( ((double)TOTAL_OP / (double) (1000000) ) / ((double) cublasTime_fp64)/1000 ) ); 
	 printf(" FP32 RPeak: %.2f TFLOPS  ",      (double) FP32_OP /(1000000)   );  
   printf("  SGEMM  :  %.2f TFLOPS\n",      (double) ( ((double)TOTAL_OP / (double) (1000000) ) / ((double) cublasTime_fp32)/1000 ) ); 

   cudaEventDestroy(startcublas_fp64);
   cudaEventDestroy(startcublas_fp32);
   cudaEventDestroy(stopcublas_fp64);             
   cudaEventDestroy(stopcublas_fp32);
   
   cudaFree(a_fp64);
   cudaFree(b_fp64);
	 cudaFree(c_fp64);
	 
   cudaFree(a_fp32);
   cudaFree(b_fp32);
   cudaFree(c_fp32);
	 
   cudaFree(c_cublas_fp64);
   cudaFree(c_cublas_fp32);
   free(c_host_cublas_fp64);
	 free(c_host_cublas_fp32);

   cudaDeviceReset();
   return 0;
}
