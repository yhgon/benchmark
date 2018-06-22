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

// build with command  nvcc -lcublas -lcudart -lcurand -arch=sm_70 tensorcore_benchmark_half.cu
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
#define num_TC 8
#define num_FMA 2
#define num_mma 64
#define FP16_OP  num_clock*num_SM*num_TC*num_FMA*num_mma
#define TOTAL_OP  MATRIX_M * MATRIX_N * MATRIX_K * 2
#define TOTAL_OP2 (MATRIX_M*MATRIX_N) * (2*MATRIX_K+2) 

__global__ void convertFp32ToFp16 (half  *out, float *in, int n);
__global__ void convertFp16ToFp32 (float *out, half  *in, int n);

int main(int argc, char* argv[]) {

   printf("FP32 Matrix Memory Size : %f \n",  (float) (sizeof(float) * (float) (MATRIX_M*MATRIX_M)  / ( 1024 * 1024 ) )  );
   printf("FP16 Matrix Memory Size : %f \n",  (float) (sizeof(half)  * (float) (MATRIX_M*MATRIX_M)  / ( 1024 * 1024 ) )  );
   float *a_fp32;
   float *b_fp32;
   float *c_fp32;
   
   half *a_fp16;
   half *b_fp16;
   half *c_fp16;

   float *c_cublas_fp32;
   float *c_host_cublas_fp32;

   half *c_cublas_fp16;
   half *c_host_cublas_fp16;

   printf(" Step1. Initialize GPU API handles...\n");
   curandGenerator_t gen;

   cublasHandle_t cublasHandle;
   cublasCreate(&cublasHandle);
       
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;      
   cudaEventCreate(&startcublas);
   cudaEventCreate(&stopcublas);
   
   // Use tensor cores
   cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

   printf(" Step2. Memory Mallocation ...\n");
   cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float));
   cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float));

   cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
   cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));
   cudaMalloc((void**)&c_fp16, MATRIX_M * MATRIX_N * sizeof(half));

   cudaMalloc((void**)&c_cublas_fp32, MATRIX_M * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_cublas_fp16, MATRIX_M * MATRIX_N * sizeof(half));

   c_host_cublas_fp32 = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   printf(" Step3. Data init with cuRAND ...\n");
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, 1337ULL);

   curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K);
   curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N);
   curandGenerateUniform(gen, c_fp32, MATRIX_M * MATRIX_N);
   cudaMemcpy(c_cublas_fp32, c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice);
   
   printf(" Stsep4. convert FP32 to FP16 for FP16 benchmark...\n");
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_fp16, c_fp32, MATRIX_M * MATRIX_N);


   curandDestroyGenerator(gen);

   half alpha = 2.0f;
   half beta = 2.0f;

   printf(" Step5. Ready to Run...\n");
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // Now using cuBLAS

   printf("warm up...");
   cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, MATRIX_M,
                b_fp16, MATRIX_K,
                &beta, 
                c_fp16, MATRIX_M);

   printf(" Step6.  Running with cuBLAS...\n");
   cudaEventRecord(startcublas);
   cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, MATRIX_M,
                b_fp16,  MATRIX_K,
                &beta, 
                c_cublas_fp16, MATRIX_M);
   cudaEventRecord(stopcublas);

   // Error checking
   printf(" Step7. Download results...\n");
   convertFp16ToFp32 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_cublas_fp32, c_cublas_fp16, MATRIX_M * MATRIX_N);

   cudaMemcpy( c_host_cublas_fp32, c_cublas_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
   float cublasTime;
   cudaEventSynchronize(stopcublas);
   cudaEventElapsedTime(&cublasTime, startcublas, stopcublas);

   printf("cublas took %fms", cublasTime);

   printf(" with Operation  %.2f\n", (double) TOTAL_OP );

   printf(" RPeak FP16 TFLOPS: %.2f with max clock  %d Mhz \n",      (double) FP16_OP /(1000000) , num_clock ); 
   printf("  RMax FP16 TFLOPS   %.2f\n",      (double) ( ((double)TOTAL_OP / (double) (1000000) ) / ((double) cublasTime)/1000 ) ); 

   cudaEventDestroy(startcublas);             
   cudaEventDestroy(stopcublas);
   
   cudaFree(a_fp32);
   cudaFree(b_fp32);
   cudaFree(c_fp32);
   cudaFree(a_fp16);
   cudaFree(b_fp16);
   cudaFree(c_fp16);

   cudaFree(c_cublas_fp32);
   cudaFree(c_cublas_fp16);

   
   free(c_host_cublas_fp32);

   cudaDeviceReset();
   return 0;
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}


