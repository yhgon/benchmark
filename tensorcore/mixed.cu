// build with command  nvcc -lcublas -lcudart -lcurand -arch=sm_70 mixed.cu
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
__global__ void convertFp32ToFp16 (half *out, float *in, int n);

int main(int argc, char* argv[]) {

   printf("FP32 Matrix Memory Size : %f \n",  (float) (sizeof(float) * (float) (MATRIX_M*MATRIX_M)  / ( 1024 * 1024 ) )  );
   printf("FP16 Matrix Memory Size : %f \n",  (float) (sizeof(half)  * (float) (MATRIX_M*MATRIX_M)  / ( 1024 * 1024 ) )  );
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_host_cublas;

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
   cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
   cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));

   cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   printf(" Step3. Data init with cuRAND ...\n");
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, 1337ULL);

   curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K);
   curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N);
   curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N);
   cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice);
   
   printf(" Step4. convert FP32 to FP16 for FP16 benchmark...\n");
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   curandDestroyGenerator(gen);

   float alpha = 2.0f;
   float beta = 2.0f;

   printf(" Step5. Ready to Run...\n");
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // Now using cuBLAS
   printf("warm up...");
   cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);

   printf(" Step6.  Running with cuBLAS...\n");
   cudaEventRecord(startcublas);
   cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
   cudaEventRecord(stopcublas);

   printf(" Step7. Download results...\n");
   cudaMemcpy( c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
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
   cudaFree(a_fp16);
   cudaFree(b_fp16);

   cudaFree(c);
   cudaFree(c_cublas);
   
   free(c_host_cublas);

   cudaDeviceReset();
   return 0;
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
