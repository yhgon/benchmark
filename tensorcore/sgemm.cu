// build with command  nvcc -lcublas -lcudart -lcurand -arch=sm_70 gemm.cu
// use max clock 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {

int size_M, size_N, size_K ;

if(argc<4){
	printf(" plz use matrix size M, N K with command  ./sgemm  8192 8192 8192  \n");
	printf(" will run default value M=N=K=8192\n\n\n");
	size_M=8192;
	size_K=8192;
	size_N=8192;

	} else{

	size_M =  atoi(argv[1]) ;
	size_K =  atoi(argv[2]) ;
	size_N =  atoi(argv[3]) ;
}


#define SIZE size  //  4096 8192 10240 16384 24576
#define MATRIX_M size_M
#define MATRIX_N size_N
#define MATRIX_K size_K

#define num_clock 1530 // V100 16GB SMX
#define num_SM 80
#define num_CUDA 64
#define num_FMA 2
#define num_DP_ratio 2
#define FP32_OP  num_clock*num_SM*num_CUDA*num_FMA
#define FP64_OP  FP32_OP / num_DP_ratio
#define TOTAL_OP  MATRIX_M * MATRIX_N * MATRIX_K * 2
#define TOTAL_OP2 (MATRIX_M*MATRIX_N) * (2*MATRIX_K+2) 


   printf("FP32 Matrix Memory Size A %dx%d : %.1f MB   \n", MATRIX_M, MATRIX_K,   (float) (sizeof(float) * (float) (MATRIX_M*MATRIX_K)  / ( 1024 * 1024 ) )  );
   printf("FP32 Matrix Memory Size B %dx%d : %.1f MB   \n", MATRIX_K, MATRIX_N,    (float) (sizeof(float) * (float) (MATRIX_K*MATRIX_N)  / ( 1024 * 1024 ) )  );
   printf("FP32 Matrix Memory Size C %dx%d : %.1f MB   \n", MATRIX_M, MATRIX_K,    (float) (sizeof(float) * (float) (MATRIX_M*MATRIX_N)  / ( 1024 * 1024 ) )  );

	
   float *a_fp32;
   float *b_fp32;
   float *c_fp32;

   printf(" Step1. Initialize GPU API handles...\n");
   curandGenerator_t gen;

   cublasHandle_t cublasHandle;
   cublasCreate(&cublasHandle);
       
   cudaEvent_t startcublas_fp32;
   cudaEvent_t stopcublas_fp32;      

   cudaEventCreate(&startcublas_fp32);
   cudaEventCreate(&stopcublas_fp32);
   
   // Use tensor cores
   // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

   printf(" Step2. Memory Mallocation ...\n");
	 
   cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float));
   cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float));
	 


   printf(" Step3. Data init with cuRAND ...\n");
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, 1337ULL);

   curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K);
   curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N);
   curandGenerateUniform(gen, c_fp32, MATRIX_M * MATRIX_N);
	 
   curandDestroyGenerator(gen);

   float alpha_fp32 = 2.0f;
   float  beta_fp32 = 2.0f;

   printf(" Step5. Ready to Run...\n");
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha_fp32, beta_fp32);

   // Now using cuBLAS
   printf(" Step6. warm up...\n");
   cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
               MATRIX_M, MATRIX_N, MATRIX_K, 
  	       &alpha_fp32, 
	       a_fp32, MATRIX_M, 
	       b_fp32, MATRIX_K, 
	       &beta_fp32, 
	       c_fp32, MATRIX_M);
								
    printf(" Step7.  Running with cuBLAS... sgemm\n");
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

	 
	 
   float cublasTime_fp32 ;
   cudaEventElapsedTime(&cublasTime_fp32, startcublas_fp32, stopcublas_fp32);

   printf("\n\nSGEMM cublas took %fms ", cublasTime_fp32);
   printf(" with   %f OP clock %d Mhz \n", (double) TOTAL_OP,  num_clock );

  double RPEAK = (double) FP32_OP /(1000000) ;
  double RMAX = (double) ( ((double)TOTAL_OP / (double) (1000000) ) / ((double) cublasTime_fp32)/1000 );

   printf("FP32 RPeak: %.2f TFLOPS",  RPEAK );  
   printf(" SGEMM : %.2f TFLOPS\n",   RMAX  ); 

   printf("Ratio of Real/Theoretic %.2f   \n",  RMAX/RPEAK );



   cudaEventDestroy(startcublas_fp32);
   cudaEventDestroy(stopcublas_fp32);
   
	 
   cudaFree(a_fp32);
   cudaFree(b_fp32);
   cudaFree(c_fp32);

   cudaDeviceReset();
   return 0;
}
