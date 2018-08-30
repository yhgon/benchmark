// build with command  nvcc -lcublas -lcudart -lcurand -arch=sm_70 dgemm.cu -o dgemm
// use max application clock  with nvidia-smi -ac 877,1530 for Tesla V100


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


   printf("FP64 Matrix Memory Size A %dx%d : %.1f MB   \n", MATRIX_M, MATRIX_K,   (float) (sizeof(double) * (float) (MATRIX_M*MATRIX_K)  / ( 1024 * 1024 ) )  );
   printf("FP64 Matrix Memory Size B %dx%d : %.1f MB   \n", MATRIX_K, MATRIX_N,    (float) (sizeof(double) * (float) (MATRIX_K*MATRIX_N)  / ( 1024 * 1024 ) )  );
   printf("FP64 Matrix Memory Size C %dx%d : %.1f MB   \n", MATRIX_M, MATRIX_K,    (float) (sizeof(double) * (float) (MATRIX_M*MATRIX_N)  / ( 1024 * 1024 ) )  );

	
   double *a_fp64;
   double *b_fp64;
   double *c_fp64;

   printf(" Step1. Initialize GPU API handles...\n");
   curandGenerator_t gen;

   cublasHandle_t cublasHandle;
   cublasCreate(&cublasHandle);
       
   cudaEvent_t startcublas_fp64;
   cudaEvent_t stopcublas_fp64;      

   cudaEventCreate(&startcublas_fp64);
   cudaEventCreate(&stopcublas_fp64);
   
   // Use tensor cores
   // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

   printf(" Step2. Memory Mallocation ...\n");
	 
   cudaMalloc((void**)&a_fp64, MATRIX_M * MATRIX_K * sizeof(double));
   cudaMalloc((void**)&b_fp64, MATRIX_K * MATRIX_N * sizeof(double));
   cudaMalloc((void**)&c_fp64, MATRIX_M * MATRIX_N * sizeof(double));
	 


   printf(" Step3. Data init with cuRAND ...\n");
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, 1337ULL);

   curandGenerateUniformDouble(gen, a_fp64, MATRIX_M * MATRIX_K);
   curandGenerateUniformDouble(gen, b_fp64, MATRIX_K * MATRIX_N);
   curandGenerateUniformDouble(gen, c_fp64, MATRIX_M * MATRIX_N);
	 
   curandDestroyGenerator(gen);

   double alpha_fp64 = 2.0f;
   double  beta_fp64 = 2.0f;

   printf(" Step5. Ready to Run...\n");
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha_fp64, beta_fp64);

   // Now using cuBLAS
   printf(" Step6. warm up...\n");
   cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
               MATRIX_M, MATRIX_N, MATRIX_K, 
  	       &alpha_fp64, 
	       a_fp64, MATRIX_M, 
	       b_fp64, MATRIX_K, 
	       &beta_fp64, 
	       c_fp64, MATRIX_M);
								
    printf(" Step7.  Running with cuBLAS... sgemm\n");
    cudaEventRecord(startcublas_fp64);
    cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
	        &alpha_fp64, 
		a_fp64, MATRIX_M, 
		b_fp64, MATRIX_K, 
		&beta_fp64, 
		c_fp64, MATRIX_M);
   cudaEventRecord(stopcublas_fp64);							
   cudaEventSynchronize(stopcublas_fp64);

	 
	 
   float cublasTime_fp64 ;
   cudaEventElapsedTime(&cublasTime_fp64, startcublas_fp64, stopcublas_fp64);

   printf("\n\nDGEMM cublas took %fms ", cublasTime_fp64);
   printf(" with   %f OP clock %d Mhz \n", (double) TOTAL_OP,  num_clock );

  double RPEAK = (double) FP64_OP /(1000000) ;
  double RMAX = (double) ( ((double)TOTAL_OP / (double) (1000000) ) / ((double) cublasTime_fp64)/1000 );

   printf("FP64 RPeak: %.2f TFLOPS",  RPEAK );  
   printf(" DGEMM : %.2f TFLOPS\n",   RMAX  ); 

   printf("Ratio of Real/Theoretic %.2f   \n",  RMAX/RPEAK );



   cudaEventDestroy(startcublas_fp64);
   cudaEventDestroy(stopcublas_fp64);
   
	 
   cudaFree(a_fp64);
   cudaFree(b_fp64);
   cudaFree(c_fp64);

   cudaDeviceReset();
   return 0;
}
