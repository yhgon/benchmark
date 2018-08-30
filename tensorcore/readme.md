# introduction 
this code use cuBLAS API in CUDA 9.2 with 396.26  ( with CUDA 9.0 only two benchmark is available) 
Half Precision Benchmark use cublasHgemm API in cuBLAS

the application clock is fixed in the max application clock for tesla V100. plz modify if you want to test Tesla P100 or other GPUs.

## FP16
```
cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
             MATRIX_M, MATRIX_N, MATRIX_K,
             &alpha,
             a_fp16, MATRIX_M,
             b_fp16,  MATRIX_K,
             &beta,
             c_cublas_fp16, MATRIX_M);
```

Mixed Precision Benchmark use cublasDgemmEx  API in cuBLAS

```
cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
             MATRIX_M, MATRIX_N, MATRIX_K,
             &alpha,
             a_fp16, CUDA_R_16F, MATRIX_M,
             b_fp16, CUDA_R_16F, MATRIX_K,
             &beta,
             c_cublas, CUDA_R_32F, MATRIX_M,
             CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
```

before compile the source code, modify  matrix size manually `#define SIZE 8192  //  4096 8192 10240 16384 24576 `  in line#40 [half_precision](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed.cu#L40)


in line#99  benchmark code use curand API to generate random matrix [L99](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed.cu#L99)

in line#121  benchmark code warm up the GPU [L121](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed.cu#L121)

moreover, for best performance, use CUDA 9.2 and recent cublas patch. from [nvidia dev site](http://developer.nvidia.com) You also could use docker images from [docker hub](https://hub.docker.com/r/nvidia/cuda/tags/) 

this code don't include comparing the result from CPU to save benchmark time.  


below script show how to compile and run the benchmark.  Moreover, the theoratical number is base on Volta SMX2 16GB with boost clock 

```
module load cuda/9.2.88.1
nvcc -lcublas -lcurand -lcudart -arch=sm_70 mixed.cu -o mixed-8k
nvcc -lcublas -lcurand -lcudart -arch=sm_70 half.cu -o half-8k
 
nvidia-smi -ac 877,1530
./mixed-8k | grep RMax
 ./half-8k | grep RMax
```

##  GEMM
For double precision and single precision, single code do benchmark. 

```
module load cuda/9.2.88.1
nvcc -lcublas -lcurand -lcudart -arch=sm_70 gemm.cu -o gemm-8k

nvidia-smi -ac 877,1530
./gemm-8k | grep SGEMM
./gemm-8k | grep DGEMM
```

##  FP32 
For single precision benchmark,  you need to configure matrix size M,N,K for simplicity, use 8192

```
nvcc -lcublas -lcurand -lcudart -arch=sm_70 sgemm.cu -o sgemm

nvidia-smi -ac 877,1530

./sgemm 8192 8192 8192 | grep SGEMM

SGEMM cublas took 79.954941ms  with   1099511627776.000000 OP clock 1530 Mhz 
FP32 RPeak: 15.67 TFLOPS SGEMM : 13.75 TFLOPS

./sgemm 8192 8192 8192
FP32 Matrix Memory Size A 8192x8192 : 256.0 MB   
FP32 Matrix Memory Size B 8192x8192 : 256.0 MB   
FP32 Matrix Memory Size C 8192x8192 : 256.0 MB   
 Step1. Initialize GPU API handles...
 Step2. Memory Mallocation ...
 Step3. Data init with cuRAND ...
 Step5. Ready to Run...

M = 8192, N = 8192, K = 8192. alpha = 2.000000, beta = 2.000000

 Step6. warm up...
 Step7.  Running with cuBLAS... sgemm


SGEMM cublas took 80.276482ms  with   1099511627776.000000 OP clock 1530 Mhz 
FP32 RPeak: 15.67 TFLOPS SGEMM : 13.70 TFLOPS
Ratio of Real/Theoretic 0.87 



```

## Malloc test 
for malloc test, 
complie with `nvcc ./malloc_test.cu -o malloc_test` and launch with starting memory size(MB) and increment size(MB) with command `./malloc_test 1024 4 ` which means starting 1024MB increasing 4 MB for each iteration. 

```
below is example of usage 
/*
$ export CUDA_VISIBLE_DEVICES=2;./malloc_test 
 plz use below command 
 ./malloc_test 1024 10  
 to  malloc 1024MB and increment would be 10MB 

11GB 11264MB
15GB 15360MB 
23GB 23552MB
31GB 31744MB

current free memory is 15721.0

~$ export CUDA_VISIBLE_DEVICES=2;./malloc_test 15710 1
------------------------------------------------------------------
	Total(MB)=	Free(MB)+	init(MB)+	Alloc(MB)
0	16149.9    =	15721.0+    	428.9     	 <------  initial used memory 
------------------------------------------------------------------
0	16149.9    =	13.0+    	428.9+    	15708.0 
1	16149.9    =	11.0+    	428.9+    	15709.0 
2	16149.9    =	11.0+    	428.9+    	15710.0 
3	16149.9    =	9.0+    	428.9+    	15711.0 
4	16149.9    =	9.0+    	428.9+    	15712.0 
5	16149.9    =	7.0+    	428.9+    	15713.0 
6	16149.9    =	7.0+    	428.9+    	15714.0 
7	16149.9    =	5.0+    	428.9+    	15715.0 
8	16149.9    =	5.0+    	428.9+    	15716.0 
9	16149.9    =	3.0+    	428.9+    	15717.0 
10	16149.9    =	3.0+    	428.9+    	15718.0 
11	16149.9    =	15721.0+    	428.9+    	15719.0 
couldn't allocate 15719.0 MB Err : out of memory

```

