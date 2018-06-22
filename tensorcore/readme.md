# introduction 
this code use cuBLAS API in CUDA 9.2 with 396.26
Half Precision Benchmark use cublasHgemm API in cuBLAS

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

before compile the source code, modify  matrix size manually `#define SIZE 8192  //  4096 8192 10240 16384 24576 `  in line#40 [half_precision](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed_precision.cu#L40)


in line#99  benchmark code use curand API to generate random matrix [L99](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed_precision.cu#L99)

in line#121  benchmark code warm up the GPU [L121](https://github.com/yhgon/benchmark/blob/master/tensorcore/mixed_precision.cu#L121)

moreover, for best performance, use CUDA 9.2 and recent cublas patch. from [nvidia dev site](http://developer.nvidia.com) You also could use docker images from [docker hub](https://hub.docker.com/r/nvidia/cuda/tags/) 

this code don't include comparing the result. 

below script show how to compile and run the benchmark. 
```
module load cuda/9.2.88.1
 
nvcc -lcublas -lcurand -lcudart -arch=sm_70 mixed_precision.cu -o mixed-8k
 
nvcc -lcublas -lcurand -lcudart -arch=sm_70 half_precision.cu -o half-8k
 
nvidia-smi -ac 877,1530
 
./mixed-8k | grep RMax
 
./half-8k | grep RMax
```

this code is my initial draft code. if you find any bug, inform me
