/*
compile : nvcc malloc_test.cu -o malloc_test
exec : ./malloc_test 512 1
starting 512MB , increase 1MB 
if you have multiple GPU, use below
export CUDA_VISIBLE_DEVICES=3;./malloc_test 512 1

MB convert table would be : 
GB	MB
1	1024
2	2048
3	3072
4	4096
5	5120
6	6144
7	7168
8	8192
9	9216
10	10240
11	11264
12	12288
13	13312
14	14336
15	15360
16	16384
17	17408
18	18432
19	19456
20	20480
21	21504
22	22528
23	23552
24	24576
25	25600
26	26624
27	27648
28	28672
29	29696
30	30720
31	31744
32	32768
*/

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

	int *devPtr= NULL;
	size_t mem_size, free, total , start, inc, size ;
	float free_f, total_f, used_f, used_start_f, used_now_f;

        cudaMemGetInfo(&free,&total);
        free_f  = float(free)   / (1024*1024) ; 
        total_f = float(total)  / (1024*1024) ;
        used_f =  total_f-free_f ;
	used_start_f  = used_f;

	if(argc<3){
	printf(" plz use below command \n ./malloc_test 1024 10  \n to  malloc 1024MB and increment would be 10MB \n");
	printf("\n11GB 11264MB\n15GB 15360MB \n23GB 23552MB\n31GB 31744MB\n");
	printf("\ncurrent free memory is %.1f\n", free_f);

	return 0;
	}

        start =  atoi(argv[1]) / sizeof(int) ;
        inc = atoi(argv[2])  ;
        size = start;
	
        printf("------------------------------------------------------------------\n");
        printf("\tTotal(MB)=\tFree(MB)+\tinit(MB)+\tAlloc(MB)\n"); 
        printf("0\t%.1f    =\t%.1f+    \t%.1f     \t <------  initial used memory \n", total_f,free_f, used_f );
        printf("------------------------------------------------------------------\n");
	int i = 0;

	do {
		
		mem_size = sizeof(int) * size * (1024*1024) + (inc*i) * (1024*1024) ; 
		cudaMalloc(&devPtr, mem_size ); // MB
                cudaMemGetInfo(&free,&total);
	        free_f  = float(free)   / (1024*1024) ;
        	total_f = float(total)  / (1024*1024) ;
	        used_f =  total_f-free_f ;
		used_now_f = (float)mem_size/(1024*1024)   ; 
        printf("%d\t%.1f    =\t%.1f+    \t%.1f+    \t%.1f \n", i, total_f, free_f, used_start_f, used_now_f);
		if(devPtr == NULL) {
			printf("couldn't allocate %.1f MB ", used_now_f);
			printf("Err : %s\n", cudaGetErrorString(cudaGetLastError()) );
			return 0;	
		} else {
			//printf("Allocated %d int's.\n", int(size));
		}
		cudaFree(devPtr);
		size = (size* sizeof(int) + inc )/sizeof(int) ;  
		mem_size = sizeof(int) * size  ;
		
		i=i+1;
	} while(1);

}
