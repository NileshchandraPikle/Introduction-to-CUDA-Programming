/*   Program name: HelloGPU.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418  
         Webpage: https://piklenileshchandra.wixsite.com/personal
*/

#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
#define SIZE 16777216

#define CHECK(call)                                                     \
{									\
  const cudaError_t error = call;					\
  if(error != cudaSuccess)						\
  {									\
    printf("Error %s %d", __FILE__, __LINE__);				\
    printf("\n Code %d Reason %s \n",error, cudaGetErrorString(error));	\
    exit(1);								\
  }									\
}	
 __global__ void reductionAtomicFloatKernel(float *d_a, float *d_sum)
 {
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   if(gid < SIZE)
   {
    atomicAdd(d_sum, d_a[gid]); 
   }
  
 }
 int main()
 {
   int i;
   float *h_a = (float *)malloc(SIZE * sizeof(float));
   float *h_sum = (float *)malloc(sizeof(float));
   for(i = 0; i < SIZE; i++)
   {
     h_a[i] = 1.0;
   }
   float sum = 0.0;
   h_sum[0] = 0.0;
   clock_t start, end;
   double cpu_time_used;
   start = clock();
   for(i = 0; i < SIZE; i++)
   {
     sum += h_a[i];
   }
   end = clock();
   cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
   printf("\n Sequential execution time = %f\n",cpu_time_used);
   
   
   float *d_a, *d_sum;
   CHECK(cudaMalloc((void **)&d_a, SIZE*sizeof(float)));
   CHECK(cudaMalloc((void **)&d_sum, sizeof(float)));
   CHECK(cudaMemcpy(d_a, h_a, SIZE*sizeof(float), cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_sum, h_sum, sizeof(float), cudaMemcpyHostToDevice));
   
   int numB, numT;
   numT = 128;
   numB = SIZE/(float)numT + 1;
   CHECK(cudaMalloc((void **)&d_sum, sizeof(float)));

  float milliseconds = 0; 
  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);

   reductionAtomicFloatKernel<<<numB, numT>>>(d_a, d_sum);
   
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);  
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  printf("\n Parallel execution time using AtomicAdd float = %f\n",milliseconds*0.001);

   CHECK(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));  

   printf("\n h_sum = %f",*h_sum);
  
   free(h_a);
   cudaFree(d_a);
   cudaFree(d_sum);

   return(0);
 }
