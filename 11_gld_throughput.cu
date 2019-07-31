/*
   Program name: gld_throughput.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418 
    
    Purpose: Program to demonstrate global memory efficiency

    Description: A simple vector addition kernel is written which performs strided access to arrays. 
                 Because of different offset values global memory load effeciency varies. The data 
                 type is float (single precision) hence each thread request 4 bytes of data. 
                 Depends on L1 cache size and different values of offset observe global memory 
                 load efficiency. Use following profiling command

                 1.  nvprof --devices 0 --metrics gld_transactions ./a.out
                     This metrics returns total number of global memory load transactions
                 2. nvprof --devices 0 --metrics gld_efficiency ./a.out
                     This metrics returns global memory load efficiency

    To understand Memory coalesing refer following linksW
    Link1: https://www.youtube.com/watch?v=mLxZyWOI340
    Link2: https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/
    Link3: https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved

*/

#include<stdio.h>
#include<stdlib.h>

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

   __global__ void testKernel(float *d_A, float *d_B, float *d_C, int N, int offset)
   {
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     int k = gid + offset;
     if(k < N)
     {
       d_C[gid] = d_A[k] + d_B[k];
     }

   }
 void initData(float *h_A, float *h_B, float *hostRef, float *gpuRef, int N)
 {
   for(int i = 0; i < N; i++)
   {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
    hostRef[i] = 0.0;
    gpuRef[i] = 0.0;
   }
 }

 void hostFunc(float *h_A, float *h_B, float *hostRef, int N, int offset)
 { 
   int idx;
   for(int k = 0, idx = offset; idx < N; idx++,k++)
   {
     hostRef[k] = h_A[idx] + h_B[idx];
   }
 }

  int compareResults(float *hostRef,float *gpuRef, int N)
  {
   for(int i = 0; i < N; i++)
   {
    if(hostRef[i] != gpuRef[i])
    {
     return 1;
    }
   }
   return 0;
  }  
 int main()
 {
  int N;
  N = 2 << 20; // # elements in arrays
  size_t nBytes = N * sizeof(float); // Size reqiored to stoare arrays

  int offset;// To determine the stride size in kernel
  printf("\n Enter offset value: ");
  scanf("%d",&offset);

  /********************* Memory allocation at host side ****************************/
  float *h_A = (float *)malloc(nBytes); // Host side input array h_A
  float *h_B = (float *)malloc(nBytes); // Host side input array h_B
  float *hostRef = (float *)malloc(nBytes); // Host side reference output array hostRef
  float *gpuRef = (float *)malloc(nBytes);  // Device side reference output array gpuRef

  initData(h_A,h_B,hostRef, gpuRef, N); // Data initialization function
  
  hostFunc(h_A, h_B, hostRef, N, offset);
  /********************* Memory allocation at device side ****************************/

  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc((void **)&d_A, nBytes)); // Device side input array d_A
  CHECK(cudaMalloc((void **)&d_B, nBytes)); // Device side input array d_B
  CHECK(cudaMalloc((void **)&d_C, nBytes)); // Device side output array d_C

  /********************* Data transfer from host to device ****************************/
  
  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
  
  /******************************* Kernel Launch **************************************/

  int numT, numB;
  numT = 32; // # threads per block
  numB = ceil(N/(float)numT); // # blocks

  testKernel<<<numB,numT>>>(d_A, d_B, d_C, N, offset); // Kernel to check global memory efficiency

  /********************* Data transfer from host to device ****************************/
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)); // Transfer data from device to host  
  
  int check = compareResults(hostRef,gpuRef, N);  
  if(check!= 0)
  {
    printf("\n ALERT!!! CPU and GPU side results are not matching!!!");
  }

  /******************************* Free device and Host Memories ********************************/
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return(0);
 }
